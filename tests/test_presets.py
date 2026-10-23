"""Complete image presets remain reproducible with local Python dependencies."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from luma_fatal import configuration
from luma_fatal.cli import main as cli_main
from scripts import check_presets


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_presets_apply_full_effect(path):
    assert configuration.Config.load(path).parameters.fade == 0


def test_frozen_recipes_are_complete_and_independent_of_device_defaults(monkeypatch):
    _, presets = check_presets.load_contract()
    assert [entry["name"] for entry, _ in presets] == ["Fattal 1", "Fattal 2"]
    monkeypatch.setitem(configuration.DEVICE_MAX_EDGES, "medium", 2048)
    monkeypatch.setitem(configuration.DEVICE_INPUT_PIXELS, "medium", 12_000_000)
    for _, config in presets:
        assert config.parameters.fade == 0
        assert config.processing.effective_max_edge == 1536
        assert config.processing.effective_max_input_pixels == 8_000_000
        assert config.image.color_backend == "matrix_trc"
        assert config.output.jpeg_background is None
        assert config.video.deblock == "off"
        assert config.video.deband == "off"
        assert config.video.flat_cleanup == 0


@pytest.mark.parametrize("variant", [1, 2])
def test_compressed_presets_enable_smooth_shadows_debanding_and_cleanup(variant):
    root = Path(__file__).resolve().parents[1]
    reference = configuration.Config.load(root / "configs" / f"fattal-{variant}.yaml")
    compressed = configuration.Config.load(
        root / "configs" / f"fattal-{variant}-compressed.yaml"
    )
    assert compressed == replace(
        reference,
        numerics=replace(reference.numerics, shadow_softness=6),
        video=replace(reference.video, deband="gentle", denoise=1.0, flat_cleanup=0.75),
    )


def test_bundled_regressions_run_without_external_commands(monkeypatch, tmp_path):

    def forbidden(*args, **kwargs):
        raise AssertionError("Preset regression must not execute an external command")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    output = tmp_path / "results"
    check_presets.main(["--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["all_matched"]
    assert len(report["records"]) == 2
    assert (tmp_path / "results/report.json").is_file()
    for row in report["records"]:
        with Image.open(row["output"]) as image:
            assert image.format == "JPEG"
    contract, _ = check_presets.load_contract()
    assert report["selected_samples"] == [entry["id"] for entry in contract["samples"]]
    for entry in contract["samples"]:
        assert (check_presets.ROOT / entry["source"]).is_relative_to(
            check_presets.ROOT / "tests/fixtures"
        )
        assert (
            check_presets.digest(check_presets.ROOT / entry["source"])
            == entry["source_sha256"]
        )


@pytest.mark.parametrize("variant", [1, 2])
def test_cli_applies_frozen_recipe_without_fade_or_codec_overrides(
    variant, tmp_path, capsys
):
    contract, presets = check_presets.load_contract()
    sample = next(
        entry for entry in contract["samples"] if entry["id"] == "p3-noise"
    )
    entry, config = presets[variant - 1]
    output = tmp_path / f"fattal-{variant}.jpg"
    cli_main(
        [
            "apply",
            str(check_presets.ROOT / sample["source"]),
            str(output),
            "--config",
            str(check_presets.ROOT / entry["config"]),
        ]
    )
    expected = sample["expected"][config.filter]
    assert check_presets.digest(output) == expected["encoded_sha256"]
    with Image.open(output) as exported:
        assert (
            hashlib.sha256(exported.tobytes()).hexdigest()
            == expected["decoded_rgb_sha256"]
        )
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["configuration"] == entry["configuration"]
    assert record["parameters"]["fade"] == 0
    assert record["resolved_processing"]["max_input_pixels"] == 8_000_000


def test_changed_fixture_is_rejected_before_processing_or_writing(monkeypatch, tmp_path):
    contract, presets = check_presets.load_contract()
    contract = deepcopy(contract)
    source = tmp_path / "different.jpg"
    Image.new("RGB", (16, 16), "red").save(source)
    contract["samples"][0]["source"] = str(source)
    monkeypatch.setattr(check_presets, "load_contract", lambda: (contract, presets))
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="input changed: p3-noise"):
        check_presets.check(output=output)
    assert not output.exists()


def test_checker_reports_regression_without_refreshing_expected_values(
    monkeypatch, tmp_path
):
    before = check_presets.digest(check_presets.CONTRACT)

    def altered(source, config):
        return {"rgb_sha256": "altered"}, b"not-an-export"

    monkeypatch.setattr(check_presets, "snapshot", altered)
    report = check_presets.check(samples=("p3-noise",))
    assert not report["all_matched"]
    assert all("rgb_sha256" in row["mismatched_fields"] for row in report["records"])
    with pytest.raises(SystemExit) as result:
        check_presets.main(["--sample", "p3-noise"])
    assert result.value.code == 1
    assert check_presets.digest(check_presets.CONTRACT) == before


def test_semantic_recipe_drift_is_detected(tmp_path):
    contract = json.loads(check_presets.CONTRACT.read_text())
    changed = deepcopy(contract)
    changed["presets"][0]["configuration"]["parameters"]["fade"] = 50
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration changed: Fattal 1"):
        check_presets.load_contract(path)


def test_unknown_sample_is_rejected_before_writing(tmp_path):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="Unknown regression samples"):
        check_presets.check(output=output, samples=("missing",))
    assert not output.exists()


def test_missing_fixture_is_reported_without_skipping_baselines(
    monkeypatch, tmp_path
):
    contract, presets = check_presets.load_contract()
    contract = deepcopy(contract)
    contract["samples"][0]["source"] = str(tmp_path / "missing.jpg")
    monkeypatch.setattr(check_presets, "load_contract", lambda: (contract, presets))
    with pytest.raises(ValueError, match="Missing regression input"):
        check_presets.check()
