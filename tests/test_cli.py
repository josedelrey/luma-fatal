import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from luma_fatal.cli import main


def make_input(path):
    Image.fromarray(
        np.random.default_rng(3095).integers(0, 256, (32, 40, 3), np.uint8)
    ).save(path)


def test_apply_preserves_source_and_records_provenance(tmp_path, capsys):
    source, destination = tmp_path / "input.png", tmp_path / "output.jpg"
    make_input(source)
    original = source.read_bytes()
    main(["apply", str(source), str(destination), "--filter", "fattal2", "--fade", "0"])
    metadata = json.loads(destination.with_suffix(".meta.json").read_text())
    assert metadata["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert (
        metadata["output_sha256"]
        == hashlib.sha256(destination.read_bytes()).hexdigest()
    )
    assert metadata["filter"] == "Fattal 2"
    assert metadata["parameters"]["fade"] == 0
    assert metadata["dimensions"] == [40, 32]
    assert source.read_bytes() == original
    assert "exact end-to-end equivalence not established" in metadata["fidelity_status"]
    assert metadata["runtime"]["dependencies"]["numpy"]
    assert {"core.py", "pipeline.py", "configuration.py"} <= metadata[
        "implementation_sources_sha256"
    ].keys()


def test_evaluate_accepts_original_jpeg_reference(tmp_path, capsys):
    inputs, references = tmp_path / "inputs", tmp_path / "references/fattal_1"
    inputs.mkdir()
    references.mkdir(parents=True)
    make_input(inputs / "image.png")
    Image.open(inputs / "image.png").save(
        references / "image.jpg", quality=100, subsampling=2
    )
    output = tmp_path / "results"
    main(
        ["evaluate", str(inputs), str(references.parent), str(output), "--fade", "100"]
    )
    report = json.loads((output / "evaluation.json").read_text())
    assert report["compared_count"] == 1
    assert report["missing_count"] == 1
    row = next(row for row in report["records"] if row["status"] == "compared")
    assert row["metrics"]["reference"].endswith("image.jpg")
    assert (
        row["metrics"]["reference_sha256"]
        == hashlib.sha256((references / "image.jpg").read_bytes()).hexdigest()
    )


def test_evaluate_rejects_ambiguous_reference_formats(tmp_path, capsys):
    inputs, references = tmp_path / "inputs", tmp_path / "references/fattal_1"
    inputs.mkdir()
    references.mkdir(parents=True)
    make_input(inputs / "image.png")
    make_input(references / "image.png")
    Image.open(inputs / "image.png").save(references / "image.jpg")
    with pytest.raises(SystemExit) as error:
        main(
            ["evaluate", str(inputs), str(references.parent), str(tmp_path / "results")]
        )
    assert error.value.code == 2
    assert "Ambiguous reference" in capsys.readouterr().err


def test_apply_rejects_overwriting_input(tmp_path, capsys):
    source = tmp_path / "original.png"
    make_input(source)
    original = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["apply", str(source), str(source)])
    assert error.value.code == 2
    assert source.read_bytes() == original


def test_evaluate_reports_missing_references(tmp_path, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    make_input(inputs / "image.png")
    output = tmp_path / "results"
    main(["evaluate", str(inputs), str(tmp_path / "references"), str(output)])
    report = json.loads((output / "evaluation.json").read_text())
    assert report["compared_count"] == 0
    assert report["missing_count"] == 2
    assert {row["status"] for row in report["records"]} == {"missing_reference"}
    assert "mean_image_mae" not in report


def test_compare_identical_images_serializes_infinite_psnr(tmp_path, capsys):
    source = tmp_path / "input.png"
    make_input(source)
    output = tmp_path / "comparison"
    main(
        [
            "compare",
            str(source),
            str(source),
            "--original",
            str(source),
            "--output",
            str(output),
        ]
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["mae"] == 0
    assert report["psnr_db"] is None and report["psnr_infinite"] is True
    assert (output / "differences.png").exists()
