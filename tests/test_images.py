"""Image file adapters preserve sources and reject aliased destinations."""

import hashlib
import os

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, OutputConfig
from luma_fatal.cli import main
from luma_fatal.images import process_image


def make_input(path):
    pixels = np.random.default_rng(3095).integers(0, 256, (32, 40, 3), np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def assert_rejected(source, destination, entry_point, capsys):
    paths = (source, destination, destination.with_suffix(".meta.json"))
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    if entry_point == "api":
        config = Config(parameters=FilterParams(fade=0), output=OutputConfig(format="png"))
        with pytest.raises(ValueError, match="paths must differ"):
            process_image(source, destination, config=config)
    else:
        with pytest.raises(SystemExit) as error:
            main(["apply", str(source), str(destination), "--output-format", "png",
                  "--fade", "0"])
        assert error.value.code == 2
        assert "paths must differ" in capsys.readouterr().err
    for path, original in before.items():
        if original is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == original


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("pair", ["input_output", "input_sidecar", "output_sidecar"])
@pytest.mark.parametrize("link_kind", ["hard", "symbolic"])
def test_image_paths_reject_file_aliases(tmp_path, capsys, entry_point, pair, link_kind):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    sidecar = destination.with_suffix(".meta.json")
    make_input(source)
    if pair == "input_output":
        existing, alias = source, destination
    elif pair == "input_sidecar":
        existing, alias = source, sidecar
    else:
        make_input(destination)
        existing, alias = destination, sidecar
    if link_kind == "hard":
        os.link(existing, alias)
    else:
        try:
            alias.symlink_to(existing.name)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"Symbolic links are unavailable: {error}")
    assert_rejected(source, destination, entry_point, capsys)


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("target", ["output", "sidecar"])
def test_image_paths_reject_direct_source_collisions(tmp_path, capsys, entry_point, target):
    destination = tmp_path / "output.png"
    source = destination if target == "output" else destination.with_suffix(".meta.json")
    make_input(source)
    assert_rejected(source, destination, entry_point, capsys)


def test_identical_independent_files_can_be_processed(tmp_path):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()
    destination.write_bytes(original)
    config = Config(parameters=FilterParams(fade=0), output=OutputConfig(format="png"))
    record = process_image(source, destination, config=config)
    assert source.read_bytes() == original
    assert destination.read_bytes() != original
    assert record["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert record["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
