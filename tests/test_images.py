"""Image file adapters preserve sources and reject aliased destinations."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, OutputConfig
from luma_fatal.cli import main
import luma_fatal.images as image_adapter
from luma_fatal.images import process_image


def make_input(path):
    pixels = np.random.default_rng(3095).integers(0, 256, (32, 40, 3), np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def run_image(source, destination, entry_point):
    if entry_point == "cli":
        main(["apply", str(source), str(destination), "--output-format", "png",
              "--fade", "100"])
    else:
        process_image(
            source, destination,
            config=Config(parameters=FilterParams(fade=100), output=OutputConfig(format="png")),
        )


def assert_no_artifacts(source, original, destination):
    assert source.read_bytes() == original
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(destination.parent.glob(".luma-fatal-*"))


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


def test_existing_independent_file_is_preserved(tmp_path):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()
    destination.write_bytes(original)
    config = Config(parameters=FilterParams(fade=0), output=OutputConfig(format="png"))
    with pytest.raises(ValueError, match="already exists"):
        process_image(source, destination, config=config)
    assert source.read_bytes() == original
    assert destination.read_bytes() == original
    assert not destination.with_suffix(".meta.json").exists()


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("target", ["output", "sidecar"])
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_destinations_are_preserved(tmp_path, entry_point, target, kind):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()
    sidecar = destination.with_suffix(".meta.json")
    existing = destination if target == "output" else sidecar
    if kind == "file":
        existing.write_bytes(b"existing data")
    else:
        existing.mkdir()
    with pytest.raises(SystemExit if entry_point == "cli" else ValueError):
        run_image(source, destination, entry_point)
    assert source.read_bytes() == original
    if kind == "file":
        assert existing.read_bytes() == b"existing data"
    else:
        assert existing.is_dir()
        assert not list(existing.iterdir())
    other = sidecar if target == "output" else destination
    assert not other.exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.parametrize("target", ["output", "sidecar"])
def test_dangling_destination_links_are_preserved(tmp_path, target):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    existing = destination if target == "output" else destination.with_suffix(".meta.json")
    try:
        existing.symlink_to("missing.png")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Symbolic links are unavailable: {error}")
    with pytest.raises(ValueError, match="already exists"):
        run_image(source, destination, "api")
    assert existing.is_symlink()
    assert not (tmp_path / "missing.png").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("stage", ["encoding", "metadata"])
def test_staging_failures_publish_nothing(tmp_path, monkeypatch, entry_point, stage):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()

    def fail_encoding(image, path, *args):
        path.write_bytes(b"partial image")
        raise OSError("Encoding interrupted")

    def fail_metadata(path, record):
        path.write_text("partial metadata")
        raise OSError("Metadata interrupted")

    if stage == "encoding":
        monkeypatch.setattr(image_adapter, "save_image", fail_encoding)
    else:
        monkeypatch.setattr(image_adapter, "_write_json", fail_metadata)
    with pytest.raises(SystemExit if entry_point == "cli" else OSError):
        run_image(source, destination, entry_point)
    assert_no_artifacts(source, original, destination)


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("copy_number", [1, 2])
def test_publication_failures_roll_back_created_files(
    tmp_path, monkeypatch, entry_point, failure, copy_number
):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()
    original_copy = image_adapter.shutil.copyfileobj
    copies = 0

    def fail_copy(complete, output, **kwargs):
        nonlocal copies
        copies += 1
        if copies == copy_number:
            output.write(b"partial publication")
            raise failure("Publication interrupted")
        return original_copy(complete, output, **kwargs)

    monkeypatch.setattr(image_adapter.shutil, "copyfileobj", fail_copy)
    expected = SystemExit if entry_point == "cli" and failure is OSError else failure
    with pytest.raises(expected):
        run_image(source, destination, entry_point)
    assert_no_artifacts(source, original, destination)


@pytest.mark.parametrize("entry_point", ["api", "cli"])
@pytest.mark.parametrize("target", ["output", "sidecar"])
def test_publication_races_preserve_other_writer(tmp_path, monkeypatch, entry_point, target):
    source, destination = tmp_path / "input.png", tmp_path / "output.png"
    make_input(source)
    original = source.read_bytes()
    sidecar = destination.with_suffix(".meta.json")
    competing = destination if target == "output" else sidecar
    original_open = Path.open

    def competing_writer(path, mode="r", *args, **kwargs):
        if path == competing and mode == "xb":
            with original_open(path, "wb") as other:
                other.write(b"other writer")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_writer)
    with pytest.raises(SystemExit if entry_point == "cli" else FileExistsError):
        run_image(source, destination, entry_point)
    assert source.read_bytes() == original
    assert competing.read_bytes() == b"other writer"
    other = sidecar if target == "output" else destination
    assert not other.exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))
