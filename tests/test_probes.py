import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from luma_fatal.probes import generate_probes


def test_probe_manifest_is_complete_and_reproducible(tmp_path):
    first = generate_probes(tmp_path / "first", size=256, seed=17)
    second = generate_probes(tmp_path / "second", size=256, seed=17)
    assert first == second
    assert len(first["probes"]) >= 24
    names = {entry["name"] for entry in first["probes"]}
    assert {
        "ramp_gray_horizontal",
        "rgb_cube",
        "impulses_black",
        "checker_multiscale",
        "noise_rgb",
        "test_scene",
    } <= names
    assert json.loads((tmp_path / "first" / "manifest.json").read_text()) == first
    for entry in first["probes"]:
        path = tmp_path / "first" / entry["file"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            assert image.size == (256, 256)
            assert image.mode == "RGB"
            assert "icc_profile" not in image.info
            assert "exif" not in image.info


def test_probe_pixels_encode_known_signals(tmp_path):
    generate_probes(tmp_path, size=256)
    with Image.open(tmp_path / "ramp_gray_horizontal.png") as image:
        ramp = np.asarray(image)
    assert np.array_equal(ramp[0, :, 0], np.arange(256, dtype=np.uint8))
    assert np.array_equal(ramp[0], ramp[-1])
    with Image.open(tmp_path / "impulses_black.png") as image:
        impulses = np.asarray(image)
    assert np.array_equal(impulses[128, 128], [255, 255, 255])
    assert np.count_nonzero(impulses) == 6
    with Image.open(tmp_path / "flat_gray_128.png") as image:
        assert np.all(np.asarray(image) == 128)
    with Image.open(tmp_path / "rgb_cube.png") as image:
        cube = np.asarray(image)
    assert len(np.unique(cube.reshape(-1, 3), axis=0)) == 256


def test_seed_changes_noise_but_preserves_analytic_probes(tmp_path):
    first = generate_probes(tmp_path / "first", size=256, seed=1)
    second = generate_probes(tmp_path / "second", size=256, seed=2)
    first_hashes = {entry["name"]: entry["sha256"] for entry in first["probes"]}
    second_hashes = {entry["name"]: entry["sha256"] for entry in second["probes"]}
    assert first_hashes["noise_rgb"] != second_hashes["noise_rgb"]
    assert first_hashes["noise_gray"] != second_hashes["noise_gray"]
    assert first_hashes["test_scene"] == second_hashes["test_scene"]


@pytest.mark.parametrize("size", [255, 0, True, 256.0])
def test_small_or_noninteger_probe_size_fails(tmp_path, size):
    with pytest.raises(ValueError, match="at least 256"):
        generate_probes(tmp_path, size=size)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_fails(tmp_path, seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        generate_probes(tmp_path, seed=seed)
