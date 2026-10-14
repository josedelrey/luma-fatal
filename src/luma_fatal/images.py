"""Still-image file adapter and reproducibility records."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from PIL import Image

from . import __version__
from .configuration import Config, resolve_config
from .pipeline import apply_filter
from .processing import working_size, enlargement_sizes
from .image_io import save_image

STATUS = "Experimental gradient-domain processing, exact end-to-end equivalence not established"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: Config | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Process one image file and record its resolved settings and fingerprints.

    config_path loads settings when config is omitted. When both are supplied,
    it records the file underlying an already resolved configuration, including
    CLI overrides. The resolved settings always appear in the output record.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    config_path = Path(config_path) if config_path is not None else None
    config_source = (
        dict(path=str(config_path), sha256=_digest(config_path))
        if config_path is not None
        else None
    )
    if config is None and config_path is not None:
        config = Config.load(config_path)
    else:
        config = resolve_config(config)
    if config.filter == "both":
        raise ValueError("process_image requires one filter per output")
    variant = int(config.filter[-1])
    params = config.parameters
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ to preserve the original")
    suffixes = (".jpg", ".jpeg") if config.output.format == "jpeg" else (".png",)
    if output_path.suffix.lower() not in suffixes:
        raise ValueError(
            f"Output extension must be {' or '.join(suffixes)} for {config.output.format}. "
            "Use --output-format to select a different encoding"
        )
    start = time.perf_counter()
    with Image.open(input_path) as original:
        source_dimensions = list(original.size)
        source_icc = original.info.get("icc_profile")
        original_mode = original.mode
        source_orientation = original.getexif().get(274)
        result = apply_filter(original, variant, config=config)
        output_icc = (
            result.info.get("icc_profile")
            if config.image.color_space == "srgb"
            else (source_icc if original_mode in ("RGB", "RGBA") else None)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path, config.output, output_icc)
    work_size = working_size(result.size, config.processing.effective_max_edge)
    record = dict(
        implementation_version=__version__,
        fidelity_status=STATUS,
        filter=f"Fattal {variant}",
        parameters=asdict(params),
        runtime=dict(
            python=platform.python_version(),
            platform=platform.system(),
            architecture=platform.machine(),
            dependencies={
                name: version(name)
                for name in ("numpy", "scipy", "Pillow", "numba", "PyYAML")
            },
        ),
        processing_profile=config.processing.profile_name,
        configuration=config.to_dict(),
        resolved_processing=dict(
            max_edge=config.processing.effective_max_edge,
            max_input_pixels=config.processing.effective_max_input_pixels,
            prepared_dimensions=list(result.size),
            effect_applied=params.fade < 100,
            working_dimensions=list(work_size) if params.fade < 100 else None,
            enlargement_stages=[
                list(size) for size in enlargement_sizes(work_size, result.size)
            ]
            if params.fade < 100
            else [],
        ),
        source_dimensions=source_dimensions,
        source_mode=original_mode,
        source_exif_orientation=source_orientation,
        source_icc_sha256=hashlib.sha256(source_icc).hexdigest()
        if source_icc
        else None,
        output_format=config.output.format,
        input_decode=result.info.get("_input_decode"),
        color_conversion=result.info.get("_color_conversion"),
        implementation_sources_sha256={
            name: _digest(Path(__file__).with_name(name))
            for name in (
                "core.py",
                "pipeline.py",
                "numerics.py",
                "processing.py",
                "configuration.py",
                "image_io.py",
                "color.py",
                "images.py",
            )
        },
        source=str(input_path),
        source_sha256=_digest(input_path),
        output=str(output_path),
        output_sha256=_digest(output_path),
        dimensions=list(result.size),
        seconds=time.perf_counter() - start,
        color_policy=asdict(config.image),
    )
    if config_source is not None:
        record["config_source"] = config_source
    _write_json(output_path.with_suffix(".meta.json"), record)
    return record
