"""Immutable settings shared by image adapters and decoded-frame processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
import json
import math
from numbers import Real
from pathlib import Path
import tomllib
import re

import numpy as np
import yaml


class _ConfigLoader(yaml.SafeLoader):
    """Read YAML with explicit booleans and reject ambiguous duplicate keys."""

    yaml_implicit_resolvers = {
        key: [
            (tag, pattern)
            for tag, pattern in values
            if tag not in ("tag:yaml.org,2002:bool", "tag:yaml.org,2002:timestamp")
        ]
        for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        values = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            if key in values:
                raise ValueError(f"Duplicate configuration key: {key}")
            values[key] = self.construct_object(value_node, deep=deep)
        return values


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false|True|False|TRUE|FALSE)$"),
    list("tTfF"),
)
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)

DEVICE_MAX_EDGES = {"low": 1024, "medium": 1536, "high": 2048}
DEVICE_INPUT_PIXELS = {"low": 4_000_000, "medium": 8_000_000, "high": 12_000_000}

VIDEO_COLOR_PROFILES = {
    "bt709": ("bt709", "bt709", "bt709"),
    "bt601-pal": ("bt470bg", "bt709", "bt470bg"),
    "bt601-ntsc": ("smpte170m", "smpte170m", "smpte170m"),
    "srgb": ("bt709", "iec61966-2-1", "bt709"),
    "bt2020-pq": ("bt2020nc", "smpte2084", "bt2020"),
    "bt2020-hlg": ("bt2020nc", "arib-std-b67", "bt2020"),
}


def _unique_json_object(pairs):
    """Reject repeated JSON keys before they can overwrite earlier settings."""
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate configuration key: {key}")
        values[key] = value
    return values


def _choice(name, value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")


def _integer(name, value, minimum, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {limit}")


def _finite(name, value, minimum, maximum, *, exclusive_min=False, exclusive_max=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
        or (exclusive_min and value == minimum)
        or (exclusive_max and value == maximum)
    ):
        raise ValueError(f"{name} must be finite and in the permitted range")


@dataclass(frozen=True)
class NumericsConfig:
    """Experimental numerical controls with deterministic reference defaults."""

    iterations: int = 80
    relaxation: float = 1.9
    pyramid_min_size: int = 32
    normalization_low: float = 0.001
    normalization_high: float = 0.995
    normalization_floor: float = 0.0001
    gradient_threshold: float = 0.0001
    log_offset: float = 0.01
    shadow_threshold: int = 6
    detail_level: int = 0
    power_mode: str = "approximate"
    shadow_softness: int = 0
    luminance_mode: str = "rounded"

    def __post_init__(self):
        _integer("iterations", self.iterations, 1)
        _finite(
            "relaxation", self.relaxation, 0, 2, exclusive_min=True, exclusive_max=True
        )
        with np.errstate(under="ignore"):
            old_weight = np.float32(1 - self.relaxation)
            new_weight = np.float32(self.relaxation)
        if not (-1 < old_weight < 1 and 0 < new_weight < 2):
            raise ValueError(
                "relaxation must keep float32 solver weights strictly inside "
                "(-1, 1) and (0, 2)"
            )
        _integer("pyramid_min_size", self.pyramid_min_size, 2)
        _finite("normalization_low", self.normalization_low, 0, 1, exclusive_max=True)
        _finite("normalization_high", self.normalization_high, 0, 1, exclusive_max=True)
        if self.normalization_low >= self.normalization_high:
            raise ValueError("normalization_low must be below normalization_high")
        for name in ("normalization_floor", "gradient_threshold", "log_offset"):
            _finite(name, getattr(self, name), 0, 1, exclusive_min=True)
            if float(np.float32(getattr(self, name))) == 0:
                raise ValueError(f"{name} must remain positive in float32")
        _integer("shadow_threshold", self.shadow_threshold, 0, 255)
        _integer("shadow_softness", self.shadow_softness, 0, 255)
        _integer("detail_level", self.detail_level, 0)
        _choice("power_mode", self.power_mode, ("approximate", "standard"))
        _choice("luminance_mode", self.luminance_mode, ("rounded", "fractional"))


@dataclass(frozen=True)
class FilterParams:
    """Effect controls on a 0..100 scale, with a positive output gamma."""

    alpha: float = 1
    beta: float = 80
    saturation: float = 100
    noise: float = 0
    fade: float = 0
    output_gamma: float = 1.0
    tone_strength: float = 100
    highlight_compression: float = 0
    black_level: float = 0
    white_level: float = 100
    vibrance: float = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "output_gamma":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 <= value <= 100
            ):
                raise ValueError(f"{name} must be finite and between 0 and 100")
        if self.alpha == 0:
            raise ValueError("alpha must be greater than zero")
        with np.errstate(under="ignore"):
            scaled_alpha = np.float32(self.alpha) / np.float32(100)
        if scaled_alpha == 0:
            raise ValueError("alpha must remain positive in float32 after division by 100")
        if self.black_level >= self.white_level:
            raise ValueError("black_level must be below white_level")
        _finite("output_gamma", self.output_gamma, 0, math.inf, exclusive_min=True)
        if not math.isfinite(1.0 / self.output_gamma):
            raise ValueError("output_gamma must have a finite reciprocal")


@dataclass(frozen=True)
class ProcessingConfig:
    """Input and working-size limits, transparency, and pixel blending."""

    device_class: str = "medium"
    max_edge: int | None = None
    max_input_pixels: int | None = None
    limit_input_pixels: bool = True
    resize: bool = True
    transparency: str = "reject"
    blend_mode: str = "normal"
    working_resampler: str = "nearest"

    def __post_init__(self):
        _choice("device_class", self.device_class, DEVICE_MAX_EDGES)
        _choice(
            "working_resampler", self.working_resampler, ("nearest", "bilinear", "bicubic")
        )
        if self.max_edge is not None:
            _integer("max_edge", self.max_edge, 2)
        if self.max_input_pixels is not None:
            _integer("max_input_pixels", self.max_input_pixels, 4)
        if not isinstance(self.limit_input_pixels, bool):
            raise ValueError("limit_input_pixels must be a boolean")
        if self.max_input_pixels is not None and (
            not self.resize or not self.limit_input_pixels
        ):
            raise ValueError(
                "max_input_pixels requires resize=true and limit_input_pixels=true"
            )
        if not isinstance(self.resize, bool):
            raise ValueError("resize must be a boolean")
        if not self.resize and self.max_edge is not None:
            raise ValueError("max_edge cannot be combined with resize=false")
        _choice("transparency", self.transparency, ("reject", "straight"))
        if self.blend_mode != "normal":
            raise ValueError('Only blend_mode="normal" is supported')

    @property
    def effective_max_edge(self):
        if not self.resize:
            return None
        return (
            self.max_edge
            if self.max_edge is not None
            else DEVICE_MAX_EDGES[self.device_class]
        )

    @property
    def effective_max_input_pixels(self):
        if not self.resize or not self.limit_input_pixels:
            return None
        return (
            self.max_input_pixels
            if self.max_input_pixels is not None
            else DEVICE_INPUT_PIXELS[self.device_class]
        )

    @property
    def profile_name(self):
        return "core" if not self.resize else f"bounded-{self.effective_max_edge}"


@dataclass(frozen=True)
class ImageConfig:
    """Color conversion, EXIF orientation, and decode sampling."""

    color_space: str = "srgb"
    color_backend: str = "matrix_trc"
    exif_orientation: str = "apply"
    decode_sampling: bool = True

    def __post_init__(self):
        _choice("color_space", self.color_space, ("encoded", "srgb"))
        _choice("color_backend", self.color_backend, ("matrix_trc", "lcms"))
        _choice("exif_orientation", self.exif_orientation, ("ignore", "apply"))
        if not isinstance(self.decode_sampling, bool):
            raise ValueError("decode_sampling must be a boolean")


@dataclass(frozen=True)
class OutputConfig:
    """File encoding. JPEG options do not affect in-memory filter outputs."""

    format: str = "jpeg"
    jpeg_quality: int = 100
    jpeg_subsampling: str = "420"
    embed_icc: bool = False
    jpeg_background: tuple[int, int, int] | None = None

    def __post_init__(self):
        _choice("output.format", self.format, ("png", "jpeg"))
        _integer("jpeg_quality", self.jpeg_quality, 1, 100)
        _choice("jpeg_subsampling", self.jpeg_subsampling, ("444", "422", "420"))
        if not isinstance(self.embed_icc, bool):
            raise ValueError("embed_icc must be a boolean")
        if self.jpeg_background is not None:
            if (
                not isinstance(self.jpeg_background, (tuple, list))
                or len(self.jpeg_background) != 3
            ):
                raise ValueError("jpeg_background must contain three RGB integers")
            for value in self.jpeg_background:
                _integer("jpeg_background channel", value, 0, 255)
            object.__setattr__(self, "jpeg_background", tuple(self.jpeg_background))

    @property
    def suffix(self):
        return ".png" if self.format == "png" else ".jpg"


@dataclass(frozen=True)
class VideoConfig:
    """Video preprocessing and H.264 export settings."""

    crf: int = 18
    preset: str = "medium"
    deblock: str = "off"
    deband: str = "off"
    flat_cleanup: float = 0.0
    flat_cleanup_mode: str = "standard"
    flat_cleanup_coarse_strength: float = 1.0
    flat_cleanup_sigma: float = 1.2
    pre_noise_rms: float = 0.0
    pre_noise_kind: str = "highpass"
    pre_noise_masked: bool = True
    pre_noise_seed: int = 20261017
    frame_rate: str = "auto"
    input_color: str = "auto"
    input_range: str = "auto"
    encoder: str = "libx264"
    nvenc_cq: int = 18
    nvenc_preset: str = "p4"
    denoise: float = 0.0
    filter_backend: str = "cpu"
    calibration_cache: bool = False
    gpu_iterations: int | None = None
    lighting_scale: float = 1.0
    decode_prefetch: int = 0
    pinned_frames: bool = False
    gpu_buffer_reuse: bool = False
    solver_halo: int = 96
    pipeline_depth: int = 0
    gpu_cleanup_graph: bool = False
    validation_mode: str = "scan"

    def __post_init__(self):
        _choice("video filter_backend", self.filter_backend, ("cpu", "auto"))
        if not isinstance(self.calibration_cache, bool):
            raise ValueError("video calibration_cache must be a boolean")
        if self.gpu_iterations is not None:
            _integer("video gpu_iterations", self.gpu_iterations, 1)
        _finite("video lighting_scale", self.lighting_scale, 0.25, 1.0)
        _integer("video decode_prefetch", self.decode_prefetch, 0, 16)
        if not isinstance(self.pinned_frames, bool):
            raise ValueError("video pinned_frames must be a boolean")
        if not isinstance(self.gpu_buffer_reuse, bool):
            raise ValueError("video gpu_buffer_reuse must be a boolean")
        if not isinstance(self.gpu_cleanup_graph, bool):
            raise ValueError("video gpu_cleanup_graph must be a boolean")
        _integer("video solver_halo", self.solver_halo, 32, 96)
        _integer("video pipeline_depth", self.pipeline_depth, 0, 8)
        _choice("video validation_mode", self.validation_mode, ("scan", "stream"))
        _choice("video encoder", self.encoder, ("libx264", "h264_nvenc"))
        _integer("video nvenc_cq", self.nvenc_cq, 0, 51)
        _choice("video nvenc_preset", self.nvenc_preset,
                ("p1", "p2", "p3", "p4", "p5", "p6", "p7"))
        if type(self.crf) is not int or not 0 <= self.crf <= 51:
            raise ValueError("Video crf must be an integer between 0 and 51")
        if self.preset not in (
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ):
            raise ValueError("Unsupported H.264 preset")
        if self.deblock not in ("off", "weak4", "weak8"):
            raise ValueError("Video deblock must be off, weak4 or weak8")
        if self.deband not in ("off", "gentle"):
            raise ValueError("Video deband must be off or gentle")
        _finite("video denoise", self.denoise, 0, 1)
        _finite("video flat_cleanup", self.flat_cleanup, 0, 1)
        _choice("video flat_cleanup_mode", self.flat_cleanup_mode,
                ("standard", "multiscale"))
        _finite("video flat_cleanup_coarse_strength",
                self.flat_cleanup_coarse_strength, 0, 1)
        _finite("video flat_cleanup_sigma", self.flat_cleanup_sigma, 0.01, 16)
        _finite("video pre_noise_rms", self.pre_noise_rms, 0, 4)
        _choice("video pre_noise_kind", self.pre_noise_kind, ("highpass", "fbm"))
        if not isinstance(self.pre_noise_masked, bool):
            raise ValueError("video pre_noise_masked must be a boolean")
        _integer("video pre_noise_seed", self.pre_noise_seed, 0)
        _choice("video input_color", self.input_color, ("auto", *VIDEO_COLOR_PROFILES))
        _choice("video input_range", self.input_range, ("auto", "limited", "full"))
        if self.frame_rate != "auto":
            try:
                if not isinstance(self.frame_rate, str) or Fraction(self.frame_rate) <= 0:
                    raise ValueError
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError("Video frame_rate must be auto or a positive rate string") from error
            object.__setattr__(self, "frame_rate", str(Fraction(self.frame_rate)))


@dataclass(frozen=True)
class Config:
    """Validated settings for filtering, input preparation, and export."""

    filter: str = "fattal1"
    parameters: FilterParams = field(default_factory=FilterParams)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    schema_version: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported configuration schema_version")
        _choice("filter", self.filter, ("fattal1", "fattal2", "both"))
        for name, expected in (
            ("parameters", FilterParams),
            ("processing", ProcessingConfig),
            ("image", ImageConfig),
            ("output", OutputConfig),
            ("numerics", NumericsConfig),
            ("video", VideoConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a {expected.__name__}")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Configuration must be an object with named sections")
        types = {
            "parameters": FilterParams,
            "processing": ProcessingConfig,
            "image": ImageConfig,
            "output": OutputConfig,
            "numerics": NumericsConfig,
            "video": VideoConfig,
        }
        if any(not isinstance(key, str) for key in data):
            raise ValueError("Configuration keys must be strings")
        unknown = set(data) - set(types) - {"filter", "schema_version"}
        if unknown:
            raise ValueError(
                f"Unknown configuration sections: {', '.join(sorted(unknown))}"
            )
        values = {key: data[key] for key in ("filter", "schema_version") if key in data}
        for name, section in data.items():
            if name in ("filter", "schema_version"):
                continue
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be an object")
            if any(not isinstance(key, str) for key in section):
                raise ValueError(f"{name} keys must be strings")
            unknown = set(section) - {item.name for item in fields(types[name])}
            if unknown:
                raise ValueError(
                    f"Unknown {name} options: {', '.join(sorted(unknown))}"
                )
            values[name] = types[name](**section)
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(cls._load_data(Path(path), ()))

    @classmethod
    def source_paths(cls, path: str | Path) -> tuple[Path, ...]:
        """List a configuration and its inherited files in resolution order."""
        sources = []
        cls._load_data(Path(path), (), sources)
        return tuple(sources)

    @classmethod
    def _load_data(
        cls, path: Path, ancestors: tuple[Path, ...], sources: list[Path] | None = None
    ):
        resolved = path.resolve()
        if resolved in ancestors:
            raise ValueError(f"Configuration inheritance cycle at {path}")
        path = Path(path)
        try:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
            elif path.suffix.lower() == ".toml":
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".json":
                data = json.loads(
                    path.read_text(encoding="utf-8"),
                    object_pairs_hook=_unique_json_object,
                )
            else:
                raise ValueError(
                    "Configuration files must use .yaml, .yml, .toml or .json"
                )
        except (yaml.YAMLError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid configuration file {path}: {error}") from error
        if sources is not None:
            sources.append(resolved)
        if not isinstance(data, dict):
            return data
        own = data.copy()
        if "extends" not in own:
            return own
        parent_name = own.pop("extends")
        if not isinstance(parent_name, str) or not parent_name.strip():
            raise ValueError("extends must name a configuration file")
        parent = cls._load_data(
            path.parent / parent_name, (*ancestors, resolved), sources
        )
        if not isinstance(parent, dict):
            raise ValueError("Inherited configuration must be an object")
        merged = parent.copy()
        for name, value in own.items():
            if name in ("parameters", "processing", "image", "output", "numerics", "video"):
                inherited = parent.get(name, {})
                if not isinstance(value, dict) or not isinstance(inherited, dict):
                    raise ValueError(f"Inherited {name} section must be an object")
                merged[name] = {**inherited, **value}
            else:
                merged[name] = value
        return merged

    def to_dict(self):
        return asdict(self)

    def to_yaml(self):
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def to_toml(self):
        lines = [
            f"schema_version = {self.schema_version}",
            f"filter = {json.dumps(self.filter)}",
            "",
        ]
        for name, section in self.to_dict().items():
            if name in ("filter", "schema_version"):
                continue
            lines.append(f"[{name}]")
            for key, value in section.items():
                if value is not None:
                    # All values here have equivalent JSON and TOML literal forms.
                    lines.append(f"{key} = {json.dumps(value)}")
            lines.append("")
        return "\n".join(lines)


def resolve_config(config=None, profile=None):
    """Resolve explicit settings or a convenience processing profile."""
    if config is not None and profile is not None:
        raise ValueError("Use config or the profile argument, not both")
    if config is not None:
        if not isinstance(config, Config):
            raise ValueError("config must be a Config")
        return config
    if profile is None or profile == "bounded-1536":
        return Config()
    if profile == "core":
        return replace(
            Config(), processing=ProcessingConfig(resize=False, transparency="straight")
        )
    raise ValueError("profile must be core or bounded-1536")
