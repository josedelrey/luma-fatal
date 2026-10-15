"""Immutable settings shared by image adapters and decoded-frame processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
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

    def __post_init__(self):
        _integer("iterations", self.iterations, 1)
        _finite(
            "relaxation", self.relaxation, 0, 2, exclusive_min=True, exclusive_max=True
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


@dataclass(frozen=True)
class FilterParams:
    """Effect controls on a 0..100 scale, with a positive output gamma."""

    alpha: float = 1
    beta: float = 80
    saturation: float = 100
    noise: float = 0
    fade: float = 50
    output_gamma: float = 1.0

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
        _finite("output_gamma", self.output_gamma, 0, math.inf, exclusive_min=True)
        if not math.isfinite(1.0 / self.output_gamma):
            raise ValueError("output_gamma must have a finite reciprocal")


@dataclass(frozen=True)
class ProcessingConfig:
    """Device-class sizing and explicitly supported pixel-composition policies."""

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
    """Photo interpretation using display color and orientation by default."""

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
class Config:
    """Complete reusable run configuration with strict sections and defaults."""

    filter: str = "fattal1"
    parameters: FilterParams = field(default_factory=FilterParams)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    schema_version: int = 1

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
        path = Path(path)
        try:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
            elif path.suffix.lower() == ".toml":
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ValueError(
                    "Configuration files must use .yaml, .yml, .toml or .json"
                )
        except (yaml.YAMLError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid configuration file {path}: {error}") from error
        return cls.from_dict(data)

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
