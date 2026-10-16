"""Configurable gradient-domain image processing."""

from .configuration import (
    Config,
    FilterParams,
    ImageConfig,
    NumericsConfig,
    OutputConfig,
    ProcessingConfig,
    VideoConfig,
)
from .filters import fattal1, fattal2
from .pipeline import apply_filter, process_frames

__all__ = [
    "Config",
    "FilterParams",
    "ImageConfig",
    "NumericsConfig",
    "OutputConfig",
    "ProcessingConfig",
    "VideoConfig",
    "apply_filter",
    "fattal1",
    "fattal2",
    "process_frames",
]
__version__ = "0.1.0"
