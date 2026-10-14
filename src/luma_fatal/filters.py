"""Convenience functions for the two Fattal variants."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from .configuration import Config, FilterParams, resolve_config
from .pipeline import apply_filter


def fattal1(
    image: Image.Image | np.ndarray,
    *,
    profile: str | None = None,
    config: Config | None = None,
    **parameters: float,
) -> Image.Image | np.ndarray:
    """Apply Fattal 1 with effect parameters expressed on a 0..100 scale."""
    resolved = resolve_config(config, profile)
    return apply_filter(
        image, 1, replace(resolved.parameters, **parameters), config=resolved
    )


def fattal2(
    image: Image.Image | np.ndarray,
    *,
    profile: str | None = None,
    config: Config | None = None,
    **parameters: float,
) -> Image.Image | np.ndarray:
    """Apply Fattal 2 with effect parameters expressed on a 0..100 scale."""
    resolved = resolve_config(config, profile)
    return apply_filter(
        image, 2, replace(resolved.parameters, **parameters), config=resolved
    )
