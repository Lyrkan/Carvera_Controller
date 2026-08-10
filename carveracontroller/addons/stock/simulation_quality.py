"""Preset tables for stock cut-simulation quality / memory trade-offs."""

from __future__ import annotations

from typing import Any

# Target voxels along the stock's longest axis
DEFAULT_VOXEL_RESOLUTION = "low"
VOXEL_TARGET_BY_LEVEL = {
    "low": 100,
    "medium": 300,
    "high": 500,
}

# Equally spaced restore-point slots along the toolpath
DEFAULT_CHECKPOINT_LEVEL = "medium"
CHECKPOINT_SLOTS_BY_LEVEL = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
}


def normalize_voxel_resolution(value: Any) -> str:
    """Return a known voxel-resolution level, defaulting to ``low``."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in VOXEL_TARGET_BY_LEVEL:
            return key
    return DEFAULT_VOXEL_RESOLUTION


def normalize_checkpoint_level(value: Any) -> str:
    """Return a known checkpoint-retention level, defaulting to ``medium``."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in CHECKPOINT_SLOTS_BY_LEVEL:
            return key
    return DEFAULT_CHECKPOINT_LEVEL


def checkpoint_slots_for_level(level: Any) -> int:
    """Slot count for a retention level (after normalization)."""
    return CHECKPOINT_SLOTS_BY_LEVEL[normalize_checkpoint_level(level)]
