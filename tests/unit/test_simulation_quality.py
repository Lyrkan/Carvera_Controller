"""Unit tests for stock simulation quality presets."""

from carveracontroller.addons.stock.simulation_quality import (
    CHECKPOINT_SLOTS_BY_LEVEL,
    DEFAULT_CHECKPOINT_LEVEL,
    DEFAULT_VOXEL_RESOLUTION,
    VOXEL_TARGET_BY_LEVEL,
    checkpoint_slots_for_level,
    normalize_checkpoint_level,
    normalize_voxel_resolution,
)
from carveracontroller.addons.stock.stock_defaults import (
    checkpoint_level_from_settings,
    default_settings,
    mesh_while_playing_from_settings,
    voxel_resolution_from_settings,
)
from carveracontroller.addons.stock.voxel.stock_simulator import CheckpointStore


def test_voxel_preset_defaults():
    assert VOXEL_TARGET_BY_LEVEL["low"] == 100
    assert DEFAULT_VOXEL_RESOLUTION == "low"


def test_checkpoint_slots_match_presets_and_store_default():
    assert CHECKPOINT_SLOTS_BY_LEVEL == {"low": 256, "medium": 1024, "high": 4096}
    store = CheckpointStore()
    assert store.slot_count == CHECKPOINT_SLOTS_BY_LEVEL["medium"]
    assert store.base_interval == 4
    assert checkpoint_slots_for_level("high") == 4096
    assert checkpoint_slots_for_level("bogus") == CHECKPOINT_SLOTS_BY_LEVEL[DEFAULT_CHECKPOINT_LEVEL]
    assert DEFAULT_CHECKPOINT_LEVEL == "medium"


def test_normalize_voxel_resolution_fallbacks():
    assert normalize_voxel_resolution("HIGH") == "high"
    assert normalize_voxel_resolution("medium") == "medium"
    assert normalize_voxel_resolution(None) == DEFAULT_VOXEL_RESOLUTION
    assert normalize_voxel_resolution("nope") == DEFAULT_VOXEL_RESOLUTION
    assert normalize_voxel_resolution(12) == DEFAULT_VOXEL_RESOLUTION


def test_normalize_checkpoint_level_fallbacks():
    assert normalize_checkpoint_level("LOW") == "low"
    assert normalize_checkpoint_level("high") == "high"
    assert normalize_checkpoint_level(None) == DEFAULT_CHECKPOINT_LEVEL
    assert normalize_checkpoint_level("") == DEFAULT_CHECKPOINT_LEVEL
    assert normalize_checkpoint_level({}) == DEFAULT_CHECKPOINT_LEVEL


def test_default_settings_include_quality_presets():
    settings = default_settings()
    assert settings["voxel_resolution"] == DEFAULT_VOXEL_RESOLUTION
    assert settings["checkpoint_level"] == DEFAULT_CHECKPOINT_LEVEL
    assert settings["mesh_while_playing"] is False
    assert voxel_resolution_from_settings(settings) == DEFAULT_VOXEL_RESOLUTION
    assert checkpoint_level_from_settings(settings) == DEFAULT_CHECKPOINT_LEVEL
    assert mesh_while_playing_from_settings(settings) is False


def test_settings_helpers_normalize_bad_values():
    assert voxel_resolution_from_settings({}) == DEFAULT_VOXEL_RESOLUTION
    assert checkpoint_level_from_settings({"checkpoint_level": "bogus"}) == DEFAULT_CHECKPOINT_LEVEL
    assert voxel_resolution_from_settings({"voxel_resolution": "High"}) == "high"
    assert mesh_while_playing_from_settings({}) is False
    assert mesh_while_playing_from_settings({"mesh_while_playing": False}) is False
    assert mesh_while_playing_from_settings({"mesh_while_playing": 0}) is False
    assert mesh_while_playing_from_settings({"mesh_while_playing": True}) is True
