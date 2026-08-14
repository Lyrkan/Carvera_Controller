"""Stock playback presentation helpers (no full Kivy widget tree)."""

from unittest.mock import MagicMock

import pytest

from carveracontroller.GcodeViewer import GCodeViewer


def _viewer(**kwargs) -> GCodeViewer:
    viewer = GCodeViewer.__new__(GCodeViewer)
    viewer.simulate_cut = True
    viewer.dynamic_display = False
    viewer.stock_mesh_while_playing = False
    viewer._defer_carved_stock = False
    viewer.raw_positions = [0.0, 0.0, 0.0]
    viewer.cur_line_index = 0
    viewer._stock_simulator = MagicMock()
    for key, value in kwargs.items():
        setattr(viewer, key, value)
    return viewer


def test_carved_stock_hidden_while_deferred_after_pause():
    viewer = _viewer(dynamic_display=False, _defer_carved_stock=True)
    assert viewer._carved_stock_visible() is False


def test_carved_stock_visible_when_paused_after_flush():
    viewer = _viewer(dynamic_display=False, _defer_carved_stock=False)
    assert viewer._carved_stock_visible() is True


def test_carved_stock_hidden_while_playing_without_live_mesh():
    viewer = _viewer(dynamic_display=True, stock_mesh_while_playing=False)
    assert viewer._carved_stock_visible() is False


def test_pause_defers_carved_stock_when_live_mesh_off():
    viewer = _viewer()
    viewer._sync_play_mesh_policy = MagicMock()

    viewer._on_dynamic_display_changed(None, False)

    assert viewer._defer_carved_stock is True
    viewer._sync_play_mesh_policy.assert_called_once()
    viewer._stock_simulator.request_mesh_flush.assert_called_once()


def test_pause_does_not_defer_when_live_mesh_on():
    viewer = _viewer(stock_mesh_while_playing=True)
    viewer._sync_play_mesh_policy = MagicMock()

    viewer._on_dynamic_display_changed(None, False)

    assert viewer._defer_carved_stock is False
    viewer._stock_simulator.request_mesh_flush.assert_called_once()


def test_play_clears_deferred_carved_stock():
    viewer = _viewer(_defer_carved_stock=True)
    viewer._sync_play_mesh_policy = MagicMock()

    viewer._on_dynamic_display_changed(None, True)

    assert viewer._defer_carved_stock is False
    viewer._stock_simulator.request_mesh_flush.assert_not_called()


def test_mesh_apply_reveals_carved_stock_after_deferred_pause():
    viewer = _viewer(_defer_carved_stock=True)
    viewer._voxel_chunk_meshes = {}
    viewer.voxelmesh = MagicMock()
    viewer.voxelmesh.children = []
    viewer._voxel_mesh_anchor = object()
    viewer._update_voxel_uniforms = MagicMock()
    viewer._rebuild_stock_mesh = MagicMock()
    viewer._ensure_stock_on_canvas = MagicMock()

    viewer._apply_stock_meshes({})

    assert viewer._defer_carved_stock is False
    viewer._rebuild_stock_mesh.assert_called_once()
    viewer._ensure_stock_on_canvas.assert_called_once()


def test_clear_all_does_not_reveal_deferred_carved_stock():
    viewer = _viewer(_defer_carved_stock=True)
    viewer._clear_voxel_chunk_meshes = MagicMock()
    viewer._rebuild_stock_mesh = MagicMock()
    viewer._ensure_stock_on_canvas = MagicMock()

    viewer._apply_stock_meshes({"__clear_all__": None})

    assert viewer._defer_carved_stock is False
    viewer._clear_voxel_chunk_meshes.assert_called_once()
    viewer._rebuild_stock_mesh.assert_not_called()
    viewer._ensure_stock_on_canvas.assert_not_called()


def test_stock_uniforms_keep_a_axis_rotation():
    """Orbit/resize must not reset stock A rotation to identity."""
    from kivy.graphics.transformation import Matrix

    rot = Matrix().rotate(0.8, 1, 0, 0)
    viewer = _viewer(
        _stock_rotation_mat=rot,
        lines_center=[0.0, 0.0, 0.0],
        m_viewMatrix=Matrix(),
        _proj_matrix=Matrix(),
        stock_bounds_mm=None,
    )
    viewer.stockmesh = {}
    viewer.voxelmesh = {}
    viewer._stock_scale = lambda: 1.0

    GCodeViewer._update_stock_uniforms(viewer)
    GCodeViewer._update_voxel_uniforms(viewer)

    assert viewer.stockmesh["rotation_mat"] is rot
    assert viewer.voxelmesh["rotation_mat"] is rot


def test_raw_feed_z_max_uses_unrotated_wcs_z():
    """A-baked display Z is near 0 at A=90; stock estimate must use machine Z."""
    raw_zs = [10.0] * 2 + [22.0] * 18
    raw: list[float] = []
    baked: list[float] = []
    feeds: list[float] = []
    for i, z in enumerate(raw_zs):
        raw.extend((float(i), 0.0, z))
        baked.extend((float(i), -z, 0.0))
        feeds.append(1000.0)
    viewer = _viewer(
        raw_positions=raw,
        positions=baked,
        raw_feed_rates=feeds,
        z_max_mm=0.0,
    )
    assert GCodeViewer.raw_feed_z_max_mm(viewer) == pytest.approx(22.0)
