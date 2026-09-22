"""Field-texture stock display: packing, static grids, shader source, strategy."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from carveracontroller.addons.stock.simulator.carvers.cylindrical.backend import shell_floor_radius_mm
from carveracontroller.addons.stock.simulator.carvers.field_mesh import (
    FLAG_OUTSIDE,
    FLAG_SOLID,
    FLAG_THROUGH,
    build_heightmap_field_mesh,
    build_rotary_field_mesh,
    display_radius,
    pack_height_field,
    pack_radius_field,
    radius_span_mm,
    unpack_field,
)
from carveracontroller.addons.stock.simulator.carvers.heightmap import HeightmapBackend
from carveracontroller.addons.stock.simulator.display import (
    FieldDisplay,
    RemeshDisplay,
    select_stock_display,
)
from carveracontroller.addons.stock.simulator.worker import StockSimulator
from carveracontroller.addons.stock.stock_geometry import StockBounds

_SHADER = Path(__file__).resolve().parents[2] / "carveracontroller" / "shaders" / "carved_field.glsl"


def test_height_field_roundtrip_marks_outside_and_through_cut():
    heights = np.array(
        [
            [5.0, -1e30],
            [0.0, 1.25],
        ],
        dtype=np.float32,
    )
    pixels, width, height, lo, span = pack_height_field(heights, 0.0, 5.0)
    assert (width, height) == (2, 2)
    assert lo == 0.0
    assert span == 5.0
    values, flags = unpack_field(pixels, width, height, lo, span)
    assert flags[0, 0] == FLAG_SOLID
    assert flags[0, 1] == FLAG_OUTSIDE
    assert flags[1, 0] == FLAG_THROUGH
    assert flags[1, 1] == FLAG_SOLID
    step = span / 65535.0
    assert values[0, 0] == 5.0
    assert abs(values[1, 1] - 1.25) <= step
    assert values[1, 0] == 0.0


def test_shader_height_scale_matches_packed_bytes():
    """Vertex shader reconstructs the 16-bit field without a hi*256 overflow."""
    heights = np.array([[10.0, 1.25], [0.0, 6.0]], dtype=np.float32)
    pixels, width, height, lo, span = pack_height_field(heights, 0.0, 10.0)
    rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    unit = (rgba[:, :, 0] / 255.0) * (65280.0 / 65535.0) + (rgba[:, :, 1] / 255.0) * (255.0 / 65535.0)
    decoded = lo + unit * span
    assert abs(float(decoded[0, 0]) - 10.0) < 1e-4
    assert abs(float(decoded[1, 0]) - 1.25) < span / 65535.0
    src = _SHADER.read_text().split("---fragment", 1)[0]
    assert "65280.0 / 65535.0" in src


def test_radius_field_roundtrip_and_shell_floor():
    radii = np.array([[-1.0, 0.01, 3.0]], dtype=np.float32)
    span = radius_span_mm(3.0, 10.0, 10.0)
    assert span == np.hypot(5.0, 5.0)
    pixels, width, height, lo, packed_span = pack_radius_field(radii, span)
    values, flags = unpack_field(pixels, width, height, lo, packed_span)
    assert flags[0].tolist() == [FLAG_OUTSIDE, FLAG_SOLID, FLAG_SOLID]
    assert values[0, 0] == 0.0
    assert abs(values[0, 2] - 3.0) <= packed_span / 65535.0
    cell = 1.0
    floor = shell_floor_radius_mm(cell)
    assert display_radius(-1.0, cell) == 0.0
    assert display_radius(0.01, cell) == floor
    assert display_radius(3.0, cell) == 3.0
    assert floor > 0.01


def test_heightmap_indices_ignore_heights_and_stay_in_uint16():
    nx, ny = 300, 40
    base = np.full((nx, ny), 8.0, dtype=np.float32)
    carved = base.copy()
    carved[10:40, 5:20] = 1.5
    first = build_heightmap_field_mesh(base, min_x=-4.0, min_y=-2.0, min_z=0.0, cell_size=0.2)
    second = build_heightmap_field_mesh(carved, min_x=-4.0, min_y=-2.0, min_z=0.0, cell_size=0.2)
    assert first and len(first) == len(second)
    assert len(first) > 2  # grid is wider than one uint16 chunk
    for (verts_a, idx_a, fmt_a), (verts_b, idx_b, fmt_b) in zip(first, second):
        assert fmt_a == fmt_b
        assert idx_a == idx_b
        assert verts_a == verts_b
        count = len(verts_a) // 10
        assert 0 < count <= 65500
        assert max(idx_a) < count


def _top_vertices(meshes):
    for verts, _idx, _fmt in meshes:
        grid = np.frombuffer(verts, dtype=np.float32).reshape(-1, 10)
        if grid.size and grid[0, 8] == 0.0:
            yield grid


def _step_normals(meshes, u0, u1, z0, z1):
    """Normals of triangles that join the two cells, after those cells are lifted."""
    found = []
    for verts, idx, _fmt in meshes:
        grid = np.frombuffer(verts, dtype=np.float32).reshape(-1, 10)
        if grid.size == 0 or not np.isclose(grid[0, 8], 2.0):
            continue
        tris = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        u = grid[:, 6]
        tri_u = u[tris]
        join = np.isclose(tri_u, u0).any(axis=1) & np.isclose(tri_u, u1).any(axis=1)
        if not np.any(join):
            continue
        # Both edges are lifted. A through-cut side is a wall, so it is not discarded.
        assert np.all(grid[tris[join], 9] > 0.5)
        lifted = grid[:, 0:3].copy()
        lifted[np.isclose(u, u0), 2] = z0
        lifted[np.isclose(u, u1), 2] = z1
        for ids in tris[join]:
            found.append(np.cross(lifted[ids[1]] - lifted[ids[0]], lifted[ids[2]] - lifted[ids[0]]))
    return np.stack(found) if found else np.zeros((0, 3))


def test_heightmap_steps_are_vertical_and_two_sided():
    """A height change is a vertical face, visible from either side."""
    heights = np.full((2, 1), 5.0, dtype=np.float32)
    meshes = build_heightmap_field_mesh(heights, min_x=0.0, min_y=0.0, min_z=0.0, cell_size=1.0)
    verts = np.concatenate(list(_top_vertices(meshes)))
    cell0 = verts[np.isclose(verts[:, 6], 0.25)]
    cell1 = verts[np.isclose(verts[:, 6], 0.75)]
    assert cell0.shape[0] == 4
    assert np.isclose(cell0[:, 0].min(), 0.0) and np.isclose(cell0[:, 0].max(), 1.0)
    assert np.isclose(cell1[:, 0].min(), 1.0) and np.isclose(cell1[:, 0].max(), 2.0)
    normals = _step_normals(meshes, 0.25, 0.75, 10.0, 2.0)
    assert len(normals) == 4
    assert np.allclose(normals[:, 1], 0.0, atol=1e-5)
    assert np.allclose(normals[:, 2], 0.0, atol=1e-5)
    assert normals[:, 0].min() < 0.0 and normals[:, 0].max() > 0.0


def test_heightmap_y_steps_are_vertical_and_two_sided():
    heights = np.full((1, 2), 5.0, dtype=np.float32)
    meshes = build_heightmap_field_mesh(heights, min_x=0.0, min_y=0.0, min_z=0.0, cell_size=1.0)
    normals = []
    for verts, idx, _fmt in meshes:
        grid = np.frombuffer(verts, dtype=np.float32).reshape(-1, 10)
        if grid.size == 0 or not np.isclose(grid[0, 8], 2.0):
            continue
        tris = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        v = grid[:, 7]
        tri_v = v[tris]
        join = np.isclose(tri_v, 0.25).any(axis=1) & np.isclose(tri_v, 0.75).any(axis=1)
        if not np.any(join):
            continue
        assert np.all(grid[tris[join], 9] > 0.5)
        lifted = grid[:, 0:3].copy()
        lifted[np.isclose(v, 0.25), 2] = 10.0
        lifted[np.isclose(v, 0.75), 2] = 2.0
        normals.extend(np.cross(lifted[ids[1]] - lifted[ids[0]], lifted[ids[2]] - lifted[ids[0]]) for ids in tris[join])
    normals = np.stack(normals)
    assert len(normals) == 4
    assert np.allclose(normals[:, 0], 0.0, atol=1e-5)
    assert np.allclose(normals[:, 2], 0.0, atol=1e-5)
    assert normals[:, 1].min() < 0.0 and normals[:, 1].max() > 0.0


def test_heightmap_steps_meet_across_chunks_and_stop_at_outside():
    wide = np.full((200, 1), 4.0, dtype=np.float32)
    meshes = build_heightmap_field_mesh(wide, min_x=0.0, min_y=0.0, min_z=0.0, cell_size=1.0)
    verts = np.concatenate(list(_top_vertices(meshes)))
    left = verts[np.isclose(verts[:, 6], 125.5 / 200.0)]
    right = verts[np.isclose(verts[:, 6], 126.5 / 200.0)]
    assert np.isclose(left[:, 0].max(), 126.0)
    assert np.isclose(right[:, 0].min(), 126.0)
    normals = _step_normals(meshes, 125.5 / 200.0, 126.5 / 200.0, 8.0, 1.0)
    assert len(normals) == 4
    assert np.allclose(normals[:, 2], 0.0, atol=1e-4)

    split = np.full((3, 1), 5.0, dtype=np.float32)
    split[1, 0] = -1e30
    meshes = build_heightmap_field_mesh(split, min_x=0.0, min_y=0.0, min_z=0.0, cell_size=1.0)
    verts = np.concatenate(list(_top_vertices(meshes)))
    end = verts[np.isclose(verts[:, 6], 0.5 / 3.0)]
    other = verts[np.isclose(verts[:, 6], 2.5 / 3.0)]
    assert np.isclose(end[:, 0].max(), 1.0)
    assert np.isclose(other[:, 0].min(), 2.0)
    assert len(_step_normals(meshes, 0.5 / 3.0, 2.5 / 3.0, 5.0, 5.0)) == 0


def test_heightmap_perimeter_walls_face_outward():
    heights = np.full((1, 1), 4.0, dtype=np.float32)
    meshes = build_heightmap_field_mesh(heights, min_x=0.0, min_y=0.0, min_z=0.0, cell_size=1.0)
    assert len(meshes) == 3
    wall_verts, wall_idx, _fmt = meshes[2]
    verts = np.frombuffer(wall_verts, dtype=np.float32).reshape(-1, 10)
    assert len(wall_idx) == 4 * 6
    # Stored Z is the bottom plane; the shader lifts vertices with extra > 0.
    lifted = verts[:, 0:3].copy()
    lifted[verts[:, 9] > 0.5, 2] = 4.0
    edge = np.cross(lifted[1] - lifted[0], lifted[2] - lifted[0])
    assert edge[0] < 0.0
    assert abs(edge[1]) < 1e-5
    assert abs(edge[2]) < 1e-5
    assert np.allclose(verts[0, 3:6], (-1.0, 0.0, 0.0))


def test_rotary_mesh_is_a_unit_cylinder_independent_of_radius():
    first = build_rotary_field_mesh(nx=3, n_theta=32, min_x=0.0, cell_size=1.0, axis_y=5.0, axis_z=-2.0)
    second = build_rotary_field_mesh(nx=3, n_theta=32, min_x=0.0, cell_size=1.0, axis_y=5.0, axis_z=-2.0)
    assert len(first) == len(second) >= 3
    for (verts_a, idx_a, _), (verts_b, idx_b, _) in zip(first, second):
        assert idx_a == idx_b
        assert verts_a == verts_b
        count = len(verts_a) // 10
        assert max(idx_a) < count <= 65500
    shell = np.frombuffer(first[0][0], dtype=np.float32).reshape(-1, 10)
    assert np.allclose(shell[:, 8], 3.0)
    assert np.allclose(np.hypot(shell[:, 1], shell[:, 2]), 1.0)


def test_field_shader_samples_texture_and_keeps_stock_local_two_tone():
    src = _SHADER.read_text()
    vertex, fragment = src.split("---fragment", 1)
    assert "texture2D(texture2" in vertex
    assert "field_tap(v_uv)" in vertex
    header_at = fragment.find("$HEADER$")
    assert fragment.find("GL_OES_standard_derivatives") < header_at
    assert "dFdx" in fragment
    assert "step(stock_z_max - surface_z_eps, stock_pos.z)" in fragment
    assert "step(stock_z_max - surface_z_eps, world_z)" not in fragment
    assert "is_surface = (1.0 - cap) * outward * at_od;" in fragment


class _Sim:
    def __init__(self, backend):
        self._lock = threading.RLock()
        self._generation = 1
        self._enabled = True
        self._backend = backend
        self._coalesce_gpu_keys: set[tuple[int, int, int]] = set()
        self.received: list[dict] = []
        self._on_meshes_ready = self.received.append


def test_vertex_texture_units_select_field_payload_or_mesh_tiles():
    assert isinstance(select_stock_display(0), RemeshDisplay)
    field = select_stock_display(8)
    assert isinstance(field, FieldDisplay)
    assert isinstance(StockSimulator()._display, RemeshDisplay)

    bounds = StockBounds(min_x=0, min_y=0, min_z=0, max_x=4, max_y=3, max_z=2)
    backend = HeightmapBackend(bounds, 1.0)
    sim = _Sim(backend)
    assert field.publish(sim, backend, {(0, 0, 0)}, sim._generation, replace=True)
    payload = sim.received[-1]["__field__"]
    assert payload["kind"] == "heightmap"
    assert payload["meshes"]
    assert "__replace__" not in sim.received[-1]

    sim.received.clear()
    assert field.publish(sim, backend, {(0, 0, 0)}, sim._generation, replace=False)
    assert sim.received[-1]["__field__"]["meshes"] is None

    sim.received.clear()
    remesh = select_stock_display(0)
    assert remesh.publish(sim, backend, {(0, 0, 0)}, sim._generation, replace=True)
    assert "__replace__" in sim.received[-1]
    assert "__field__" not in sim.received[-1]

    class _Voxel:
        kind = "voxel"

        def expand_dirty(self, keys):
            return set(keys)

        def copy_tiles(self, keys):
            return self

        def mesh_tiles(self, keys):
            return {key: ([1.0, 2.0, 3.0], [0, 1, 2], []) for key in keys}

        def take_laser_dirty(self):
            return False

    voxel = _Voxel()
    sim._backend = voxel
    sim.received.clear()
    assert field.publish(sim, voxel, {(0, 0, 0)}, sim._generation, replace=False)
    assert "__field__" not in sim.received[-1]
    assert (0, 0, 0) in sim.received[-1]
