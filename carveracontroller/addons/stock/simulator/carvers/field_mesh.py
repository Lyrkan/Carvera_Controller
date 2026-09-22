"""Static grids and RGBA8 field textures for heightmap and rotary stock.

The mesh is built once from the grid and the outside mask. Heights and radii
live in the texture; the vertex shader displaces the grid. Indices do not
depend on the field values.
"""

from __future__ import annotations

import array
import math

import numpy as np

from carveracontroller.addons.stock.simulator.carvers.array_mesh import MAX_KIVY_MESH_VERTS
from carveracontroller.addons.stock.simulator.carvers.cylindrical.backend import shell_floor_radius_mm

# Heightmap occupancy sentinel is -1e30. Anything below half of that is empty.
_OUTSIDE_CUTOFF = float(np.float32(-1e30) * np.float32(0.5))
# Matches the remesher's zero-thickness cutoff.
_THROUGH_EPS = 1e-4

FLAG_OUTSIDE = 0
FLAG_SOLID = 1
FLAG_THROUGH = 2

ROLE_TOP = 0.0
ROLE_BOTTOM = 1.0
ROLE_WALL = 2.0
ROLE_SHELL = 3.0
ROLE_CAP = 4.0

FIELD_VERTEX_FORMAT = [
    (b"v_pos", 3, "float"),
    (b"v_normal", 3, "float"),
    (b"v_uv", 2, "float"),
    (b"v_role", 1, "float"),
    (b"v_extra", 1, "float"),
]
_FLOATS = 10
_MAX_QUADS = MAX_KIVY_MESH_VERTS // 4

FieldMesh = tuple[array.array, array.array, list]


def display_radius(radius: float, cell_size_mm: float) -> float:
    """Radius the rotary shader draws. Empty cells stay off the axis."""
    if float(radius) < 0.0:
        return 0.0
    return max(float(radius), shell_floor_radius_mm(cell_size_mm))


def radius_span_mm(stock_radius_mm: float, size_y_mm: float, size_z_mm: float) -> float:
    """Stable unpack span covering the stock radius and the YZ box corner."""
    corner = math.hypot(0.5 * float(size_y_mm), 0.5 * float(size_z_mm))
    return max(float(stock_radius_mm), corner, 1e-6)


def pack_height_field(
    heights: np.ndarray,
    z_min: float,
    z_max: float,
) -> tuple[bytes, int, int, float, float]:
    """Pack ``heights`` ``(nx, ny)`` into an RGBA8 texture (row = Y).

    R/G are a 16-bit fixed height over ``[z_min, z_max]``. B is the cell flag.
    """
    field = np.asarray(heights, dtype=np.float32)
    outside = ~np.isfinite(field) | (field <= _OUTSIDE_CUTOFF)
    through = (~outside) & (field <= float(z_min) + _THROUGH_EPS)
    solid = (~outside) & (~through)
    flags = np.zeros(field.shape, dtype=np.uint8)
    flags[solid] = FLAG_SOLID
    flags[through] = FLAG_THROUGH
    span = max(float(z_max) - float(z_min), 1e-6)
    norm = np.zeros(field.shape, dtype=np.float64)
    np.subtract(field.astype(np.float64), float(z_min), out=norm, where=solid)
    norm /= span
    np.clip(norm, 0.0, 1.0, out=norm)
    norm[~solid] = 0.0
    return _pack_rgba(norm, flags), int(field.shape[0]), int(field.shape[1]), float(z_min), span


def pack_radius_field(
    radii: np.ndarray,
    span: float,
) -> tuple[bytes, int, int, float, float]:
    """Pack ``radii`` ``(nx, n_theta)``. Negative radius is outside; zero stays solid."""
    field = np.asarray(radii, dtype=np.float32)
    outside = ~np.isfinite(field) | (field < 0.0)
    flags = np.zeros(field.shape, dtype=np.uint8)
    flags[~outside] = FLAG_SOLID
    span_f = max(float(span), 1e-6)
    norm = np.zeros(field.shape, dtype=np.float64)
    np.divide(field.astype(np.float64), span_f, out=norm, where=~outside)
    np.clip(norm, 0.0, 1.0, out=norm)
    norm[outside] = 0.0
    return _pack_rgba(norm, flags), int(field.shape[0]), int(field.shape[1]), 0.0, span_f


def unpack_field(
    pixels: bytes,
    width: int,
    height: int,
    lo: float,
    span: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the packers. Returns ``(values, flags)`` shaped ``(width, height)``."""
    rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(int(height), int(width), 4)
    units = rgba[:, :, 0].astype(np.uint16) * np.uint16(256) + rgba[:, :, 1].astype(np.uint16)
    values = float(lo) + (units.astype(np.float64) / 65535.0) * float(span)
    flags = rgba[:, :, 2].astype(np.int16)
    return values.T.copy(), flags.T.copy()


def build_heightmap_field_mesh(
    heights: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    cell_size: float,
) -> list[FieldMesh]:
    """Flat per-cell top, vertical steps, shared bottom, and perimeter walls.

    Vertex positions do not store the height. Only the outside mask affects
    topology. A step is its own wall, including where a cut goes through the
    stock, so the opening keeps sides down to the bottom.
    """
    field = np.asarray(heights)
    nx, ny = int(field.shape[0]), int(field.shape[1])
    if nx < 1 or ny < 1:
        return []
    inside = np.isfinite(field) & (field > _OUTSIDE_CUTOFF)
    cell = float(cell_size)
    z_bot = float(min_z)
    meshes: list[FieldMesh] = []
    meshes.extend(_heightmap_tops(inside, min_x=float(min_x), min_y=float(min_y), z_bot=z_bot, cell=cell))
    edge = _max_quad_edge()
    for y0 in range(0, ny, edge):
        qh = min(edge, ny - y0)
        for x0 in range(0, nx, edge):
            qw = min(edge, nx - x0)
            meshes.append(
                _grid_chunk(
                    x0,
                    y0,
                    qw,
                    qh,
                    origin_x=float(min_x),
                    origin_y=float(min_y),
                    z=z_bot,
                    cell=cell,
                    nu=nx,
                    nv=ny,
                    role=ROLE_BOTTOM,
                    outward_up=False,
                )
            )
    meshes.extend(_heightmap_walls(inside, min_x=float(min_x), min_y=float(min_y), z_bot=z_bot, cell=cell))
    meshes.extend(_heightmap_steps(inside, min_x=float(min_x), min_y=float(min_y), z_bot=z_bot, cell=cell))
    return meshes


def build_rotary_field_mesh(
    *,
    nx: int,
    n_theta: int,
    min_x: float,
    cell_size: float,
    axis_y: float,
    axis_z: float,
) -> list[FieldMesh]:
    """Shared ``(x, θ)`` cylinder plus end caps. Radius comes from the texture."""
    del axis_y, axis_z  # the shader places the axis; the mesh stores sin/cos
    nx_i, n_th = int(nx), int(n_theta)
    if nx_i < 1 or n_th < 1:
        return []
    cell = float(cell_size)
    d_theta = 360.0 / n_th
    theta = np.deg2rad(np.arange(n_th, dtype=np.float64) * d_theta)
    sin_t = np.sin(theta).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    meshes: list[FieldMesh] = []
    stride = n_th
    qw_max = max(1, (MAX_KIVY_MESH_VERTS // stride) - 1)
    while qw_max > 1 and (qw_max + 1) * stride > MAX_KIVY_MESH_VERTS:
        qw_max -= 1
    x0 = 0
    while x0 < nx_i:
        qw = min(qw_max, nx_i - x0)
        meshes.append(
            _shell_chunk(
                x0,
                qw,
                n_th,
                sin_t,
                cos_t,
                min_x=float(min_x),
                cell=cell,
                nx=nx_i,
            )
        )
        x0 += qw
    x_start = float(min_x)
    x_end = float(min_x) + nx_i * cell
    meshes.append(_cap_mesh(x_start, sin_t, cos_t, n_th, nx_i, end=False))
    meshes.append(_cap_mesh(x_end, sin_t, cos_t, n_th, nx_i, end=True))
    return meshes


def _pack_rgba(norm: np.ndarray, flags: np.ndarray) -> bytes:
    units = np.rint(np.asarray(norm, dtype=np.float64) * 65535.0).astype(np.uint16)
    rgba = np.zeros((norm.shape[1], norm.shape[0], 4), dtype=np.uint8)
    src = units.T
    rgba[:, :, 0] = (src >> np.uint16(8)).astype(np.uint8)
    rgba[:, :, 1] = (src & np.uint16(0xFF)).astype(np.uint8)
    rgba[:, :, 2] = flags.T
    rgba[:, :, 3] = 255
    return np.ascontiguousarray(rgba).tobytes()


def _as_f32(data: np.ndarray) -> array.array:
    out = array.array("f")
    out.frombytes(np.ascontiguousarray(data, dtype=np.float32).ravel().tobytes())
    return out


def _as_u16(data: np.ndarray) -> array.array:
    out = array.array("H")
    out.frombytes(np.ascontiguousarray(data, dtype=np.uint16).ravel().tobytes())
    return out


def _pack_indexed(
    pos: np.ndarray,
    normal: np.ndarray,
    uv: np.ndarray,
    role: float,
    extra: np.ndarray,
    indices: np.ndarray,
) -> FieldMesh:
    n = int(pos.shape[0])
    if n > MAX_KIVY_MESH_VERTS or n < 1:
        raise ValueError(f"field mesh has {n} vertices")
    verts = np.empty((n, _FLOATS), dtype=np.float32)
    verts[:, 0:3] = pos
    verts[:, 3:6] = normal
    verts[:, 6:8] = uv
    verts[:, 8] = float(role)
    verts[:, 9] = extra
    return _as_f32(verts), _as_u16(indices), FIELD_VERTEX_FORMAT


def _top_chunk_cells() -> int:
    """Owned cells on a side. Four corners per cell must fit in one mesh."""
    edge = 126
    while edge > 1 and 4 * edge * edge > MAX_KIVY_MESH_VERTS:
        edge -= 1
    return max(1, edge)


def _heightmap_tops(
    inside: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> list[FieldMesh]:
    nx, ny = inside.shape
    edge = _top_chunk_cells()
    meshes: list[FieldMesh] = []
    y0 = 0
    while y0 < ny:
        qh = min(edge, ny - y0)
        x0 = 0
        while x0 < nx:
            qw = min(edge, nx - x0)
            mesh = _top_chunk(
                inside,
                x0,
                y0,
                qw,
                qh,
                min_x=min_x,
                min_y=min_y,
                z=z_bot,
                cell=float(cell),
            )
            if mesh is not None:
                meshes.append(mesh)
            x0 += qw
        y0 += qh
    return meshes


def _top_chunk(
    inside: np.ndarray,
    x0: int,
    y0: int,
    qw: int,
    qh: int,
    *,
    min_x: float,
    min_y: float,
    z: float,
    cell: float,
) -> FieldMesh | None:
    """One flat quad per inside cell. Steps between cells are separate walls."""
    nx, ny = int(inside.shape[0]), int(inside.shape[1])
    xs = min_x + (x0 + np.arange(qw, dtype=np.float64)) * cell
    ys = min_y + (y0 + np.arange(qh, dtype=np.float64)) * cell
    corner_x = np.stack((xs, xs + cell, xs + cell, xs), axis=1)
    corner_y = np.stack((ys, ys, ys + cell, ys + cell), axis=1)
    xx = np.broadcast_to(corner_x[None, :, :], (qh, qw, 4))
    yy = np.broadcast_to(corner_y[:, None, :], (qh, qw, 4))
    pos = np.stack((xx, yy, np.full((qh, qw, 4), z)), axis=-1).reshape(-1, 3)
    u = (x0 + np.arange(qw, dtype=np.float64) + 0.5) / float(nx)
    v = (y0 + np.arange(qh, dtype=np.float64) + 0.5) / float(ny)
    uv = np.stack(
        (
            np.broadcast_to(u[None, :, None], (qh, qw, 4)),
            np.broadcast_to(v[:, None, None], (qh, qw, 4)),
        ),
        axis=-1,
    ).reshape(-1, 2)

    oi, oj = np.meshgrid(np.arange(qw, dtype=np.int32), np.arange(qh, dtype=np.int32), indexing="xy")
    owned = inside[x0 + oi, y0 + oj]
    oi, oj = oi[owned], oj[owned]
    if oi.size == 0:
        return None
    base = (oj * qw + oi) * np.int32(4)
    indices = np.stack((base, base + 1, base + 2, base, base + 2, base + 3), axis=1).ravel()
    normal = np.zeros((pos.shape[0], 3), dtype=np.float32)
    normal[:, 2] = 1.0
    extra = np.zeros((pos.shape[0],), dtype=np.float32)
    return _pack_indexed(
        pos.astype(np.float32),
        normal,
        uv.astype(np.float32),
        ROLE_TOP,
        extra,
        indices.astype(np.uint16),
    )


def _max_quad_edge() -> int:
    edge = int(math.sqrt(MAX_KIVY_MESH_VERTS)) - 1
    while edge > 1 and (edge + 1) * (edge + 1) > MAX_KIVY_MESH_VERTS:
        edge -= 1
    return max(1, edge)


def _grid_indices(qw: int, qh: int, *, outward_up: bool) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(qw, dtype=np.int32), np.arange(qh, dtype=np.int32), indexing="xy")
    v00 = (jj * (qw + 1) + ii).ravel()
    v10 = v00 + 1
    v01 = v00 + (qw + 1)
    v11 = v01 + 1
    if outward_up:
        tris = np.stack((v00, v10, v11, v00, v11, v01), axis=1)
    else:
        tris = np.stack((v00, v01, v11, v00, v11, v10), axis=1)
    return tris.astype(np.uint16).ravel()


def _grid_chunk(
    x0: int,
    y0: int,
    qw: int,
    qh: int,
    *,
    origin_x: float,
    origin_y: float,
    z: float,
    cell: float,
    nu: int,
    nv: int,
    role: float,
    outward_up: bool,
) -> FieldMesh:
    xs = origin_x + (x0 + np.arange(qw + 1, dtype=np.float64)) * cell
    ys = origin_y + (y0 + np.arange(qh + 1, dtype=np.float64)) * cell
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pos = np.stack((xx, yy, np.full_like(xx, z)), axis=-1).reshape(-1, 3).astype(np.float32)
    uu = (x0 + np.arange(qw + 1, dtype=np.float64)) / float(nu)
    vv = (y0 + np.arange(qh + 1, dtype=np.float64)) / float(nv)
    ug, vg = np.meshgrid(uu, vv, indexing="xy")
    uv = np.stack((ug, vg), axis=-1).reshape(-1, 2).astype(np.float32)
    normal = np.zeros_like(pos)
    normal[:, 2] = 1.0 if outward_up else -1.0
    extra = np.zeros((pos.shape[0],), dtype=np.float32)
    return _pack_indexed(pos, normal, uv, role, extra, _grid_indices(qw, qh, outward_up=outward_up))


def _heightmap_steps(
    inside: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> list[FieldMesh]:
    """Vertical faces on every inside/inside edge, as walls rather than top faces.

    A through-cut discards the top and the bottom. The side has to be a wall or
    the half that samples the hole is discarded with them, which is the missing
    lower part of a through-pocket.
    """
    nx, ny = int(inside.shape[0]), int(inside.shape[1])
    blocks: list[np.ndarray] = []
    if nx > 1:
        ii, jj = np.nonzero(inside[:-1, :] & inside[1:, :])
        if ii.size:
            blocks.append(_x_step_quads(ii, jj, nx=nx, ny=ny, min_x=min_x, min_y=min_y, z_bot=z_bot, cell=cell))
    if ny > 1:
        ii, jj = np.nonzero(inside[:, :-1] & inside[:, 1:])
        if ii.size:
            blocks.append(_y_step_quads(ii, jj, nx=nx, ny=ny, min_x=min_x, min_y=min_y, z_bot=z_bot, cell=cell))
    if not blocks:
        return []
    return _pack_two_sided_quads(np.concatenate(blocks, axis=0))


def _x_step_quads(
    ii: np.ndarray,
    jj: np.ndarray,
    *,
    nx: int,
    ny: int,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> np.ndarray:
    """``(n, 4, 10)`` quads on the edge between cell ``(ii, jj)`` and ``(ii+1, jj)``."""
    x = min_x + (ii.astype(np.float64) + 1.0) * cell
    y0 = min_y + jj.astype(np.float64) * cell
    y1 = y0 + cell
    u0 = (ii.astype(np.float64) + 0.5) / float(nx)
    u1 = (ii.astype(np.float64) + 1.5) / float(nx)
    v = (jj.astype(np.float64) + 0.5) / float(ny)
    out = np.zeros((ii.shape[0], 4, _FLOATS), dtype=np.float32)
    out[:, :, 0] = x[:, None]
    out[:, 0, 1] = y0
    out[:, 3, 1] = y0
    out[:, 1, 1] = y1
    out[:, 2, 1] = y1
    out[:, :, 2] = z_bot
    out[:, :, 3] = 1.0
    out[:, 0, 6] = u0
    out[:, 1, 6] = u0
    out[:, 2, 6] = u1
    out[:, 3, 6] = u1
    out[:, :, 7] = v[:, None]
    out[:, :, 9] = 1.0
    return out


def _y_step_quads(
    ii: np.ndarray,
    jj: np.ndarray,
    *,
    nx: int,
    ny: int,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> np.ndarray:
    """``(n, 4, 10)`` quads on the edge between cell ``(ii, jj)`` and ``(ii, jj+1)``."""
    y = min_y + (jj.astype(np.float64) + 1.0) * cell
    x0 = min_x + ii.astype(np.float64) * cell
    x1 = x0 + cell
    v0 = (jj.astype(np.float64) + 0.5) / float(ny)
    v1 = (jj.astype(np.float64) + 1.5) / float(ny)
    u = (ii.astype(np.float64) + 0.5) / float(nx)
    out = np.zeros((ii.shape[0], 4, _FLOATS), dtype=np.float32)
    out[:, 0, 0] = x0
    out[:, 3, 0] = x0
    out[:, 1, 0] = x1
    out[:, 2, 0] = x1
    out[:, :, 1] = y[:, None]
    out[:, :, 2] = z_bot
    out[:, :, 4] = 1.0
    out[:, :, 6] = u[:, None]
    out[:, 0, 7] = v0
    out[:, 1, 7] = v0
    out[:, 2, 7] = v1
    out[:, 3, 7] = v1
    out[:, :, 9] = 1.0
    return out


def _pack_two_sided_quads(quads: np.ndarray) -> list[FieldMesh]:
    """Pack quads as walls with both windings. Every vertex is a lifted edge."""
    meshes: list[FieldMesh] = []
    pattern = np.array([0, 1, 2, 0, 2, 3, 0, 3, 2, 0, 2, 1], dtype=np.uint16)
    for start in range(0, quads.shape[0], _MAX_QUADS):
        chunk = quads[start : start + _MAX_QUADS]
        n = int(chunk.shape[0])
        indices = np.empty((n, pattern.shape[0]), dtype=np.uint16)
        base = (np.arange(n, dtype=np.uint16) * np.uint16(4))[:, None]
        indices[:, :] = base + pattern
        meshes.append(
            _pack_indexed(
                chunk[:, :, 0:3].reshape(-1, 3),
                chunk[:, :, 3:6].reshape(-1, 3),
                chunk[:, :, 6:8].reshape(-1, 2),
                ROLE_WALL,
                chunk[:, :, 9].reshape(-1),
                indices.ravel(),
            )
        )
    return meshes


def _heightmap_walls(
    inside: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> list[FieldMesh]:
    nx, ny = inside.shape
    blocks: list[np.ndarray] = []

    def _add(mask: np.ndarray, side: str) -> None:
        ii, jj = np.nonzero(mask)
        if ii.size == 0:
            return
        blocks.append(_wall_quads(ii, jj, side, nx=nx, ny=ny, min_x=min_x, min_y=min_y, z_bot=z_bot, cell=cell))

    left = inside.copy()
    left[1:, :] &= ~inside[:-1, :]
    right = inside.copy()
    right[:-1, :] &= ~inside[1:, :]
    down = inside.copy()
    down[:, 1:] &= ~inside[:, :-1]
    up = inside.copy()
    up[:, :-1] &= ~inside[:, 1:]
    _add(left, "left")
    _add(right, "right")
    _add(down, "down")
    _add(up, "up")
    if not blocks:
        return []
    quads = np.concatenate(blocks, axis=0)
    meshes: list[FieldMesh] = []
    for start in range(0, quads.shape[0], _MAX_QUADS):
        chunk = quads[start : start + _MAX_QUADS]
        n = int(chunk.shape[0])
        pos = chunk[:, :, 0:3].reshape(-1, 3)
        normal = chunk[:, :, 3:6].reshape(-1, 3)
        uv = chunk[:, :, 6:8].reshape(-1, 2)
        extra = chunk[:, :, 9].reshape(-1)
        indices = np.empty((n, 6), dtype=np.uint16)
        base = (np.arange(n, dtype=np.uint16) * np.uint16(4))[:, None]
        indices[:, :] = base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
        meshes.append(_pack_indexed(pos, normal, uv, ROLE_WALL, extra, indices.ravel()))
    return meshes


def _wall_quads(
    ii: np.ndarray,
    jj: np.ndarray,
    side: str,
    *,
    nx: int,
    ny: int,
    min_x: float,
    min_y: float,
    z_bot: float,
    cell: float,
) -> np.ndarray:
    """``(n, 4, 10)`` wall quads, CCW when seen from outside the stock."""
    x0 = min_x + ii.astype(np.float64) * cell
    y0 = min_y + jj.astype(np.float64) * cell
    x1 = x0 + cell
    y1 = y0 + cell
    # Top edge first (extra=1), then the same XY on the bottom plane (extra=0).
    if side == "left":
        xs = (x0, x0, x0, x0)
        ys = (y0, y1, y1, y0)
        sign, axis = -1.0, 0
    elif side == "right":
        xs = (x1, x1, x1, x1)
        ys = (y1, y0, y0, y1)
        sign, axis = 1.0, 0
    elif side == "down":
        xs = (x1, x0, x0, x1)
        ys = (y0, y0, y0, y0)
        sign, axis = -1.0, 1
    else:
        xs = (x0, x1, x1, x0)
        ys = (y1, y1, y1, y1)
        sign, axis = 1.0, 1
    n = int(ii.shape[0])
    out = np.zeros((n, 4, _FLOATS), dtype=np.float32)
    # Cell center, so the lifted edge matches the plateau instead of a corner average.
    u = (ii.astype(np.float64) + 0.5) / float(nx)
    v = (jj.astype(np.float64) + 0.5) / float(ny)
    for k in range(4):
        out[:, k, 0] = xs[k]
        out[:, k, 1] = ys[k]
        out[:, k, 2] = float(z_bot)
        out[:, k, 3 + axis] = sign
        out[:, k, 6] = u
        out[:, k, 7] = v
        out[:, k, 8] = ROLE_WALL
        out[:, k, 9] = 1.0 if k < 2 else 0.0
    return out


def _shell_chunk(
    x0: int,
    qw: int,
    n_theta: int,
    sin_t: np.ndarray,
    cos_t: np.ndarray,
    *,
    min_x: float,
    cell: float,
    nx: int,
) -> FieldMesh:
    ix = np.arange(qw + 1, dtype=np.float64)
    xs = min_x + (x0 + ix) * cell
    xx = np.broadcast_to(xs[:, None], (qw + 1, n_theta))
    ss = np.broadcast_to(sin_t[None, :], (qw + 1, n_theta))
    cc = np.broadcast_to(cos_t[None, :], (qw + 1, n_theta))
    pos = np.stack((xx, ss, cc), axis=-1).reshape(-1, 3).astype(np.float32)
    uu = (x0 + ix) / float(nx)
    vv = np.arange(n_theta, dtype=np.float64) / float(n_theta)
    ug = np.broadcast_to(uu[:, None], (qw + 1, n_theta))
    vg = np.broadcast_to(vv[None, :], (qw + 1, n_theta))
    uv = np.stack((ug, vg), axis=-1).reshape(-1, 2).astype(np.float32)
    normal = np.zeros_like(pos)
    normal[:, 1] = pos[:, 1]
    normal[:, 2] = pos[:, 2]
    extra = np.zeros((pos.shape[0],), dtype=np.float32)
    ii = np.repeat(np.arange(qw, dtype=np.int32), n_theta)
    tt = np.tile(np.arange(n_theta, dtype=np.int32), qw)
    tt1 = (tt + 1) % n_theta
    v00 = ii * n_theta + tt
    v10 = (ii + 1) * n_theta + tt
    v11 = (ii + 1) * n_theta + tt1
    v01 = ii * n_theta + tt1
    if int(v11.max()) >= (qw + 1) * n_theta:
        raise ValueError("rotary shell index does not fit a Kivy mesh")
    indices = np.stack((v00, v10, v11, v00, v11, v01), axis=1).astype(np.uint16).ravel()
    return _pack_indexed(pos, normal, uv, ROLE_SHELL, extra, indices)


def _cap_mesh(
    x: float,
    sin_t: np.ndarray,
    cos_t: np.ndarray,
    n_theta: int,
    nx: int,
    *,
    end: bool,
) -> FieldMesh:
    t = np.arange(n_theta, dtype=np.int32)
    t1 = (t + 1) % n_theta
    rim_a = t1 if end else t
    rim_b = t if end else t1
    n = n_theta
    pos = np.zeros((n * 3, 3), dtype=np.float32)
    uv = np.zeros((n * 3, 2), dtype=np.float32)
    extra = np.zeros((n,), dtype=np.float32)
    extra_full = np.zeros((n * 3,), dtype=np.float32)
    u_edge = 1.0 if end else 0.0
    u_cell = (nx - 0.5) / float(nx) if end else 0.5 / float(nx)
    pos[0::3, 0] = x
    uv[0::3, 0] = u_cell
    uv[0::3, 1] = (t.astype(np.float64) + 0.5) / float(n_theta)
    pos[1::3, 0] = x
    pos[1::3, 1] = sin_t[rim_a]
    pos[1::3, 2] = cos_t[rim_a]
    extra_full[1::3] = 1.0
    uv[1::3, 0] = u_edge
    uv[1::3, 1] = rim_a.astype(np.float64) / float(n_theta)
    pos[2::3, 0] = x
    pos[2::3, 1] = sin_t[rim_b]
    pos[2::3, 2] = cos_t[rim_b]
    extra_full[2::3] = 1.0
    uv[2::3, 0] = u_edge
    uv[2::3, 1] = rim_b.astype(np.float64) / float(n_theta)
    normal = np.zeros_like(pos)
    normal[:, 0] = 1.0 if end else -1.0
    indices = np.arange(n * 3, dtype=np.uint16)
    del extra
    return _pack_indexed(pos, normal, uv, ROLE_CAP, extra_full, indices)
