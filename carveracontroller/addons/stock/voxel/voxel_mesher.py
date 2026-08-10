"""Face-culling mesher for carved voxel chunks -> Kivy Mesh buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .voxel_grid import ChunkCoord, ChunkedVoxelGrid

# Same layout as tool meshes: pos(3) + normal(3) + color(4) + tc(2)
VERTEX_FORMAT = [
    (b"v_pos", 3, "float"),
    (b"v_normal", 3, "float"),
    (b"v_color", 4, "float"),
    (b"v_tc0", 2, "float"),
]

# Default color for voxel chunks
DEFAULT_COLOR = (0.75, 0.72, 0.55, 1.0)

# Face offsets: (axis, sign, normal, corner offsets in the two tangent axes)
# Each face is four corners of a unit cube face, in CCW order when viewed from outside.
_FACES = (
    # +X
    (0, 1, (1.0, 0.0, 0.0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    # -X
    (0, -1, (-1.0, 0.0, 0.0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    # +Y
    (1, 1, (0.0, 1.0, 0.0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    # -Y
    (1, -1, (0.0, -1.0, 0.0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    # +Z
    (2, 1, (0.0, 0.0, 1.0), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    # -Z
    (2, -1, (0.0, 0.0, -1.0), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def mesh_chunk(
    grid: ChunkedVoxelGrid,
    coord: ChunkCoord,
    occupancy: np.ndarray,
    color: tuple[float, float, float, float] = DEFAULT_COLOR,
) -> tuple[list[float], list[int], list] | None:
    """Build a triangle mesh for the exposed faces of one chunk.

    Returns ``(vertices, indices, VERTEX_FORMAT)`` or ``None`` if nothing to draw.
    Vertex positions are in **world millimetres** (caller scales into viewer space).
    """
    cs = grid.chunk_size
    if occupancy.shape != (cs, cs, cs):
        raise ValueError(f"occupancy shape {occupancy.shape} != ({cs},{cs},{cs})")

    # Ignore padding outside the logical stock grid (those voxels must stay empty).
    solid = occupancy.astype(bool) & grid._valid_mask(coord)
    if not solid.any():
        return None

    # Neighbour occupancy for face culling. Out-of-chunk neighbours consult the
    # parent grid; out-of-stock is treated as empty (exposed).
    neighbors = _neighbor_solid_maps(grid, coord, solid)

    positions: list[float] = []
    normals: list[float] = []
    colors: list[float] = []
    indices: list[int] = []
    vert_count = 0

    ox, oy, oz = grid.chunk_world_origin(coord)
    vs = grid.voxel_size
    cr, cg, cb, ca = color

    # Iterate solid voxels; for each exposed face, emit a quad (2 tris).
    # Vectorized mask per face direction first, then gather voxel indices.
    for axis, sign, normal, corners in _FACES:
        if sign > 0:
            own = solid
            other = neighbors[axis][1]  # positive neighbour map, same shape
            exposed = own & ~other
        else:
            own = solid
            other = neighbors[axis][0]
            exposed = own & ~other

        if not exposed.any():
            continue

        ixs, iys, izs = np.nonzero(exposed)
        nx, ny, nz = normal
        for lx, ly, lz in zip(ixs.tolist(), iys.tolist(), izs.tolist()):
            base = vert_count
            for cx, cy, cz in corners:
                positions.extend(
                    [
                        ox + (lx + cx) * vs,
                        oy + (ly + cy) * vs,
                        oz + (lz + cz) * vs,
                    ]
                )
                normals.extend([nx, ny, nz])
                colors.extend([cr, cg, cb, ca])
                vert_count += 1
            # Two triangles
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    if vert_count == 0:
        return None

    vertices = _pack_vertices(positions, normals, colors)
    return vertices, indices, VERTEX_FORMAT


def _neighbor_solid_maps(
    grid: ChunkedVoxelGrid,
    coord: ChunkCoord,
    solid: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each axis, return (neg_neighbor_solid, pos_neighbor_solid) aligned to ``solid``.

    For a voxel at local (i,j,k), the +X neighbour solidness is in ``pos[i,j,k]``
    (i.e. solidness of voxel i+1, or the adjacent chunk's edge, or False if OOB).
    """
    from .voxel_grid import ChunkCoord as CC

    cs = grid.chunk_size
    result: list[tuple[np.ndarray, np.ndarray]] = []

    for axis in range(3):
        # Shift within chunk
        if axis == 0:
            pos_internal = np.zeros_like(solid)
            pos_internal[:-1, :, :] = solid[1:, :, :]
            neg_internal = np.zeros_like(solid)
            neg_internal[1:, :, :] = solid[:-1, :, :]
            # Edge strips from neighbouring chunks
            pos_edge = _chunk_face_solid(grid, CC(coord.cx + 1, coord.cy, coord.cz), face=("x", 0))
            neg_edge = _chunk_face_solid(grid, CC(coord.cx - 1, coord.cy, coord.cz), face=("x", -1))
            if pos_edge is not None:
                pos_internal[-1, :, :] = pos_edge
            if neg_edge is not None:
                neg_internal[0, :, :] = neg_edge
        elif axis == 1:
            pos_internal = np.zeros_like(solid)
            pos_internal[:, :-1, :] = solid[:, 1:, :]
            neg_internal = np.zeros_like(solid)
            neg_internal[:, 1:, :] = solid[:, :-1, :]
            pos_edge = _chunk_face_solid(grid, CC(coord.cx, coord.cy + 1, coord.cz), face=("y", 0))
            neg_edge = _chunk_face_solid(grid, CC(coord.cx, coord.cy - 1, coord.cz), face=("y", -1))
            if pos_edge is not None:
                pos_internal[:, -1, :] = pos_edge
            if neg_edge is not None:
                neg_internal[:, 0, :] = neg_edge
        else:
            pos_internal = np.zeros_like(solid)
            pos_internal[:, :, :-1] = solid[:, :, 1:]
            neg_internal = np.zeros_like(solid)
            neg_internal[:, :, 1:] = solid[:, :, :-1]
            pos_edge = _chunk_face_solid(grid, CC(coord.cx, coord.cy, coord.cz + 1), face=("z", 0))
            neg_edge = _chunk_face_solid(grid, CC(coord.cx, coord.cy, coord.cz - 1), face=("z", -1))
            if pos_edge is not None:
                pos_internal[:, :, -1] = pos_edge
            if neg_edge is not None:
                neg_internal[:, :, 0] = neg_edge

        result.append((neg_internal, pos_internal))
    return result


def _chunk_face_solid(grid, coord, face) -> np.ndarray | None:
    """Return a 2D boolean face from a neighbouring chunk, or None if empty/OOB.

    ``face`` is ``('x'|'y'|'z', 0|-1)`` selecting the low (0) or high (-1) face.
    """
    from .voxel_grid import CHUNK_EMPTY, CHUNK_FULL

    cx, cy, cz = coord.cx, coord.cy, coord.cz
    if not (0 <= cx < grid.n_chunks_x and 0 <= cy < grid.n_chunks_y and 0 <= cz < grid.n_chunks_z):
        return None  # outside stock → treated as empty (caller leaves zeros)

    state = grid.get_chunk_state(coord)
    axis, index = face
    cs = grid.chunk_size

    if state is CHUNK_EMPTY:
        return np.zeros((cs, cs), dtype=bool)
    if state is CHUNK_FULL:
        # Untouched solid
        return np.ones((cs, cs), dtype=bool)
    if not isinstance(state, np.ndarray):
        return np.ones((cs, cs), dtype=bool)

    arr = state
    if axis == "x":
        return arr[index, :, :].astype(bool)
    if axis == "y":
        return arr[:, index, :].astype(bool)
    return arr[:, :, index].astype(bool)


def _pack_vertices(positions, normals, colors) -> list[float]:
    vertices: list[float] = []
    for i in range(len(positions) // 3):
        p = 3 * i
        c = 4 * i
        vertices.extend(
            [
                positions[p],
                positions[p + 1],
                positions[p + 2],
                normals[p],
                normals[p + 1],
                normals[p + 2],
                colors[c],
                colors[c + 1],
                colors[c + 2],
                colors[c + 3],
                0.0,
                0.0,
            ]
        )
    return vertices


def solid_occupancy_for_chunk(grid: ChunkedVoxelGrid, coord: ChunkCoord) -> np.ndarray:
    """Build a solid occupancy array for a FULL (or missing) chunk without storing it."""
    cs = grid.chunk_size
    arr = np.zeros((cs, cs, cs), dtype=np.uint8)
    arr[grid._valid_mask(coord)] = 1
    return arr


def mesh_chunk_state(
    grid: ChunkedVoxelGrid,
    coord: ChunkCoord,
    state: object | None = None,
) -> tuple[list[float], list[int], list] | None:
    """Mesh one chunk from its current state (FULL / EMPTY / array)."""
    from .voxel_grid import CHUNK_EMPTY, CHUNK_FULL

    if state is None:
        state = grid.get_chunk_state(coord)
    if state is CHUNK_EMPTY:
        return None
    if state is CHUNK_FULL:
        return mesh_chunk(grid, coord, solid_occupancy_for_chunk(grid, coord))
    if isinstance(state, np.ndarray):
        return mesh_chunk(grid, coord, state)
    return None


def mesh_dirty_chunks(
    grid: ChunkedVoxelGrid,
    dirty_chunk_coords: set[tuple[int, int, int]],
) -> dict[tuple[int, int, int], tuple | None]:
    """Mesh all dirty chunks. ``None`` means the chunk mesh should be removed."""
    from .voxel_grid import CHUNK_EMPTY, CHUNK_FULL, ChunkCoord

    result: dict[tuple[int, int, int], tuple | None] = {}
    for key in dirty_chunk_coords:
        coord = ChunkCoord(*key)
        if not (
            0 <= coord.cx < grid.n_chunks_x and 0 <= coord.cy < grid.n_chunks_y and 0 <= coord.cz < grid.n_chunks_z
        ):
            continue
        state = grid.get_chunk_state(coord)
        if state is CHUNK_EMPTY:
            result[key] = None
            continue
        if isinstance(state, np.ndarray):
            grid.maybe_collapse_chunk(coord, state)
            state = grid.get_chunk_state(coord)
            if state is CHUNK_EMPTY:
                result[key] = None
                continue
        packed = mesh_chunk_state(grid, coord, state)
        result[key] = packed
    return result
