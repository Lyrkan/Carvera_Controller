"""Material-removal simulator driven by toolpath segments + tool profiles."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np

from carveracontroller.addons.stock.simulation_quality import (
    CHECKPOINT_SLOTS_BY_LEVEL,
    DEFAULT_CHECKPOINT_LEVEL,
    DEFAULT_VOXEL_RESOLUTION,
    VOXEL_TARGET_BY_LEVEL,
)
from carveracontroller.addons.stock.stock_geometry import StockBounds, rotate_yz
from carveracontroller.addons.stock.stock_shape import RectangularStock, StockShape
from carveracontroller.addons.tool_visualization.mesh_builder import (
    effective_tool_diameter,
    fallback_tool_dimensions,
    profile_length,
    resolve_section_lengths,
    tool_profile,
)
from carveracontroller.addons.tool_visualization.tool_definition import ToolDefinition

from .stock_occupancy import non_full_chunk_keys, seed_shape_occupancy
from .voxel_grid import (
    CHUNK_EMPTY,
    CHUNK_FULL,
    ChunkCoord,
    ChunkedVoxelGrid,
    pick_voxel_size_mm,
)

logger = logging.getLogger(__name__)

# Sentinel to shut down the worker.
_STOP = object()
# Latest-wins poke: carve/rewind toward ``_display_vertex`` (no per-frame job flood).
_DISPLAY_FOLLOW = object()
# Emit pending dirty meshes immediately (pause catch-up).
_MESH_FLUSH = object()

# Throttle while the carved mesh is following playback (pause uses constructor default).
DEFAULT_MESH_THROTTLE_S = 0.15
PLAY_MESH_THROTTLE_S = 0.5

_NEIGHBOR_OFFSETS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# Cut-segment merge knobs
_MERGE_TOL_VOXEL_FRAC = 0.25
_MERGE_MAX_MM_FLOOR = 8.0
_MERGE_MAX_MM_VOXEL_MULT = 16.0
_MERGE_MAX_SPAN = 64
# Subdivide wrapping / helical A so each constant-A carve is a few degrees.
_MAX_DA_DEG = 2.0


def _exterior_chunk_keys(grid: ChunkedVoxelGrid) -> set[tuple[int, int, int]]:
    """Chunk coords on the outer shell of the grid (have at least one stock face)."""
    keys: set[tuple[int, int, int]] = set()
    nx, ny, nz = grid.n_chunks_x, grid.n_chunks_y, grid.n_chunks_z
    for cx in range(nx):
        for cy in range(ny):
            for cz in range(nz):
                if cx in (0, nx - 1) or cy in (0, ny - 1) or cz in (0, nz - 1):
                    keys.add((cx, cy, cz))
    return keys


def _expand_dirty_with_neighbors(
    grid: ChunkedVoxelGrid,
    dirty: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Include 6-connected neighbours so pocket walls on FULL chunks are remeshed."""
    expanded = set(dirty)
    nx, ny, nz = grid.n_chunks_x, grid.n_chunks_y, grid.n_chunks_z
    for cx, cy, cz in dirty:
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            x, y, z = cx + dx, cy + dy, cz + dz
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                expanded.add((x, y, z))
    return expanded


def _xyz_dist(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _point_to_segment_dist_sq(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    p: tuple[float, float, float],
) -> float:
    """Squared distance from ``p`` to closed segment ``A→B``."""
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    abz = b[2] - a[2]
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    apz = p[2] - a[2]
    ab_len_sq = abx * abx + aby * aby + abz * abz
    if ab_len_sq < 1e-18:
        return apx * apx + apy * apy + apz * apz
    t = (apx * abx + apy * aby + apz * abz) / ab_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    dx = a[0] + t * abx - p[0]
    dy = a[1] + t * aby - p[1]
    dz = a[2] + t * abz - p[2]
    return dx * dx + dy * dy + dz * dz


def _can_extend_collinear_merge(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    tol_mm: float,
) -> bool:
    """True if ``B`` lies within ``tol_mm`` of chord ``A→C`` (open projection).

    Local gate only; :func:`_clamp_collinear_run_end` enforces the final chord.
    """
    ab_len = _xyz_dist(a, b)
    if ab_len < 1e-12:
        return False
    acx = c[0] - a[0]
    acy = c[1] - a[1]
    acz = c[2] - a[2]
    ac_len_sq = acx * acx + acy * acy + acz * acz
    if ac_len_sq < 1e-18:
        return False
    bax = b[0] - a[0]
    bay = b[1] - a[1]
    baz = b[2] - a[2]
    t = (bax * acx + bay * acy + baz * acz) / ac_len_sq
    if t <= 0.0 or t >= 1.0:
        return False
    qx = a[0] + t * acx
    qy = a[1] + t * acy
    qz = a[2] + t * acz
    dx = b[0] - qx
    dy = b[1] - qy
    dz = b[2] - qz
    return (dx * dx + dy * dy + dz * dz) <= (tol_mm * tol_mm)


def _collinear_intermediates_within_tol(
    point_at: Callable[[int], tuple[float, float, float]],
    run_start: int,
    run_end: int,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    tol_sq: float,
) -> bool:
    """True if every vertex in ``[run_start, run_end)`` lies within ``tol`` of chord ``p0→p1``."""
    return all(_point_to_segment_dist_sq(p0, p1, point_at(k)) <= tol_sq for k in range(run_start, run_end))


def _clamp_collinear_run_end(
    point_at: Callable[[int], tuple[float, float, float]],
    run_start: int,
    run_end: int,
    p0: tuple[float, float, float],
    tol_mm: float,
) -> tuple[int, tuple[float, float, float]]:
    """Longest prefix of ``[run_start, run_end]`` within ``tol_mm`` of the final chord."""
    if run_end <= run_start:
        return run_end, point_at(run_end)
    tol_sq = float(tol_mm) * float(tol_mm)
    p1 = point_at(run_end)
    if _collinear_intermediates_within_tol(point_at, run_start, run_end, p0, p1, tol_sq):
        return run_end, p1

    best = run_start
    lo = run_start + 1
    hi = run_end - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p_mid = point_at(mid)
        if _collinear_intermediates_within_tol(point_at, run_start, mid, p0, p_mid, tol_sq):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best, point_at(best)


@dataclass(frozen=True)
class CarveJob:
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    tool_def: ToolDefinition | None
    is_cut: bool
    end_vertex: int = 0  # path vertex at segment end (checkpoints / seek)
    a0: float = 0.0
    a1: float = 0.0


@dataclass(frozen=True)
class CarveRangeJob:
    """Carve cut segments in ``(from_vertex, to_vertex]`` from the loaded toolpath."""

    from_vertex: int
    to_vertex: int
    # Idle ahead-carve: cancelled per-segment when real work is submitted.
    low_priority: bool = False


@dataclass(frozen=True)
class RecarveJob:
    """Restore nearest checkpoint ≤ target and carve the remainder on the worker."""

    target_vertex: int = 0


@dataclass(frozen=True)
class PathSnapshot:
    """Read-only toolpath refs for one worker job (copied under the lock)."""

    positions: list[float] | None
    vertex_types: list[float] | None
    tools: list[int] | None
    tool_table: dict | None
    tool_scale: float = 1.0
    angles: list[float] | None = None

    def vertex_count(self) -> int:
        if not self.positions:
            return 0
        return len(self.positions) // 3


# Packed-checkpoint sentinels (compare with ``is``). ``_REMOVED`` = back to FULL.
_PACKED_EMPTY = "EMPTY"
_REMOVED = "REMOVED"


def _pack_chunk_state(state: object) -> object:
    """Bit-pack a chunk's occupancy array (8x smaller than the uint8 grid)."""
    if state is CHUNK_EMPTY:
        return _PACKED_EMPTY
    return np.packbits(np.ascontiguousarray(state).reshape(-1))


def _unpack_chunk_state(packed: object, chunk_size: int) -> object:
    if packed is _PACKED_EMPTY:
        return CHUNK_EMPTY
    n = chunk_size * chunk_size * chunk_size
    arr = np.unpackbits(packed, count=n)
    return arr.reshape(chunk_size, chunk_size, chunk_size).astype(np.uint8)


def _packed_nbytes(value: object) -> int:
    if value is _PACKED_EMPTY or value is _REMOVED:
        return 16  # small fixed overhead for the sentinel + dict key
    return int(value.nbytes)


def _packed_equal(a: object, b: object) -> bool:
    if a is _PACKED_EMPTY or b is _PACKED_EMPTY:
        return a is b
    return bool(a.shape == b.shape and np.array_equal(a, b))


@dataclass(frozen=True)
class GridCheckpoint:
    vertex: int
    # Base: all non-FULL chunks. Delta: changed since previous CP in the GOP
    # (packed array, ``_PACKED_EMPTY``, or ``_REMOVED``).
    is_base: bool
    data: dict[tuple[int, int, int], object]
    nbytes: int


def _equal_spaced_targets(path_vertex_count: int, slot_count: int) -> list[int]:
    """Unique sorted vertex indices evenly spaced over ``(0, V-1]``."""
    v = int(path_vertex_count)
    n = max(1, int(slot_count))
    if v < 2:
        return []
    last = v - 1
    return list(dict.fromkeys(t for t in (int(round(i * last / n)) for i in range(1, n + 1)) if t > 0))


class CheckpointStore:
    """Sparse bit-packed grid snapshots at equal-spaced toolpath targets.

    Most entries are deltas since the previous checkpoint; every
    ``base_interval``-th is a full base so seeks replay at most one GOP.
    """

    def __init__(self, slot_count: int = 1024, base_interval: int = 4):
        self.slot_count = max(1, int(slot_count))
        self.base_interval = max(1, int(base_interval))
        self._path_vertex_count = 0
        self._targets: list[int] = []
        self._next_target_idx = 0
        self._items: list[GridCheckpoint] = []
        # Incremental packed non-FULL map as of the last recorded checkpoint.
        self._last_full_map: dict[tuple[int, int, int], object] = {}

    def clear(self) -> None:
        self._items.clear()
        self._last_full_map = {}
        self._next_target_idx = 0

    def set_path_vertex_count(self, n: int) -> None:
        """Recompute targets for ``n`` vertices; no-op when ``n`` is unchanged."""
        n = max(0, int(n))
        if n == self._path_vertex_count:
            return
        self._path_vertex_count = n
        self._targets = _equal_spaced_targets(n, self.slot_count)
        self.clear()

    @property
    def target_count(self) -> int:
        """Effective slot capacity after dedupe (HUD denominator)."""
        return len(self._targets)

    def vertices(self) -> list[int]:
        return [cp.vertex for cp in self._items]

    def next_unrecorded_target(self) -> int | None:
        """Next equal-spaced target vertex, or ``None`` if all recorded."""
        if self._next_target_idx >= len(self._targets):
            return None
        return int(self._targets[self._next_target_idx])

    def __len__(self) -> int:
        return len(self._items)

    def best_at_or_before(self, vertex: int) -> GridCheckpoint | None:
        best: GridCheckpoint | None = None
        for cp in self._items:
            if cp.vertex <= vertex and (best is None or cp.vertex > best.vertex):
                best = cp
        return best

    def resolve_chunks(self, checkpoint: GridCheckpoint, chunk_size: int) -> dict[tuple[int, int, int], object]:
        """Unpack non-FULL chunks for ``checkpoint`` (base + deltas in its GOP)."""
        idx = self._items.index(checkpoint)
        base_idx = idx
        while not self._items[base_idx].is_base:
            base_idx -= 1

        merged: dict[tuple[int, int, int], object] = dict(self._items[base_idx].data)
        for cp in self._items[base_idx + 1 : idx + 1]:
            for key, val in cp.data.items():
                if val is _REMOVED:
                    merged.pop(key, None)
                else:
                    merged[key] = val

        return {key: _unpack_chunk_state(val, chunk_size) for key, val in merged.items()}

    def maybe_record(
        self,
        vertex: int,
        grid: ChunkedVoxelGrid,
        changed_keys: set[tuple[int, int, int]] | None = None,
    ) -> bool:
        """Snapshot at ``vertex`` if the next equal-spaced target is reached.

        ``changed_keys`` are chunks carved since the last CP (``None`` → base).
        """
        vertex = int(vertex)
        if vertex <= 0 or self._next_target_idx >= len(self._targets):
            return False
        target = self._targets[self._next_target_idx]
        if vertex < target:
            return False
        if self._items and vertex <= self._items[-1].vertex:
            return False

        force_base = changed_keys is None or not self._items or len(self._items) % self.base_interval == 0

        if force_base:
            raw, _ = grid.snapshot_non_full()
            packed = {key: _pack_chunk_state(state) for key, state in raw.items()}
            nbytes = sum(_packed_nbytes(v) for v in packed.values())
            cp = GridCheckpoint(vertex=vertex, is_base=True, data=packed, nbytes=nbytes)
            self._last_full_map = dict(packed)
        else:
            delta: dict[tuple[int, int, int], object] = {}
            for key in changed_keys:
                state = grid.get_chunk_state(ChunkCoord(*key))
                if state is CHUNK_FULL:
                    if key in self._last_full_map:
                        delta[key] = _REMOVED
                    continue
                packed_state = _pack_chunk_state(state)
                prev = self._last_full_map.get(key)
                if prev is None or not _packed_equal(prev, packed_state):
                    delta[key] = packed_state
            # Empty deltas are valid seek anchors (air cuts / no-op carves).
            nbytes = sum(_packed_nbytes(v) for v in delta.values())
            cp = GridCheckpoint(vertex=vertex, is_base=False, data=delta, nbytes=nbytes)
            for key, val in delta.items():
                if val is _REMOVED:
                    self._last_full_map.pop(key, None)
                else:
                    self._last_full_map[key] = val

        self._items.append(cp)
        self._next_target_idx += 1
        while self._next_target_idx < len(self._targets) and self._targets[self._next_target_idx] <= vertex:
            self._next_target_idx += 1
        return True


MeshReadyCallback = Callable[[dict[tuple[int, int, int], tuple | None]], None]
ProgressCallback = Callable[[int], None]
CheckpointsCallback = Callable[[list[int]], None]


def _max_profile_radius(profile: list[tuple[float, float]]) -> float:
    if not profile:
        return 0.0
    return max(r for _z, r in profile)


def _truncate_profile_to_cut(profile: list[tuple[float, float]], cut_z: float) -> list[tuple[float, float]]:
    """Keep only the cutting section of a tool profile (exclude shank)."""
    if not profile or cut_z <= 0:
        return profile
    out: list[tuple[float, float]] = []
    for z, r in profile:
        if z <= cut_z + 1e-9:
            out.append((z, r))
        else:
            break
    if not out:
        return profile[:2] if len(profile) >= 2 else profile
    if out[-1][0] < cut_z - 1e-9:
        out.append((cut_z, out[-1][1]))
    return out


def _resolve_profile(
    tool_def: ToolDefinition | None,
    tool_unit_scale: float = 1.0,
) -> list[tuple[float, float]]:
    """Cutting profile in mm (flute/shoulder only).

    Build/truncate in file units, then scale ``(z, r)`` by ``tool_unit_scale``.
    Do not pass that scale into :func:`tool_profile`. Null-tool fallback is mm.
    """
    if tool_def is None:
        diameter, length = fallback_tool_dimensions(1.0)
        return [(0.0, diameter / 2.0), (length, diameter / 2.0)]

    profile = list(tool_profile(tool_def, scale=1.0))
    diameter = effective_tool_diameter(tool_def, scale=1.0)
    fallback_overall = profile_length(tool_def, scale=1.0, diameter=diameter)
    flute_z, shoulder_z, _overall = resolve_section_lengths(tool_def, fallback_overall)
    cut_z = max(flute_z, shoulder_z)
    profile = _truncate_profile_to_cut(profile, cut_z)
    s = float(tool_unit_scale) if tool_unit_scale else 1.0
    if s != 1.0:
        profile = [(z * s, r * s) for z, r in profile]
    return profile


class StockSimulator:
    """Background voxel carver synced to G-code playback.

    Carving / meshing / checkpoints run on a daemon thread. UI callers should
    only queue work (:meth:`set_toolpath`, :meth:`set_display_vertex`,
    :meth:`submit_range`, :meth:`submit_recarve_to`,
    :meth:`submit_idle_precompute`). Callbacks fire on the worker — hop to the
    Kivy clock before touching GL/widgets.

    Occupancy follows :meth:`set_display_vertex` (latest-wins) so the carved
    mesh can patch during play. Idle ahead-carve uses a second sparse bake
    grid so bookmarks can be recorded past the playhead without moving the
    displayed occupancy. It is skipped when :meth:`set_idle_ahead_allowed`
    is False.
    """

    def __init__(
        self,
        on_meshes_ready: MeshReadyCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_checkpoints: CheckpointsCallback | None = None,
        mesh_throttle_s: float = DEFAULT_MESH_THROTTLE_S,
    ):
        self._on_meshes_ready = on_meshes_ready
        self._on_progress = on_progress
        self._on_checkpoints = on_checkpoints
        self._mesh_throttle_s = mesh_throttle_s
        self._grid: ChunkedVoxelGrid | None = None
        self._bake_grid: ChunkedVoxelGrid | None = None
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.RLock()
        self._generation = 0
        self._resimulating = False
        self._enabled = False
        # Cancel in-flight idle without bumping generation (preserves CP jump).
        self._idle_cancel = threading.Event()
        self._checkpoints = CheckpointStore()
        # Furthest carved vertex on the live / bake grids (worker-owned).
        self._grid_carved_vertex = 0
        self._bake_carved_vertex = 0
        # Latest-wins playhead the worker carves toward (UI-owned writes).
        self._display_vertex = 0
        self._display_poke_pending = False
        self._mesh_flush = False
        # Next mesh emit replaces all GPU chunks (after reset / recarve clear).
        self._force_mesh_replace = False
        # Toolpath published from UI; worker copies refs per job under lock.
        self._path_positions: list[float] | None = None
        self._path_vertex_types: list[float] | None = None
        self._path_tools: list[int] | None = None
        self._path_angles: list[float] | None = None
        self._tool_table: dict | None = None
        self._tool_scale: float = 1.0
        # When False, carve/checkpoint continue but mesh callbacks are skipped.
        self._mesh_updates_enabled = True
        # Viewer enables this only while paused so bake can fill checkpoints.
        self._idle_ahead_allowed = True
        # Placed solid used to seed non-box occupancy (default = full AABB).
        self._shape: StockShape = RectangularStock(width_mm=1.0, length_mm=1.0, height_mm=1.0)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def resimulating(self) -> bool:
        return self._resimulating

    @property
    def grid(self) -> ChunkedVoxelGrid | None:
        return self._grid

    @property
    def carved_vertex(self) -> int:
        """Furthest vertex the live grid has been carved to (UI-thread safe)."""
        with self._lock:
            return int(self._grid_carved_vertex)

    @property
    def bake_grid(self) -> ChunkedVoxelGrid | None:
        """Sparse occupancy used only to record ahead checkpoints."""
        with self._lock:
            return self._bake_grid

    @property
    def bake_carved_vertex(self) -> int:
        """Furthest vertex the bake grid has been carved to (UI-thread safe)."""
        with self._lock:
            return int(self._bake_carved_vertex)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def hud_stats(self) -> dict | None:
        """Viewer HUD snapshot, or ``None`` when simulation is off / has no grid."""
        with self._lock:
            if not self._enabled or self._grid is None:
                return None
            grid = self._grid
            verts = self._checkpoints.vertices()
            head_vertex = max(verts) if verts else 0
            return {
                "voxel_size_mm": float(grid.voxel_size),
                "grid_nx": int(grid.nx),
                "grid_ny": int(grid.ny),
                "grid_nz": int(grid.nz),
                "checkpoint_count": len(self._checkpoints),
                "checkpoint_max_count": int(self._checkpoints.target_count),
                "checkpoint_head_vertex": int(head_vertex),
                "resimulating": bool(self._resimulating),
            }

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="StockSimulator", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._queue.put(_STOP)
        worker = self._worker
        if worker is None:
            return
        worker.join(timeout=2.0)
        if worker.is_alive():
            # Keep the handle so start()/reset() cannot spawn a second worker.
            logger.warning("StockSimulator worker did not stop within timeout")
            return
        if self._worker is worker:
            self._worker = None

    def reset(
        self,
        bounds: StockBounds,
        voxel_size_mm: float | None = None,
        enable: bool = True,
        voxel_target: int | None = None,
        checkpoint_slots: int | None = None,
        shape: StockShape | None = None,
    ) -> None:
        """(Re)initialize the voxel grid. Clears any pending carve jobs."""
        target = voxel_target if voxel_target is not None else VOXEL_TARGET_BY_LEVEL[DEFAULT_VOXEL_RESOLUTION]
        size = voxel_size_mm if voxel_size_mm is not None else pick_voxel_size_mm(bounds, target=target)
        slots = (
            int(checkpoint_slots)
            if checkpoint_slots is not None
            else CHECKPOINT_SLOTS_BY_LEVEL[DEFAULT_CHECKPOINT_LEVEL]
        )
        with self._lock:
            self._generation += 1
            self._shape = (
                shape
                if shape is not None
                else RectangularStock(
                    width_mm=max(bounds.size[0], 1e-6),
                    length_mm=max(bounds.size[1], 1e-6),
                    height_mm=max(bounds.size[2], 1e-6),
                )
            )
            self._grid = ChunkedVoxelGrid(bounds, size)
            seed_shape_occupancy(self._grid, self._shape)
            self._bake_grid = None
            self._enabled = enable
            self._resimulating = False
            self._grid_carved_vertex = 0
            self._bake_carved_vertex = 0
            self._display_vertex = 0
            self._display_poke_pending = False
            self._mesh_flush = False
            self._force_mesh_replace = True
            self._mesh_updates_enabled = True
            self._checkpoints = CheckpointStore(slot_count=slots)
            if self._path_positions is not None:
                self._checkpoints.set_path_vertex_count(len(self._path_positions) // 3)
        self._drain_queue()
        self.start()
        self._emit_checkpoints()
        if enable:
            self._emit_initial_surface()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._generation += 1
            self._grid = None
            self._bake_grid = None
            self._resimulating = False
            self._grid_carved_vertex = 0
            self._bake_carved_vertex = 0
            self._display_vertex = 0
            self._display_poke_pending = False
            self._mesh_flush = False
            self._force_mesh_replace = False
            self._mesh_updates_enabled = True
            self._checkpoints.clear()
        self._drain_queue()
        if self._on_meshes_ready:
            self._on_meshes_ready({"__clear_all__": None})
        self._emit_progress(0)
        self._emit_checkpoints()

    def _emit_progress(self, vertex: int) -> None:
        if not self._on_progress:
            return
        try:
            self._on_progress(int(vertex))
        except Exception:
            logger.exception("stock progress callback failed")

    def _emit_checkpoints(self) -> None:
        if not self._on_checkpoints:
            return
        with self._lock:
            vertices = self._checkpoints.vertices()
        try:
            self._on_checkpoints(vertices)
        except Exception:
            logger.exception("stock checkpoints callback failed")

    def _maybe_record_checkpoint(
        self,
        vertex: int,
        grid: ChunkedVoxelGrid,
        changed_keys: set[tuple[int, int, int]] | None = None,
    ) -> bool:
        recorded = self._checkpoints.maybe_record(vertex, grid, changed_keys)
        if recorded:
            self._emit_checkpoints()
        return recorded

    def set_toolpath(
        self,
        positions: list[float] | None,
        vertex_types: list[float] | None,
        tools: list[int] | None,
        tool_table: dict | None,
        tool_scale: float = 1.0,
        angles: list[float] | None = None,
    ) -> None:
        """Publish the active toolpath (UI thread). Buffers must stay read-only.

        ``tool_scale`` is document-unit → mm (e.g. 25.4); applied after building
        the file-unit cutting profile, not passed into :func:`tool_profile`.
        """
        with self._lock:
            self._path_positions = positions
            self._path_vertex_types = vertex_types
            self._path_tools = tools
            self._path_angles = angles
            self._tool_table = tool_table
            self._tool_scale = float(tool_scale) if tool_scale else 1.0
            n = 0 if positions is None else len(positions) // 3
            self._checkpoints.set_path_vertex_count(n)
        self._emit_checkpoints()

    def clear_toolpath(self) -> None:
        self.set_toolpath(None, None, None, None, tool_scale=1.0)

    def set_display_vertex(self, vertex: int) -> None:
        """Latest-wins playhead: the worker carves or rewinds toward ``vertex``.

        Safe to call every frame. Coalesces to at most one queued poke.
        Preempts idle bake; the worker rearms it after catching up when idle
        ahead is allowed (paused).
        """
        if not self._enabled or self._grid is None:
            return
        v = max(0, int(vertex))
        with self._lock:
            if v == int(self._display_vertex) and v == int(self._grid_carved_vertex):
                return
            self._display_vertex = v
            already = self._display_poke_pending
            self._display_poke_pending = True
        self._idle_cancel.set()
        if not already:
            self._queue.put(_DISPLAY_FOLLOW)

    def set_mesh_throttle_s(self, seconds: float) -> None:
        """Change how long the worker batches dirty chunks before remeshing."""
        with self._lock:
            self._mesh_throttle_s = max(0.0, float(seconds))

    def request_mesh_flush(self) -> None:
        """Emit pending dirty meshes on the next worker wake (pause catch-up)."""
        if not self._enabled or self._grid is None:
            return
        with self._lock:
            self._mesh_flush = True
        self._queue.put(_MESH_FLUSH)

    def submit_range(self, from_vertex: int, to_vertex: int) -> None:
        """Queue carving for ``(from_vertex, to_vertex]`` (expanded on the worker)."""
        if not self._enabled or self._grid is None:
            return
        frm = max(0, int(from_vertex))
        to = max(0, int(to_vertex))
        if to <= frm:
            return
        self._idle_cancel.set()
        with self._lock:
            self._display_vertex = to
        self._queue.put(CarveRangeJob(from_vertex=frm, to_vertex=to))

    def submit_idle_precompute(self, from_vertex: int) -> None:
        """Queue low-priority bake-grid carving to record ahead checkpoints.

        Does not move the display grid. No-op when idle-ahead is disallowed.
        Pre-empted when any non-idle carve is submitted.
        """
        if not self._enabled or self._grid is None:
            return
        with self._lock:
            if not self._idle_ahead_allowed:
                return
            hint = max(0, int(from_vertex))
            to = self._path_vertex_count_locked() - 1
        if to <= 0:
            return
        self._idle_cancel.clear()
        self._queue.put(CarveRangeJob(from_vertex=hint, to_vertex=to, low_priority=True))

    def _maybe_rearm_idle_bake(self) -> None:
        """Resume bake after display follow preempted it (paused scrub)."""
        with self._lock:
            if not self._enabled or not self._idle_ahead_allowed or self._grid is None:
                return
            hint = int(self._display_vertex)
        self.submit_idle_precompute(hint)

    def cancel_idle_precompute(self) -> None:
        """Stop in-flight idle precompute at the next segment boundary."""
        self._idle_cancel.set()

    def set_mesh_updates_enabled(self, enabled: bool) -> None:
        """Toggle mesh emits without stopping carve/checkpoint work.

        Dirty chunks are kept while suppressed. Re-enabling does not remesh;
        use :meth:`request_mesh_flush` for display.
        """
        with self._lock:
            self._mesh_updates_enabled = bool(enabled)

    def set_idle_ahead_allowed(self, allowed: bool) -> None:
        """Allow :meth:`submit_idle_precompute` (pause-only bake grid).

        The viewer enables this while paused and disables it while playing.
        """
        with self._lock:
            self._idle_ahead_allowed = bool(allowed)

    def submit_recarve_to(self, target_vertex: int = 0) -> None:
        """Seek-back / reload: restore checkpoint and carve up to ``target_vertex`` on the worker."""
        if not self._enabled or self._grid is None:
            return
        self._idle_cancel.set()
        with self._lock:
            self._generation += 1
            self._resimulating = True
            self._grid.reset()
            seed_shape_occupancy(self._grid, self._shape)
            self._grid_carved_vertex = 0
            self._display_vertex = max(0, int(target_vertex))
            self._force_mesh_replace = True
        self._drain_queue()
        if self._on_meshes_ready:
            try:
                self._on_meshes_ready({"__clear_all__": None})
            except Exception:
                logger.exception("stock mesh clear callback failed")
        self._queue.put(RecarveJob(target_vertex=max(0, int(target_vertex))))

    def _emit_initial_surface(self) -> None:
        """Queue an empty recarve so the worker meshes the solid exterior shell."""
        if not self._enabled or self._grid is None:
            return
        with self._lock:
            self._resimulating = True
            self._force_mesh_replace = True
        self._queue.put(RecarveJob(target_vertex=0))

    def _seed_grid(self, grid: ChunkedVoxelGrid) -> set[tuple[int, int, int]]:
        """Re-apply shape occupancy after ``grid.reset()``. Returns dirty chunk keys."""
        return seed_shape_occupancy(grid, self._shape)

    def _prepare_bake_grid(self, gen: int, hint_vertex: int) -> tuple[ChunkedVoxelGrid, int] | None:
        """Create/restore the bake grid to the latest bookmark ≤ the bake head.

        ``hint_vertex`` is the display playhead so bake can jump forward if
        playback passed it. Returns ``(bake_grid, start_vertex)``.
        """
        with self._lock:
            if self._generation != gen or not self._enabled or self._grid is None:
                return None
            display = self._grid
            if self._bake_grid is None:
                self._bake_grid = ChunkedVoxelGrid(
                    display.bounds,
                    display.voxel_size,
                    display.chunk_size,
                )
                seed_shape_occupancy(self._bake_grid, self._shape)
                self._bake_carved_vertex = 0
            bake = self._bake_grid
            bake_carved = int(self._bake_carved_vertex)
            hint = max(0, int(hint_vertex))
            target_head = max(bake_carved, hint)
            cp = self._checkpoints.best_at_or_before(target_head)
            if cp is not None:
                self._apply_checkpoint(bake, cp)
                start = int(cp.vertex)
            else:
                bake.reset()
                seed_shape_occupancy(bake, self._shape)
                start = 0
            self._bake_carved_vertex = start
            return bake, start

    def _initial_surface_dirty_keys(self, grid: ChunkedVoxelGrid) -> set[tuple[int, int, int]]:
        """Chunks that must remesh for an uncarved solid (AABB shell + non-box interiors)."""
        dirty = _exterior_chunk_keys(grid)
        dirty.update(non_full_chunk_keys(grid))
        return dirty

    def _path_snapshot_locked(self) -> PathSnapshot:
        """Copy path buffer refs under the lock (caller must hold ``_lock``)."""
        return PathSnapshot(
            positions=self._path_positions,
            vertex_types=self._path_vertex_types,
            tools=self._path_tools,
            tool_table=self._tool_table,
            tool_scale=self._tool_scale,
            angles=self._path_angles,
        )

    def _path_vertex_count_locked(self) -> int:
        """Path length under the lock (caller must hold ``_lock``)."""
        positions = self._path_positions
        if not positions:
            return 0
        return len(positions) // 3

    def _restore_checkpoint_at_or_before(
        self,
        grid: ChunkedVoxelGrid,
        vertex: int,
    ) -> tuple[int, set[tuple[int, int, int]]]:
        """Restore nearest checkpoint ≤ ``vertex``. Caller must hold ``_lock``.

        Returns ``(start_vertex, restored_keys)``. Resets to solid if none.
        """
        cp = self._checkpoints.best_at_or_before(vertex)
        if cp is None:
            grid.reset()
            self._seed_grid(grid)
            return 0, self._initial_surface_dirty_keys(grid)
        chunks = self._checkpoints.resolve_chunks(cp, grid.chunk_size)
        grid.restore_non_full(chunks)
        return int(cp.vertex), set(chunks.keys())

    def _apply_checkpoint(
        self,
        grid: ChunkedVoxelGrid,
        checkpoint: GridCheckpoint,
    ) -> set[tuple[int, int, int]]:
        """Apply ``checkpoint`` to ``grid``. Caller must hold ``_lock``."""
        chunks = self._checkpoints.resolve_chunks(checkpoint, grid.chunk_size)
        grid.restore_non_full(chunks)
        return set(chunks.keys())

    @staticmethod
    def _is_cut_vertex(path: PathSnapshot, idx: int) -> bool:
        types = path.vertex_types
        if not types or idx < 0 or idx >= len(types):
            return False
        return float(types[idx]) <= 1.5

    @staticmethod
    def _tool_number_at_vertex(path: PathSnapshot, idx: int) -> int | None:
        tools = path.tools
        if not tools or idx < 0 or idx >= len(tools):
            return None
        return int(tools[idx])

    @staticmethod
    def _tool_def_at_vertex(path: PathSnapshot, idx: int) -> ToolDefinition | None:
        tools = path.tools
        table = path.tool_table
        if not tools or not table or idx < 0 or idx >= len(tools):
            return None
        return table.get(int(tools[idx]))

    @staticmethod
    def _raw_xyz(path: PathSnapshot, idx: int) -> tuple[float, float, float]:
        pos = path.positions
        base = idx * 3
        return (float(pos[base]), float(pos[base + 1]), float(pos[base + 2]))

    @staticmethod
    def _raw_angle(path: PathSnapshot, idx: int) -> float:
        angs = path.angles
        if not angs or idx < 0 or idx >= len(angs):
            return 0.0
        return float(angs[idx])

    def _carve_job(
        self,
        path: PathSnapshot,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        tool_def: ToolDefinition | None,
        start_idx: int,
        end_idx: int,
    ) -> CarveJob:
        return CarveJob(
            p0=p0,
            p1=p1,
            tool_def=tool_def,
            is_cut=True,
            end_vertex=end_idx,
            a0=self._raw_angle(path, start_idx),
            a1=self._raw_angle(path, end_idx),
        )

    def _iter_cut_jobs(
        self,
        path: PathSnapshot,
        from_vertex: int,
        to_vertex: int,
        *,
        voxel_size_mm: float,
        checkpoints: CheckpointStore,
    ) -> Iterator[CarveJob]:
        """Yield merged cut jobs with ``end_vertex`` in ``(from_vertex, to_vertex]``.

        Near-collinear same-tool cuts collapse to a chord, flushing at the next
        unrecorded checkpoint target or length/span caps. After greedy extend,
        :func:`_clamp_collinear_run_end` keeps intermediates within tolerance.
        """
        n = path.vertex_count()
        if n < 2:
            return
        # Segment ending at i connects vertices i-1 → i.
        start = max(1, int(from_vertex) + 1)
        end = min(n - 1, int(to_vertex))
        merge_tol = _MERGE_TOL_VOXEL_FRAC * float(voxel_size_mm)
        max_merge_mm = max(_MERGE_MAX_MM_FLOOR, _MERGE_MAX_MM_VOXEL_MULT * float(voxel_size_mm))
        max_span = _MERGE_MAX_SPAN

        def point_at(idx: int) -> tuple[float, float, float]:
            return self._raw_xyz(path, idx)

        i = start
        while i <= end:
            if not self._is_cut_vertex(path, i):
                i += 1
                continue

            run_end = i
            p0 = self._raw_xyz(path, i - 1)
            p1 = self._raw_xyz(path, run_end)
            tool_num = self._tool_number_at_vertex(path, run_end)
            tool_def = self._tool_def_at_vertex(path, run_end)
            path_len = _xyz_dist(p0, p1)
            span = 1
            # Re-peek so interleaved maybe_record advances are visible.
            break_at = checkpoints.next_unrecorded_target()
            # Target already behind this segment: flush one job so recording can catch up.
            if break_at is not None and i > break_at:
                yield self._carve_job(path, p0, p1, tool_def, i - 1, run_end)
                i = run_end + 1
                continue

            j = i + 1
            while j <= end:
                if not self._is_cut_vertex(path, j):
                    break
                if self._tool_number_at_vertex(path, j) != tool_num:
                    break
                if break_at is not None and j > break_at:
                    break

                p_new = self._raw_xyz(path, j)
                step = _xyz_dist(p1, p_new)
                a_only = _xyz_dist(p0, p_new) <= merge_tol and step <= merge_tol
                if not a_only and not _can_extend_collinear_merge(p0, p1, p_new, merge_tol):
                    break

                if (not a_only) and (path_len + step > max_merge_mm or span + 1 > max_span):
                    break
                if a_only and span + 1 > max_span:
                    break

                p1 = p_new
                run_end = j
                path_len += step
                span += 1
                j += 1

                if break_at is not None and run_end >= break_at:
                    break

            if run_end > i and not (_xyz_dist(p0, p1) <= merge_tol):
                run_end, p1 = _clamp_collinear_run_end(point_at, i, run_end, p0, merge_tol)

            yield self._carve_job(path, p0, p1, tool_def, i - 1, run_end)
            i = run_end + 1

    def _rewind_grid_to(
        self,
        grid: ChunkedVoxelGrid,
        gen: int,
        vertex: int,
    ) -> tuple[int, set[tuple[int, int, int]]] | None:
        """Restore occupancy to a checkpoint ≤ ``vertex``. Returns ``(start, dirty)``."""
        with self._lock:
            if self._generation != gen or grid is not self._grid:
                return None
            old_keys = non_full_chunk_keys(grid)
            start_vertex, restored_keys = self._restore_checkpoint_at_or_before(grid, vertex)
            self._grid_carved_vertex = start_vertex
            return start_vertex, old_keys | restored_keys

    def _maybe_jump_checkpoint(
        self,
        grid: ChunkedVoxelGrid,
        gen: int,
        start_vertex: int,
        to_vertex: int,
    ) -> tuple[int, set[tuple[int, int, int]]]:
        """If a checkpoint sits between ``start_vertex`` and ``to_vertex``, restore it."""
        with self._lock:
            if self._generation != gen or grid is not self._grid:
                return start_vertex, set()
            cp = self._checkpoints.best_at_or_before(to_vertex)
            if cp is None or cp.vertex <= start_vertex:
                return start_vertex, set()
            old_keys = non_full_chunk_keys(grid)
            restored_keys = self._apply_checkpoint(grid, cp)
            self._grid_carved_vertex = cp.vertex
            return cp.vertex, old_keys | restored_keys

    def _carve_segments_to(
        self,
        grid: ChunkedVoxelGrid,
        path: PathSnapshot,
        gen: int,
        start_vertex: int,
        to_vertex: int,
        pending_dirty: set[tuple[int, int, int]],
        changed_since_cp: set[tuple[int, int, int]],
        *,
        idle: bool = False,
        follow_display: bool = False,
    ) -> tuple[int, bool]:
        """Carve ``(start_vertex, to_vertex]``. Returns ``(last_end, interrupted)``."""
        last_end_vertex = start_vertex
        interrupted = False
        last_progress_t = 0.0
        for seg in self._iter_cut_jobs(
            path,
            start_vertex,
            to_vertex,
            voxel_size_mm=grid.voxel_size,
            checkpoints=self._checkpoints,
        ):
            if self._generation != gen or self._resimulating:
                interrupted = True
                break
            if idle and self._idle_cancel.is_set():
                interrupted = True
                break
            if follow_display:
                with self._lock:
                    live_target = int(self._display_vertex)
                if live_target < seg.end_vertex:
                    interrupted = True
                    break
            with self._lock:
                if self._generation != gen or self._resimulating:
                    interrupted = True
                    break
                dirty = self._carve_one(grid, seg, path.tool_scale)
                pending_dirty.update(dirty)
                changed_since_cp.update(dirty)
                if self._maybe_record_checkpoint(seg.end_vertex, grid, changed_since_cp):
                    changed_since_cp.clear()
                if idle:
                    self._bake_carved_vertex = seg.end_vertex
                else:
                    self._grid_carved_vertex = seg.end_vertex
            last_end_vertex = seg.end_vertex
            if not idle:
                now_p = time.monotonic()
                if now_p - last_progress_t >= 0.05:
                    self._emit_progress(seg.end_vertex)
                    last_progress_t = now_p
        return last_end_vertex, interrupted

    def _follow_display_vertex(
        self,
        grid: ChunkedVoxelGrid,
        path: PathSnapshot,
        gen: int,
        pending_dirty: set[tuple[int, int, int]],
        changed_since_cp: set[tuple[int, int, int]],
    ) -> None:
        """Carve or rewind the live grid to ``_display_vertex``."""
        while self._generation == gen and not self._resimulating:
            with self._lock:
                if self._generation != gen or grid is not self._grid:
                    return
                target = int(self._display_vertex)
                carved = int(self._grid_carved_vertex)
            if target == carved:
                return
            start_vertex = carved
            if target < carved:
                rewound = self._rewind_grid_to(grid, gen, target)
                if rewound is None:
                    return
                start_vertex, dirty = rewound
                pending_dirty.update(dirty)
                changed_since_cp.clear()
                self._emit_progress(start_vertex)
                if target <= start_vertex:
                    with self._lock:
                        if self._generation == gen:
                            self._grid_carved_vertex = int(target)
                    self._emit_progress(target)
                    return
            start_vertex, jump_dirty = self._maybe_jump_checkpoint(grid, gen, start_vertex, target)
            if jump_dirty:
                pending_dirty.update(jump_dirty)
                changed_since_cp.clear()
                self._emit_progress(start_vertex)
            _last_end, interrupted = self._carve_segments_to(
                grid,
                path,
                gen,
                start_vertex,
                target,
                pending_dirty,
                changed_since_cp,
                follow_display=True,
            )
            if self._generation != gen:
                return
            with self._lock:
                live_target = int(self._display_vertex)
                if not interrupted:
                    self._grid_carved_vertex = int(target)
            if not interrupted:
                self._emit_progress(target)
            if live_target == target and not interrupted:
                return
            # Target moved (ahead or behind) or a mid-carve interrupt: loop.

    def _worker_loop(self) -> None:
        pending_dirty: set[tuple[int, int, int]] = set()
        # Chunks carved since the last recorded checkpoint (delta candidates).
        display_changed_since_cp: set[tuple[int, int, int]] = set()
        bake_changed_since_cp: set[tuple[int, int, int]] = set()
        dirty_gen = self._generation
        last_emit = 0.0

        while True:
            with self._lock:
                throttle = self._mesh_throttle_s
            try:
                item = self._queue.get(timeout=throttle)
            except queue.Empty:
                item = None

            if item is _STOP:
                break

            with self._lock:
                gen = self._generation
                grid = self._grid
                enabled = self._enabled
                resimulating = self._resimulating
                path = self._path_snapshot_locked()
                display_vertex = int(self._display_vertex)
                carved_vertex = int(self._grid_carved_vertex)
                do_flush = self._mesh_flush
                force_replace = self._force_mesh_replace
                mesh_updates = self._mesh_updates_enabled

            if gen != dirty_gen:
                pending_dirty.clear()
                display_changed_since_cp.clear()
                bake_changed_since_cp.clear()
                dirty_gen = gen

            if grid is None or not enabled:
                pending_dirty.clear()
                display_changed_since_cp.clear()
                bake_changed_since_cp.clear()
                continue

            # During seek/reset, only RecarveJob may mutate the display grid.
            if resimulating and not isinstance(item, RecarveJob):
                pending_dirty.clear()
                display_changed_since_cp.clear()
                continue

            finished_recarve = False
            recarve_finished = False
            did_display_follow = False
            if item is _DISPLAY_FOLLOW:
                with self._lock:
                    self._display_poke_pending = False
                self._follow_display_vertex(grid, path, gen, pending_dirty, display_changed_since_cp)
                did_display_follow = True

            elif item is _MESH_FLUSH:
                pass

            elif item is None and mesh_updates and display_vertex != carved_vertex:
                self._follow_display_vertex(grid, path, gen, pending_dirty, display_changed_since_cp)
                did_display_follow = True

            elif isinstance(item, RecarveJob):
                with self._lock:
                    if self._generation != gen:
                        continue
                    if grid is not self._grid:
                        continue
                    self._resimulating = True
                    start_vertex, restored_keys = self._restore_checkpoint_at_or_before(grid, item.target_vertex)
                    self._grid_carved_vertex = start_vertex
                pending_dirty.clear()
                pending_dirty.update(restored_keys)
                display_changed_since_cp.clear()
                dirty_gen = gen
                self._emit_progress(start_vertex)
                last_end, _interrupted = self._carve_segments_to(
                    grid,
                    path,
                    gen,
                    start_vertex,
                    item.target_vertex,
                    pending_dirty,
                    display_changed_since_cp,
                )
                if self._generation == gen:
                    final_vertex = int(item.target_vertex if item.target_vertex else last_end)
                    self._emit_progress(final_vertex)
                    with self._lock:
                        self._grid_carved_vertex = final_vertex
                    # Remesh the stock shell so __replace__ does not leave uncarved voids.
                    pending_dirty.update(self._initial_surface_dirty_keys(grid))
                with self._lock:
                    if self._generation == gen:
                        finished_recarve = True
                        recarve_finished = True
                    else:
                        pending_dirty.clear()
                        continue
                last_emit = 0.0

            elif isinstance(item, CarveRangeJob) and item.low_priority:
                if self._generation != gen or self._resimulating:
                    continue
                if self._idle_cancel.is_set():
                    continue
                prepared = self._prepare_bake_grid(gen, item.from_vertex)
                if prepared is None:
                    continue
                bake, start_vertex = prepared
                bake_changed_since_cp.clear()
                if start_vertex >= item.to_vertex:
                    continue
                bake_dirty: set[tuple[int, int, int]] = set()
                self._carve_segments_to(
                    bake,
                    path,
                    gen,
                    start_vertex,
                    item.to_vertex,
                    bake_dirty,
                    bake_changed_since_cp,
                    idle=True,
                )
                continue

            elif isinstance(item, CarveRangeJob):
                if self._generation != gen or self._resimulating:
                    pending_dirty.clear()
                    display_changed_since_cp.clear()
                    continue

                start_vertex = item.from_vertex

                if self._grid_carved_vertex > item.from_vertex:
                    rewound = self._rewind_grid_to(grid, gen, item.from_vertex)
                    if rewound is None:
                        continue
                    start_vertex, dirty_keys = rewound
                    pending_dirty.update(dirty_keys)
                    display_changed_since_cp.clear()
                    dirty_gen = gen

                start_vertex, jump_dirty = self._maybe_jump_checkpoint(grid, gen, start_vertex, item.to_vertex)
                if jump_dirty:
                    pending_dirty.update(jump_dirty)
                    display_changed_since_cp.clear()
                    dirty_gen = gen

                last_end_vertex, interrupted = self._carve_segments_to(
                    grid,
                    path,
                    gen,
                    start_vertex,
                    item.to_vertex,
                    pending_dirty,
                    display_changed_since_cp,
                )

                if self._generation == gen:
                    final_vertex = last_end_vertex if interrupted else item.to_vertex
                    self._emit_progress(final_vertex)
                    with self._lock:
                        self._grid_carved_vertex = int(final_vertex)

            now = time.monotonic()
            with self._lock:
                mesh_updates = self._mesh_updates_enabled
                do_flush = self._mesh_flush
                force_replace = self._force_mesh_replace
                throttle = self._mesh_throttle_s
            replace = finished_recarve or force_replace
            should_emit = replace or do_flush or (pending_dirty and (now - last_emit) >= throttle)
            if should_emit:
                if not mesh_updates:
                    # Keep dirty so pause can flush visited chunks without a
                    # full-shell replace. Consume flush to avoid a busy loop.
                    last_emit = now
                    with self._lock:
                        self._mesh_flush = False
                else:
                    dirty = set(pending_dirty)
                    pending_dirty.clear()
                    if self._try_mesh_and_emit(grid, dirty, gen, replace=replace, force_emit=do_flush):
                        last_emit = now
                        with self._lock:
                            if replace:
                                self._force_mesh_replace = False
                            self._mesh_flush = False

            if recarve_finished:
                with self._lock:
                    if self._generation == gen:
                        self._resimulating = False
                with self._lock:
                    poke_display = self._display_poke_pending
                if poke_display:
                    self._queue.put(_DISPLAY_FOLLOW)

            if did_display_follow:
                self._maybe_rearm_idle_bake()

        if pending_dirty:
            with self._lock:
                grid = self._grid
                gen = self._generation
                enabled = self._enabled
                mesh_updates = self._mesh_updates_enabled
            if enabled and grid is not None and mesh_updates:
                self._try_mesh_and_emit(grid, pending_dirty, gen, replace=False)
            pending_dirty.clear()

    def _try_mesh_and_emit(
        self,
        grid: ChunkedVoxelGrid,
        dirty: set[tuple[int, int, int]],
        gen: int,
        *,
        replace: bool,
        force_emit: bool = False,
    ) -> bool:
        """Copy mesh-relevant chunks under lock, mesh unlocked, emit if still current.

        Copies the 2-ring around ``dirty`` so unlocked meshing never touches the
        live grid. Generation is stamped under lock to match the emitted meshes.
        ``force_emit`` (pause flush) notifies even when nothing is dirty so the
        viewer can swap AABB fill for voxels.
        """
        if not self._on_meshes_ready:
            return False

        with self._lock:
            if self._generation != gen or grid is not self._grid or not self._enabled or self._grid is None:
                return False
            dirty_keys = set(dirty)
            if dirty_keys:
                mesh_keys = _expand_dirty_with_neighbors(grid, dirty_keys)
                copy_keys = _expand_dirty_with_neighbors(grid, mesh_keys)
                tmp = grid.copy_chunks(copy_keys)
            else:
                mesh_keys = set()
                tmp = None

        if dirty_keys and tmp is not None:
            from .voxel_mesher import mesh_dirty_chunks

            meshes = mesh_dirty_chunks(tmp, mesh_keys)
        else:
            meshes = {}

        with self._lock:
            if self._generation != gen or grid is not self._grid:
                return False
            try:
                if replace:
                    if not dirty_keys and not meshes:
                        if force_emit:
                            self._on_meshes_ready({})
                        return True
                    self._on_meshes_ready({"__replace__": meshes})
                elif meshes or force_emit:
                    self._on_meshes_ready(meshes)
            except Exception:
                logger.exception("stock mesh callback failed")
            return True

    def _carve_one(
        self,
        grid: ChunkedVoxelGrid,
        job: CarveJob,
        tool_unit_scale: float = 1.0,
    ) -> set[tuple[int, int, int]]:
        if not job.is_cut:
            return set()
        kwargs = {}
        if abs(float(job.a0)) > 1e-9 or abs(float(job.a1)) > 1e-9:
            kwargs["a0"] = job.a0
            kwargs["a1"] = job.a1
        return carve_segment_into_grid(
            grid,
            job.p0,
            job.p1,
            job.tool_def,
            tool_unit_scale=tool_unit_scale,
            **kwargs,
        )


def _local_z_slab(oz: float, vs: float, cs: int, tip_z_lo: float, tip_z_hi: float) -> tuple[int, int]:
    """Local Z index range ``[iz_lo, iz_hi)`` whose centres may meet the cutter.

    Centres are ``oz + (iz + 0.5) * vs``. ``tip_z_lo`` / ``tip_z_hi`` already include
    a one-voxel pad; ``floor`` on both ends keeps every centre the un-slabbed dense
    path could have accepted.
    """
    if vs <= 0 or cs <= 0:
        return 0, max(0, cs)
    lo = (float(tip_z_lo) - float(oz)) / float(vs) - 0.5
    hi = (float(tip_z_hi) - float(oz)) / float(vs) - 0.5
    iz_lo = max(0, int(np.floor(lo)))
    iz_hi = min(cs, int(np.floor(hi)) + 1)
    return iz_lo, iz_hi


def _constant_profile_radius(profile_rs: np.ndarray) -> float | None:
    """Return the shared radius when every profile knot is the same, else ``None``."""
    if profile_rs.size == 0:
        return None
    rmin = float(np.min(profile_rs))
    rmax = float(np.max(profile_rs))
    if rmax - rmin < 1e-12:
        return rmax
    return None


def _aabb_rotated_around_x(
    aabb: tuple[float, float, float, float, float, float],
    angle_deg: float,
) -> tuple[float, float, float, float, float, float]:
    """Axis-aligned bounds of *aabb* after rotating YZ around X by *angle_deg*."""
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    ys: list[float] = []
    zs: list[float] = []
    for y in (min_y, max_y):
        for z in (min_z, max_z):
            y2, z2 = rotate_yz(y, z, angle_deg)
            ys.append(y2)
            zs.append(z2)
    return (min_x, min(ys), min(zs), max_x, max(ys), max(zs))


def _rotate_yz_np(yy: np.ndarray, zz: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(float(angle_deg))
    c = np.cos(rad)
    s = np.sin(rad)
    return yy * c - zz * s, yy * s + zz * c


def carve_segment_into_grid(
    grid: ChunkedVoxelGrid,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    tool_def: ToolDefinition | None,
    tool_unit_scale: float = 1.0,
    *,
    sparse_solid_frac: float = 0.5,
    a0: float = 0.0,
    a1: float = 0.0,
) -> set[tuple[int, int, int]]:
    """Carve one toolpath segment into ``grid``.

    Returns chunks whose occupancy changed. Remesh callers should expand with
    :func:`_expand_dirty_with_neighbors`. Tip follows ``p0→p1``; cutting body is
    the upright surface of revolution from :func:`tool_profile`. A voxel clears
    iff some tip pose on the segment contains it (Z-valid tip window + XY-closest
    tip, then profile radius). ``sparse_solid_frac`` selects the solid-only gather
    path (tests may force ``0.0`` / ``1.0``).

    ``a0`` / ``a1`` are A-axis angles in degrees at the segment ends. Non-zero
    A rotates the stock: voxels are tested in machine XYZ via ``Rx(-A)``.
    """
    a0 = float(a0)
    a1 = float(a1)
    da = a1 - a0
    if abs(da) > _MAX_DA_DEG:
        n = max(1, int(math.ceil(abs(da) / _MAX_DA_DEG)))
        dirty: set[tuple[int, int, int]] = set()
        dxs = p1[0] - p0[0]
        dys = p1[1] - p0[1]
        dzs = p1[2] - p0[2]
        for i in range(n):
            t0 = i / n
            t1 = (i + 1) / n
            pi0 = (p0[0] + t0 * dxs, p0[1] + t0 * dys, p0[2] + t0 * dzs)
            pi1 = (p0[0] + t1 * dxs, p0[1] + t1 * dys, p0[2] + t1 * dzs)
            ai = a0 + 0.5 * (t0 + t1) * da
            dirty |= carve_segment_into_grid(
                grid,
                pi0,
                pi1,
                tool_def,
                tool_unit_scale,
                sparse_solid_frac=sparse_solid_frac,
                a0=ai,
                a1=ai,
            )
        return dirty

    profile = _resolve_profile(tool_def, tool_unit_scale=tool_unit_scale)
    max_r = _max_profile_radius(profile)
    flute_z = profile[-1][0] if profile else 0.0
    if max_r <= 0 and flute_z <= 0:
        return set()

    profile_zs = np.array([z for z, _r in profile], dtype=np.float64) if profile else np.zeros(0)
    profile_rs = np.array([r for _z, r in profile], dtype=np.float64) if profile else np.zeros(0)

    xs = (p0[0], p1[0])
    ys = (p0[1], p1[1])
    zs = (p0[2], p1[2])
    pad = max_r + grid.voxel_size
    aabb = (
        min(xs) - pad,
        min(ys) - pad,
        min(zs) - pad,
        max(xs) + pad,
        max(ys) + pad,
        max(zs) + flute_z + pad,
    )
    angle = a0
    rotate = abs(angle) > 1e-9
    if rotate:
        aabb = _aabb_rotated_around_x(aabb, angle)

    dirty: set[tuple[int, int, int]] = set()
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    seg_len_sq = dx * dx + dy * dy + dz * dz
    xy_len_sq = dx * dx + dy * dy
    z0 = float(profile[0][0]) if profile else 0.0
    z1 = float(profile[-1][0]) if profile else 0.0
    eps = 1e-9
    vs = grid.voxel_size
    tip_z_lo = min(p0[2], p1[2]) - vs
    tip_z_hi = max(p0[2], p1[2]) + flute_z + vs
    xy_reject_r = max_r + vs

    for coord in grid.iter_chunks_in_aabb(*aabb):
        state = grid.get_chunk_state(coord)
        if state is CHUNK_EMPTY:
            continue

        cs = grid.chunk_size
        ox, oy, oz = grid.chunk_world_origin(coord)
        cx1 = ox + cs * vs
        cy1 = oy + cs * vs
        cz1 = oz + cs * vs

        if not rotate:
            if cz1 < tip_z_lo or oz > tip_z_hi:
                continue
            if _segment_aabb_dist_xy(p0[0], p0[1], p1[0], p1[1], ox, oy, cx1, cy1) > xy_reject_r:
                continue

        arr = grid.get_or_create_chunk(coord)
        if arr is None:
            continue

        solid_count = int(arr.sum())
        if solid_count == 0:
            grid.maybe_collapse_chunk(coord, arr)
            continue

        chunk_voxels = cs * cs * cs
        use_sparse = solid_count <= chunk_voxels * sparse_solid_frac

        ix0 = coord.cx * cs
        iy0 = coord.cy * cs
        iz0 = coord.cz * cs
        valid_x = (ix0 + np.arange(cs)) < grid.nx
        valid_y = (iy0 + np.arange(cs)) < grid.ny
        valid_z = (iz0 + np.arange(cs)) < grid.nz
        z_lo, z_hi = _local_z_slab(oz, vs, cs, tip_z_lo, tip_z_hi)
        if rotate:
            z_lo, z_hi = 0, cs

        if use_sparse:
            ixs, iys, izs = np.nonzero(arr)
            XX = ox + (ixs.astype(np.float64) + 0.5) * vs
            YY = oy + (iys.astype(np.float64) + 0.5) * vs
            ZZ = oz + (izs.astype(np.float64) + 0.5) * vs
            if rotate:
                YY, ZZ = _rotate_yz_np(YY, ZZ, -angle)
            inside = _voxels_inside_tool(
                XX,
                YY,
                ZZ,
                p0,
                dx,
                dy,
                dz,
                seg_len_sq,
                xy_len_sq,
                z0,
                z1,
                profile,
                profile_zs,
                profile_rs,
                eps,
                max_r,
            )
            # Mask out voxels outside the global grid.
            inside &= valid_x[ixs] & valid_y[iys] & valid_z[izs]
            if not inside.any():
                grid.maybe_collapse_chunk(coord, arr)
                continue
            n_hit = int(np.count_nonzero(inside))
            arr[ixs[inside], iys[inside], izs[inside]] = 0
            after = solid_count - n_hit
        else:
            if z_lo >= z_hi:
                grid.maybe_collapse_chunk(coord, arr)
                continue
            lx = ox + (np.arange(cs, dtype=np.float64) + 0.5) * vs
            ly = oy + (np.arange(cs, dtype=np.float64) + 0.5) * vs
            lz = oz + (np.arange(z_lo, z_hi, dtype=np.float64) + 0.5) * vs
            XX = lx[:, None, None]
            YY = ly[None, :, None]
            ZZ = lz[None, None, :]
            if rotate:
                YY, ZZ = _rotate_yz_np(YY, ZZ, -angle)
            inside = _voxels_inside_tool(
                XX,
                YY,
                ZZ,
                p0,
                dx,
                dy,
                dz,
                seg_len_sq,
                xy_len_sq,
                z0,
                z1,
                profile,
                profile_zs,
                profile_rs,
                eps,
                max_r,
            )
            inside &= valid_x[:, None, None] & valid_y[None, :, None] & valid_z[z_lo:z_hi][None, None, :]
            slab = arr[:, :, z_lo:z_hi]
            inside &= slab.astype(bool)
            if not inside.any():
                grid.maybe_collapse_chunk(coord, arr)
                continue
            n_hit = int(np.count_nonzero(inside))
            slab[inside] = 0
            after = solid_count - n_hit

        grid.maybe_collapse_chunk(coord, arr)
        if after != solid_count:
            dirty.add(coord.as_tuple())

    return dirty


def _voxels_inside_tool(
    XX: np.ndarray,
    YY: np.ndarray,
    ZZ: np.ndarray,
    p0: tuple[float, float, float],
    dx: float,
    dy: float,
    dz: float,
    seg_len_sq: float,
    xy_len_sq: float,
    z0: float,
    z1: float,
    profile: list[tuple[float, float]],
    profile_zs: np.ndarray,
    profile_rs: np.ndarray,
    eps: float,
    max_r: float,
) -> np.ndarray:
    """Shared tip-window / radius mask for dense or sparse coordinate layouts.

    ``max_r`` is the profile peak; centres farther than that in XY are rejected
    before radius interpolation. A constant-radius profile (flat mill) skips
    ``searchsorted`` entirely.
    """
    XX, YY, ZZ = np.broadcast_arrays(
        np.asarray(XX, dtype=np.float64),
        np.asarray(YY, dtype=np.float64),
        np.asarray(ZZ, dtype=np.float64),
    )
    const_r = _constant_profile_radius(profile_rs)

    def _inside_from_xy(
        window_ok: np.ndarray,
        dist_xy: np.ndarray,
        *,
        z_rel: np.ndarray | None = None,
        band_lo: np.ndarray | None = None,
        band_hi: np.ndarray | None = None,
    ) -> np.ndarray:
        window_ok, dist_xy = np.broadcast_arrays(window_ok, dist_xy)
        candidate = window_ok & (dist_xy <= max_r + eps)
        if not np.any(candidate):
            return np.zeros(candidate.shape, dtype=bool)
        if const_r is not None:
            return candidate if const_r > 0 else np.zeros(candidate.shape, dtype=bool)
        radii = np.zeros(candidate.shape, dtype=np.float64)
        if band_lo is not None and band_hi is not None:
            lo_b, hi_b, cand_b = np.broadcast_arrays(band_lo, band_hi, candidate)
            radii[cand_b] = _max_profile_radii_in_band(
                profile, lo_b[cand_b], hi_b[cand_b], zs=profile_zs, rs=profile_rs
            )
            dist_b = np.broadcast_to(dist_xy, cand_b.shape)
            return cand_b & (dist_b <= radii + eps) & (radii > 0)
        if z_rel is None:
            return np.zeros(candidate.shape, dtype=bool)
        z_b, cand_b = np.broadcast_arrays(z_rel, candidate)
        radii[cand_b] = _sample_profile_radii(profile, z_b[cand_b], zs=profile_zs, rs=profile_rs)
        dist_b = np.broadcast_to(dist_xy, cand_b.shape)
        return cand_b & (dist_b <= radii + eps) & (radii > 0)

    if seg_len_sq < 1e-18:
        z_rel = ZZ - p0[2]
        dist_xy = np.sqrt((XX - p0[0]) ** 2 + (YY - p0[1]) ** 2)
        if profile_zs.size:
            window_ok = (z_rel >= profile_zs[0] - 1e-9) & (z_rel <= profile_zs[-1] + 1e-9)
        else:
            window_ok = np.zeros(ZZ.shape, dtype=bool)
        return _inside_from_xy(window_ok, dist_xy, z_rel=z_rel)

    if xy_len_sq < 1e-18:
        # Pure plunge — clear if within max radius over the z_rel band.
        tip_z_lo = min(p0[2], p0[2] + dz)
        tip_z_hi = max(p0[2], p0[2] + dz)
        z_rel_lo = ZZ - tip_z_hi
        z_rel_hi = ZZ - tip_z_lo
        band_lo = np.maximum(z_rel_lo, z0)
        band_hi = np.minimum(z_rel_hi, z1)
        window_ok = band_lo <= band_hi + eps
        dist_xy = np.sqrt((XX - p0[0]) ** 2 + (YY - p0[1]) ** 2)
        return _inside_from_xy(window_ok, dist_xy, band_lo=band_lo, band_hi=band_hi)

    # XY motion: Z-valid tip window, then XY-closest tip in that window.
    if abs(dz) < 1e-18:
        z_rel_fixed = ZZ - p0[2]
        window_ok = (z_rel_fixed >= z0 - eps) & (z_rel_fixed <= z1 + eps)
        t_lo = np.where(window_ok, 0.0, 1.0)
        t_hi = np.where(window_ok, 1.0, 0.0)
    else:
        t_a = (ZZ - z1 - p0[2]) / dz
        t_b = (ZZ - z0 - p0[2]) / dz
        t_lo = np.maximum(np.minimum(t_a, t_b), 0.0)
        t_hi = np.minimum(np.maximum(t_a, t_b), 1.0)
        window_ok = t_lo <= t_hi + eps

    t = ((XX - p0[0]) * dx + (YY - p0[1]) * dy) / xy_len_sq
    t = np.minimum(np.maximum(t, t_lo), t_hi)
    tip_x = p0[0] + t * dx
    tip_y = p0[1] + t * dy
    tip_z = p0[2] + t * dz
    z_rel = ZZ - tip_z
    dist_xy = np.sqrt((XX - tip_x) ** 2 + (YY - tip_y) ** 2)
    return _inside_from_xy(window_ok, dist_xy, z_rel=z_rel)


def _segments_intersect_2d(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """True if closed segments AB and CD intersect (including touching)."""

    def _orient(px, py, qx, qy, rx, ry) -> float:
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

    def _on_seg(px, py, qx, qy, rx, ry) -> bool:
        return min(px, rx) - 1e-12 <= qx <= max(px, rx) + 1e-12 and min(py, ry) - 1e-12 <= qy <= max(py, ry) + 1e-12

    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True
    if abs(o1) <= 1e-12 and _on_seg(ax, ay, cx, cy, bx, by):
        return True
    if abs(o2) <= 1e-12 and _on_seg(ax, ay, dx, dy, bx, by):
        return True
    if abs(o3) <= 1e-12 and _on_seg(cx, cy, ax, ay, dx, dy):
        return True
    return abs(o4) <= 1e-12 and _on_seg(cx, cy, bx, by, dx, dy)


def _segment_aabb_dist_xy(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> float:
    """Minimum 2D distance from tip segment AB to an axis-aligned rectangle."""

    def _inside(px: float, py: float) -> bool:
        return xmin <= px <= xmax and ymin <= py <= ymax

    if _inside(ax, ay) or _inside(bx, by):
        return 0.0

    edges = (
        (xmin, ymin, xmax, ymin),
        (xmax, ymin, xmax, ymax),
        (xmax, ymax, xmin, ymax),
        (xmin, ymax, xmin, ymin),
    )
    for ex0, ey0, ex1, ey1 in edges:
        if _segments_intersect_2d(ax, ay, bx, by, ex0, ey0, ex1, ey1):
            return 0.0

    def _point_to_aabb(px: float, py: float) -> float:
        cx = min(max(px, xmin), xmax)
        cy = min(max(py, ymin), ymax)
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _point_to_seg(px: float, py: float) -> float:
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-18:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))
        qx = ax + t * abx
        qy = ay + t * aby
        return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5

    d = min(_point_to_aabb(ax, ay), _point_to_aabb(bx, by))
    for cx, cy in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)):
        d = min(d, _point_to_seg(cx, cy))
    return float(d)


def _max_profile_radii_in_band(
    profile: list[tuple[float, float]],
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    *,
    zs: np.ndarray | None = None,
    rs: np.ndarray | None = None,
) -> np.ndarray:
    """Max profile radius over each voxel's achievable ``[band_lo, band_hi]``."""
    # Endpoints cover flat mills exactly; for ball/V, also sample interior knots.
    r_lo = _sample_profile_radii(profile, band_lo, zs=zs, rs=rs)
    r_hi = _sample_profile_radii(profile, band_hi, zs=zs, rs=rs)
    out = np.maximum(r_lo, r_hi)
    if len(profile) <= 2:
        return out
    knot_zs = zs[1:-1] if zs is not None and len(zs) == len(profile) else None
    if knot_zs is None:
        knot_iter = (float(z) for z, _r in profile[1:-1])
    else:
        knot_iter = (float(z) for z in knot_zs)
    for z in knot_iter:
        z_arr = np.full_like(band_lo, z)
        in_band = (z_arr >= band_lo - 1e-9) & (z_arr <= band_hi + 1e-9)
        if not in_band.any():
            continue
        r_knot = _sample_profile_radii(profile, z_arr, zs=zs, rs=rs)
        out = np.where(in_band, np.maximum(out, r_knot), out)
    return out


def _sample_profile_radii(
    profile: list[tuple[float, float]],
    z_rel: np.ndarray,
    *,
    zs: np.ndarray | None = None,
    rs: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized profile radius lookup for an array of tool-relative Z values."""
    if not profile:
        return np.zeros_like(z_rel, dtype=np.float64)

    if zs is None or rs is None:
        zs = np.array([z for z, _r in profile], dtype=np.float64)
        rs = np.array([r for _z, r in profile], dtype=np.float64)

    flat = z_rel.ravel()
    out = np.zeros_like(flat, dtype=np.float64)

    # Below tip or above stick-out → 0
    below = flat < zs[0] - 1e-9
    above = flat > zs[-1] + 1e-9
    mid = ~(below | above)

    if mid.any():
        idx = np.searchsorted(zs, flat[mid], side="right") - 1
        idx = np.clip(idx, 0, len(zs) - 2)
        z0 = zs[idx]
        z1 = zs[idx + 1]
        r0 = rs[idx]
        r1 = rs[idx + 1]
        denom = z1 - z0
        t = np.zeros_like(flat[mid], dtype=np.float64)
        nonzero = np.abs(denom) >= 1e-12
        if nonzero.any():
            t[nonzero] = (flat[mid][nonzero] - z0[nonzero]) / denom[nonzero]
        out[mid] = r0 + t * (r1 - r0)

    return out.reshape(z_rel.shape)
