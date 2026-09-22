"""One stock drawing strategy, chosen once from vertex-texture support.

``FieldDisplay`` keeps a static grid and blits the height or radius field.
``RemeshDisplay`` is the existing tile mesher. Carvers do not branch on GL.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from carveracontroller.addons.stock.simulator.carver_select import (
    BACKEND_CYLINDRICAL,
    BACKEND_HEIGHTMAP,
)
from carveracontroller.addons.stock.simulator.carvers.cylindrical.backend import shell_floor_radius_mm
from carveracontroller.addons.stock.simulator.carvers.field_mesh import (
    build_heightmap_field_mesh,
    build_rotary_field_mesh,
    pack_height_field,
    pack_radius_field,
    radius_span_mm,
)

logger = logging.getLogger(__name__)

_GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS = 35660


def vertex_texture_image_units() -> int | None:
    """Vertex texture units, or None when no GL context is current."""
    try:
        from kivy.graphics.opengl import glGetError, glGetIntegerv
    except Exception:
        return None
    try:
        for _ in range(8):
            if int(glGetError()) == 0:
                break
        raw = glGetIntegerv(_GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)
        err = int(glGetError())
    except Exception:
        return None
    if err != 0:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return 0
        raw = raw[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def select_stock_display(vertex_texture_units: int):
    """Field displacement when the GPU can sample a texture in the vertex shader."""
    if int(vertex_texture_units) > 0:
        logger.info("stock display: field displacement (%s vertex texture units)", int(vertex_texture_units))
        return FieldDisplay()
    logger.info("stock display: remesh (no vertex texture fetch)")
    return RemeshDisplay()


class RemeshDisplay:
    """Today's ``mesh_tiles`` path, including the cylindrical full-shell coalesce."""

    def publish(
        self,
        sim: Any,
        backend: Any,
        dirty: set[tuple[int, int, int]],
        gen: int,
        *,
        replace: bool,
        force_emit: bool = False,
    ) -> bool:
        if not sim._on_meshes_ready:
            return True

        with sim._lock:
            if sim._generation != gen or not sim._enabled or sim._backend is None:
                return False
            if backend is not sim._backend:
                return False
            dirty_keys = set(dirty)
            # Cylindrical wraps in θ and is drawn as one field. Playback patches
            # that field in place (no __replace__) so the viewer does not
            # destroy GPU meshes every throttle window. Heightmap/voxel tiles
            # patch incrementally.
            coalesce = str(getattr(backend, "kind", "")) == BACKEND_CYLINDRICAL
            if dirty_keys:
                if coalesce:
                    mesh_keys = backend.initial_surface_keys()
                    tmp = backend.copy_tiles(mesh_keys)
                else:
                    mesh_keys = backend.expand_dirty(dirty_keys)
                    copy_keys = backend.expand_dirty(mesh_keys)
                    tmp = backend.copy_tiles(copy_keys)
            else:
                mesh_keys = set()
                tmp = None

        if dirty_keys and tmp is not None:
            meshes = tmp.mesh_tiles(mesh_keys)
        else:
            meshes = {}

        with sim._lock:
            if sim._generation != gen or sim._backend is None:
                return False
            include_laser = False
            laser_payload = None
            take = getattr(backend, "take_laser_dirty", None)
            get = getattr(backend, "laser_gpu_payload", None)
            if take is not None and take():
                include_laser = True
                laser_payload = get() if get is not None else None
            elif replace and get is not None:
                include_laser = True
                laser_payload = get()
            try:
                if coalesce and meshes:
                    current = {k for k, packed in meshes.items() if packed is not None}
                    if not replace:
                        for old in sim._coalesce_gpu_keys - current:
                            meshes[old] = None
                    sim._coalesce_gpu_keys = current
                out: dict = {}
                if replace:
                    if dirty_keys or meshes:
                        out["__replace__"] = meshes
                    if include_laser:
                        out["__laser__"] = laser_payload
                    if out or force_emit:
                        sim._on_meshes_ready(out)
                    return True
                out = dict(meshes)
                if include_laser:
                    out["__laser__"] = laser_payload
                if out or force_emit:
                    sim._on_meshes_ready(out)
            except Exception:
                logger.exception("stock mesh callback failed")
            return True

    def apply(self, viewer: Any, meshes: dict) -> None:
        viewer._apply_stock_meshes(meshes)


class FieldDisplay:
    """Static grid plus an RGBA8 field texture. Voxel carving still remeshes."""

    def __init__(self) -> None:
        self._remesh = RemeshDisplay()

    def publish(
        self,
        sim: Any,
        backend: Any,
        dirty: set[tuple[int, int, int]],
        gen: int,
        *,
        replace: bool,
        force_emit: bool = False,
    ) -> bool:
        kind = str(getattr(backend, "kind", ""))
        if kind not in (BACKEND_HEIGHTMAP, BACKEND_CYLINDRICAL):
            return self._remesh.publish(sim, backend, dirty, gen, replace=replace, force_emit=force_emit)
        if not sim._on_meshes_ready:
            return True

        with sim._lock:
            if sim._generation != gen or not sim._enabled or sim._backend is None:
                return False
            if backend is not sim._backend:
                return False
            source = _copy_field_source(backend)
            dirty_keys = set(dirty)

        if source is None:
            return True
        payload = _field_payload(source, include_mesh=replace)

        with sim._lock:
            if sim._generation != gen or sim._backend is None:
                return False
            include_laser = False
            laser_payload = None
            take = getattr(backend, "take_laser_dirty", None)
            get = getattr(backend, "laser_gpu_payload", None)
            if take is not None and take():
                include_laser = True
                laser_payload = get() if get is not None else None
            elif replace and get is not None:
                include_laser = True
                laser_payload = get()
            try:
                out: dict = {}
                if dirty_keys or replace or force_emit or include_laser:
                    out["__field__"] = payload
                if include_laser:
                    out["__laser__"] = laser_payload
                if out or force_emit:
                    sim._on_meshes_ready(out)
            except Exception:
                logger.exception("stock field callback failed")
            return True

    def apply(self, viewer: Any, meshes: dict) -> None:
        if meshes and "__field__" in meshes:
            viewer._apply_field_payload(meshes)
            return
        viewer._apply_stock_meshes(meshes)


def _copy_field_source(backend: Any) -> dict | None:
    kind = str(getattr(backend, "kind", ""))
    bounds = backend.bounds
    cell = float(backend.cell_size)
    if kind == BACKEND_HEIGHTMAP:
        values = np.array(backend.heights, copy=True, dtype=np.float32)
        return {
            "kind": kind,
            "values": values,
            "z_min": float(bounds.min_z),
            "z_max": float(bounds.max_z),
            "min_x": float(bounds.min_x),
            "min_y": float(bounds.min_y),
            "cell": cell,
            "axis_y": 0.0,
            "axis_z": 0.0,
            "floor": 0.0,
            "span": 0.0,
        }
    if kind == BACKEND_CYLINDRICAL:
        values = np.array(backend.radii, copy=True, dtype=np.float32)
        return {
            "kind": kind,
            "values": values,
            "z_min": 0.0,
            "z_max": 0.0,
            "min_x": float(bounds.min_x),
            "min_y": float(bounds.min_y),
            "cell": cell,
            "axis_y": float(backend.axis_y),
            "axis_z": float(backend.axis_z),
            "floor": shell_floor_radius_mm(cell),
            "span": radius_span_mm(float(backend.stock_radius), float(bounds.size[1]), float(bounds.size[2])),
            "nx": int(backend.nx),
            "n_theta": int(backend.n_theta),
        }
    return None


def _field_payload(source: dict, *, include_mesh: bool) -> dict:
    kind = source["kind"]
    values = source["values"]
    if kind == BACKEND_HEIGHTMAP:
        pixels, width, height, lo, span = pack_height_field(values, source["z_min"], source["z_max"])
        meshes = None
        if include_mesh and values.size:
            meshes = build_heightmap_field_mesh(
                values,
                min_x=source["min_x"],
                min_y=source["min_y"],
                min_z=source["z_min"],
                cell_size=source["cell"],
            )
    else:
        pixels, width, height, lo, span = pack_radius_field(values, source["span"])
        meshes = None
        if include_mesh and int(source["nx"]) > 0 and int(source["n_theta"]) > 0:
            meshes = build_rotary_field_mesh(
                nx=int(source["nx"]),
                n_theta=int(source["n_theta"]),
                min_x=source["min_x"],
                cell_size=source["cell"],
                axis_y=source["axis_y"],
                axis_z=source["axis_z"],
            )
    return {
        "kind": kind,
        "width": int(width),
        "height": int(height),
        "pixels": pixels,
        "lo": float(lo),
        "span": float(span),
        "floor": float(source["floor"]),
        "cell": float(source["cell"]),
        "axis_y": float(source["axis_y"]),
        "axis_z": float(source["axis_z"]),
        "meshes": meshes,
    }
