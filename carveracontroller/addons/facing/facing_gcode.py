"""
Facing toolpath generator

Set WCS XY origin at the stock corner chosen on the wizard; width follows + or - X,
length follows + or - Y from that corner. Z0 at the top surface; cuts use negative Z.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from carveracontroller.translation import tr

from .facing_path import (
    PathElement,
    fillet_polyline,
    path_to_gcode,
    path_to_preview_xy,
)
from .stock_geometry import (
    CORNER_BL,
    CORNER_BR,
    CORNER_TL,
    CORNER_TR,
    stock_rect_from_origin_corner,
)

MILLING_CLIMB = "climb"
MILLING_CONVENTIONAL = "conventional"
MILLING_BOTH = "both"

PATTERN_RASTER_X = "raster_x"
PATTERN_RASTER_Y = "raster_y"
PATTERN_SPIRAL = "spiral"
PATTERN_SPIRAL_ROUND = "spiral_round"

_SPIRAL_PATTERNS = (PATTERN_SPIRAL, PATTERN_SPIRAL_ROUND)

_SPIRAL_CHORD_ERR_MM = 0.05
_SPIRAL_MAX_DTHETA = math.pi / 8.0
_SPIRAL_MIN_DTHETA = 1e-4
_SPIRAL_MAX_POINTS = 50000


@dataclass
class FacingParams:
    stock_width_mm: float
    stock_length_mm: float
    stock_origin_corner: str
    margin_x_mm: float
    margin_y_mm: float
    margin_z_mm: float
    tool_diameter_mm: float
    clearance_z_mm: float
    spindle_rpm: float
    spindle_spinup_dwell_s: int
    pattern: str
    milling_direction: str
    rough_feed_mm_min: float
    rough_plunge_feed_mm_min: float
    rough_stepover_mm: float
    path_radius_mm: float
    rough_depth_per_pass_mm: float
    rough_total_depth_mm: float
    finish_enabled: bool
    finish_feed_mm_min: float
    finish_stepover_mm: float
    finish_depth_mm: float
    ext_port_enabled: bool
    ext_port_pwm: int


@dataclass(frozen=True)
class FacingEnvelope:
    """Stock+margins rectangle and inset facing rectangle (mm, work coordinates)."""

    origin_corner: str
    stock_x0: float
    stock_x1: float
    stock_y0: float
    stock_y1: float
    facing_xa: float
    facing_xb: float
    facing_ya: float
    facing_yb: float
    rough_stepover_mm: float
    path_radius_mm: float
    pattern: str
    milling_direction: str


def _inset_span(p0: float, p1: float, r: float) -> tuple[float, float]:
    return p0 + r, p1 - r


def compute_facing_envelope(p: FacingParams) -> FacingEnvelope:
    w = p.stock_width_mm
    sl = p.stock_length_mm
    mx = p.margin_x_mm
    my = p.margin_y_mm
    rad = p.tool_diameter_mm / 2.0

    pattern = p.pattern.strip().lower()
    if pattern not in (PATTERN_RASTER_X, PATTERN_RASTER_Y, PATTERN_SPIRAL, PATTERN_SPIRAL_ROUND):
        raise ValueError(tr._("Unknown facing pattern."))
    if pattern in _SPIRAL_PATTERNS and p.milling_direction == MILLING_BOTH:
        raise ValueError(tr._("Spiral facing requires Climb or Conventional milling, not Both."))

    nx0, ny0, nx1, ny1 = stock_rect_from_origin_corner(w, sl, p.stock_origin_corner)
    x0 = nx0 - mx
    x1 = nx1 + mx
    y0 = ny0 - my
    y1 = ny1 + my

    xa, xb = _inset_span(x0, x1, rad)
    ya, yb = _inset_span(y0, y1, rad)
    if xa >= xb - 1e-6 or ya >= yb - 1e-6:
        raise ValueError(tr._("Facing area too small for tool diameter (after margins and radius)."))

    total_cut = p.rough_total_depth_mm + p.margin_z_mm
    if total_cut <= 0:
        raise ValueError(tr._("Total facing depth must be positive."))

    rough_step = max(p.rough_stepover_mm, 0.05)
    return FacingEnvelope(
        origin_corner=p.stock_origin_corner.strip().lower(),
        stock_x0=x0,
        stock_x1=x1,
        stock_y0=y0,
        stock_y1=y1,
        facing_xa=xa,
        facing_xb=xb,
        facing_ya=ya,
        facing_yb=yb,
        rough_stepover_mm=rough_step,
        path_radius_mm=max(0.0, p.path_radius_mm),
        pattern=pattern,
        milling_direction=p.milling_direction,
    )


def _rough_z_levels(p: FacingParams) -> list[float]:
    total_cut = p.rough_total_depth_mm + p.margin_z_mm
    doc = max(p.rough_depth_per_pass_mm, 0.01)
    z_levels: list[float] = []
    z = -doc
    while z > -total_cut + 1e-6:
        z_levels.append(z)
        z -= doc
    z_levels.append(-total_cut)
    return z_levels


def compute_facing_z_levels(p: FacingParams) -> list[float]:
    """Rough Z cutting depths (negative), including final level at -total_cut."""
    compute_facing_envelope(p)
    return _rough_z_levels(p)


def _rows_along_x(corner: str, ya: float, yb: float, step: float) -> list[float]:
    if corner in (CORNER_BL, CORNER_BR):
        ys: list[float] = []
        y = ya
        while y <= yb + 1e-6:
            ys.append(y)
            y += step
        return ys
    ys = []
    y = yb
    while y >= ya - 1e-6:
        ys.append(y)
        y -= step
    return ys


def _cols_along_y(corner: str, xa: float, xb: float, step: float) -> list[float]:
    if corner in (CORNER_BL, CORNER_TL):
        xs: list[float] = []
        x = xa
        while x <= xb + 1e-6:
            xs.append(x)
            x += step
        return xs
    xs = []
    x = xb
    while x >= xa - 1e-6:
        xs.append(x)
        x -= step
    return xs


def _climb_is_forward(corner: str, raster_along_x: bool) -> bool:
    """Whether forward (xa->xb / ya->yb = increasing coordinate) is the climb direction

    Raster along X, stepover +Y (bl/br): climb = -X  -> forward=False
    Raster along X, stepover -Y (tl/tr): climb = +X  -> forward=True
    Raster along Y, stepover +X (bl/tl): climb = +Y  -> forward=True
    Raster along Y, stepover -X (br/tr): climb = -Y  -> forward=False
    """
    if raster_along_x:
        return corner in (CORNER_TL, CORNER_TR)
    return corner in (CORNER_BL, CORNER_TL)


def iter_raster_passes(
    env: FacingEnvelope,
    step_mm: float,
) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
    xa, xb = env.facing_xa, env.facing_xb
    ya, yb = env.facing_ya, env.facing_yb
    step = max(step_mm, 0.05)
    c = env.origin_corner
    md = env.milling_direction
    along_x = env.pattern == PATTERN_RASTER_X

    climb_fwd = _climb_is_forward(c, along_x)
    if md == MILLING_CLIMB:
        fixed_forward = climb_fwd
    elif md == MILLING_CONVENTIONAL:
        fixed_forward = not climb_fwd
    else:
        fixed_forward = None

    if along_x:
        forward = fixed_forward if fixed_forward is not None else c in (CORNER_BL, CORNER_TL)
        for y in _rows_along_x(c, ya, yb, step):
            xs = xa if forward else xb
            xe = xb if forward else xa
            yield (xs, y), (xe, y)
            if fixed_forward is None:
                forward = not forward
    else:
        forward = fixed_forward if fixed_forward is not None else c in (CORNER_BL, CORNER_BR)
        for x in _cols_along_y(c, xa, xb, step):
            ys = ya if forward else yb
            ye = yb if forward else ya
            yield (x, ys), (x, ye)
            if fixed_forward is None:
                forward = not forward


def _connected_raster_polyline(
    env: FacingEnvelope,
    step_mm: float,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for (sx, sy), (ex, ey) in iter_raster_passes(env, step_mm):
        if not pts:
            pts.append((sx, sy))
        elif math.hypot(sx - pts[-1][0], sy - pts[-1][1]) > 1e-7:
            pts.append((sx, sy))
        pts.append((ex, ey))
    return pts


def iter_spiral_rect_passes(
    env: FacingEnvelope,
    step_mm: float,
) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
    """Nested rectangles outside -> inwards.
    When another loop remains, the last side stops at the next loop's inset
    and steps in axis-aligned. Filleting that 90° is a single arc.
    """
    xa, xb = env.facing_xa, env.facing_xb
    ya, yb = env.facing_ya, env.facing_yb
    step = max(step_mm, 0.05)
    ccw = env.milling_direction == MILLING_CLIMB

    k = 0
    while True:
        L = xa + k * step
        R = xb - k * step
        B = ya + k * step
        T = yb - k * step
        if R - L < 1e-6 or T - B < 1e-6:
            break

        L2 = xa + (k + 1) * step
        R2 = xb - (k + 1) * step
        B2 = ya + (k + 1) * step
        T2 = yb - (k + 1) * step
        has_next = R2 - L2 >= 1e-6 and T2 - B2 >= 1e-6

        if ccw:
            if has_next:
                segments = (
                    ((L, B), (L, T)),
                    ((L, T), (R, T)),
                    ((R, T), (R, B)),
                    ((R, B), (L2, B)),
                )
            else:
                segments = (
                    ((L, B), (L, T)),
                    ((L, T), (R, T)),
                    ((R, T), (R, B)),
                    ((R, B), (L, B)),
                )
        else:
            if has_next:
                segments = (
                    ((L, B), (R, B)),
                    ((R, B), (R, T)),
                    ((R, T), (L, T)),
                    ((L, T), (L, B2)),
                    ((L, B2), (L2, B2)),
                )
            else:
                segments = (
                    ((L, B), (R, B)),
                    ((R, B), (R, T)),
                    ((R, T), (L, T)),
                    ((L, T), (L, B)),
                )
        yield from segments
        if not has_next:
            break
        k += 1


def rect_spiral_polyline(
    env: FacingEnvelope,
    step_mm: float,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for (sx, sy), (ex, ey) in iter_spiral_rect_passes(env, step_mm):
        if not pts:
            pts.append((sx, sy))
        elif math.hypot(sx - pts[-1][0], sy - pts[-1][1]) > 1e-7:
            pts.append((sx, sy))
        pts.append((ex, ey))
    return pts


def _dtheta_for_radius(r: float) -> float:
    r_eff = max(r, _SPIRAL_CHORD_ERR_MM)
    dt = math.sqrt(8.0 * _SPIRAL_CHORD_ERR_MM / r_eff)
    return min(_SPIRAL_MAX_DTHETA, max(_SPIRAL_MIN_DTHETA, dt))


def archimedean_spiral_polyline(
    env: FacingEnvelope,
    step_mm: float,
) -> list[tuple[float, float]]:
    """Outside-in circular Archimedean spiral inscribed in the facing rectangle.
    Climb starts at the leftmost point and decreases polar angle (up the left side).
    Conventional starts at the bottom and increases polar angle (along the bottom first).
    """
    xa, xb = env.facing_xa, env.facing_xb
    ya, yb = env.facing_ya, env.facing_yb
    step = max(step_mm, 0.05)
    cx = (xa + xb) * 0.5
    cy = (ya + yb) * 0.5
    r_max = min((xb - xa) * 0.5, (yb - ya) * 0.5)
    r_min = min(step * 0.5, r_max * 0.5)
    r_min = max(r_min, 1e-3)
    if r_max <= r_min + 1e-9:
        return [(cx, cy)]

    b = step / (2.0 * math.pi)
    climb = env.milling_direction == MILLING_CLIMB
    sign = -1.0 if climb else 1.0
    theta0 = math.pi if climb else -0.5 * math.pi
    theta = theta0
    r = r_max
    pts: list[tuple[float, float]] = []
    n = 0
    while r > r_min and n < _SPIRAL_MAX_POINTS:
        p = (cx + r * math.cos(theta), cy + r * math.sin(theta))
        if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-7:
            pts.append(p)
        dtheta = _dtheta_for_radius(r)
        theta += sign * dtheta
        r = r_max - b * abs(theta - theta0)
        n += 1

    if pts:
        last = pts[-1]
        if math.hypot(last[0] - cx, last[1] - cy) > 1e-6:
            pts.append((cx, cy))
    else:
        pts.append((cx, cy))
    return pts


def facing_layer_xy_paths(
    env: FacingEnvelope,
    step_mm: float,
) -> list[list[PathElement]]:
    """Stay-down XY paths for one Z layer (retract between list entries)."""
    radius = env.path_radius_mm
    retract_between_passes = (
        env.pattern
        in (
            PATTERN_RASTER_X,
            PATTERN_RASTER_Y,
        )
        and env.milling_direction != MILLING_BOTH
    )
    if env.pattern == PATTERN_SPIRAL_ROUND:
        return [fillet_polyline(archimedean_spiral_polyline(env, step_mm), 0.0)]
    if env.pattern == PATTERN_SPIRAL:
        return [fillet_polyline(rect_spiral_polyline(env, step_mm), radius)]
    if retract_between_passes:
        paths: list[list[PathElement]] = []
        for a, b in iter_raster_passes(env, step_mm):
            paths.append(fillet_polyline((a, b), radius))
        return paths
    return [fillet_polyline(_connected_raster_polyline(env, step_mm), radius)]


def facing_toolpath_xy_polyline(env: FacingEnvelope) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for path in facing_layer_xy_paths(env, env.rough_stepover_mm):
        pts.extend(path_to_preview_xy(path))
    return pts


def generate_facing_gcode(p: FacingParams) -> str:
    lines: list[str] = []
    env = compute_facing_envelope(p)

    z_levels = _rough_z_levels(p)
    total_cut = p.rough_total_depth_mm + p.margin_z_mm
    rough_step = env.rough_stepover_mm
    finish_step = max(p.finish_stepover_mm, 0.05) if p.finish_enabled else rough_step

    lines.append("(Facing)")
    lines.append("(WCS XY origin = stock corner from wizard; Z0 top of stock)")
    lines.append("G21")
    lines.append("G90")
    lines.append("G17")
    lines.append("G94")
    if p.ext_port_enabled:
        lines.append(f"M851 S{p.ext_port_pwm:d}")
    lines.append(f"G0 Z{p.clearance_z_mm:.4f}")
    lines.append(f"M3 S{p.spindle_rpm:.0f}")
    if p.spindle_spinup_dwell_s > 0:
        lines.append(f"G4 P{p.spindle_spinup_dwell_s:d}")

    def emit_layer(z_cut: float, step: float, feed: float, plunge_f: float) -> None:
        first = True
        for elements in facing_layer_xy_paths(env, step):
            if not elements:
                continue
            el0 = elements[0]
            sx, sy = el0.x0, el0.y0
            if first:
                lines.append(f"G0 X{sx:.4f} Y{sy:.4f}")
                lines.append(f"G1 Z{z_cut:.4f} F{plunge_f:.1f}")
                first = False
            else:
                lines.append(f"G0 Z{p.clearance_z_mm:.4f}")
                lines.append(f"G0 X{sx:.4f} Y{sy:.4f}")
                lines.append(f"G1 Z{z_cut:.4f} F{plunge_f:.1f}")
            lines.extend(path_to_gcode(elements, feed))
        lines.append(f"G0 Z{p.clearance_z_mm:.4f}")

    for z_cut in z_levels:
        lines.append(f"(Rough pass Z{z_cut:.4f})")
        emit_layer(
            z_cut,
            rough_step,
            p.rough_feed_mm_min,
            p.rough_plunge_feed_mm_min,
        )

    if p.finish_enabled and p.finish_depth_mm > 1e-6:
        z_fin = -total_cut - p.finish_depth_mm
        lines.append(f"(Finish pass Z{z_fin:.4f})")
        fd = p.finish_feed_mm_min
        pf = min(p.rough_plunge_feed_mm_min, fd)
        emit_layer(z_fin, finish_step, fd, pf)

    lines.append("M5")
    if p.ext_port_enabled:
        lines.append("M852")
    lines.append("G28")
    lines.append("M2")
    return "\n".join(lines) + "\n"
