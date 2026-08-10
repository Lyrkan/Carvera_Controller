"""Place a stock shape into WCS using a StockOrigin."""

from __future__ import annotations

from dataclasses import dataclass

from carveracontroller.addons.facing.stock_geometry import stock_rect_from_origin_corner

from .stock_origin import Z_BOTTOM, Z_TOP, StockOrigin
from .stock_shape import StockShape


@dataclass(frozen=True)
class StockBounds:
    """Axis-aligned stock AABB in work coordinates (mm)."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.min_x + self.max_x),
            0.5 * (self.min_y + self.max_y),
            0.5 * (self.min_z + self.max_z),
        )


def compute_wcs_bounds(shape: StockShape, origin: StockOrigin) -> StockBounds:
    """Map a local stock shape into WCS using the given origin placement."""
    (_min_local, (dx, dy, dz)) = shape.local_bounds()
    min_x, min_y, max_x, max_y = stock_rect_from_origin_corner(dx, dy, origin.xy_corner)

    if origin.z_reference == Z_TOP:
        # Z0 at top surface; material extends downward (negative Z), matching CAM convention.
        min_z = -dz
        max_z = 0.0
    elif origin.z_reference == Z_BOTTOM:
        min_z = 0.0
        max_z = dz
    else:
        raise ValueError(f"unsupported z_reference: {origin.z_reference!r}")

    ox = origin.offset_x_mm
    oy = origin.offset_y_mm
    oz = origin.offset_z_mm
    return StockBounds(
        min_x=min_x + ox,
        min_y=min_y + oy,
        min_z=min_z + oz,
        max_x=max_x + ox,
        max_y=max_y + oy,
        max_z=max_z + oz,
    )


def wcs_to_local(bounds: StockBounds, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Map a WCS point into the shape's local frame (min corner at origin)."""
    return (x - bounds.min_x, y - bounds.min_y, z - bounds.min_z)


def contains_wcs(shape: StockShape, bounds: StockBounds, x: float, y: float, z: float) -> bool:
    """True if the WCS point lies inside the placed solid (not just the AABB)."""
    lx, ly, lz = wcs_to_local(bounds, x, y, z)
    return shape.contains_local(lx, ly, lz)
