"""1D slab geometry: contiguous regions along x, vacuum on both sides.

Region r occupies [bounds[r], bounds[r+1]]. Surface s sits at bounds[s], so
surface 0 is the left vacuum boundary and surface n_regions the right one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numba import njit

from .materials import Material, pack_materials


@dataclass(frozen=True)
class SlabGeometry:
    bounds: Tuple[float, ...]
    region_materials: Tuple[Material, ...]

    def __init__(self, bounds: Sequence[float], region_materials: Sequence[Material]):
        bounds = tuple(float(b) for b in bounds)
        region_materials = tuple(region_materials)
        if len(bounds) < 2:
            raise ValueError("need at least one region (two bounds)")
        if len(region_materials) != len(bounds) - 1:
            raise ValueError(
                f"{len(bounds) - 1} regions but {len(region_materials)} materials")
        if not all(np.isfinite(bounds)):
            raise ValueError("bounds must be finite")
        if any(b1 <= b0 for b0, b1 in zip(bounds, bounds[1:])):
            raise ValueError("bounds must be strictly increasing")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "region_materials", region_materials)

    @classmethod
    def uniform(cls, material: Material, x0: float, x1: float, n_regions: int = 1):
        """A single material split into n equal-width regions."""
        return cls(np.linspace(x0, x1, n_regions + 1), [material] * n_regions)

    @property
    def n_regions(self) -> int:
        return len(self.region_materials)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(np.asarray(self.bounds, dtype=np.float64))

    def region_of(self, x: float) -> int:
        """Region containing x, with an interface assigned to the region on
        its right. x must lie in [bounds[0], bounds[-1])."""
        b = self.bounds
        if not b[0] <= x < b[-1]:
            raise ValueError(f"x={x} outside [{b[0]}, {b[-1]})")
        return int(np.searchsorted(b, x, side="right")) - 1

    def pack(self):
        """-> bounds, mat_of_region, sig_t, sig_s, sig_a as plain arrays."""
        unique = []
        mat_of_region = np.empty(self.n_regions, dtype=np.int64)
        for r, m in enumerate(self.region_materials):
            if m not in unique:
                unique.append(m)
            mat_of_region[r] = unique.index(m)
        sig_t, sig_s, sig_a = pack_materials(unique)
        return (np.asarray(self.bounds, dtype=np.float64), mat_of_region,
                sig_t, sig_s, sig_a)


@njit(cache=True)
def distance_to_boundary(x, mu, x_lo, x_hi):
    """Distance along direction mu from x to the edge of [x_lo, x_hi]."""
    if mu > 0.0:
        return (x_hi - x) / mu
    elif mu < 0.0:
        return (x_lo - x) / mu
    return np.inf
