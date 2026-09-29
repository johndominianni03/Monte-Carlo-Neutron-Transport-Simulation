"""Source definitions (Python side) and source sampling (kernel side)."""
from __future__ import annotations

from dataclasses import dataclass

from numba import njit

from .rng import prn

SRC_BEAM = 0         # monodirectional, mu = 1, entering at the left face
SRC_ISO_PLANE = 1    # isotropic plane source at x0


@dataclass(frozen=True)
class BeamSource:
    """Monodirectional beam entering the left face (x = bounds[0]) with mu = 1."""

    def pack(self, geometry):
        return SRC_BEAM, float(geometry.bounds[0]), 0


@dataclass(frozen=True)
class IsotropicPlaneSource:
    """Isotropic plane source at x0 (in 1D, a point and a plane are the same).
    x0 must lie in [left face, right face). A source exactly on an interface
    is placed in the region to its right; mu < 0 particles then cross the
    interface after a zero-length step, which is correct."""
    x0: float

    def pack(self, geometry):
        return SRC_ISO_PLANE, float(self.x0), geometry.region_of(self.x0)


@njit(cache=True)
def sample_source(src_type, src_x, src_region, rng):
    """-> (x, mu, region). The beam draws no random numbers."""
    if src_type == SRC_BEAM:
        return src_x, 1.0, src_region
    mu = 2.0 * prn(rng) - 1.0
    return src_x, mu, src_region
