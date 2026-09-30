"""Synthetic nuclides for testing, built as the same nucdata dataclasses the
HDF5 reader returns. They go through exactly the same packing and kernels
as real nuclides.

A synthetic nuclide has constant (energy-independent) cross sections on a
grid from 1e-5 eV to 20 MeV and isotropic CM angular distributions. Its
kT is 0, meaning 0 K: the target is at rest. It can have:

  - elastic scattering (MT 2), always present;
  - capture (MT 102): no outgoing neutron, so it is an absorption;
  - discrete levels (MT 51..): level law, Q < 0, yield 1;
  - multiplying reactions (e.g. MT 16 with yield 2, MT 17 with yield 3),
    modelled as a level-type two-body law with the given Q. This is not
    realistic (n,2n) physics. It exists to exercise the secondary bank and
    the neutron balance, which do not depend on the law.

A reaction with Q < 0 is exactly 0 up to its kinematic threshold
-Q (A+1)/A (a grid point, as in real data, where every reaction's first
tabulated value is 0). It rises to its constant value at the next grid
point, thr (1 + 1e-9), and stays constant above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from . import nucdata as N

E_MIN = 1.0e-5
E_MAX = 2.0e7
_STEP = 1.0e-9          # relative width of the threshold ramp


def _neutron_product(yield_value: float, level: N.LevelInelastic = None) -> N.Product:
    law = N.UncorrelatedAngleEnergy(angle=None, energy=level)
    desc = "uncorrelated(angle=isotropic, energy=%s)" % ("level" if level else "none")
    return N.Product("neutron", "prompt", "Polynomial", float(yield_value), (desc,),
                     laws_read=True, yield_coefficients=(float(yield_value),),
                     laws=(law,))


def nuclide(name: str, awr: float, elastic_b: float, capture_b: float = 0.0,
            reactions: Sequence[Tuple[int, float, float, int]] = ()) -> N.Nuclide:
    """A synthetic nuclide.

    reactions: (MT, Q in eV (< 0), sigma in b, neutron yield) for each
    non-elastic neutron-emitting reaction, e.g. (51, -1.0e6, 0.5, 1) for a
    level or (16, -5.0e6, 0.2, 2) for an (n,2n)."""
    thresholds = []
    for mt, q, _, _ in reactions:
        if not q < 0.0:
            raise ValueError(f"MT {mt}: synthetic reactions need Q < 0")
        thresholds.append(-q * (awr + 1.0) / awr)
    points = {E_MIN, E_MAX}
    for thr in thresholds:
        if not E_MIN < thr < E_MAX / (1.0 + _STEP):
            raise ValueError(f"threshold {thr} eV outside the synthetic grid")
        points.update((thr, thr * (1.0 + _STEP)))
    grid = np.array(sorted(points), dtype=np.float64)

    rx: Dict[int, N.Reaction] = {}
    rx[2] = N.Reaction(2, "(n,elastic)", 0.0, True, False, 0,
                       np.full(grid.size, float(elastic_b)), (_neutron_product(1.0),))
    if capture_b > 0.0:
        rx[102] = N.Reaction(102, "(n,gamma)", 0.0, False, False, 0,
                             np.full(grid.size, float(capture_b)), ())
    for (mt, q, sigma, y), thr in zip(reactions, thresholds):
        i0 = int(np.searchsorted(grid, thr))
        assert grid[i0] == thr
        xs = np.full(grid.size - i0, float(sigma))
        xs[0] = 0.0
        level = N.LevelInelastic(threshold=thr, mass_ratio=(awr / (awr + 1.0)) ** 2)
        rx[mt] = N.Reaction(mt, f"synthetic MT {mt}", float(q), True, False, i0, xs,
                            (_neutron_product(y, level),))
    return N.Nuclide(name=name, Z=0, A=int(round(awr)), metastable=0, awr=float(awr),
                     temperature="0K", kT=0.0, energy=grid, reactions=rx,
                     path="", library="synthetic")


@dataclass(eq=False)
class SyntheticLibrary:
    """Duck-types nucdata.Library for synthetic nuclides."""
    nuclides: Dict[str, N.Nuclide]
    label: str = "synthetic"

    def load(self, name, temperature=None, distributions=False):
        return self.nuclides[name]


@dataclass(frozen=True)
class SyntheticMaterial:
    """A material given directly as (nuclide, atoms per barn-cm) pairs."""
    name: str
    atom_densities: Tuple[Tuple[str, float], ...] = field(default=())

    @property
    def nuclide_names(self):
        return tuple(n for n, _ in self.atom_densities)

    def number_densities(self, awr=None):
        return tuple(self.atom_densities)
