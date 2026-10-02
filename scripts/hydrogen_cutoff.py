#!/usr/bin/env python3
"""Hydrogen energy-cutoff illustration, before and after free-gas scattering.

    MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5 ./venv/bin/python scripts/hydrogen_cutoff.py

The README / Part 1 problem, with identical inputs: pure H-1 at
0.0708 g/cm^3 (294 K data), slab [0, 30] cm, 1 MeV beam at x = 0,
10 batches x 1000 histories, seed 3, default energy cutoff (1e-5 eV).

- "before": free_gas=False, the target always at rest, as in Part 1. The
  script asserts the Part 1 balance exactly (commit cf0a925): source 10000,
  absorbed 1013, leaked 1421 / 625, cutoff 6941, residual 0.
- "after": free-gas target motion as in OpenMC (H-1 has awr < 1, so at
  every energy). Neutrons now thermalise near kT = 0.0253 eV instead of
  slowing down indefinitely, and end by capture or leakage.

What still reaches the cutoff is the sub-1e-5 eV tail of the thermal
population. Estimate: one thermal H collision sends the neutron below
eps_c = 1e-5 eV / kT = 4e-4 with probability about
(4 / (3 sqrt(pi))) eps_c^(3/2) / (eps sigma_eff/sigma_f) ~ 4e-6 (the
hydrogen free-gas kernel, tests/test_freegas.py), and a thermalised neutron
makes about 80 collisions before capture. That gives a cutoff fraction of
order 1e-4. OpenMC would keep transporting these neutrons on extrapolated
cross sections (docs/deviations_from_openmc.md, deviation 2).

This is a diagnostic comparison, not a statistical test; the seed is the
one used before.
"""
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mcslab import tallies as T                              # noqa: E402
from mcslab.ce_materials import CEMaterial                   # noqa: E402
from mcslab.config_ce import MonoEnergetic                   # noqa: E402
from mcslab.config_kin import (EnergyCutoffWarning,          # noqa: E402
                               KinRunConfig, run_kin)
from mcslab.geometry import SlabGeometry                     # noqa: E402
from mcslab.nucdata import Library                           # noqa: E402
from mcslab.rng import STRIDE                                # noqa: E402
from mcslab.sources import BeamSource                        # noqa: E402

PART1 = {"source": 10000.0, "created": 0.0, "absorbed": 1013.0, "leak_left": 1421.0,
         "leak_right": 625.0, "cutoff": 6941.0, "zero_yield": 0.0, "lost": 0.0,
         "residual": 0.0}


def config(free_gas):
    h = CEMaterial("H", 0.0708, (("H1", 1.0),))
    return KinRunConfig(SlabGeometry([0.0, 30.0], [h]), BeamSource(), MonoEnergetic(1.0e6),
                        Library.open(), n_batches=10, histories_per_batch=1000, seed=3,
                        free_gas=free_gas)


def main():
    rows = []
    for label, free_gas in (("before (target at rest)", False), ("after (free gas)", True)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EnergyCutoffWarning)
            res = run_kin(config(free_gas))
        b = res.balance()
        assert b["residual"] == 0.0 and b["lost"] == 0.0
        if not free_gas:
            assert b == PART1, f"free_gas=False no longer reproduces Part 1: {b}"
        rows.append((label, b, res.max_draws, res.count(T.K_FREE_GAS),
                     res.count(T.K_ELASTIC)))
    print("H-1, 0.0708 g/cm^3, slab [0, 30] cm, 1 MeV beam, 10 x 1000, seed 3, "
          "cutoff 1e-5 eV\n")
    print(f"{'run':<26}{'absorbed':>10}{'leak L':>8}{'leak R':>8}{'cutoff':>8}"
          f"{'cutoff/src':>12}{'residual':>10}{'max draws':>11}{'free-gas / elastic':>21}")
    for label, b, md, fg, el in rows:
        n = b["source"]
        print(f"{label:<26}{b['absorbed']:>10.0f}{b['leak_left']:>8.0f}{b['leak_right']:>8.0f}"
              f"{b['cutoff']:>8.0f}{b['cutoff'] / n:>12.4g}{b['residual']:>10.0f}{md:>11}"
              f"{fg:>11} / {el}")
    print(f"\nsource {rows[0][1]['source']:.0f} in both runs; STRIDE = {STRIDE}. "
          "'before' reproduces Part 1 exactly (asserted).")


if __name__ == "__main__":
    main()
