"""Response cross sections for the Phase 3 tallies: packing (Python) and
the per-energy refresh (Numba).

Each tally score equals an OpenMC v0.16.0 score in a neutron-only run
(src/reaction.cpp REACTION_TYPE_MAP; src/tallies/tally_scoring.cpp):

    R_HEATING        MT 301  "heating"          eV-b   (get_nuclide_neutron_heating)
    R_HEATING_LOCAL  MT 901  "heating-local"    eV-b   (get_nuclide_xs)
    R_DAMAGE         MT 444  "damage-energy"    eV-b   (get_nuclide_xs)
    R_H3             MT 205  "H3-production"    b      (get_nuclide_xs)
    R_HE4            MT 207  "He4-production"   b      (get_nuclide_xs)
    R_ABSORPTION     sum of the disappearance reactions, "absorption"; the
                     kernel's own absn table, the set OpenMC calls absorption

A nuclide without one of the MTs scores exactly 0 for it, as OpenMC's
get_nuclide_xs returns 0 when reaction_index_ is C_NONE (in the D7 data:
MT 205 for every W isotope). `present[k, s]` records which data exist, so
a zero from absent data is not mistaken for a measured zero.

The values are interpolated with the grid index and factor (ci, cf) that
transport_kin.macro_total already computed for the flight's energy, via
xs.interp_at: 0 below the reaction's threshold index, lin-lin above. OpenMC
(Reaction::xs, src/reaction.cpp) uses the same index and factor, and
evaluates (1-f) x[i] + f x[i+1] where interp_at evaluates
x[i] + f (x[i+1] - x[i]): equal in exact arithmetic, about one ulp apart.

Kernel layout:
    rxs[roff[k, s] : roff[k, s] + n_k - rthr[k, s]]   values of response s
                                                      of nuclide k on grid
                                                      points rthr..n_k - 1
An absent reaction has rthr = n_k, so interp_at returns 0 at every grid
index and never reads rxs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

from .tallies import N_RESP, R_ABSORPTION
from .xs import interp_at

RESPONSE_MTS = (301, 901, 444, 205, 207)          # R_HEATING .. R_HE4
N_MT_RESP = len(RESPONSE_MTS)


@dataclass(frozen=True)
class PackedResponses:
    rxs: np.ndarray        # concatenated response values
    roff: np.ndarray       # (n_nuclides, N_MT_RESP) int64 offsets into rxs
    rthr: np.ndarray       # (n_nuclides, N_MT_RESP) int64 threshold indices
    present: np.ndarray    # (n_nuclides, N_RESP) bool; absorption always True


def pack_responses(nuclide_names: Sequence[str], nuclides: dict) -> PackedResponses:
    """Pack the response MTs of the packed nuclides (xs.PackedXS order)."""
    n_nuc = len(nuclide_names)
    roff = np.zeros((n_nuc, N_MT_RESP), dtype=np.int64)
    rthr = np.zeros((n_nuc, N_MT_RESP), dtype=np.int64)
    present = np.zeros((n_nuc, N_RESP), dtype=bool)
    present[:, R_ABSORPTION] = True
    parts, size = [], 0
    for k, name in enumerate(nuclide_names):
        nuc = nuclides[name]
        n = nuc.energy.size
        for s, mt in enumerate(RESPONSE_MTS):
            rx = nuc.reactions.get(mt)
            if rx is None:
                rthr[k, s] = n
                roff[k, s] = 0
                continue
            if rx.threshold_idx + rx.xs.size != n:
                raise ValueError(f"{name} MT {mt}: values do not fill the grid from "
                                 "the threshold index")
            rthr[k, s] = rx.threshold_idx
            roff[k, s] = size
            parts.append(np.asarray(rx.xs, dtype=np.float64))
            size += rx.xs.size
            present[k, s] = True
    rxs = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)
    return PackedResponses(rxs, roff, rthr, present)


@njit(cache=True)
def refresh_responses(m, m_off, mat_nuc, mat_dens, e_off, absn, rxs, roff, rthr,
                      ci, cf, ns, ntot):
    """Macroscopic responses of material m at the energy of the current
    (ci, cf): ns[jj, s] = N_j sigma_s(E) for material slot jj, and
    ntot[s] = sum over the slots in the material's order."""
    for s in range(ns.shape[1]):
        ntot[s] = 0.0
    j0 = m_off[m]
    for j in range(j0, m_off[m + 1]):
        jj = j - j0
        k = mat_nuc[j]
        n = e_off[k + 1] - e_off[k]
        i = ci[jj]
        f = cf[jj]
        dens = mat_dens[j]
        for s in range(roff.shape[1]):
            ns[jj, s] = dens * interp_at(i, f, n, rxs, roff[k, s], rthr[k, s])
        ns[jj, R_ABSORPTION] = dens * interp_at(i, f, n, absn, e_off[k], 0)
        for s in range(ns.shape[1]):
            ntot[s] += ns[jj, s]
