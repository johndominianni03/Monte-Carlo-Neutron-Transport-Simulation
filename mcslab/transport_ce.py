"""Continuous-energy transport kernels, Phase 2a: the first collision ends
the history. Plain arrays and scalars only.

There are no scattering kinematics yet (Phase 2b), so a neutron keeps its
source energy until it collides or leaks, and every collision terminates it.
At the collision, absorption is scored as its expected value
w * Sigma_a(E) / Sigma_t(E). That costs no random number.

Random-number consumption per event (part of the regression contract):
  source energy   : monoenergetic -> 0; log-uniform -> 1 (drawn first)
  source direction: beam -> 0; isotropic -> 1
  flight          : 1 (distance), skipped in void (Sigma_t = 0)
  collision       : 0 (ends the history)
"""
import numpy as np
from numba import njit

from .geometry import distance_to_boundary
from .rng import RNG_DRAWS, RNG_SIZE, init_history, prn
from .sources import sample_source
from .tallies import (ABSORPTION, COLL_ESTIMATOR, COLLISION, DIAG_LOST,
                      DIAG_MAX_DRAWS, NEG, POS, TRACK_LENGTH)
from .xs import macro_xs

SRC_E_MONO = 0         # all particles at e_lo
SRC_E_LOG_UNIFORM = 1  # density proportional to 1/E on [e_lo, e_hi)


@njit(cache=True)
def sample_energy(e_type, e_lo, e_hi, rng):
    if e_type == SRC_E_MONO:
        return e_lo
    return e_lo * np.exp(prn(rng) * np.log(e_hi / e_lo))


@njit(cache=True)
def transport_history_ce(history, master_seed, bounds, mat_of_region,
                         egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                         src_type, src_x, src_region, e_type, e_lo, e_hi,
                         reg, surf, rng):
    """Follow one source particle to its first collision or until it leaks.
    Returns (draws used, lost flag)."""
    init_history(rng, master_seed, np.uint64(history))
    n_regions = mat_of_region.shape[0]
    E = sample_energy(e_type, e_lo, e_hi, rng)
    x, mu, r = sample_source(src_type, src_x, src_region, rng)
    w = 1.0
    lost = 0

    while True:
        m = mat_of_region[r]
        st = macro_xs(m, E, egrid, e_off, tot, m_off, mat_nuc, mat_dens)
        if st > 0.0:
            d_coll = -np.log(1.0 - prn(rng)) / st
        else:
            d_coll = np.inf
        d_bdy = distance_to_boundary(x, mu, bounds[r], bounds[r + 1])

        if d_coll < d_bdy:
            sa = macro_xs(m, E, egrid, e_off, absn, m_off, mat_nuc, mat_dens)
            reg[TRACK_LENGTH, r] += w * d_coll
            reg[COLLISION, r] += w
            reg[COLL_ESTIMATOR, r] += w / st
            reg[ABSORPTION, r] += w * (sa / st)
            break                                # no kinematics yet
        else:
            if d_bdy == np.inf:
                lost = 1
                break
            reg[TRACK_LENGTH, r] += w * d_bdy
            if mu > 0.0:
                surf[POS, r + 1] += w
                r += 1
                x = bounds[r]
            else:
                surf[NEG, r] += w
                x = bounds[r]
                r -= 1
            if r < 0 or r >= n_regions:
                break
    return int(rng[RNG_DRAWS]), lost


@njit(cache=True)
def run_batches_ce(batch_start, batch_end, histories_per_batch, master_seed,
                   bounds, mat_of_region,
                   egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                   src_type, src_x, src_region, e_type, e_lo, e_hi,
                   region_sums, surface_sums, diagnostics):
    """Run batches [batch_start, batch_end), writing only their rows (same
    structure and guarantees as transport.run_batches)."""
    for b in range(batch_start, batch_end):
        reg = region_sums[b]
        surf = surface_sums[b]
        reg[:, :] = 0.0
        surf[:, :] = 0.0
        rng = np.zeros(RNG_SIZE, np.uint64)
        max_draws = 0
        n_lost = 0
        first = b * histories_per_batch
        for k in range(histories_per_batch):
            draws, lost = transport_history_ce(
                first + k, master_seed, bounds, mat_of_region,
                egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                src_type, src_x, src_region, e_type, e_lo, e_hi, reg, surf, rng)
            if draws > max_draws:
                max_draws = draws
            n_lost += lost
        diagnostics[b, DIAG_MAX_DRAWS] = max_draws
        diagnostics[b, DIAG_LOST] = n_lost
