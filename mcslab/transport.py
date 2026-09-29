"""Analog one-group transport kernels. Plain arrays and scalars only.

Random-number consumption per event (this order is part of the regression
contract; changing it changes every result):
  source   : isotropic -> 1 (mu); beam -> 0
  flight   : 1 (distance), skipped in void (Sigma_t = 0)
  collision: 1 (absorb vs scatter), then 1 more (new mu) if it scatters
"""
import numpy as np
from numba import njit

from .geometry import distance_to_boundary
from .rng import RNG_DRAWS, RNG_SIZE, init_history, prn
from .sources import sample_source
from .tallies import (ABSORPTION, COLL_ESTIMATOR, COLLISION, DIAG_LOST,
                      DIAG_MAX_DRAWS, NEG, POS, TRACK_LENGTH)


@njit(cache=True)
def transport_history(history, master_seed, bounds, mat_of_region, sig_t, sig_a,
                      src_type, src_x, src_region, reg, surf, rng):
    """Follow one source particle until it is absorbed or leaks.

    reg  : [N_REGION_SCORES, n_regions] view of this batch's region tallies
    surf : [2, n_regions + 1] view of this batch's surface tallies
    Returns (draws used, lost flag).
    """
    init_history(rng, master_seed, np.uint64(history))
    n_regions = mat_of_region.shape[0]
    x, mu, r = sample_source(src_type, src_x, src_region, rng)
    w = 1.0
    lost = 0

    while True:
        m = mat_of_region[r]
        st = sig_t[m]
        if st > 0.0:
            d_coll = -np.log(1.0 - prn(rng)) / st
        else:
            d_coll = np.inf
        d_bdy = distance_to_boundary(x, mu, bounds[r], bounds[r + 1])

        if d_coll < d_bdy:
            # move to the collision site and collide
            x += d_coll * mu
            reg[TRACK_LENGTH, r] += w * d_coll
            reg[COLLISION, r] += w
            reg[COLL_ESTIMATOR, r] += w / st
            if prn(rng) * st < sig_a[m]:
                reg[ABSORPTION, r] += w
                break
            mu = 2.0 * prn(rng) - 1.0          # isotropic lab-frame scatter
        else:
            if d_bdy == np.inf:
                # mu == 0 in a void: the particle can never move. Only
                # possible for a source in a void with mu exactly 0.
                lost = 1
                break
            # move to the boundary; place x exactly on it and step the region
            # index by direction (never re-locate from x, so round-off cannot
            # misplace the particle), then resample in the new region.
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
                break                            # leaked through a vacuum boundary
    return int(rng[RNG_DRAWS]), lost


@njit(cache=True)
def run_batches(batch_start, batch_end, histories_per_batch, master_seed,
                bounds, mat_of_region, sig_t, sig_a,
                src_type, src_x, src_region,
                region_sums, surface_sums, diagnostics):
    """Run batches [batch_start, batch_end), writing only their rows.

    Batch b runs histories b*n .. b*n + n - 1 in order. Every history's RNG
    stream depends only on (master_seed, history id), and each batch
    accumulates into its own row, so this loop can become prange without
    changing a single bit of the output.
    """
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
            draws, lost = transport_history(
                first + k, master_seed, bounds, mat_of_region, sig_t, sig_a,
                src_type, src_x, src_region, reg, surf, rng)
            if draws > max_draws:
                max_draws = draws
            n_lost += lost
        diagnostics[b, DIAG_MAX_DRAWS] = max_draws
        diagnostics[b, DIAG_LOST] = n_lost
