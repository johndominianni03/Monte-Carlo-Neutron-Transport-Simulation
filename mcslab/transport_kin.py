"""Continuous-energy transport with collision kinematics (Phase 2b).
Plain arrays and scalars only.

A neutron flies, collides with a nuclide chosen in proportion to
N_k sigma_t,k, and is either absorbed or scattered by an elastic or
inelastic channel chosen in proportion to its cross section (collision.py,
after OpenMC v0.16.0 src/physics.cpp). Absorption is analog: it ends the
particle and scores its weight. A channel with yield y > 1 creates y - 1
secondaries, identical copies of the outgoing neutron, as in OpenMC
(physics.cpp inelastic_scatter). They go on a per-history LIFO bank and
are transported after their parent dies, continuing the history's random
number stream (OpenMC particle.cpp event_check_limit_and_revive). A history
family's results therefore depend on (master_seed, history id) only.

Energy cutoff (approved deviation D5). After each collision, a neutron below
e_cut is killed, as OpenMC's collision() does with its (default 0) cutoff. A
secondary born below e_cut is not banked, as in OpenMC's create_secondary.
Neither is dropped silently: its weight is scored in cutoff_weight[r] for
the region of the collision and counted in K_CUTOFF, so

    sources + created == absorbed + leaked_left + leaked_right + cutoff

holds exactly (tests/test_kinematics.py test d).

Direction is a full unit vector (u, v, w) with u along the slab normal. The
source sets (mu, sqrt(1 - mu^2), 0) and draws nothing extra. The slab is
invariant under rotation about x and every collision is rotationally
covariant, so this choice of azimuth changes no tally.

Time: t += d / speed(E), with OpenMC's relativistic speed
c sqrt(E (E + 2m)) / (E + m) (particle.cpp Particle::speed; constants.h,
CODATA 2018). It is carried for track recording (Part 2) and scores
nothing.

Random-number consumption per event (part of the regression contract):
  source energy   : monoenergetic 0; log-uniform 1 (drawn first)
  source direction: beam 0; isotropic 1
  flight          : 1 (distance), skipped in void (Sigma_t = 0)
  collision       : nuclide 1; absorption 1 if sigma_a > 0 for that
                    nuclide; scatter channel 1; then
                    elastic (target at rest): angle 2 (isotropic 1) + phi 1
                    level inelastic:          angle 2 (isotropic 1) + phi 1
"""
import math

import numpy as np
from numba import njit

from . import collision as C
from .geometry import distance_to_boundary
from .rng import RNG_DRAWS, RNG_SIZE, init_history, prn
from .sources import sample_source
from .tallies import (ABSORPTION, COLL_ESTIMATOR, COLLISION, K_ABSORBED, K_COLLISIONS,
                      K_CREATED, K_CUTOFF, K_ELASTIC, K_INELASTIC, K_LEAK_LEFT,
                      K_LEAK_RIGHT, K_LOST, K_MAX_BANK, K_MAX_DRAWS, K_SOURCE, NEG,
                      POS, SPEC_COLL, SPEC_TL, TRACK_LENGTH)
from .transport_ce import sample_energy
from .xs import grid_locate, interp_at

C_LIGHT = 2.99792458e10          # cm/s (OpenMC constants.h C_LIGHT)
MASS_NEUTRON_EV = 939.56542052e6  # eV/c^2 (OpenMC constants.h, CODATA 2018)

# secondary bank columns
B_X, B_U, B_V, B_W, B_E, B_T, B_NCOL = 0, 1, 2, 3, 4, 5, 6


@njit(cache=True)
def speed(E):
    """Neutron speed (cm/s) at kinetic energy E (eV), OpenMC Particle::speed."""
    return C_LIGHT * math.sqrt(E * (E + 2.0 * MASS_NEUTRON_EV)) / (E + MASS_NEUTRON_EV)


@njit(cache=True)
def energy_bin(edges, E):
    """g with edges[g] <= E < edges[g+1], or -1 outside [edges[0], edges[-1])."""
    n = edges.shape[0]
    if not (edges[0] <= E < edges[n - 1]):
        return -1
    a = 0
    b = n - 1
    while b - a > 1:
        m = (a + b) >> 1
        if edges[m] <= E:
            a = m
        else:
            b = m
    return a


@njit(cache=True)
def macro_total(m, E, egrid, e_off, tot, m_off, mat_nuc, mat_dens, ci, cf, ct):
    """Sigma_t (1/cm) of material m at E, summed in the material's nuclide
    order exactly as xs.macro_xs does. For position jj of the material it
    also stores the grid index ci[jj], factor cf[jj] and micro total ct[jj],
    so the collision reuses them without another search."""
    s = 0.0
    j0 = m_off[m]
    for j in range(j0, m_off[m + 1]):
        k = mat_nuc[j]
        g0 = e_off[k]
        n = e_off[k + 1] - g0
        i, f = grid_locate(egrid, g0, n, E)
        mt = interp_at(i, f, n, tot, g0, 0)
        ci[j - j0] = i
        cf[j - j0] = f
        ct[j - j0] = mt
        s += mat_dens[j] * mt
    return s


@njit(cache=True)
def transport_history_kin(history, master_seed, bounds, mat_of_region,
                          egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                          nuc_awr, ch_off, ch_int, ch_q, chxs, ip, fp,
                          src_type, src_x, src_region, e_type, e_lo, e_hi,
                          e_cut, spec_edges,
                          reg, surf, spec, cutw, cnt, chev, chcr,
                          bank_f, bank_r, ci, cf, ct, rng):
    """Follow one source neutron and all its secondaries. Tallies go to this
    batch's rows (reg, surf, spec, cutw, cnt, chev, chcr). Returns
    (draws used by the family, lost flag)."""
    init_history(rng, master_seed, np.uint64(history))
    n_regions = mat_of_region.shape[0]
    cap = bank_f.shape[0]
    E = sample_energy(e_type, e_lo, e_hi, rng)
    x, mu, r = sample_source(src_type, src_x, src_region, rng)
    u = mu
    v = math.sqrt(max(0.0, 1.0 - mu * mu))
    w = 0.0
    t = 0.0
    wgt = 1.0
    cnt[K_SOURCE] += 1
    n_bank = 0
    lost = 0
    last_m = -1
    last_E = -1.0
    st = 0.0

    while True:
        # ---- transport the current particle until it dies
        while True:
            m = mat_of_region[r]
            if m != last_m or E != last_E:
                st = macro_total(m, E, egrid, e_off, tot, m_off, mat_nuc, mat_dens,
                                 ci, cf, ct)
                last_m = m
                last_E = E
            if st > 0.0:
                d_coll = -math.log(1.0 - prn(rng)) / st
            else:
                d_coll = np.inf
            d_bdy = distance_to_boundary(x, u, bounds[r], bounds[r + 1])
            g = energy_bin(spec_edges, E)

            if d_coll < d_bdy:
                x += d_coll * u
                t += d_coll / speed(E)
                reg[TRACK_LENGTH, r] += wgt * d_coll
                reg[COLLISION, r] += wgt
                reg[COLL_ESTIMATOR, r] += wgt / st
                if g >= 0:
                    spec[SPEC_TL, r, g] += wgt * d_coll
                    spec[SPEC_COLL, r, g] += wgt / st
                cnt[K_COLLISIONS] += 1

                # -- nuclide, then absorption, then scatter channel (OpenMC order)
                j = C.sample_nuclide(m, m_off, mat_dens, ct, st, rng)
                jj = j - m_off[m]
                k = mat_nuc[j]
                n = e_off[k + 1] - e_off[k]
                i = ci[jj]
                f = cf[jj]
                micro_t = ct[jj]
                micro_a = interp_at(i, f, n, absn, e_off[k], 0)
                if micro_a > 0.0:
                    if micro_a > prn(rng) * micro_t:
                        reg[ABSORPTION, r] += wgt
                        cnt[K_ABSORBED] += 1
                        break
                c = C.sample_scatter_channel(k, i, f, n, micro_t, micro_a,
                                             ch_off, ch_int, chxs, rng)
                chev[c] += 1
                prod = ch_int[c, C.CH_PROD]
                if prod < 0:
                    raise RuntimeError("sampled a reaction whose law is not implemented")
                if c == ch_off[k]:
                    E, u, v, w, mu_lab = C.elastic_scatter(E, u, v, w, nuc_awr[k],
                                                           ip, fp, prod, rng)
                    cnt[K_ELASTIC] += 1
                    n_out = 1
                else:
                    E, u, v, w, mu_lab, n_out = C.inelastic_scatter(
                        E, u, v, w, nuc_awr[k], ch_q[c], ch_int[c, C.CH_CM],
                        ip, fp, prod, rng)
                    cnt[K_INELASTIC] += 1

                if n_out > 1:
                    extra = n_out - 1
                    chcr[c] += extra
                    cnt[K_CREATED] += extra
                    if E < e_cut:
                        # born below the cutoff: not banked, but scored
                        cutw[r] += wgt * extra
                        cnt[K_CUTOFF] += extra
                    else:
                        if n_bank + extra > cap:
                            raise RuntimeError("secondary bank overflow (raise bank_capacity)")
                        for _ in range(extra):
                            bank_f[n_bank, B_X] = x
                            bank_f[n_bank, B_U] = u
                            bank_f[n_bank, B_V] = v
                            bank_f[n_bank, B_W] = w
                            bank_f[n_bank, B_E] = E
                            bank_f[n_bank, B_T] = t
                            bank_r[n_bank] = r
                            n_bank += 1
                        if n_bank > cnt[K_MAX_BANK]:
                            cnt[K_MAX_BANK] = n_bank
                if E < e_cut:
                    cutw[r] += wgt
                    cnt[K_CUTOFF] += 1
                    break
            else:
                if d_bdy == np.inf:
                    # u == 0 in a void: the particle can never move. Only
                    # possible for a source in a void with u exactly 0.
                    lost = 1
                    cnt[K_LOST] += 1
                    break
                reg[TRACK_LENGTH, r] += wgt * d_bdy
                if g >= 0:
                    spec[SPEC_TL, r, g] += wgt * d_bdy
                t += d_bdy / speed(E)
                # place x exactly on the boundary and step the region index
                # by direction (as in the Phase 1 kernel)
                if u > 0.0:
                    surf[POS, r + 1] += wgt
                    r += 1
                    x = bounds[r]
                else:
                    surf[NEG, r] += wgt
                    x = bounds[r]
                    r -= 1
                if r < 0:
                    cnt[K_LEAK_LEFT] += 1
                    break
                if r >= n_regions:
                    cnt[K_LEAK_RIGHT] += 1
                    break

        # ---- next particle of the family: LIFO
        if n_bank == 0:
            break
        n_bank -= 1
        x = bank_f[n_bank, B_X]
        u = bank_f[n_bank, B_U]
        v = bank_f[n_bank, B_V]
        w = bank_f[n_bank, B_W]
        E = bank_f[n_bank, B_E]
        t = bank_f[n_bank, B_T]
        r = bank_r[n_bank]
    return int(rng[RNG_DRAWS]), lost


@njit(cache=True)
def run_batches_kin(batch_start, batch_end, histories_per_batch, master_seed,
                    bounds, mat_of_region,
                    egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                    nuc_awr, ch_off, ch_int, ch_q, chxs, ip, fp,
                    src_type, src_x, src_region, e_type, e_lo, e_hi,
                    e_cut, spec_edges, bank_capacity,
                    region_sums, surface_sums, spectrum, cutoff_weight, counts,
                    chan_events, chan_created):
    """Run batches [batch_start, batch_end), writing only their rows. Same
    structure and guarantees as transport.run_batches: batch b runs
    histories b*n .. b*n + n - 1, each seeded from (master_seed, id) only."""
    max_nuc = 0
    for m in range(m_off.shape[0] - 1):
        max_nuc = max(max_nuc, m_off[m + 1] - m_off[m])
    for b in range(batch_start, batch_end):
        reg = region_sums[b]
        surf = surface_sums[b]
        spec = spectrum[b]
        cutw = cutoff_weight[b]
        cnt = counts[b]
        chev = chan_events[b]
        chcr = chan_created[b]
        reg[:, :] = 0.0
        surf[:, :] = 0.0
        spec[:, :, :] = 0.0
        cutw[:] = 0.0
        cnt[:] = 0
        chev[:] = 0
        chcr[:] = 0
        rng = np.zeros(RNG_SIZE, np.uint64)
        bank_f = np.zeros((bank_capacity, B_NCOL), dtype=np.float64)
        bank_r = np.zeros(bank_capacity, dtype=np.int64)
        ci = np.zeros(max(max_nuc, 1), dtype=np.int64)
        cf = np.zeros(max(max_nuc, 1), dtype=np.float64)
        ct = np.zeros(max(max_nuc, 1), dtype=np.float64)
        max_draws = 0
        first = b * histories_per_batch
        for h in range(histories_per_batch):
            draws, lost = transport_history_kin(
                first + h, master_seed, bounds, mat_of_region,
                egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                nuc_awr, ch_off, ch_int, ch_q, chxs, ip, fp,
                src_type, src_x, src_region, e_type, e_lo, e_hi,
                e_cut, spec_edges,
                reg, surf, spec, cutw, cnt, chev, chcr,
                bank_f, bank_r, ci, cf, ct, rng)
            if draws > max_draws:
                max_draws = draws
        cnt[K_MAX_DRAWS] = max_draws
