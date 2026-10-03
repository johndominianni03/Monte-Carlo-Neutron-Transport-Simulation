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

Target motion. Elastic scattering uses OpenMC's free-gas (cxs) model when
E < free_gas_threshold * kT or awr <= 1 (collision.elastic_collision), with
kT that of the nuclide's data temperature, selected per material. Otherwise,
and for kT = 0 synthetic nuclides, the target is at rest. Inelastic
channels always see a target at rest, as in OpenMC.

Multiplicity (approved deviation D2). A channel's yield y at the incident
energy gives n outgoing neutrons (collision.multiplicity): n >= 2 banks
n - 1 copies; n = 0 ends the neutron (a zero-yield event, e.g. Fe or W
MT 5 below about 6 MeV, where y = 0). OpenMC multiplies the weight by y
there, so its particle dies with weight 0. mcslab scores the ended weight
in zero_yield_weight[r] and counts it (K_ZERO_YIELD, chan_zero); it is
never dropped silently (approved P2).

Energy cutoff (approved deviation D5). After each collision, a neutron below
e_cut is killed, as OpenMC's collision() does with its (default 0) cutoff. A
secondary born below e_cut is not banked, as in OpenMC's create_secondary.
Neither is dropped silently: its weight is scored in cutoff_weight[r] for
the region of the collision and counted in K_CUTOFF, so

    sources + created == absorbed + leaked_left + leaked_right + cutoff
                         + zero_yield

holds exactly (tests/test_kinematics.py test d, tests/test_laws.py test g).

Direction is a full unit vector (u, v, w) with u along the slab normal. The
source sets (mu, sqrt(1 - mu^2), 0) and draws nothing extra. The slab is
invariant under rotation about x and every collision is rotationally
covariant, so this choice of azimuth changes no tally.

Time: t += d / speed(E), with OpenMC's relativistic speed
c sqrt(E (E + 2m)) / (E + m) (particle.cpp Particle::speed; constants.h,
CODATA 2018). It is carried for track recording and scores nothing; a
secondary starts at its parent's time.

Track recording (opt-in). History h < n_track owns slot h of the track
arrays: trk_f[h, j] = (x, u, E, t) and trk_i[h, j] = (particle id in the
family, parent id, event code EV_*, MT, region) of its j-th event. The
primary is particle 0; each secondary gets the next id when it is
created. record_event only reads the particle state: it draws no random
number and writes no tally, so tallies are bit-identical with recording
on or off (tests/test_tracks.py). A full slot range counts the dropped
events in trk_trunc[h]; physics is unaffected.

Response tallies (Phase 3, opt-in). With a depth mesh (n_bins > 0), every
flight segment is split across the depth bins of its region after the fact
(mcslab/depth_mesh.py) and scores w l N_k sigma_s(E) per nuclide k and
response s (track length), and every collision scores w N_k sigma_s(E) /
Sigma_t(E) in the bin of its position (collision estimator), with E the
pre-collision energy, as OpenMC's track-length and collision estimators do
(src/tallies/tally_scoring.cpp). sigma_s is interpolated with the grid index
and factor macro_total already computed (mcslab/responses.py). The
uncollided estimators score only the primary before its first collision.
Like track recording, the tallies only read particle state: they draw no
random number and write only their own rows, so every other output is
bit-identical with them on or off (tests/test_tally_reproducibility.py).

Random-number consumption per event (part of the regression contract):
  source energy   : monoenergetic 0; log-uniform 1 (drawn first)
  source direction: beam 0; isotropic 1
  flight          : 1 (distance), skipped in void (Sigma_t = 0)
  collision       : nuclide 1; absorption 1 if sigma_a > 0 for that
                    nuclide; scatter channel 1; then
                    elastic (target at rest): angle 2 (isotropic 1) + phi 1
                    elastic (free gas): per rejection-loop iteration
                      3 + [1] + 2, then target direction 1, then angle
                      2 (isotropic 1) + phi 1
                    other channels: applicability 1 (only if the product
                    has several laws), then the law:
                      level:              angle 2 (isotropic 1)
                      continuous (law 4): angle 2 + energy 2
                      correlated (law 61): 3
                    then phi 1, then 1 for a non-integer yield
"""
import math

import numpy as np
from numba import njit

from . import collision as C
from .depth_mesh import bin_of, segment_pieces
from .geometry import distance_to_boundary
from .responses import refresh_responses
from .rng import RNG_DRAWS, RNG_SIZE, init_history, prn
from .sources import sample_source
from .tracks import (EV_ABSORB, EV_BORN, EV_COLLISION, EV_CUTOFF, EV_LEAK, EV_LOST,
                     EV_SOURCE, EV_SURFACE, EV_ZERO_YIELD, TF_E, TF_T, TF_U, TF_X,
                     TI_EVENT, TI_MT, TI_PARENT, TI_PID, TI_REGION)
from .tallies import (ABSORPTION, COLL_ESTIMATOR, COLLISION, EST_COLL, EST_COLL_UNC,
                      EST_TL, EST_TL_UNC, K_ABSORBED,
                      K_BORN_BELOW_CUTOFF, K_COLLISIONS, K_CREATED, K_CUTOFF, K_ELASTIC,
                      K_INELASTIC, K_LEAK_LEFT, K_LEAK_RIGHT, K_LOST, K_MAX_BANK,
                      K_FREE_GAS, K_MAX_DRAWS, K_SOURCE, K_ZERO_YIELD, N_RESP, NEG, POS,
                      SPEC_COLL, SPEC_TL, TRACK_LENGTH)
from .transport_ce import sample_energy
from .xs import grid_locate, interp_at

C_LIGHT = 2.99792458e10          # cm/s (OpenMC constants.h C_LIGHT)
MASS_NEUTRON_EV = 939.56542052e6  # eV/c^2 (OpenMC constants.h, CODATA 2018)

# secondary bank columns
B_X, B_U, B_V, B_W, B_E, B_T, B_NCOL = 0, 1, 2, 3, 4, 5, 6

MT_ABSORPTION = 101  # OpenMC's event MT for an absorption (N_DISAPPEAR)


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
def record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent, ev, mt, r, x, u, E, t):
    """Append one event to recorded history `slot`. Reads the particle state
    only: no random number, no tally. When the slot range is full, the event
    is counted in trk_trunc instead."""
    j = trk_n[slot]
    if j >= trk_f.shape[1]:
        trk_trunc[slot] += 1
        return
    trk_f[slot, j, TF_X] = x
    trk_f[slot, j, TF_U] = u
    trk_f[slot, j, TF_E] = E
    trk_f[slot, j, TF_T] = t
    trk_i[slot, j, TI_PID] = pid
    trk_i[slot, j, TI_PARENT] = parent
    trk_i[slot, j, TI_EVENT] = ev
    trk_i[slot, j, TI_MT] = mt
    trk_i[slot, j, TI_REGION] = r
    trk_n[slot] = j + 1


@njit(cache=True)
def score_track(tal, flx, m_edges, m_eoff, m_boff, r, x0, u, d, wgt, unc, j0, j1, mat_nuc,
                ns, ntot, pb, pl):
    """Track-length estimators of one flight (x0, u, length d) in region r:
    for each piece of length l in depth bin b, w l into the flux and
    (w l) N_k sigma_s(E) per nuclide k (material slots j0 .. j1 - 1) and for
    the material total (last nuclide slot); also into the uncollided
    estimator when unc is set. ns, ntot hold N sigma at the flight energy."""
    e0 = m_eoff[r]
    npc = segment_pieces(m_edges, e0, m_eoff[r + 1] - e0 - 1, x0, u, d, pb, pl)
    tot = tal.shape[1] - 1
    b0 = m_boff[r]
    for q in range(npc):
        b = b0 + pb[q]
        wl = wgt * pl[q]
        flx[b, EST_TL] += wl
        if unc:
            flx[b, EST_TL_UNC] += wl
        for j in range(j0, j1):
            k = mat_nuc[j]
            jj = j - j0
            for s in range(N_RESP):
                v = wl * ns[jj, s]
                tal[b, k, s, EST_TL] += v
                if unc:
                    tal[b, k, s, EST_TL_UNC] += v
        for s in range(N_RESP):
            v = wl * ntot[s]
            tal[b, tot, s, EST_TL] += v
            if unc:
                tal[b, tot, s, EST_TL_UNC] += v


@njit(cache=True)
def score_collision(tal, flx, m_edges, m_eoff, m_boff, r, x, c, unc, j0, j1, mat_nuc,
                    ns, ntot):
    """Collision estimators at a collision at x in region r, with
    c = w / Sigma_t(E) at the pre-collision energy: c into the flux and
    c N_k sigma_s(E) per nuclide and for the total, in the depth bin of x;
    also into the first-collision estimator when unc is set."""
    e0 = m_eoff[r]
    b = m_boff[r] + bin_of(m_edges, e0, m_eoff[r + 1] - e0 - 1, x)
    tot = tal.shape[1] - 1
    flx[b, EST_COLL] += c
    if unc:
        flx[b, EST_COLL_UNC] += c
    for j in range(j0, j1):
        k = mat_nuc[j]
        jj = j - j0
        for s in range(N_RESP):
            v = c * ns[jj, s]
            tal[b, k, s, EST_COLL] += v
            if unc:
                tal[b, k, s, EST_COLL_UNC] += v
    for s in range(N_RESP):
        v = c * ntot[s]
        tal[b, tot, s, EST_COLL] += v
        if unc:
            tal[b, tot, s, EST_COLL_UNC] += v


@njit(cache=True)
def transport_history_kin(history, master_seed, bounds, mat_of_region,
                          egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                          nuc_awr, nuc_kT, fg_threshold, free_gas,
                          ch_off, ch_int, ch_q, chxs, ip, fp,
                          src_type, src_x, src_region, e_type, e_lo, e_hi,
                          e_cut, spec_edges,
                          reg, surf, spec, cutw, cnt, chev, chcr, zyw, chz,
                          bank_f, bank_r, bank_p, ci, cf, ct, rng,
                          n_track, trk_f, trk_i, trk_n, trk_trunc,
                          rxs, roff, rthr, m_edges, m_eoff, m_boff, tal, flx, ns, ntot,
                          pb, pl):
    """Follow one source neutron and all its secondaries. Tallies go to this
    batch's rows (reg, surf, spec, cutw, cnt, chev, chcr, zyw, chz, and the
    response rows tal, flx when the depth mesh has bins). If
    history < n_track its events are recorded in track slot `history`.
    Returns (draws used by the family, lost flag)."""
    do_tal = flx.shape[0] > 0
    unc = 1          # the primary is uncollided until its first collision
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
    slot = history if history < n_track else -1
    pid = 0
    parent = -1
    next_pid = 1
    if slot >= 0:
        record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent, EV_SOURCE, 0, r,
                     x, u, E, t)
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
                if do_tal:
                    refresh_responses(m, m_off, mat_nuc, mat_dens, e_off, absn, rxs, roff,
                                      rthr, ci, cf, ns, ntot)
                last_m = m
                last_E = E
            if st > 0.0:
                d_coll = -math.log(1.0 - prn(rng)) / st
            else:
                d_coll = np.inf
            d_bdy = distance_to_boundary(x, u, bounds[r], bounds[r + 1])
            g = energy_bin(spec_edges, E)

            if d_coll < d_bdy:
                if do_tal:
                    score_track(tal, flx, m_edges, m_eoff, m_boff, r, x, u, d_coll, wgt, unc,
                                m_off[m], m_off[m + 1], mat_nuc, ns, ntot, pb, pl)
                x += d_coll * u
                t += d_coll / speed(E)
                reg[TRACK_LENGTH, r] += wgt * d_coll
                reg[COLLISION, r] += wgt
                reg[COLL_ESTIMATOR, r] += wgt / st
                if g >= 0:
                    spec[SPEC_TL, r, g] += wgt * d_coll
                    spec[SPEC_COLL, r, g] += wgt / st
                cnt[K_COLLISIONS] += 1
                if do_tal:
                    score_collision(tal, flx, m_edges, m_eoff, m_boff, r, x, wgt / st, unc,
                                    m_off[m], m_off[m + 1], mat_nuc, ns, ntot)
                unc = 0

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
                        if slot >= 0:
                            record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent,
                                         EV_ABSORB, MT_ABSORPTION, r, x, u, E, t)
                        break
                c = C.sample_scatter_channel(k, i, f, n, micro_t, micro_a,
                                             ch_off, ch_int, chxs, rng)
                chev[c] += 1
                prod = ch_int[c, C.CH_PROD]
                if prod < 0:
                    raise RuntimeError("sampled a reaction whose law is not implemented")
                if c == ch_off[k]:
                    E, u, v, w, mu_lab, fg = C.elastic_collision(
                        E, u, v, w, nuc_awr[k], nuc_kT[k], fg_threshold, free_gas,
                        ip, fp, prod, rng)
                    cnt[K_ELASTIC] += 1
                    cnt[K_FREE_GAS] += fg
                    n_out = 1
                else:
                    E, u, v, w, mu_lab, n_out = C.inelastic_scatter(
                        E, u, v, w, nuc_awr[k], ch_q[c], ch_int[c, C.CH_CM],
                        ip, fp, prod, rng)
                    cnt[K_INELASTIC] += 1
                if slot >= 0:
                    record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent,
                                 EV_COLLISION, ch_int[c, C.CH_MT], r, x, u, E, t)

                if n_out == 0:
                    # zero-yield event: the neutron ends, scored (never dropped)
                    zyw[r] += wgt
                    cnt[K_ZERO_YIELD] += 1
                    chz[c] += 1
                    if slot >= 0:
                        record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent,
                                     EV_ZERO_YIELD, ch_int[c, C.CH_MT], r, x, u, E, t)
                    break
                if n_out > 1:
                    extra = n_out - 1
                    chcr[c] += extra
                    cnt[K_CREATED] += extra
                    if E < e_cut:
                        # born below the cutoff: not banked, but scored
                        cutw[r] += wgt * extra
                        cnt[K_CUTOFF] += extra
                        cnt[K_BORN_BELOW_CUTOFF] += extra
                        for _ in range(extra):
                            if slot >= 0:
                                record_event(trk_f, trk_i, trk_n, trk_trunc, slot, next_pid,
                                             pid, EV_BORN, 0, r, x, u, E, t)
                                record_event(trk_f, trk_i, trk_n, trk_trunc, slot, next_pid,
                                             pid, EV_CUTOFF, 0, r, x, u, E, t)
                            next_pid += 1
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
                            bank_p[n_bank, 0] = next_pid
                            bank_p[n_bank, 1] = pid
                            next_pid += 1
                            n_bank += 1
                        if n_bank > cnt[K_MAX_BANK]:
                            cnt[K_MAX_BANK] = n_bank
                if E < e_cut:
                    cutw[r] += wgt
                    cnt[K_CUTOFF] += 1
                    if slot >= 0:
                        record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent,
                                     EV_CUTOFF, 0, r, x, u, E, t)
                    break
            else:
                if d_bdy == np.inf:
                    # u == 0 in a void: the particle can never move. Only
                    # possible for a source in a void with u exactly 0.
                    lost = 1
                    cnt[K_LOST] += 1
                    if slot >= 0:
                        record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent,
                                     EV_LOST, 0, r, x, u, E, t)
                    break
                if do_tal:
                    score_track(tal, flx, m_edges, m_eoff, m_boff, r, x, u, d_bdy, wgt, unc,
                                m_off[m], m_off[m + 1], mat_nuc, ns, ntot, pb, pl)
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
                if slot >= 0:
                    ev = EV_LEAK if (r < 0 or r >= n_regions) else EV_SURFACE
                    record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent, ev, 0, r,
                                 x, u, E, t)
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
        pid = bank_p[n_bank, 0]
        parent = bank_p[n_bank, 1]
        unc = 0
        if slot >= 0:
            record_event(trk_f, trk_i, trk_n, trk_trunc, slot, pid, parent, EV_BORN, 0, r,
                         x, u, E, t)
    return int(rng[RNG_DRAWS]), lost


@njit(cache=True)
def run_batches_kin(batch_start, batch_end, histories_per_batch, master_seed,
                    bounds, mat_of_region,
                    egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                    nuc_awr, nuc_kT, fg_threshold, free_gas,
                    ch_off, ch_int, ch_q, chxs, ip, fp,
                    src_type, src_x, src_region, e_type, e_lo, e_hi,
                    e_cut, spec_edges, bank_capacity,
                    region_sums, surface_sums, spectrum, cutoff_weight, counts,
                    chan_events, chan_created, zero_yield_weight, chan_zero,
                    n_track, trk_f, trk_i, trk_n, trk_trunc,
                    rxs, roff, rthr, m_edges, m_eoff, m_boff, tally, mesh_flux):
    """Run batches [batch_start, batch_end), writing only their rows. Same
    structure and guarantees as transport.run_batches: batch b runs
    histories b*n .. b*n + n - 1, each seeded from (master_seed, id) only.
    Track slots of the histories run here are cleared first. tally and
    mesh_flux have zero depth bins when the response tallies are off."""
    max_nuc = 0
    for m in range(m_off.shape[0] - 1):
        max_nuc = max(max_nuc, m_off[m + 1] - m_off[m])
    max_bins = 1
    for r in range(m_eoff.shape[0] - 1):
        max_bins = max(max_bins, m_eoff[r + 1] - m_eoff[r] - 1)
    for b in range(batch_start, batch_end):
        reg = region_sums[b]
        surf = surface_sums[b]
        spec = spectrum[b]
        cutw = cutoff_weight[b]
        cnt = counts[b]
        chev = chan_events[b]
        chcr = chan_created[b]
        zyw = zero_yield_weight[b]
        chz = chan_zero[b]
        reg[:, :] = 0.0
        surf[:, :] = 0.0
        spec[:, :, :] = 0.0
        cutw[:] = 0.0
        cnt[:] = 0
        chev[:] = 0
        chcr[:] = 0
        zyw[:] = 0.0
        chz[:] = 0
        tal = tally[b]
        flx = mesh_flux[b]
        tal[:, :, :, :] = 0.0
        flx[:, :] = 0.0
        ns = np.zeros((max(max_nuc, 1), N_RESP), dtype=np.float64)
        ntot = np.zeros(N_RESP, dtype=np.float64)
        pb = np.zeros(max_bins, dtype=np.int64)
        pl = np.zeros(max_bins, dtype=np.float64)
        rng = np.zeros(RNG_SIZE, np.uint64)
        bank_f = np.zeros((bank_capacity, B_NCOL), dtype=np.float64)
        bank_r = np.zeros(bank_capacity, dtype=np.int64)
        bank_p = np.zeros((bank_capacity, 2), dtype=np.int64)
        ci = np.zeros(max(max_nuc, 1), dtype=np.int64)
        cf = np.zeros(max(max_nuc, 1), dtype=np.float64)
        ct = np.zeros(max(max_nuc, 1), dtype=np.float64)
        max_draws = 0
        first = b * histories_per_batch
        for h in range(first, min(first + histories_per_batch, n_track)):
            trk_f[h, :, :] = 0.0
            trk_i[h, :, :] = 0
            trk_n[h] = 0
            trk_trunc[h] = 0
        for h in range(histories_per_batch):
            draws, lost = transport_history_kin(
                first + h, master_seed, bounds, mat_of_region,
                egrid, e_off, tot, absn, m_off, mat_nuc, mat_dens,
                nuc_awr, nuc_kT, fg_threshold, free_gas,
                ch_off, ch_int, ch_q, chxs, ip, fp,
                src_type, src_x, src_region, e_type, e_lo, e_hi,
                e_cut, spec_edges,
                reg, surf, spec, cutw, cnt, chev, chcr, zyw, chz,
                bank_f, bank_r, bank_p, ci, cf, ct, rng,
                n_track, trk_f, trk_i, trk_n, trk_trunc,
                rxs, roff, rthr, m_edges, m_eoff, m_boff, tal, flx, ns, ntot, pb, pl)
            if draws > max_draws:
                max_draws = draws
        cnt[K_MAX_DRAWS] = max_draws
