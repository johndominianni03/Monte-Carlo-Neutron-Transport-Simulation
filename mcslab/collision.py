"""Collision physics for the continuous-energy kinematic kernel.

Transcribes OpenMC v0.16.0 (MIT license) src/physics.cpp and
src/math_functions.cpp; each function cites its original. Paths are
relative to the OpenMC repository.

Reaction channels. The scatter channels of nuclide k are
ch_off[k] .. ch_off[k+1]-1. The first is elastic (MT 2). The rest are the
non-redundant reactions with an outgoing neutron, in increasing MT. That is
OpenMC's index_inelastic_scatter_ order (src/nuclide.cpp, reactions read in
file order; src/endf.cpp is_inelastic_scatter). For these data it is
exactly the set of non-redundant reactions that are not in the absorption
("disappearance") set. The absorption cross section comes from the
pre-summed `absn` table of xs.pack.

    ch_int[c] = [mt, thr, xoff, cm, prod, nuc]
        thr, xoff : channel xs values chxs[xoff : xoff + n - thr] on grid
                    points thr .. n-1 of the nuclide grid
        cm        : 1 if the law's frame is centre of mass
        prod      : PRODUCT record in the distribution pools, or -1 if the
                    law is not implemented (the driver refuses problems
                    that can reach it, and the kernel raises if one is hit)
    ch_q[c]   = Q value (eV)

Velocities use OpenMC's units, v = sqrt(E) with E in eV.

The slab kernel carries a full direction (u, v, w) with u along the slab
normal, so the 3D vector code below is OpenMC's own.

Random numbers used per collision, in order:
    nuclide 1; absorption 1 if sigma_a > 0; scatter channel 1; then
    elastic (target at rest): angle (2, or 1 if isotropic) + phi 1
    inelastic:                law (see distributions.py) + phi 1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numba import njit

from . import distributions as D
from . import nucdata as N
from .rng import prn
from .xs import interp_at

CH_MT = 0
CH_THR = 1
CH_XOFF = 2
CH_CM = 3
CH_PROD = 4
CH_NUC = 5
CH_NCOL = 6

ELASTIC_MT = 2


# ---------------------------------------------------------------------------
# Packing (Python side)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackedPhysics:
    """Plain arrays for the kinematic kernel, plus per-channel metadata for
    the driver. Nuclide order is that of xs.pack (first use)."""
    nuc_awr: np.ndarray        # (n_nuclides,)
    nuc_kT: np.ndarray         # (n_nuclides,) eV
    ch_off: np.ndarray         # (n_nuclides + 1,)
    ch_int: np.ndarray         # (n_channels, CH_NCOL) int64
    ch_q: np.ndarray           # (n_channels,) eV
    chxs: np.ndarray           # channel cross sections (b)
    ip: np.ndarray             # distribution pools
    fp: np.ndarray
    labels: Tuple[Tuple[str, int], ...]      # (nuclide, MT) per channel
    unsupported: Tuple[Tuple[int, str], ...] # (channel, reason)
    first_positive: np.ndarray  # energy above which each channel's xs > 0


def _first_positive_energy(grid, rx) -> float:
    """Energy above which the reaction's cross section is positive: the grid
    point before its first positive value (lin-lin makes the xs > 0 just
    above it). inf if the xs is identically zero."""
    pos = np.flatnonzero(rx.xs > 0.0)
    if pos.size == 0:
        return math.inf
    j = rx.threshold_idx + int(pos[0])
    return float(grid[max(j - 1, 0)])


def _is_level(law) -> bool:
    return (isinstance(law, N.UncorrelatedAngleEnergy)
            and isinstance(law.energy, N.LevelInelastic))


def _check_laws(name, mt, product):
    """Elastic must be a single uncorrelated law without an energy law and
    with yield 1 (OpenMC elastic_scatter reads only its angle). Every other
    channel needs an energy distribution."""
    for law in product.laws:
        has_energy = (isinstance(law, N.CorrelatedAngleEnergy)
                      or (isinstance(law, N.UncorrelatedAngleEnergy) and law.energy is not None))
        if mt == ELASTIC_MT and (has_energy or len(product.laws) != 1
                                 or product.yield_coefficients != (1.0,)):
            raise ValueError(f"{name} MT 2: elastic must be one angle-only law with yield 1")
        if mt != ELASTIC_MT and not has_energy:
            raise D.Unsupported("non-elastic law without an energy distribution")


def _kernel_ready(product) -> bool:
    """Laws the transport kernel handles at this commit: Part 1's (one
    uncorrelated law, isotropic or tabular angle, no energy law or a level
    law, constant positive integer yield). The samplers handle every
    packed law; the kernel learns zero-yield events in the next commit."""
    y = product.yield_coefficients
    return (product.yield_table is None and len(y) == 1 and y[0] >= 1.0
            and math.floor(y[0]) == y[0] and len(product.laws) == 1
            and isinstance(product.laws[0], N.UncorrelatedAngleEnergy)
            and (product.laws[0].energy is None or _is_level(product.laws[0])))


def pack_physics(names: Sequence[str], nuclides: dict) -> PackedPhysics:
    """Scatter channels and their laws for the nuclides `names` (in xs.pack
    order). Laws mcslab cannot sample are recorded in `unsupported`."""
    pools = D.Pools()
    ch_off = [0]
    rows, qs, xs_parts, labels, unsupported, first_pos = [], [], [], [], [], []
    awr, kT = [], []
    xoff = 0
    for k, name in enumerate(names):
        nuc = nuclides[name]
        awr.append(nuc.awr)
        kT.append(nuc.kT)
        absorption = set(nuc.absorption_mts)
        mts = [m for m in nuc.partial_mts if m not in absorption]
        if not mts or mts[0] != ELASTIC_MT:
            raise ValueError(f"{name}: no elastic (MT 2) reaction")
        for mt in mts:
            rx = nuc.reactions[mt]
            products = rx.products
            if not products or products[0].particle != "neutron":
                # OpenMC samples products_[0] (physics.cpp inelastic_scatter)
                raise ValueError(f"{name} MT {mt}: product 0 is not the neutron")
            c = len(rows)
            try:
                prod = D.pack_product(pools, products[0])
                _check_laws(name, mt, products[0])
            except D.Unsupported as exc:
                prod = -1
                unsupported.append((c, f"{name} MT {mt}: {exc}"))
            if prod >= 0 and not _kernel_ready(products[0]):
                unsupported.append((c, f"{name} MT {mt}: law packed but not yet wired "
                                       "into the transport kernel"))
            if prod >= 0 and any(_is_level(law) for law in products[0].laws):
                # level law: E_cm >= 0 needs E >= -Q (A+1)/A wherever xs > 0
                thr = -rx.q_value * (nuc.awr + 1.0) / nuc.awr
                if _first_positive_energy(nuc.energy, rx) < thr:
                    raise ValueError(f"{name} MT {mt}: cross section positive below "
                                     f"the kinematic threshold {thr} eV")
            n = nuc.energy.size
            rows.append([mt, rx.threshold_idx, xoff, int(rx.center_of_mass), prod, k])
            qs.append(rx.q_value)
            xs_parts.append(rx.xs)
            xoff += rx.xs.size
            assert rx.threshold_idx + rx.xs.size == n
            labels.append((name, mt))
            first_pos.append(_first_positive_energy(nuc.energy, rx))
        ch_off.append(len(rows))
    return PackedPhysics(
        np.array(awr, dtype=np.float64), np.array(kT, dtype=np.float64),
        np.array(ch_off, dtype=np.int64),
        np.array(rows, dtype=np.int64).reshape(-1, CH_NCOL),
        np.array(qs, dtype=np.float64),
        np.concatenate(xs_parts) if xs_parts else np.zeros(0),
        pools.ints(), pools.floats(), tuple(labels), tuple(unsupported),
        np.array(first_pos, dtype=np.float64))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def rotate_angle(u, v, w, mu, rng):
    """OpenMC src/math_functions.cpp rotate_angle with phi sampled uniformly
    on [0, 2 pi) (1 draw). Rotates (u, v, w) by polar cosine mu."""
    phi = 2.0 * math.pi * prn(rng)
    sinphi = math.sin(phi)
    cosphi = math.cos(phi)
    a = math.sqrt(max(0.0, 1.0 - mu * mu))
    b = math.sqrt(max(0.0, 1.0 - w * w))
    if b > 1e-10:
        return (mu * u + a * (u * w * cosphi - v * sinphi) / b,
                mu * v + a * (v * w * cosphi + u * sinphi) / b,
                mu * w - a * b * cosphi)
    b = math.sqrt(1.0 - v * v)
    return (mu * u + a * (-u * v * sinphi + w * cosphi) / b,
            mu * v + a * b * sinphi,
            mu * w - a * (v * w * sinphi + u * cosphi) / b)


@njit(cache=True)
def sample_nuclide(m, m_off, mat_dens, micro_tot, st, rng):
    """OpenMC physics.cpp sample_nuclide (1 draw). micro_tot[j - m_off[m]]
    holds the micro totals the caller summed into st in this same order, so
    the cumulative sum reaches st exactly and the loop cannot fall through.
    A nuclide with zero cross section is never chosen (OpenMC could choose
    one only when the draw is exactly 0)."""
    cutoff = prn(rng) * st
    prob = 0.0
    last = -1
    for j in range(m_off[m], m_off[m + 1]):
        s = mat_dens[j] * micro_tot[j - m_off[m]]
        prob += s
        if s > 0.0:
            last = j
            if prob >= cutoff:
                return j
    return last


@njit(cache=True)
def channel_xs(c, i, f, n, ch_int, chxs):
    return interp_at(i, f, n, chxs, ch_int[c, CH_XOFF], ch_int[c, CH_THR])


@njit(cache=True)
def sample_scatter_channel(k, i, f, n, micro_tot, micro_abs, ch_off, ch_int, chxs, rng):
    """OpenMC physics.cpp scatter, channel choice (1 draw):
    cutoff = prn (sigma_t - sigma_a); elastic if sigma_el > cutoff, else the
    first inelastic channel at which the running sum reaches the cutoff.

    Round-off guard: the partials are interpolated separately from the
    pre-summed total, so the running sum can fall short of the cutoff. OpenMC
    then uses the last reaction it looked at, which may have zero cross
    section here. mcslab takes the last channel with a positive cross
    section instead (docs/deviations_from_openmc.md)."""
    cutoff = prn(rng) * (micro_tot - micro_abs)
    c0 = ch_off[k]
    c1 = ch_off[k + 1]
    prob = channel_xs(c0, i, f, n, ch_int, chxs)
    if prob > cutoff:
        return c0
    chosen = -1
    chosen_xs = 0.0
    c = c0 + 1
    while c < c1 and prob < cutoff:
        chosen_xs = channel_xs(c, i, f, n, ch_int, chxs)
        prob += chosen_xs
        chosen = c
        c += 1
    if chosen >= 0 and prob >= cutoff and chosen_xs > 0.0:
        return chosen
    last = -1
    for c in range(c0, c1):
        if channel_xs(c, i, f, n, ch_int, chxs) > 0.0:
            last = c
    if last < 0:
        raise RuntimeError("no scatter channel with a positive cross section")
    return last


@njit(cache=True)
def elastic_scatter(E, u, v, w, awr, ip, fp, prod, rng):
    """OpenMC physics.cpp elastic_scatter with the target at rest. The CM
    cosine is sampled at the lab energy E, as in OpenMC.
    Returns (E', u', v', w', mu_lab)."""
    vel = math.sqrt(E)
    vnx = vel * u
    vny = vel * v
    vnz = vel * w
    # velocity of the centre of mass (target at rest: v_t = 0)
    ap1 = awr + 1.0
    vcx = vnx / ap1
    vcy = vny / ap1
    vcz = vnz / ap1
    # transform to the CM frame
    vnx -= vcx
    vny -= vcy
    vnz -= vcz
    vel = math.sqrt(vnx * vnx + vny * vny + vnz * vnz)
    law = D.product_law(ip, prod)
    mu_cm = D.sample_mu(ip, fp, ip[law + 1], E, rng)
    ux, uy, uz = rotate_angle(vnx / vel, vny / vel, vnz / vel, mu_cm, rng)
    # back to the lab frame
    vnx = vel * ux + vcx
    vny = vel * uy + vcy
    vnz = vel * uz + vcz
    E_out = vnx * vnx + vny * vny + vnz * vnz
    vel = math.sqrt(E_out)
    mu_lab = (u * vnx + v * vny + w * vnz) / vel
    if abs(mu_lab) > 1.0:
        mu_lab = math.copysign(1.0, mu_lab)
    return E_out, vnx / vel, vny / vel, vnz / vel, mu_lab


@njit(cache=True)
def cm_to_lab(E_in, E_cm, mu_cm, awr):
    """OpenMC physics.cpp inelastic_scatter, CM -> lab for a neutron of CM
    energy E_cm and CM cosine mu_cm (target at rest)."""
    ap1 = awr + 1.0
    E = E_cm + (E_in + 2.0 * mu_cm * ap1 * math.sqrt(E_in * E_cm)) / (ap1 * ap1)
    mu = mu_cm * math.sqrt(E_cm / E) + 1.0 / ap1 * math.sqrt(E_in / E)
    return E, mu


@njit(cache=True)
def multiplicity(y, rng):
    """Number of outgoing neutrons for yield y (approved deviation D2).
    OpenMC (physics.cpp inelastic_scatter) creates y - 1 copies for an
    integer y > 0 and otherwise multiplies the weight by y. mcslab keeps
    weights at 1: an integer y (including 0) gives exactly y neutrons with
    no draw; otherwise one draw xi gives floor(y) + [xi < y - floor(y)],
    whose mean is exactly y."""
    fl = math.floor(y)
    if fl == y:
        return int(fl)
    if prn(rng) < y - fl:
        return int(fl) + 1
    return int(fl)


@njit(cache=True)
def inelastic_scatter(E_in, u, v, w, awr, q, cm, ip, fp, prod, rng):
    """OpenMC physics.cpp inelastic_scatter. Returns
    (E', u', v', w', mu_lab, neutrons out). The law is sampled, transformed
    from CM to lab if needed and the direction rotated (OpenMC's order);
    then the yield at E_in gives the number of outgoing neutrons
    (multiplicity, D2). The caller banks n - 1 identical copies for n >= 2,
    as OpenMC does, and ends the neutron for n = 0."""
    E, mu = D.sample_product(ip, fp, prod, E_in, awr, q, rng)
    if cm:
        E, mu = cm_to_lab(E_in, E, mu, awr)
    if abs(mu) > 1.0:
        mu = math.copysign(1.0, mu)
    u, v, w = rotate_angle(u, v, w, mu, rng)
    n_out = multiplicity(D.product_yield(ip, fp, prod, E_in), rng)
    return E, u, v, w, mu, n_out
