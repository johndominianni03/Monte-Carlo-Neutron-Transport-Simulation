"""Secondary-neutron laws: packing into plain arrays, and Numba samplers.

The samplers transcribe OpenMC v0.16.0 (MIT license); each cites the
function it follows. Paths are relative to the OpenMC repository.

Every law is packed into two pools, ACE-style: `ip` (int64) holds records
of offsets and codes, and `fp` (float64) holds the numbers. A record is
addressed by its offset in `ip`. Layouts:

    TABULAR  [n, interp, fx]         x = fp[fx:fx+n], p = fp[fx+n:fx+2n],
                                     c = fp[fx+2n:fx+3n]; p and c already
                                     divided by c[n-1] (OpenMC Tabular::init)
    EOUT     [n, interp, fx]         as TABULAR, but p and c as stored: OpenMC
                                     samples outgoing-energy tables with the
                                     stored CDF and does not normalise it
    TAB1     [n, interp, fx]         Tabulated1D: x = fp[fx:fx+n],
                                     y = fp[fx+n:fx+2n]
    ANGLE    [n, fe, t_0 .. t_n-1]   incident energies fp[fe:fe+n];
                                     t_i = TABULAR record for energy i
    CONT     [n, fe, t_0 .. t_n-1]   ContinuousTabular (ACE law 4): incident
                                     energies fp[fe:fe+n]; t_i = EOUT record
    CORRTAB  [n, interp, fx, a_0 .. a_n-1]
                                     an EOUT table (same first three slots)
                                     followed by one TABULAR angle record per
                                     outgoing point
    LAW      uncorrelated: [LAW_UNCORRELATED, angle, ekind, erec]
                                     angle = ANGLE record, or -1 for
                                     isotropic; ekind E_NONE | E_LEVEL |
                                     E_CONTINUOUS; erec = CONT record or -1
             correlated:   [LAW_CORRELATED, n, fe, c_0 .. c_n-1]
                                     (ACE law 61) incident energies
                                     fp[fe:fe+n]; c_i = CORRTAB record
    PRODUCT  [ykind, yref, n_law, app_0, law_0, app_1, law_1, ..]
                                     ykind Y_CONST: yield fp[yref];
                                     Y_TAB1: yref = TAB1 record. app_j = TAB1
                                     record of law j's applicability, or -1
                                     when there is one law (OpenMC reads it
                                     only when n_law > 1)

Level inelastic (E_LEVEL) takes no data from the file. Its CM energy comes
from the nuclide's AWR and the reaction's Q, as exact two-body kinematics:

    E_cm = (A/(A+1))^2 (E - thr),   thr = -Q (A+1)/A

OpenMC uses the stored, rounded `mass_ratio` and `threshold` instead
(src/distribution_energy.cpp, LevelInelastic::sample). They differ by up to
4.7e-7 relative in these data. See docs/deviations_from_openmc.md.

Random numbers used (in this order, as in OpenMC):
    applicability: 1, only if the product has more than one law
    angle        : 1 (choose incident table, drawn even when r = 0) + 1 (CDF);
                   isotropic: 1
    level        : 0
    continuous   : 1 (choose incident table) + 1 (CDF); the angle is drawn
                   first
    correlated   : 1 (choose incident table) + 1 (E_out CDF) + 1 (mu)
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from numba import njit

from . import nucdata
from .rng import prn

LAW_UNCORRELATED = 1
LAW_CORRELATED = 2
E_NONE = 0
E_LEVEL = 1
E_CONTINUOUS = 2
Y_CONST = 0
Y_TAB1 = 1

# ---------------------------------------------------------------------------
# Packing (Python side)
# ---------------------------------------------------------------------------


class Pools:
    """Growing int / float pools. `ints()` and `floats()` give the arrays."""

    def __init__(self):
        self._ip: List[int] = []
        self._fp: List[float] = []

    def put_floats(self, values) -> int:
        off = len(self._fp)
        self._fp.extend(float(v) for v in np.asarray(values, dtype=np.float64).ravel())
        return off

    def put_ints(self, values) -> int:
        off = len(self._ip)
        self._ip.extend(int(v) for v in values)
        return off

    def reserve_ints(self, n) -> int:
        off = len(self._ip)
        self._ip.extend([0] * n)
        return off

    def set_int(self, i, value):
        self._ip[i] = int(value)

    def ints(self) -> np.ndarray:
        return np.array(self._ip, dtype=np.int64)

    def floats(self) -> np.ndarray:
        return np.array(self._fp, dtype=np.float64)


def pack_tabular(pools: Pools, tab: nucdata.Tabular) -> int:
    # OpenMC Tabular::init normalises p and c by c[n-1] (src/distribution.cpp).
    # The stored CDFs of these data end at exactly 1.0, so this changes no bit.
    integral = float(tab.c[-1])
    fx = pools.put_floats(np.concatenate([tab.x, tab.p / integral, tab.c / integral]))
    return pools.put_ints([tab.x.size, tab.interpolation, fx])


def pack_eout(pools: Pools, tab: nucdata.Tabular) -> int:
    """EOUT record: an outgoing-energy table as stored (OpenMC
    ContinuousTabular and CorrelatedAngleEnergy keep the stored CDF,
    src/distribution_energy.cpp and src/secondary_correlated.cpp)."""
    fx = pools.put_floats(np.concatenate([tab.x, tab.p, tab.c]))
    return pools.put_ints([tab.x.size, tab.interpolation, fx])


def pack_tab1(pools: Pools, tab: nucdata.Tabulated1D) -> int:
    if tab.x.size < 2:
        raise Unsupported("Tabulated1D with fewer than 2 points")
    fx = pools.put_floats(np.concatenate([tab.x, tab.y]))
    return pools.put_ints([tab.x.size, tab.interpolation, fx])


def pack_angle(pools: Pools, ang: nucdata.AngleDistribution) -> int:
    tabs = [pack_tabular(pools, t) for t in ang.mu]
    fe = pools.put_floats(ang.energy)
    return pools.put_ints([ang.energy.size, fe] + tabs)


def pack_continuous(pools: Pools, law: nucdata.ContinuousTabular) -> int:
    tabs = [pack_eout(pools, t) for t in law.tables]
    fe = pools.put_floats(law.energy)
    return pools.put_ints([law.energy.size, fe] + tabs)


def pack_correlated(pools: Pools, law: nucdata.CorrelatedAngleEnergy) -> int:
    recs = []
    for tab, angles in zip(law.tables, law.mu):
        if len(angles) != tab.x.size:
            raise ValueError("correlated law: one angle table per outgoing energy expected")
        arecs = [pack_tabular(pools, a) for a in angles]
        fx = pools.put_floats(np.concatenate([tab.x, tab.p, tab.c]))
        recs.append(pools.put_ints([tab.x.size, tab.interpolation, fx] + arecs))
    fe = pools.put_floats(law.energy)
    return pools.put_ints([LAW_CORRELATED, law.energy.size, fe] + recs)


class Unsupported(Exception):
    """Raised by the packer for a law or yield mcslab does not implement."""


def pack_law(pools: Pools, law) -> int:
    if isinstance(law, nucdata.CorrelatedAngleEnergy):
        return pack_correlated(pools, law)
    if not isinstance(law, nucdata.UncorrelatedAngleEnergy):
        raise Unsupported(getattr(law, "description", type(law).__name__))
    if law.energy is None:
        ekind, erec = E_NONE, -1
    elif isinstance(law.energy, nucdata.LevelInelastic):
        ekind, erec = E_LEVEL, -1
    elif isinstance(law.energy, nucdata.ContinuousTabular):
        ekind, erec = E_CONTINUOUS, pack_continuous(pools, law.energy)
    else:
        raise Unsupported(f"uncorrelated energy law {type(law.energy).__name__}")
    angle = -1 if law.angle is None else pack_angle(pools, law.angle)
    return pools.put_ints([LAW_UNCORRELATED, angle, ekind, erec])


def pack_product(pools: Pools, product: nucdata.Product) -> int:
    """PRODUCT record for a neutron product, or raise Unsupported."""
    if not product.laws_read:
        raise ValueError("product was read without distributions=True")
    if product.yield_table is not None:
        ykind, yref = Y_TAB1, pack_tab1(pools, product.yield_table)
        if (product.yield_table.y < 0.0).any():
            raise Unsupported("negative tabulated yield")
    else:
        if len(product.yield_coefficients) != 1:
            raise Unsupported("polynomial yield of degree > 0")
        y = product.yield_coefficients[0]
        if not y >= 0.0:
            raise Unsupported(f"negative yield {y}")
        ykind, yref = Y_CONST, pools.put_floats([y])
    n_law = len(product.laws)
    if n_law < 1:
        raise Unsupported("product without a distribution")
    if n_law > 1 and len(product.applicability) != n_law:
        raise ValueError("several distributions without applicability")
    entries = []
    for j, law in enumerate(product.laws):
        app = pack_tab1(pools, product.applicability[j]) if n_law > 1 else -1
        entries += [app, pack_law(pools, law)]
    return pools.put_ints([ykind, yref, n_law] + entries)


# ---------------------------------------------------------------------------
# Samplers (Numba)
# ---------------------------------------------------------------------------


@njit(cache=True)
def lower_bound_index(a, lo, hi, value):
    """OpenMC include/openmc/search.h lower_bound_index, on a[lo:hi]:
    0 if a[lo] == value, else (first index with a >= value) - lo - 1."""
    if a[lo] == value:
        return 0
    first = lo
    last = hi
    while first < last:
        mid = (first + last) >> 1
        if a[mid] < value:
            first = mid + 1
        else:
            last = mid
    return first - lo - 1


@njit(cache=True)
def get_energy_index(fp, fe, n, E):
    """OpenMC src/math_functions.cpp get_energy_index on energies
    fp[fe:fe+n]: index i and lin-lin factor f. Below the first energy,
    i = 0 and f = 0. Above the last, i = n - 1 and f = 0."""
    i = 0
    f = 0.0
    if E >= fp[fe]:
        i = lower_bound_index(fp, fe, fe + n, E)
        if i + 1 < n:
            f = (E - fp[fe + i]) / (fp[fe + i + 1] - fp[fe + i])
    return i, f


@njit(cache=True)
def sample_tabular(ip, fp, t, rng):
    """OpenMC src/distribution.cpp Tabular::sample_unbiased (1 draw)."""
    n = ip[t]
    interp = ip[t + 1]
    fx = ip[t + 2]
    fpp = fx + n
    fc = fx + 2 * n
    c = prn(rng)
    # std::lower_bound(c_.begin() + 1, c_.end(), c): first k >= 1 with c_k >= c
    lo = fc + 1
    hi = fc + n
    while lo < hi:
        mid = (lo + hi) >> 1
        if fp[mid] < c:
            lo = mid + 1
        else:
            hi = mid
    i = lo - fc - 1
    c_i = fp[fc + i]
    x_i = fp[fx + i]
    p_i = fp[fpp + i]
    if interp == 1:
        if p_i > 0.0:
            return x_i + (c - c_i) / p_i
        return x_i
    x_i1 = fp[fx + i + 1]
    p_i1 = fp[fpp + i + 1]
    m = (p_i1 - p_i) / (x_i1 - x_i)
    if m == 0.0:
        return x_i + (c - c_i) / p_i
    return x_i + (math.sqrt(max(0.0, p_i * p_i + 2.0 * m * (c - c_i))) - p_i) / m


@njit(cache=True)
def sample_angle(ip, fp, a, E, rng):
    """OpenMC src/distribution_angle.cpp AngleDistribution::sample (2 draws):
    pick table i or i+1 with probability 1-r / r, sample it, clamp to
    [-1, 1]."""
    n = ip[a]
    fe = ip[a + 1]
    i, r = get_energy_index(fp, fe, n, E)
    if r > prn(rng):
        i += 1
    mu = sample_tabular(ip, fp, ip[a + 2 + i], rng)
    if abs(mu) > 1.0:
        mu = math.copysign(1.0, mu)
    return mu


@njit(cache=True)
def sample_mu(ip, fp, angle, E, rng):
    """Angle of an uncorrelated law: tabular, or isotropic when angle < 0
    (OpenMC UncorrelatedAngleEnergy::sample: uniform_distribution(-1, 1))."""
    if angle < 0:
        return -1.0 + 2.0 * prn(rng)
    return sample_angle(ip, fp, angle, E, rng)


@njit(cache=True)
def level_cm_energy(E, awr, q):
    """CM outgoing energy of a discrete level, E_cm = (A/(A+1))^2 (E - thr),
    thr = -Q (A+1)/A (exact two-body kinematics; see the module docstring)."""
    ap1 = awr + 1.0
    thr = -q * ap1 / awr
    r = awr / ap1
    return r * r * (E - thr)


@njit(cache=True)
def tab1d_eval(ip, fp, rec, x):
    """OpenMC src/endf.cpp Tabulated1D::operator() for one histogram or
    lin-lin region: the end values outside the table, no extrapolation."""
    n = ip[rec]
    fx = ip[rec + 2]
    fy = fx + n
    if x < fp[fx]:
        return fp[fy]
    if x > fp[fx + n - 1]:
        return fp[fy + n - 1]
    i = lower_bound_index(fp, fx, fx + n, x)
    if ip[rec + 1] == 1:
        return fp[fy + i]
    x0 = fp[fx + i]
    x1 = fp[fx + i + 1]
    y0 = fp[fy + i]
    y1 = fp[fy + i + 1]
    r = (x - x0) / (x1 - x0)
    return y0 + r * (y1 - y0)


@njit(cache=True)
def eout_bin(fp, fc, n, r1):
    """The outgoing-energy bin search shared by OpenMC's
    ContinuousTabular::sample and CorrelatedAngleEnergy::sample_dist, for
    tables without discrete lines: -> (k, c_k, c_k1). c_k1 starts at +inf
    (as in secondary_correlated.cpp) and keeps the value of the last
    comparison. If the search runs to the last bin, c_k1 equals c_k."""
    c_k = fp[fc]
    k = 0
    c_k1 = np.inf
    for j in range(n - 2):
        k = j
        c_k1 = fp[fc + k + 1]
        if r1 < c_k1:
            break
        k = j + 1
        c_k = c_k1
    return k, c_k, c_k1


@njit(cache=True)
def eout_value(fp, fx, n, interp, k, c_k, r1, guard_zero_width):
    """E_out inside bin k of an EOUT table by inverting its stored CDF, as in
    OpenMC (histogram, or lin-lin with the quadratic solution).

    A zero-width lin-lin bin divides by zero in OpenMC's correlated sampler.
    ContinuousTabular guards it (E_out = E_k); guard_zero_width selects that
    behaviour. Otherwise mcslab raises: in these data such a bin is a final
    zero-mass pair after c = 1, which no draw can select."""
    fpp = fx + n
    E_k = fp[fx + k]
    p_k = fp[fpp + k]
    if interp == 1:
        if p_k > 0.0:
            return E_k + (r1 - c_k) / p_k
        return E_k
    E_k1 = fp[fx + k + 1]
    p_k1 = fp[fpp + k + 1]
    if E_k1 == E_k:
        if guard_zero_width:
            return E_k
        raise RuntimeError("correlated law: sampled a zero-width outgoing-energy bin")
    frac = (p_k1 - p_k) / (E_k1 - E_k)
    if frac == 0.0:
        return E_k + (r1 - c_k) / p_k
    return E_k + (math.sqrt(max(0.0, p_k * p_k + 2.0 * frac * (r1 - c_k))) - p_k) / frac


@njit(cache=True)
def scale_eout(ip, fp, t_i, t_i1, r, from_i1, E_out):
    """Scaled interpolation between the incident tables i and i+1
    (ContinuousTabular::sample / CorrelatedAngleEnergy::sample_dist):
    E_1 and E_K interpolate the tables' first and last outgoing energies,
    and the sampled E_out is mapped linearly from its own table's range."""
    n_i = ip[t_i]
    fx_i = ip[t_i + 2]
    n_i1 = ip[t_i1]
    fx_i1 = ip[t_i1 + 2]
    E_i_1 = fp[fx_i]
    E_i_K = fp[fx_i + n_i - 1]
    E_i1_1 = fp[fx_i1]
    E_i1_K = fp[fx_i1 + n_i1 - 1]
    E_1 = E_i_1 + r * (E_i1_1 - E_i_1)
    E_K = E_i_K + r * (E_i1_K - E_i_K)
    if not from_i1:
        return E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1)
    return E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1)


@njit(cache=True)
def sample_continuous(ip, fp, rec, E, rng):
    """OpenMC src/distribution_energy.cpp ContinuousTabular::sample, for one
    lin-lin incident region and no discrete lines (2 draws). Note its own
    bracketing: below the first incident energy i = 0, r = 0; above the
    last, i = n - 2, r = 1."""
    n_in = ip[rec]
    fe = ip[rec + 1]
    if E < fp[fe]:
        i = 0
        r = 0.0
    elif E > fp[fe + n_in - 1]:
        i = n_in - 2
        r = 1.0
    else:
        i = lower_bound_index(fp, fe, fe + n_in, E)
        r = (E - fp[fe + i]) / (fp[fe + i + 1] - fp[fe + i])
    l = i + 1 if r > prn(rng) else i
    t = ip[rec + 2 + l]
    n = ip[t]
    fx = ip[t + 2]
    r1 = prn(rng)
    k, c_k, c_k1 = eout_bin(fp, fx + 2 * n, n, r1)
    E_out = eout_value(fp, fx, n, ip[t + 1], k, c_k, r1, True)
    return scale_eout(ip, fp, ip[rec + 2 + i], ip[rec + 2 + i + 1], r, l != i, E_out)


@njit(cache=True)
def sample_correlated(ip, fp, law, E, rng):
    """OpenMC src/secondary_correlated.cpp CorrelatedAngleEnergy::sample_dist
    and ::sample, for no discrete lines (3 draws). -> (E_out, mu).

    The angle table is that of the outgoing point closer to the sampled CDF
    value: k if r1 - c_k < c_k1 - r1, else k + 1; always k for histogram
    E_out. Mirrored exactly, including that a search ending in the last bin
    leaves c_k1 == c_k, so that bin always takes table k + 1. mu is not
    clamped here (OpenMC clamps after the CM -> lab transform)."""
    n_in = ip[law + 1]
    fe = ip[law + 2]
    i, r = get_energy_index(fp, fe, n_in, E)
    if i + 1 >= n_in:
        # OpenMC would read past its tables here (E above the last incident
        # energy). In these data that energy is the grid maximum, so lookups
        # refuse such an E before it gets here.
        raise RuntimeError("correlated law: incident energy above its last table")
    l = i + 1 if r > prn(rng) else i
    t = ip[law + 3 + l]
    n = ip[t]
    interp = ip[t + 1]
    fx = ip[t + 2]
    r1 = prn(rng)
    k, c_k, c_k1 = eout_bin(fp, fx + 2 * n, n, r1)
    E_out = eout_value(fp, fx, n, interp, k, c_k, r1, False)
    E_out = scale_eout(ip, fp, ip[law + 3 + i], ip[law + 3 + i + 1], r, l != i, E_out)
    if r1 - c_k < c_k1 - r1 or interp == 1:
        a = ip[t + 3 + k]
    else:
        a = ip[t + 3 + k + 1]
    return E_out, sample_tabular(ip, fp, a, rng)


@njit(cache=True)
def sample_law(ip, fp, law, E_in, awr, q, rng):
    """(E_out, mu) in the law's frame. Uncorrelated (OpenMC
    src/secondary_uncorrelated.cpp UncorrelatedAngleEnergy::sample): angle
    first, then energy. Correlated: sample_correlated."""
    if ip[law] == LAW_CORRELATED:
        return sample_correlated(ip, fp, law, E_in, rng)
    mu = sample_mu(ip, fp, ip[law + 1], E_in, rng)
    ekind = ip[law + 2]
    if ekind == E_LEVEL:
        return level_cm_energy(E_in, awr, q), mu
    if ekind == E_CONTINUOUS:
        return sample_continuous(ip, fp, ip[law + 3], E_in, rng), mu
    return E_in, mu


@njit(cache=True)
def product_law(ip, prod):
    """The LAW record of a product's first (for elastic: only) law."""
    return ip[prod + 4]


@njit(cache=True)
def product_yield(ip, fp, prod, E):
    """Yield of a product at incident energy E: constant (Y_CONST) or
    Tabulated1D (Y_TAB1)."""
    if ip[prod] == Y_CONST:
        return fp[ip[prod + 1]]
    return tab1d_eval(ip, fp, ip[prod + 1], E)


@njit(cache=True)
def choose_law(ip, fp, prod, E_in, rng):
    """OpenMC src/reaction_product.cpp ReactionProduct::sample_dist: with
    more than one law, 1 draw c and the first law j with
    c <= sum_{i <= j} applicability_i(E_in); if the sum falls short, the
    last law. With one law, no draw. -> LAW record."""
    n_law = ip[prod + 2]
    j = 0
    if n_law > 1:
        c = prn(rng)
        prob = 0.0
        j = n_law - 1
        for jj in range(n_law):
            prob += tab1d_eval(ip, fp, ip[prod + 3 + 2 * jj], E_in)
            if c <= prob:
                j = jj
                break
    return ip[prod + 4 + 2 * j]


@njit(cache=True)
def sample_product(ip, fp, prod, E_in, awr, q, rng):
    """OpenMC src/reaction_product.cpp ReactionProduct::sample."""
    return sample_law(ip, fp, choose_law(ip, fp, prod, E_in, rng), E_in, awr, q, rng)
