"""Secondary-neutron laws: packing into plain arrays, and Numba samplers.

The samplers transcribe OpenMC v0.16.0 (MIT license); each cites the
function it follows. Paths are relative to the OpenMC repository.

Every law is packed into two pools, ACE-style: `ip` (int64) holds records
of offsets and codes, and `fp` (float64) holds the numbers. A record is
addressed by its offset in `ip`. Layouts:

    TABULAR  [n, interp, fx]         x = fp[fx:fx+n], p = fp[fx+n:fx+2n],
                                     c = fp[fx+2n:fx+3n]; p and c already
                                     divided by c[n-1] (OpenMC Tabular::init)
    ANGLE    [n, fe, t_0 .. t_n-1]   incident energies fp[fe:fe+n];
                                     t_i = TABULAR record for energy i
    LAW      [kind, angle, ekind, erec]
                                     kind LAW_UNCORRELATED; angle = ANGLE
                                     record, or -1 for isotropic; ekind
                                     E_NONE | E_LEVEL; erec unused (-1)
    PRODUCT  [ykind, yref, n_law, app_0, law_0, ..]
                                     ykind Y_CONST with the yield in
                                     fp[yref]; app_j = -1 (applicability is
                                     Part 2, when n_law > 1 is needed)

Level inelastic (E_LEVEL) takes no data from the file. Its CM energy comes
from the nuclide's AWR and the reaction's Q, as exact two-body kinematics:

    E_cm = (A/(A+1))^2 (E - thr),   thr = -Q (A+1)/A

OpenMC uses the stored, rounded `mass_ratio` and `threshold` instead
(src/distribution_energy.cpp, LevelInelastic::sample). They differ by up to
4.7e-7 relative in these data. See docs/deviations_from_openmc.md.

Random numbers used (in this order, as in OpenMC):
    angle      : 1 (choose incident table, drawn even when r = 0) + 1 (CDF);
                 isotropic: 1
    level      : 0
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from numba import njit

from . import nucdata
from .rng import prn

LAW_UNCORRELATED = 1
E_NONE = 0
E_LEVEL = 1
Y_CONST = 0

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


def pack_angle(pools: Pools, ang: nucdata.AngleDistribution) -> int:
    tabs = [pack_tabular(pools, t) for t in ang.mu]
    fe = pools.put_floats(ang.energy)
    return pools.put_ints([ang.energy.size, fe] + tabs)


class Unsupported(Exception):
    """Raised by the packer for a law or yield mcslab does not implement."""


def pack_law(pools: Pools, law) -> int:
    if not isinstance(law, nucdata.UncorrelatedAngleEnergy):
        raise Unsupported(getattr(law, "description", type(law).__name__))
    if law.energy is not None and not isinstance(law.energy, nucdata.LevelInelastic):
        raise Unsupported(f"uncorrelated energy law {type(law.energy).__name__}")
    angle = -1 if law.angle is None else pack_angle(pools, law.angle)
    ekind = E_NONE if law.energy is None else E_LEVEL
    return pools.put_ints([LAW_UNCORRELATED, angle, ekind, -1])


def pack_product(pools: Pools, product: nucdata.Product) -> int:
    """PRODUCT record for a neutron product, or raise Unsupported."""
    if not product.laws_read:
        raise ValueError("product was read without distributions=True")
    if product.yield_table is not None or len(product.yield_coefficients) != 1:
        raise Unsupported("energy-dependent yield")
    y = product.yield_coefficients[0]
    if not (y > 0.0 and math.floor(y) == y):
        raise Unsupported(f"non-integer yield {y}")
    if len(product.laws) != 1:
        raise Unsupported(f"{len(product.laws)} distributions (applicability)")
    law = pack_law(pools, product.laws[0])
    yref = pools.put_floats([y])
    return pools.put_ints([Y_CONST, yref, 1, -1, law])


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
def sample_law(ip, fp, law, E_in, awr, q, rng):
    """(E_out, mu) in the law's frame. OpenMC src/secondary_uncorrelated.cpp
    UncorrelatedAngleEnergy::sample: angle first, then energy."""
    mu = sample_mu(ip, fp, ip[law + 1], E_in, rng)
    ekind = ip[law + 2]
    if ekind == E_LEVEL:
        return level_cm_energy(E_in, awr, q), mu
    return E_in, mu


@njit(cache=True)
def product_law(ip, prod):
    """The LAW record of a single-distribution product."""
    return ip[prod + 4]


@njit(cache=True)
def product_yield(ip, fp, prod):
    """Constant yield of a product (Y_CONST)."""
    return fp[ip[prod + 1]]


@njit(cache=True)
def sample_product(ip, fp, prod, E_in, awr, q, rng):
    """OpenMC src/reaction_product.cpp ReactionProduct::sample. With a
    single distribution no applicability draw is made."""
    return sample_law(ip, fp, product_law(ip, prod), E_in, awr, q, rng)
