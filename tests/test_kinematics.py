"""Phase 2b Part 1 validation: collision kinematics and the kinematic kernel.

Seeds were fixed in docs/phase2b_plan.md before any of these tests existed
(project conventions: no seed-shopping). Acceptance: 3 SE for means; p >= 0.0027 for
chi-square and Hotelling tests (the false-alarm rate of a two-sided 3-sigma
check); "exact" = integer identities on raw sums (all weights are 1.0) or
round-off with a stated bound.

What each test covers, and what it doesn't:
  (a2) test_a2_*: the tabular-angle sampler (distributions.sample_angle) on
       real ENDF/B-VIII.0 tables. Expected bin probabilities are computed
       here from the raw HDF5 datasets (h5py) through the sampler's exact
       inverse-CDF map, not through the packer. It covers table search,
       stochastic interpolation between incident energies, lin-lin and
       histogram tables, and the repeated-energy rule. It does not test the
       data themselves.
"""
import math
import os

import h5py
import numpy as np
import pytest
from numba import njit
from scipy import stats

from mcslab import distributions as D
from mcslab.nucdata import AngleDistribution, Library, Tabular
from mcslab.rng import RNG_SIZE, init_history, prn

P_MIN = 0.0027          # chi-square / Hotelling acceptance (3-sigma equivalent)

SEED_A2 = 20261011


def _library():
    try:
        return Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def chi2_check(report, quantity, counts, expected_counts, max_draws=None):
    """Pearson chi-square of observed vs expected counts (sum equal)."""
    counts = np.asarray(counts, dtype=np.float64)
    expected_counts = np.asarray(expected_counts, dtype=np.float64)
    assert abs(counts.sum() - expected_counts.sum()) < 1e-6 * counts.sum()
    stat = float(((counts - expected_counts) ** 2 / expected_counts).sum())
    dof = counts.size - 1
    p = float(stats.chi2.sf(stat, dof))
    report(f"{quantity} chi2 (p={p:.3f})", stat, dof, math.sqrt(2.0 * dof), max_draws)
    assert p >= P_MIN, f"{quantity}: chi2 = {stat:.1f} on {dof} dof, p = {p:.2e}"
    return p


def equiprobable_edges(cdf, lo, hi, k):
    """Edges splitting [lo, hi] into k bins of equal probability under a
    continuous non-decreasing cdf with cdf(lo) = 0, cdf(hi) = 1 (bisection)."""
    edges = [lo]
    for j in range(1, k):
        target = j / k
        a, b = lo, hi
        for _ in range(200):
            mid = 0.5 * (a + b)
            if cdf(mid) < target:
                a = mid
            else:
                b = mid
        edges.append(0.5 * (a + b))
    edges.append(hi)
    return np.array(edges)


# ---------------------------------------------------------------------------
# (a2) tabular angle sampler on real data
# ---------------------------------------------------------------------------
def raw_angle(nuclide, mt):
    """(incident energies, [(x, p, c, interp), ...]) straight from the HDF5 file."""
    lib = _library()
    path = os.path.join(str(lib.root), "neutron", nuclide + ".h5")
    with h5py.File(path, "r") as f:
        g = f[f"{nuclide}/reactions/reaction_{mt:03d}/product_0/distribution_0/angle"]
        energy = g["energy"][()]
        mu = g["mu"]
        data = mu[()]
        offsets = mu.attrs["offsets"]
        interps = mu.attrs["interpolation"]
    ends = list(offsets[1:]) + [data.shape[1]]
    tables = [(data[0, a:b], data[1, a:b], data[2, a:b], int(it))
              for a, b, it in zip(offsets, ends, interps)]
    return energy, tables


def table_cdf(table, X):
    """P(sampled mu <= X) for Tabular::sample_unbiased on (x, p, c): a draw
    c in [c_i, c_i+1) maps monotonically to x_i + d with
    Q_i(d) = c - c_i, Q_i(d) = p_i d (histogram) or p_i d + m d^2/2 (lin-lin).
    So P(mu <= X) = sum_i clip(Q_i(X - x_i), 0, c_i+1 - c_i) / c_last. This is
    exact for the sampler even where the stored c differ from the integral
    of p (by at most 5e-7 in these data)."""
    x, p, c, interp = table
    total = 0.0
    for i in range(x.size - 1):
        d = X - x[i]
        if d <= 0.0:
            break
        dc = c[i + 1] - c[i]
        if interp == 1:
            q = p[i] * d
        else:
            m = (p[i + 1] - p[i]) / (x[i + 1] - x[i])
            if m < 0.0:
                d = min(d, -p[i] / m)
            q = p[i] * d + 0.5 * m * d * d
        total += min(max(q, 0.0), dc)
    return total / c[-1]


def mixture_cdf(energy, tables, E):
    """CDF of AngleDistribution::sample at E: table i+1 with probability r,
    table i otherwise, (i, r) from OpenMC's get_energy_index."""
    n = energy.size
    i, r = 0, 0.0
    if E >= energy[0]:
        # lower_bound_index: 0 if equal to the first value, else last index
        # with energy < E
        i = 0 if energy[0] == E else int(np.searchsorted(energy, E, side="left")) - 1
        if i + 1 < n:
            r = (E - energy[i]) / (energy[i + 1] - energy[i])
    if r == 0.0:
        return lambda X: table_cdf(tables[i], X)
    return lambda X: (1.0 - r) * table_cdf(tables[i], X) + r * table_cdf(tables[i + 1], X)


@njit(cache=True)
def _sample_angles(ip, fp, a, E, seed, n):
    """n cosines from one AngleDistribution record, sample s on the stream of
    history s."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    out = np.empty(n)
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        out[s] = D.sample_angle(ip, fp, a, E, rng)
    return out


@njit(cache=True)
def _angle_draws(seed, n):
    """The two draws sample_angle makes for sample s: (table choice, CDF)."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    out = np.empty((n, 2))
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        out[s, 0] = prn(rng)
        out[s, 1] = prn(rng)
    return out


def synthetic_histogram_angle():
    """Two different, deliberately unnormalised histogram tables (the real
    data only have isotropic 2-point histograms, identical at both
    energies). Sampling must divide p and c by c[-1], as OpenMC does."""
    t0 = (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([0.1, 0.2, 0.3, 0.4, 0.4]),
          np.array([0.0, 0.05, 0.15, 0.30, 0.50]), 1)
    t1 = (np.array([-1.0, 0.0, 1.0]), np.array([0.8, 0.2, 0.2]),
          np.array([0.0, 0.8, 1.0]), 1)
    energy = np.array([1.0e6, 2.0e6])
    ang = AngleDistribution(energy, tuple(Tabular(*t) for t in (t0, t1)))
    return energy, [t0, t1], ang


def _packed(ang):
    pools = D.Pools()
    a = D.pack_angle(pools, ang)
    return pools.ints(), pools.floats(), a


def _nearest_index(energy, target):
    return int(np.argmin(np.abs(energy - target)))


A2_CASES = [
    # (nuclide or "synthetic", MT, how the incident energy is chosen)
    ("Fe56", 2, "on-grid near 14 MeV"),
    ("Fe56", 2, "midpoint near 1 MeV"),      # lin-lin tables that differ
    ("W184", 2, "on-grid near 14 MeV"),
    ("synthetic", 0, "r = 0.3"),             # histogram tables that differ
]


@pytest.mark.parametrize("nuclide,mt,where", A2_CASES)
def test_a2_tabular_angle_sampler(report, note, nuclide, mt, where):
    if nuclide == "synthetic":
        energy, tables, ang = synthetic_histogram_angle()
        E = 1.3e6
    else:
        energy, tables = raw_angle(nuclide, mt)
        ang = _library().load(nuclide, distributions=True).reactions[mt].products[0].laws[0].angle
        if where.startswith("on-grid"):
            E = float(energy[_nearest_index(energy, 14.0e6)])
        else:
            j = _nearest_index(energy, 1.0e6)
            E = 0.5 * float(energy[j] + energy[j + 1])
    cdf = mixture_cdf(energy, tables, E)
    k, n = 40, 200_000
    edges = equiprobable_edges(cdf, -1.0, 1.0, k)
    probs = np.diff([cdf(e) for e in edges])
    ip, fp, a = _packed(ang)
    mu = _sample_angles(ip, fp, a, E, np.uint64(SEED_A2), n)
    assert np.all(np.abs(mu) <= 1.0)
    counts = np.histogram(mu, bins=edges)[0]
    chi2_check(report, f"{nuclide} MT {mt} mu_cm at E={E:.6g} eV", counts, n * probs)

    # exact: the sampler is the inverse CDF of the table it picked.
    # Replay the two draws of the first 5000 samples.
    i = 0 if energy[0] == E else int(np.searchsorted(energy, E, side="left")) - 1
    r = (E - energy[i]) / (energy[i + 1] - energy[i]) if E >= energy[0] and i + 1 < energy.size else 0.0
    draws = _angle_draws(np.uint64(SEED_A2), 5000)
    worst = 0.0
    for s in range(5000):
        t = tables[i + 1] if r > draws[s, 0] else tables[i]
        worst = max(worst, abs(table_cdf(t, mu[s]) - draws[s, 1]))
    note(f"(a2) {nuclide} MT {mt} at {E:.6g} eV: max |CDF(mu) - c| over 5000 "
         f"replayed draws = {worst:.2e} (bound 1e-9)")
    assert worst <= 1e-9
