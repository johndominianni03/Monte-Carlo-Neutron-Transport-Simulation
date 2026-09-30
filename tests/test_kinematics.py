"""Phase 2b Part 1 validation: collision kinematics and the kinematic kernel.

Seeds were fixed in docs/phase2b_plan.md before any of these tests existed
(project conventions: no seed-shopping). Acceptance: 3 SE for means; p >= 0.0027 for
chi-square and Hotelling tests (the false-alarm rate of a two-sided 3-sigma
check); "exact" = integer identities on raw sums (all weights are 1.0) or
round-off with a stated bound.

What each test covers, and what it doesn't:
  (a)  test_a_*: collision.elastic_scatter (target at rest) for a synthetic
       nuclide with isotropic CM scattering, A in {1, 12, 184}: E' uniform on
       [alpha E, E] (chi-square), mean ln(E/E') = xi (3 SE), and every E'
       inside [alpha E, E] to round-off. It covers the elastic kinematics,
       the CM transform and rotate_angle. It does not cover anisotropic data
       or target motion.
  (b)  test_b_*: the full kernel in an infinite, non-absorbing medium of a
       synthetic constant-cross-section scatterer (the slab is so thick
       that leakage is asserted to be exactly 0, so the histories are
       infinite-medium histories). Per source neutron, both flux estimators
       integrated over space and an energy bin must equal
       ln(E2/E1) / (xi Sigma_s), i.e. phi(E) = 1 / (xi Sigma_s E). One
       Hotelling T^2 test per (A, estimator) over 10 log bins uses the
       batch covariance, so correlations between bins are accounted for.
       For A = 1 this is exact below the source energy. For A = 12 the
       window starts 8 collision intervals below the source, and the
       Placzek transient there is bounded by a deterministic solve in the
       test. It does not cover absorption, leakage or anisotropy.
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

from mcslab import collision as C
from mcslab import distributions as D
from mcslab import synthetic as S
from mcslab import tallies as T
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.nucdata import AngleDistribution, Library, Tabular
from mcslab.rng import RNG_SIZE, STRIDE, init_history, prn
from mcslab.sources import IsotropicPlaneSource

P_MIN = 0.0027          # chi-square / Hotelling acceptance (3-sigma equivalent)
N_SIGMA = 3.0

# Seeds fixed in docs/phase2b_plan.md before the tests were written.
SEED_A = 20261010
SEED_A2 = 20261011
SEED_B = 20261012


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


def hotelling_check(report, quantity, batches, expected, max_draws=None, cutoff=None):
    """Hotelling T^2 test that the mean of the per-batch vectors equals
    `expected`. It uses the sample covariance of the batch estimates, so
    correlated components (e.g. energy bins filled by the same histories)
    are handled correctly. F = (B - k) / (k (B - 1)) T^2 ~ F(k, B - k)."""
    X = np.asarray(batches, dtype=np.float64)
    B, k = X.shape
    d = X.mean(axis=0) - np.asarray(expected, dtype=np.float64)
    S_cov = np.cov(X, rowvar=False, ddof=1)
    t2 = float(B * d @ np.linalg.solve(S_cov, d))
    F = (B - k) / (k * (B - 1)) * t2
    p = float(stats.f.sf(F, k, B - k))
    report(f"{quantity} T2 (p={p:.3f})", t2, k, math.sqrt(2.0 * k), max_draws, cutoff)
    assert p >= P_MIN, f"{quantity}: Hotelling T2 = {t2:.2f} (k = {k}), p = {p:.2e}"
    return p


def mean_check(report, quantity, samples, expected, max_draws=None):
    """3-SE check of the mean of i.i.d. samples."""
    x = np.asarray(samples, dtype=np.float64)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    report(quantity, mean, expected, se, max_draws)
    assert abs(mean - expected) <= N_SIGMA * se, (
        f"{quantity}: {mean:.7g} vs {expected:.7g} ({(mean - expected) / se:+.2f} SE)")


def alpha_xi(A):
    """alpha = ((A-1)/(A+1))^2 and xi = 1 + alpha ln(alpha) / (1 - alpha)
    (xi = 1 for A = 1, the limit alpha ln alpha -> 0)."""
    a = ((A - 1.0) / (A + 1.0)) ** 2
    xi = 1.0 if a == 0.0 else 1.0 + a * math.log(a) / (1.0 - a)
    return a, xi


def synthetic_physics(nuclide):
    """PackedPhysics for one synthetic nuclide."""
    return C.pack_physics([nuclide.name], {nuclide.name: nuclide})


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


# ---------------------------------------------------------------------------
# (a) isotropic-CM elastic scattering off a target at rest
# ---------------------------------------------------------------------------
@njit(cache=True)
def _elastic_energies(ip, fp, prod, awr, E, seed, n):
    """n outgoing energies of collision.elastic_scatter; sample s uses the
    stream of history s. The incoming direction is arbitrary (the result
    does not depend on it)."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    out = np.empty(n)
    first_draw = np.empty(n)
    max_draws = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        E2, u, v, w, mu = C.elastic_scatter(E, 0.6, 0.8, 0.0, awr, ip, fp, prod, rng)
        out[s] = E2
        if rng[1] > max_draws:
            max_draws = rng[1]
        init_history(rng, seed, np.uint64(s))
        first_draw[s] = prn(rng)          # the draw that set mu_cm
    return out, first_draw, max_draws


@pytest.mark.parametrize("A", [1.0, 12.0, 184.0])
def test_a_elastic_isotropic_cm(report, note, A):
    E, n, k = 2.0e6, 200_000, 50
    alpha, xi = alpha_xi(A)
    nuc = S.nuclide(f"A{A:g}", awr=A, elastic_b=1.0)
    phys = synthetic_physics(nuc)
    prod = int(phys.ch_int[0, C.CH_PROD])
    e_out, xi1, draws = _elastic_energies(phys.ip, phys.fp, prod, A, E,
                                          np.uint64(SEED_A), n)
    draws = int(draws)

    # exact (round-off): kinematically allowed range, and per event the
    # two-body relation E' = E ((1 + alpha) + (1 - alpha) mu_cm) / 2 with
    # mu_cm = -1 + 2 xi_1 replayed from the sample's first draw
    assert e_out.min() >= alpha * E * (1.0 - 1e-12), (e_out.min(), alpha * E)
    assert e_out.max() <= E * (1.0 + 1e-12), (e_out.max(), E)
    mu_cm = -1.0 + 2.0 * xi1
    rel = np.abs(e_out - E * ((1.0 + alpha) + (1.0 - alpha) * mu_cm) / 2.0) / E
    assert rel.max() <= 1e-12, rel.max()
    note(f"(a) A={A:g}: max |E' - E((1+a)+(1-a)mu_cm)/2| / E over {n} events = "
         f"{rel.max():.1e} (bound 1e-12)")
    edges = np.linspace(alpha * E, E, k + 1)
    counts = np.histogram(np.clip(e_out, alpha * E, E), bins=edges)[0]
    chi2_check(report, f"A={A:g} E' uniform on [aE, E]", counts, np.full(k, n / k), draws)
    mean_check(report, f"A={A:g} mean ln(E/E') = xi", np.log(E / e_out), xi)


# ---------------------------------------------------------------------------
# (b) slowing down in an infinite non-absorbing medium
# ---------------------------------------------------------------------------
def placzek_deviation(A, n_intervals, steps_per_interval=40_000):
    """max |xi F(u) - 1| on each lethargy interval [j q, (j+1) q), q = ln(1/alpha),
    for the collision density F(u) per unit lethargy below a unit source at
    u = 0 (the source collision itself excluded), isotropic CM scattering,
    no absorption.

    With G(u) = e^u F(u):
      first interval: F = exp(u alpha/(1-alpha)) / (1-alpha)   (analytic)
      at u = q the first-flight term stops: G(q+) = G(q-) - 1/(1-alpha)
      beyond:  G'(u) = (G(u) - G(u - q)) / (1 - alpha)          (delay ODE)
    The delay ODE is integrated with the trapezoid rule (second order; the
    delayed values sit exactly on the previous interval's mesh). Halving the
    step changes the result by < 1e-9 beyond 8 intervals for A = 12."""
    from scipy.signal import lfilter
    alpha, xi = alpha_xi(A)
    q = -math.log(alpha)
    h = q / steps_per_interval
    s = 1.0 / (1.0 - alpha)
    u0 = np.arange(steps_per_interval + 1) * h
    G_prev = np.exp(u0 * s) * s
    out = [float(np.max(np.abs(xi * np.exp(-u0) * G_prev - 1.0)))]
    G_start = G_prev[-1] - s
    c = 0.5 * h * s
    c1 = (1.0 + c) / (1.0 - c)
    for j in range(1, n_intervals):
        b = -c * (G_prev[1:] + G_prev[:-1]) / (1.0 - c)
        rest = lfilter([1.0], [1.0, -c1], b, zi=np.array([c1 * G_start]))[0]
        G = np.concatenate([[G_start], rest])
        out.append(float(np.max(np.abs(xi * np.exp(-(j * q + u0)) * G - 1.0))))
        G_prev, G_start = G, G[-1]
    return out


B_CASES = {
    # A: (energy cutoff eV, window lower edge, window upper edge)
    1.0: (1.0, 1.0, 1.0e6),                       # exact for all E < E0
    12.0: (1.0e3, 1.0e3, 1.0e6 * alpha_xi(12.0)[0] ** 8),
}


@pytest.mark.parametrize("A", sorted(B_CASES))
def test_b_infinite_medium_slowing_down(report, note, A):
    E0, sigma_s, half_width = 1.0e6, 1.0, 1.0e4
    e_cut, lo, hi = B_CASES[A]
    alpha, xi = alpha_xi(A)
    if A > 1.0:
        dev = placzek_deviation(A, 10)
        bound = max(dev[8:])
        note(f"(b) A={A:g}: Placzek transient max |xi F - 1| from 8 intervals below "
             f"the source on: {bound:.1e} (bound asserted: 1e-6)")
        assert bound < 1e-6
    nuc = S.nuclide(f"B{A:g}", awr=A, elastic_b=sigma_s)
    lib = S.SyntheticLibrary({nuc.name: nuc})
    mat = S.SyntheticMaterial("scatterer", ((nuc.name, 1.0),))    # Sigma_s = 1/cm
    edges = np.exp(np.linspace(math.log(lo), math.log(hi), 11))
    cfg = KinRunConfig(SlabGeometry([-half_width, half_width], [mat]),
                       IsotropicPlaneSource(0.0), MonoEnergetic(E0), lib,
                       n_batches=100, histories_per_batch=2000, seed=SEED_B,
                       energy_cutoff=e_cut, energy_edges=tuple(edges))
    with pytest.warns(EnergyCutoffWarning):
        res = run_kin(cfg)
    n_hist = cfg.n_histories
    cutoff_per_src = float(res.cutoff_weight.sum()) / n_hist

    # exact: nothing absorbed, nothing leaks, nothing multiplies, and every
    # history ends at the energy cutoff
    assert res.lost == 0 and res.max_draws < STRIDE
    assert res.count(T.K_ABSORBED) == 0 and res.count(T.K_CREATED) == 0
    assert res.count(T.K_LEAK_LEFT) == 0 and res.count(T.K_LEAK_RIGHT) == 0
    assert res.count(T.K_CUTOFF) == n_hist and res.cutoff_weight.sum() == float(n_hist)
    assert res.balance()["residual"] == 0.0

    expected = np.log(edges[1:] / edges[:-1]) / (xi * sigma_s)
    for est, name in ((T.SPEC_COLL, "collision"), (T.SPEC_TL, "track-length")):
        hotelling_check(report, f"A={A:g} {name} phi in 10 bins",
                        res.spectrum_batches(est)[:, 0, :], expected,
                        res.max_draws, cutoff_per_src)
