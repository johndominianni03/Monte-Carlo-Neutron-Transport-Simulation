"""Phase 2b Part 2: free-gas target motion and per-material temperature.

Seeds were fixed in docs/phase2b_plan.md before any of these tests existed
(project conventions: no seed-shopping). Acceptance: 3 SE for means; p >= 0.0027 for
chi-square tests; exact = bit identity or a stated round-off bound.

Physics under test (OpenMC v0.16.0 src/physics.cpp sample_target_velocity,
sample_cxs_target_velocity, elastic_scatter): below free_gas_threshold * kT
(400 kT), or for awr <= 1 at every energy, elastic scattering sees a target
drawn from a Maxwellian gas at the nuclide's data temperature, under the
constant-cross-section (cxs) model; above it the target is at rest.

Derivations used by the expected values (units: E in eV, eps = E/kT, the
neutron mass is 1 and the target mass A; constant free-atom cross section
sigma_f; isotropic scattering in the CM frame):

1. Effective cross section. A neutron of speed v meets atoms of velocity V
   with relative speed g = |v - V|, so the collision rate per atom is
   sigma_f <g>, and sigma_eff(E) = sigma_f <g> / v. Averaging |v - V| over
   the Maxwellian (integrate over V in spherical coordinates about v) gives
       sigma_eff / sigma_f = (1 + 1/(2 a^2)) erf(a) + exp(-a^2) / (a sqrt(pi)),
       a^2 = A E / kT.
   It tends to 1 for E >> kT/A and to 2 / (a sqrt(pi)) (the 1/v law) for
   E -> 0. The test also checks it against the numerical integral of (3b).

2. Stationary collision density. A gas in equilibrium satisfies detailed
   balance: phi_M(E) sigma(E->E') = phi_M(E') sigma(E'->E), where
   phi_M ~ E exp(-E/kT) is the Maxwellian flux and
   sigma(E->E') = sigma_eff(E) P(E->E'), with P the distribution of E' per
   collision, which is what the sampler draws. Integrating over E,
       integral pi(E) P(E->E') dE = pi(E'),
       pi(E) ~ sigma_eff(E) E exp(-E/kT).
   So if E_in follows pi, E_out after one collision follows pi again. pi is
   not E exp(-E/kT): that is the flux pi / sigma_eff. The flux of an
   infinite medium is Maxwellian only if the tabulated cross section is
   sigma_eff (the 294 K data are Doppler-broadened, so they are; a
   constant tabulated cross section is not). The mean of pi: write
   v = G + (A/(A+1)) g and V = G - g/(A+1), with G the CM velocity and
   g = v - V. For two Maxwellian gases, G (mass A+1) and g (reduced mass
   m = A/(A+1)) are independent Maxwellians. A collision weights a pair by
   |g|, so |g| has density ~ g^3 exp(-m g^2 / (2 kT)), with
   <g^2> = 4 kT / m, while G stays Maxwellian, <G^2 / 2> = (3/2) kT / (A+1).
   The cross term averages to zero, so
       <E>_pi = (3/2) kT / (A+1) + (1/2) (A/(A+1))^2 4 kT (A+1)/A
              = kT (2 - 1 / (2 (A+1))):  1.75 kT (A = 1), 1.9615 kT (A = 12).

3. Single-collision kernel at a fixed E (what P5 tests).
   (a) Hydrogen (A = 1), closed form (E. P. Wigner and J. E. Wilkins,
       USAEC report AECD-2275, 1944; also in M. M. R. Williams, The Slowing
       Down and Thermalization of Neutrons, North-Holland, 1966):
           P(eps' | eps) ~ erf(sqrt(eps'))                       eps' <  eps
                           exp(eps - eps') erf(sqrt(eps))         eps' >  eps
       normalised by Z = (eps + 1/2) erf(sqrt eps) + sqrt(eps/pi) exp(-eps)
       (Z / eps = sigma_eff / sigma_f, consistent with 1). Derivation: for
       equal masses, v' = G + (g/2) Omega with Omega isotropic. By
       Archimedes' hat-box theorem, |v'|^2 is uniform between
       (|G| - g/2)^2 and (|G| + g/2)^2, an interval of length 2 |G| g, so the
       relative-speed weight g cancels and the rate density in eps' is
       ~ integral d3V exp(-V^2) [eps' in the interval] / |G|. Change to G
       and write it in two-centre coordinates s = |G|, t = |u - G|
       (d3G = 2 pi s t ds dt / |u|, |2G - u|^2 = 2 s^2 + 2 t^2 - eps). The
       condition becomes |s - t| <= min(sqrt eps, sqrt eps') and
       s + t >= max(sqrt eps, sqrt eps'). With p = s + t, q = s - t the
       integral factorises into erf(min(...)) times exp(-max(eps, eps')).
       Dividing the rate by v gives the expression above. Its CDF is closed
       form:
           X <= eps: [X erf(sqrt X) - erf(sqrt X)/2 + sqrt(X/pi) exp(-X)] / Z
           X >  eps: [Z - erf(sqrt eps) exp(eps - X)] / Z
   (b) Any A, by direct numerical integration, independent of the sampler.
       For given (v, V), the outgoing neutron is v' = V_cm + r Omega with
       r = A |v - V| / (A+1) its CM speed. By the hat-box theorem, E' = |v'|^2
       (units v = sqrt(E)) is uniform on [(|V_cm| - r)^2, (|V_cm| + r)^2].
       So P(E' <= X | E) = < g h(X) > / < g >, averaged over the Maxwellian
       target (density ~ exp(-A V^2 / kT)), with h the fraction of that
       interval below X. The test integrates this by Gauss-Legendre
       quadrature over |V| and the cosine between V and v.

What the tests cover, and what they don't:
  free gas (seed 20261022): one collision through collision.elastic_collision
       (the kernel's dispatcher) for synthetic A = 1 and A = 12 at
       kT = 0.0253 eV, with E_in drawn from pi by inverse CDF. E_out is
       chi-squared against pi on 20 equiprobable bins, and its mean is
       checked against kT (2 - 1/(2(A+1))) at 3 SE. It tests detailed
       balance of the sampled kernel at the right temperature and mass,
       but not its shape at a given E.
  kernel shape (seed 20261025, approved P5, required): fixed E_in in
       {kT, 20 kT}, A in {1, 12}. E_out is chi-squared against (3a) for
       A = 1 and (3b) for A = 12. A kernel that left E unchanged, or used
       the wrong mass or temperature, would fail. Plus an exact check at
       A = 12, E = 1000 kT: the dispatcher takes the target-at-rest path,
       bit-identical to Part 1's elastic_scatter with the same draws.
  dispatch and temperature (seed 20261026), exact: the 400 kT boundary,
       A <= 1, kT = 0 and the free_gas switch; the kernel's free-gas
       counter on real H-1 and on a synthetic nuclide; OpenMC's NEAREST
       temperature selection and per-material packing on real data; and the
       Phase 2a driver refusing per-material temperatures.
  None of them is a transport-level thermal-spectrum test (that would need
  a synthetic nuclide whose tabulated elastic cross section is sigma_eff,
  and a way to end histories in a non-absorbing medium).
"""
import math

import numpy as np
import pytest
from numba import njit
from scipy import integrate, stats
from scipy.special import erf

from mcslab import collision as C
from mcslab import synthetic as S
from mcslab import tallies as T
from mcslab.rng import RNG_SIZE, STRIDE, init_history, prn

P_MIN = 0.0027
KT = 0.0253
THRESHOLD = 400.0
# Seeds fixed in docs/phase2b_plan.md before the tests were written.
SEED_FG = 20261022
SEED_SHAPE = 20261025
SEED_DISPATCH = 20261026
N_EVENTS = 200_000


def s_eff(eps, A):
    """sigma_eff / sigma_f of a constant free-atom cross section (1)."""
    a = math.sqrt(A * eps)
    return (1.0 + 0.5 / (a * a)) * erf(a) + math.exp(-a * a) / (a * math.sqrt(math.pi))


def mean_pi(A):
    """<E>_pi / kT (2)."""
    return 2.0 - 1.0 / (2.0 * (A + 1.0))


def chi2_check(report, quantity, counts, expected, max_draws=None):
    counts = np.asarray(counts, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert abs(counts.sum() - expected.sum()) < 1e-6 * counts.sum()
    stat = float(((counts - expected) ** 2 / expected).sum())
    dof = counts.size - 1
    p = float(stats.chi2.sf(stat, dof))
    report(f"{quantity} chi2 (p={p:.3f})", stat, dof, math.sqrt(2.0 * dof), max_draws)
    assert p >= P_MIN, f"{quantity}: chi2 = {stat:.1f} on {dof} dof, p = {p:.2e}"


def scatterer(A, kT=KT):
    nuc = S.nuclide(f"G{A:g}", awr=A, elastic_b=1.0, kT=kT)
    phys = C.pack_physics([nuc.name], {nuc.name: nuc})
    return phys, int(phys.ch_int[0, C.CH_PROD])


# ---------------------------------------------------------------------------
# free gas: pi is preserved by one collision
# ---------------------------------------------------------------------------
def pi_table(A, n=200_001, eps_max=60.0):
    """(t, CDF) of pi in t = sqrt(eps) (density ~ s_eff(t^2) t^3 exp(-t^2),
    smooth at 0), by cumulative Simpson integration; and a check of its
    accuracy against scipy quad."""
    t = np.linspace(0.0, math.sqrt(eps_max), n)
    f = np.zeros(n)
    f[1:] = [s_eff(x * x, A) * x ** 3 * math.exp(-x * x) for x in t[1:]]
    cdf = np.concatenate([[0.0], integrate.cumulative_simpson(f, x=t)])
    total = cdf[-1]
    cdf /= total

    def exact(x):
        return integrate.quad(lambda y: s_eff(y * y, A) * y ** 3 * math.exp(-y * y)
                              if y > 0 else 0.0, 0.0, x, epsabs=1e-14, epsrel=1e-12,
                              limit=200)[0] / total
    probe = np.linspace(0.05, math.sqrt(eps_max) - 0.05, 40)
    table_err = max(abs(np.interp(x, t, cdf) - exact(x)) for x in probe)
    return t, cdf, table_err


@njit(cache=True)
def _collide_from_pi(t, cdf, A, kT, ip, fp, prod, seed, n):
    """E_in = kT t(xi)^2 from the first draw (inverse CDF of pi), then one
    elastic collision through the kernel's dispatcher on the same stream."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    e_in = np.empty(n)
    e_out = np.empty(n)
    used = 0
    md = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        x = np.interp(prn(rng), cdf, t)
        E = kT * x * x
        E2, u, v, w, mu, fg = C.elastic_collision(E, 0.6, 0.8, 0.0, A, kT, 400.0, True,
                                                  ip, fp, prod, rng)
        e_in[s] = E
        e_out[s] = E2
        used += fg
        md = max(md, int(rng[1]))
    return e_in, e_out, used, md


@pytest.mark.parametrize("A", [1.0, 12.0])
def test_free_gas_preserves_collision_density(report, note, A):
    t, cdf, table_err = pi_table(A)
    # (2): the closed-form mean of pi against direct integration
    num = integrate.quad(lambda e: e * s_eff(e, A) * e * math.exp(-e), 1e-12, 80.0,
                         epsabs=1e-14, limit=200)[0]
    den = integrate.quad(lambda e: s_eff(e, A) * e * math.exp(-e), 1e-12, 80.0,
                         epsabs=1e-14, limit=200)[0]
    note(f"(free gas) A={A:g}: <E>_pi = {num / den:.10f} kT by quadrature, "
         f"{mean_pi(A):.10f} kT closed form; pi CDF table error {table_err:.1e} "
         "(bound 1e-9)")
    assert abs(num / den - mean_pi(A)) < 1e-9
    assert table_err < 1e-9
    phys, prod = scatterer(A)
    e_in, e_out, used, md = _collide_from_pi(t, cdf, A, KT, phys.ip, phys.fp, prod,
                                             np.uint64(SEED_FG), N_EVENTS)
    assert used == N_EVENTS                     # every E_in < 60 kT < 400 kT
    assert md < STRIDE
    k = 20
    edges = KT * np.interp(np.linspace(0.0, 1.0, k + 1), cdf, t) ** 2
    edges[-1] = np.inf
    counts = np.histogram(e_out, bins=edges)[0]
    chi2_check(report, f"A={A:g} E_out ~ pi", counts, np.full(k, N_EVENTS / k), md)
    mean = float(e_out.mean() / KT)
    se = float(e_out.std(ddof=1) / KT / math.sqrt(N_EVENTS))
    report(f"A={A:g} mean E_out / kT", mean, mean_pi(A), se)
    assert abs(mean - mean_pi(A)) <= 3.0 * se, (mean, mean_pi(A), se)


# ---------------------------------------------------------------------------
# kernel shape at fixed incident energies (P5, required)
# ---------------------------------------------------------------------------
def hydrogen_cdf(X, eps):
    """(3a): P(eps' <= X | eps) for A = 1, closed form."""
    Z = (eps + 0.5) * erf(math.sqrt(eps)) + math.sqrt(eps / math.pi) * math.exp(-eps)
    if X <= eps:
        return (X * erf(math.sqrt(X)) - 0.5 * erf(math.sqrt(X))
                + math.sqrt(X / math.pi) * math.exp(-X)) / Z
    return (Z - erf(math.sqrt(eps)) * math.exp(eps - X)) / Z


def integrated_cdf(Xs, eps, A, n=1200, y_max=6.0):
    """(3b): P(eps' <= X | eps) by Gauss-Legendre quadrature over the
    Maxwellian target (|V| = y sqrt(kT/A), y in [0, y_max]) and the cosine
    between V and v, in units kT = 1, v = sqrt(eps). Also returns
    <g> / v (= sigma_eff / sigma_f). Nothing here uses the sampler."""
    gy, wy = np.polynomial.legendre.leggauss(n)
    y = 0.5 * y_max * (gy + 1.0)
    wy = 0.5 * y_max * wy
    gm, wm = np.polynomial.legendre.leggauss(n)
    V = (y / math.sqrt(A))[:, None]
    mu = gm[None, :]
    W = (wy * y * y * np.exp(-y * y))[:, None] * wm[None, :]
    sv = math.sqrt(eps)
    g = np.sqrt(np.maximum(eps + V * V - 2.0 * sv * V * mu, 0.0))
    vcm = np.sqrt(np.maximum(eps + 2.0 * A * sv * V * mu + A * A * V * V, 0.0)) / (A + 1.0)
    r = A * g / (A + 1.0)
    lo = (vcm - r) ** 2
    width = (vcm + r) ** 2 - lo
    norm = float((W * g).sum())
    out = np.array([float((W * g * np.clip((X - lo) / np.where(width > 0.0, width, 1.0),
                                           0.0, 1.0)).sum()) / norm for X in Xs])
    return out, norm / float(W.sum()) / sv


def bisect(cdf, lo, hi, target):
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _collide_fixed(E, A, kT, ip, fp, prod, seed, n):
    rng = np.zeros(RNG_SIZE, np.uint64)
    e_out = np.empty(n)
    used = 0
    md = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        E2, u, v, w, mu, fg = C.elastic_collision(E, 0.6, 0.8, 0.0, A, kT, 400.0, True,
                                                  ip, fp, prod, rng)
        e_out[s] = E2
        used += fg
        md = max(md, int(rng[1]))
    return e_out, used, md


SHAPE_CASES = [(1.0, 1.0), (1.0, 20.0), (12.0, 1.0), (12.0, 20.0)]


@pytest.mark.parametrize("A,eps", SHAPE_CASES)
def test_free_gas_kernel_shape(report, note, A, eps):
    k = 20
    if A == 1.0:
        cdf = lambda X: hydrogen_cdf(X, eps)          # noqa: E731
        hi = eps + 60.0
        # the independent integrator must reproduce the closed form
        probe = np.linspace(0.05 * eps, eps + 8.0, 25)
        F_int, _ = integrated_cdf(probe, eps, 1.0)
        dev = float(np.max(np.abs(F_int - [hydrogen_cdf(X, eps) for X in probe])))
        note(f"(kernel shape) A=1 eps={eps:g}: max |integrated - closed-form CDF| = "
             f"{dev:.1e} (bound 1e-6)")
        assert dev <= 1e-6
        edges = [0.0] + [bisect(cdf, 0.0, hi, j / k) for j in range(1, k)] + [np.inf]
    else:
        hi = 4.0 * eps + 40.0
        grid = np.linspace(0.0, hi, 401)
        F_grid, ratio = integrated_cdf(grid, eps, A)
        F_half, _ = integrated_cdf(grid[::4], eps, A, n=600)
        refine = float(np.max(np.abs(F_grid[::4] - F_half)))
        note(f"(kernel shape) A={A:g} eps={eps:g}: quadrature refinement 600 -> 1200 "
             f"changes the CDF by {refine:.1e} (bound 1e-6); <g>/v = {ratio:.10f} vs "
             f"sigma_eff/sigma_f = {s_eff(eps, A):.10f}")
        assert refine <= 1e-6
        assert abs(ratio - s_eff(eps, A)) <= 1e-8
        assert abs(F_grid[-1] - 1.0) <= 1e-12
        cdf = lambda X: float(np.interp(X, grid, F_grid))   # noqa: E731
        inner = [bisect(cdf, 0.0, hi, j / k) for j in range(1, k)]
        # bin edges from the interpolated CDF, probabilities from the
        # integrator itself at those edges
        F_edges, _ = integrated_cdf(inner, eps, A)
        edges = [0.0] + inner + [np.inf]
    probs = np.diff([0.0] + [float(v) for v in (np.array([cdf(e) for e in edges[1:-1]])
                                                if A == 1.0 else F_edges)] + [1.0])
    phys, prod = scatterer(A)
    e_out, used, md = _collide_fixed(eps * KT, A, KT, phys.ip, phys.fp, prod,
                                     np.uint64(SEED_SHAPE), N_EVENTS)
    assert used == N_EVENTS
    counts = np.histogram(e_out / KT, bins=np.array(edges))[0]
    chi2_check(report, f"A={A:g} E_out at E_in={eps:g} kT", counts, N_EVENTS * probs, md)


@njit(cache=True)
def _at_rest_pair(E, A, kT, ip, fp, prod, seed, n):
    """The dispatcher and Part 1's elastic_scatter on the same streams:
    worst |difference| (must be exactly 0), draws, free-gas uses."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    diff = 0.0
    used = 0
    draws_a = 0
    draws_b = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        a0, a1, a2, a3, a4, fg = C.elastic_collision(E, 0.6, 0.8, 0.0, A, kT, 400.0, True,
                                                     ip, fp, prod, rng)
        draws_a = max(draws_a, int(rng[1]))
        init_history(rng, seed, np.uint64(s))
        b0, b1, b2, b3, b4 = C.elastic_scatter(E, 0.6, 0.8, 0.0, A, ip, fp, prod, rng)
        draws_b = max(draws_b, int(rng[1]))
        diff = max(diff, abs(a0 - b0), abs(a1 - b1), abs(a2 - b2), abs(a3 - b3),
                   abs(a4 - b4))
        used += fg
    return diff, used, draws_a, draws_b


def test_free_gas_at_rest_above_threshold(note):
    """Exact: A = 12 at E = 1000 kT (> 400 kT) takes the target-at-rest path,
    bit for bit and draw for draw."""
    phys, prod = scatterer(12.0)
    diff, used, da, db = _at_rest_pair(1000.0 * KT, 12.0, KT, phys.ip, phys.fp, prod,
                                       np.uint64(SEED_SHAPE), 20_000)
    note(f"(kernel shape) A=12, E=1000 kT: dispatcher vs Part 1 at-rest elastic over "
         f"20000 events: max |difference| = {diff:g}, free-gas uses {used}, draws "
         f"{da} vs {db}")
    assert diff == 0.0 and used == 0 and da == db == 2


# ---------------------------------------------------------------------------
# dispatch rule, kernel counter, temperature selection (exact)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _one(E, A, kT, threshold, free_gas, ip, fp, prod, seed):
    rng = np.zeros(RNG_SIZE, np.uint64)
    init_history(rng, seed, np.uint64(0))
    r = C.elastic_collision(E, 0.6, 0.8, 0.0, A, kT, threshold, free_gas, ip, fp, prod, rng)
    return r[5], int(rng[1])


def test_dispatch_rule():
    seed = np.uint64(SEED_DISPATCH)
    phys12, p12 = scatterer(12.0)
    e400 = THRESHOLD * KT
    below = float(np.nextafter(e400, 0.0))
    assert _one(e400, 12.0, KT, THRESHOLD, True, phys12.ip, phys12.fp, p12, seed) == (0, 2)
    fg, draws = _one(below, 12.0, KT, THRESHOLD, True, phys12.ip, phys12.fp, p12, seed)
    assert fg == 1 and draws >= 2 + 5 + 1
    assert _one(below, 12.0, 0.0, THRESHOLD, True, phys12.ip, phys12.fp, p12, seed) == (0, 2)
    assert _one(below, 12.0, KT, THRESHOLD, False, phys12.ip, phys12.fp, p12, seed) == (0, 2)
    phys1, p1 = scatterer(1.0)
    assert _one(14.0e6, 1.0, KT, THRESHOLD, True, phys1.ip, phys1.fp, p1, seed)[0] == 1
    assert C.uses_free_gas(14.0e6, 0.99917, KT, THRESHOLD, True)       # H-1's awr
    assert not C.uses_free_gas(14.0e6, 1.0000001, KT, THRESHOLD, True)


def _library():
    from mcslab.nucdata import Library
    try:
        return Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def test_kernel_free_gas_counter():
    """Exact: through the kernel, every elastic event of pure H-1 uses free
    gas; a synthetic A = 12 nuclide never does when every collision is at
    or above 400 kT (energy cutoff = 400 kT), and does below it."""
    import warnings
    from mcslab.ce_materials import CEMaterial
    from mcslab.config_ce import MonoEnergetic
    from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource
    lib = _library()
    h = CEMaterial("H", 0.0708, (("H1", 1.0),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(KinRunConfig(SlabGeometry([0.0, 5.0], [h]), BeamSource(),
                                   MonoEnergetic(1.0e6), lib, n_batches=2,
                                   histories_per_batch=200, seed=SEED_DISPATCH))
        assert res.count(T.K_ELASTIC) > 0
        assert res.count(T.K_FREE_GAS) == res.count(T.K_ELASTIC)
        x = S.nuclide("X12", awr=12.0, elastic_b=4.0, capture_b=0.01, kT=KT)
        mat = S.SyntheticMaterial("x", (("X12", 0.1),))
        slib = S.SyntheticLibrary({"X12": x})
        counts = []
        for e_cut in (THRESHOLD * KT, 1.0):
            r = run_kin(KinRunConfig(SlabGeometry([0.0, 50.0], [mat]), BeamSource(),
                                     MonoEnergetic(1.0e3), slib, n_batches=2,
                                     histories_per_batch=200, seed=SEED_DISPATCH,
                                     energy_cutoff=e_cut))
            assert r.balance()["residual"] == 0.0
            counts.append((r.count(T.K_FREE_GAS), r.count(T.K_ELASTIC)))
    assert counts[0][0] == 0 and counts[0][1] > 0
    assert counts[1][0] > 0


def test_temperature_selection_nearest_rule():
    """Exact: OpenMC's NEAREST rule with a 10 K tolerance on the real kTs."""
    from mcslab.nucdata import nearest_temperature
    kts = _library().kts("Li7")
    assert set(kts) == {"250K", "294K", "600K", "900K", "1200K", "2500K"}
    assert nearest_temperature(kts, 293.6) == "294K"
    assert nearest_temperature(kts, 900.0) == "900K"
    assert nearest_temperature(kts, 909.99) == "900K"
    assert nearest_temperature(kts, 2500.0) == "2500K"
    for bad in (910.0, 950.0, 973.0, 100.0):
        with pytest.raises(ValueError, match="no data at or near"):
            nearest_temperature(kts, bad)


def test_per_material_temperature_packing():
    """Exact: the same nuclide in materials at two temperatures is packed
    twice, each entry with the grid and kT of its own data group, read
    back against h5py."""
    import dataclasses
    import h5py
    from mcslab import ce_materials as cm
    from mcslab.config_ce import MonoEnergetic
    from mcslab.config_kin import KinRunConfig, pack_problem
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource
    lib = _library()
    hot = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    cold = cm.CEMaterial("Li7 cold", 0.534, (("Li7", 1.0),))          # 293.6 K default
    cfg = KinRunConfig(SlabGeometry([0.0, 1.0, 2.0], [hot, cold]), BeamSource(),
                       MonoEnergetic(1.0e6), lib, n_batches=2, histories_per_batch=10)
    _, _, packed, phys, nucs = pack_problem(cfg)
    names = list(packed.nuclide_names)
    assert "Li7@900K" in names and "Li7@294K" in names and "Be9" in names
    with h5py.File(lib.nuclide_path("Li7"), "r") as f:
        for label, group in (("Li7@900K", "900K"), ("Li7@294K", "294K")):
            k = names.index(label)
            grid = packed.egrid[packed.e_off[k]:packed.e_off[k + 1]]
            assert np.array_equal(grid, f["Li7/energy"][group][()])
            assert phys.nuc_kT[k] == float(f["Li7/kTs"][group][()])
    with h5py.File(lib.nuclide_path("Be9"), "r") as f:
        assert phys.nuc_kT[names.index("Be9")] == float(f["Be9/kTs/900K"][()])


def test_phase2a_driver_refuses_other_temperatures():
    import dataclasses
    from mcslab import ce_materials as cm
    from mcslab.config_ce import CERunConfig, MonoEnergetic, pack_problem
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource
    lib = _library()
    hot = dataclasses.replace(cm.iron(), temperature=900.0)
    cfg = CERunConfig(SlabGeometry([0.0, 1.0], [hot]), BeamSource(), MonoEnergetic(1.0e6),
                      lib, n_batches=2, histories_per_batch=10)
    with pytest.raises(ValueError, match="Phase 2a runs use one temperature"):
        pack_problem(cfg)
    same = dataclasses.replace(cm.iron(), temperature=293.6)
    pack_problem(CERunConfig(SlabGeometry([0.0, 1.0], [same]), BeamSource(),
                             MonoEnergetic(1.0e6), lib, n_batches=2, histories_per_batch=10))
