"""Phase 2a validation: nuclear data reader, lookup, materials, CE kernel.

Requires the ENDF/B-VIII.0 HDF5 files (scripts/fetch_data.sh) with
MCSLAB_DATA pointing at the library root. Without it the module skips.

What each test covers, and what it doesn't:
  (a) test_a_*: lookups at grid energies return the stored values bit for
      bit, for every reaction of every nuclide. This covers binary search and
      interpolation at f = 0. It does not cover the accuracy of the data.
  (b) test_b_*: grids strictly increasing; every reaction exactly 0 below its
      threshold index (by lookup) and at every grid energy below the
      kinematic threshold -Q(A+1)/A (non-redundant reactions with Q < 0).
  (c) test_c_*: MT 1 is NOT stored in these files, so "total = sum of
      partials" cannot be checked against an evaluated total here. What is
      checked:
      - no reaction is counted twice in the total
      - every stored redundant sum (MT 4, 103-107, and Li-7 MT 205) matches
        its components within ENDF rounding
      - the kernel's pre-summed total matches the per-reaction lookups at
        off-grid energies
      The evaluated total is checked by hand against NNDC (README).
  (d) test_d_*: 14.1 MeV beam through natural iron, first collision ends the
      history. Transmission = exp(-Sigma_t L) within 3 SE, with Sigma_t
      computed here from the raw HDF5 files via h5py + np.interp and an
      inline number-density formula. It does not go through mcslab.nucdata,
      mcslab.xs or mcslab.ce_materials.
  Plus: the data files match the pinned checksums, and the material number
  densities agree with IUPAC standard atomic weights.

The seed for (d) was fixed before the first run (docs/development.md: no seed-shopping).
"""
import math
import os

import h5py
import numpy as np
import pytest

from mcslab import ce_materials as cm
from mcslab import tallies as T
from mcslab import xs
from mcslab.nucdata import Library, file_sha256
from mcslab.rng import STRIDE

try:
    LIB = Library.open()
except FileNotFoundError as exc:                       # pragma: no cover
    pytest.skip(f"nuclear data not available ({exc})", allow_module_level=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUCLIDES = ["H1", "Fe54", "Fe56", "Fe57", "Fe58", "W180", "W182", "W183", "W184",
            "W186", "Li6", "Li7", "Be9", "F19"]
E14 = 14.1e6


@pytest.fixture(scope="module")
def nucs():
    return {n: LIB.load(n) for n in NUCLIDES}


# ---------------------------------------------------------------------------
# data identity
# ---------------------------------------------------------------------------
def test_data_files_match_pinned_checksums():
    pinned = os.path.join(REPO, "scripts", "checksums", "endfb-viii.0.sha256")
    with open(pinned) as fh:
        entries = [line.split() for line in fh if line.strip()]
    for digest, rel in entries:
        assert file_sha256(LIB.root / rel) == digest, f"{rel} differs from pinned sha256"


# ---------------------------------------------------------------------------
# (a) on-grid lookups are exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NUCLIDES)
def test_a_on_grid_lookup_exact(nucs, name):
    nu = nucs[name]
    E = nu.energy
    for mt, rx in nu.reactions.items():
        got = xs.lookup(nu, mt, E[rx.threshold_idx:])
        assert np.array_equal(got, rx.xs), f"{name} MT {mt}: on-grid lookup != stored"
        # explicit end points (the last point takes a separate branch)
        assert xs.lookup(nu, mt, float(E[-1])) == rx.xs[-1]
        assert xs.lookup(nu, mt, float(E[rx.threshold_idx])) == rx.xs[0]
    tot = nu.total_on_grid()
    assert np.array_equal(xs.lookup_total(nu, E), tot)


def test_a_off_grid_energies_refused(nucs):
    nu = nucs["Fe56"]
    for bad in (np.nextafter(nu.energy[0], 0.0), np.nextafter(nu.energy[-1], np.inf),
                float("nan")):
        with pytest.raises(ValueError):
            xs.lookup(nu, 2, bad)


# ---------------------------------------------------------------------------
# (b) grids and thresholds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NUCLIDES)
def test_b_grid_strictly_increasing(nucs, name):
    d = np.diff(nucs[name].energy)
    bad = np.flatnonzero(d <= 0.0)
    assert bad.size == 0, f"{name}: grid not strictly increasing at indices {bad[:10]}"


@pytest.mark.parametrize("name", NUCLIDES)
def test_b_zero_below_threshold(nucs, name):
    nu = nucs[name]
    E = nu.energy
    for mt, rx in nu.reactions.items():
        thr = rx.threshold_idx
        if thr > 0:
            below = xs.lookup(nu, mt, E[:thr])
            assert not below.any(), f"{name} MT {mt}: nonzero on grid below threshold"
            assert xs.lookup(nu, mt, float(np.nextafter(E[thr], 0.0))) == 0.0
            # the first tabulated value is 0, so the threshold rule and lin-lin
            # interpolation agree (the kernel relies on this when it
            # interpolates the pre-summed total)
            assert rx.xs[0] == 0.0, f"{name} MT {mt}: xs at threshold = {rx.xs[0]}"
        assert (rx.xs >= 0.0).all(), f"{name} MT {mt}: negative cross section"
        if not rx.redundant and rx.q_value < 0.0:
            e_kin = -rx.q_value * (nu.awr + 1.0) / nu.awr
            full = nu.summed_on_grid([mt])
            assert not full[E < e_kin].any(), (
                f"{name} MT {mt}: nonzero below kinematic threshold {e_kin:.6g} eV")


# ---------------------------------------------------------------------------
# (c) total vs partials
# ---------------------------------------------------------------------------
# Summation MTs and their components (ENDF-6 manual, ENDF-102, Appendix B).
SUMS = {4: range(51, 92), 16: range(875, 892), 103: range(600, 650),
        104: range(650, 700), 105: range(700, 750), 106: range(750, 800),
        107: range(800, 850)}
NEVER_PARTIAL = {1, 3, 101, 203, 204, 205, 206, 207, 301, 444, 901}
# The observed remainders after the rounding bound are at most 1.1e-10 b,
# in sub-threshold tails (see docs/data_inventory.md and README).
ABS_FLOOR_B = 1e-9


def half_quantum(x):
    """Half the last-digit quantum of x when written in the ENDF 11-column
    format: 7 significant digits for a one-digit exponent (1e-9 <= |x| <
    1e10), 6 for a two-digit exponent. NJOY passes cross sections between
    modules on ENDF-format (PENDF) tapes, so every stored value carries this
    rounding."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    nz = x > 0
    e = np.floor(np.log10(x[nz]))
    digits = np.where((e >= -9) & (e < 10), 7, 6)
    out[nz] = 0.5 * 10.0 ** (e - (digits - 1))
    return out


def test_c_mt1_not_stored_and_no_double_counting(nucs, note):
    for name, nu in nucs.items():
        assert 1 not in nu.reactions, (
            f"{name} stores MT 1: compare it to the sum of partials here")
        partial = set(nu.partial_mts)
        assert not partial & NEVER_PARTIAL, f"{name}: {partial & NEVER_PARTIAL} non-redundant"
        for parent, comps in SUMS.items():
            if parent in partial:
                dup = partial & set(comps)
                assert not dup, f"{name}: MT {parent} and its components {dup} both non-redundant"
    note("(c) MT 1 is stored in none of the 14 files; the total is the sum of "
         "non-redundant reactions (no MT counted twice)")


def test_c_redundant_sums_match_components(nucs, note):
    checks = []
    for name, nu in nucs.items():
        pairs = [(p, [m for m in c if m in nu.reactions]) for p, c in SUMS.items()
                 if p in nu.reactions and nu.reactions[p].redundant]
        if name == "Li7":
            pairs.append((205, [m for m in range(52, 92) if m in nu.reactions]))
        for parent, comps in pairs:
            if not comps:
                continue
            stored = nu.summed_on_grid([parent])
            summed = nu.summed_on_grid(comps)
            bound = half_quantum(stored) + sum(half_quantum(nu.summed_on_grid([c]))
                                               for c in comps)
            diff = np.abs(stored - summed)
            excess = diff - bound
            k = int(np.argmax(diff))
            checks.append((float(diff.max()), float(excess.max()), name, parent,
                           float(nu.energy[k]), float(stored[k])))
            assert (excess <= ABS_FLOOR_B).all(), (
                f"{name} MT {parent}: |stored - sum| exceeds rounding + {ABS_FLOOR_B} b "
                f"by {excess.max():.3e} b")
    worst = max(checks)
    worst_excess = max(checks, key=lambda c: c[1])
    note(f"(c) {len(checks)} stored redundant sums checked; worst |stored - sum| = "
         f"{worst[0]:.3e} b ({worst[2]} MT {worst[3]} at {worst[4]:.4g} eV, "
         f"stored {worst[5]:.4g} b)")
    note(f"(c) worst excess over ENDF rounding bound = {max(worst_excess[1], 0.0):.3e} b "
         f"({worst_excess[2]} MT {worst_excess[3]}); tolerance rounding + {ABS_FLOOR_B:g} b")


def test_c_presummed_total_matches_partial_lookups(nucs, note):
    rng = np.random.default_rng(20261002)      # test-side sampling only
    worst = 0.0
    for name, nu in nucs.items():
        E = np.exp(rng.uniform(np.log(nu.energy[0]), np.log(nu.energy[-1]), 2000))
        E = np.concatenate([E, [E14]])
        by_parts = np.zeros_like(E)
        for mt in nu.partial_mts:
            by_parts += xs.lookup(nu, mt, E)
        tot = xs.lookup_total(nu, E)
        rel = np.abs(tot - by_parts) / np.where(by_parts > 0, by_parts, 1.0)
        worst = max(worst, float(rel.max()))
        assert (rel <= 1e-12).all(), f"{name}: max rel diff {rel.max():.3e}"
    note(f"(c) pre-summed total vs sum of partial lookups (2001 energies per nuclide): "
         f"worst relative difference {worst:.2e} (tolerance 1e-12)")


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
# IUPAC/CIAAW standard atomic weights (ciaaw.org); 6.94 is lithium's
# conventional value (its standard weight is the interval [6.938, 6.997]).
STANDARD_WEIGHT = {"Fe": 55.845, "W": 183.84, "Be": 9.0121831, "F": 18.998403,
                   "Li": 6.94}


@pytest.mark.parametrize("symbol", sorted(STANDARD_WEIGHT))
def test_mean_atomic_mass_matches_standard_weight(nucs, symbol):
    """The mean mass from IUPAC abundances x AWR x m_n reproduces the standard
    atomic weight to 2e-4 relative. This catches wrong abundances or wrong
    AWR use. Lithium's standard weight is an interval, so its conventional
    6.94 gets 1e-3."""
    comp = cm.element(symbol)
    mean = sum(f * nucs[n].awr * cm.NEUTRON_MASS_U for n, f in comp)
    tol = 1e-3 if symbol == "Li" else 2e-4
    assert abs(mean / STANDARD_WEIGHT[symbol] - 1.0) < tol, f"{symbol}: {mean}"


def test_flibe_composition_and_density(nucs):
    awr = {n: v.awr for n, v in nucs.items()}
    assert cm.flibe().density_g_cm3 == 2.413 - 4.884e-4 * 973.0
    for li6 in (None, 0.9):
        nd = dict(cm.flibe(li6).number_densities(awr))
        li = nd["Li6"] + nd["Li7"]
        assert math.isclose(li / nd["Be9"], 2.0, rel_tol=1e-12)
        assert math.isclose(nd["F19"] / nd["Be9"], 4.0, rel_tol=1e-12)
        expect_li6 = 0.0759 if li6 is None else li6
        assert math.isclose(nd["Li6"] / li, expect_li6, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# (d) end-to-end: 14.1 MeV beam through natural iron
# ---------------------------------------------------------------------------
D_SEED = 20261001
D_LENGTH_CM = 5.0
D_BATCHES = 100
D_PER_BATCH = 10_000


def raw_iron_macroscopic(E):
    """Sigma_t and Sigma_a (1/cm) of natural iron at E, straight from the HDF5
    files. Independent of mcslab.nucdata / xs / ce_materials: h5py reads,
    np.interp (lin-lin), inline number densities."""
    abundance = {"Fe54": 0.05845, "Fe56": 0.91754, "Fe57": 0.02119, "Fe58": 0.00282}
    rho, n_avogadro, m_n = 7.874, 6.02214076e23, 1.00866491595
    micro_t, micro_a, awr = {}, {}, {}
    for name in abundance:
        with h5py.File(os.path.join(str(LIB.root), "neutron", name + ".h5"), "r") as f:
            g = f[name]
            awr[name] = float(g.attrs["atomic_weight_ratio"])
            grid = g["energy/294K"][()]
            st = sa = 0.0
            for key in g["reactions"]:
                r = g["reactions"][key]
                if r.attrs["redundant"]:
                    continue
                d = r["294K/xs"]
                i0 = int(d.attrs["threshold_idx"])
                val = 0.0 if E < grid[i0] else float(np.interp(E, grid[i0:], d[()]))
                st += val
                has_neutron = any(r[p].attrs["particle"] == b"neutron"
                                  for p in r if p.startswith("product_"))
                if not has_neutron:
                    sa += val
            micro_t[name], micro_a[name] = st, sa
    mean_mass = sum(abundance[n] * awr[n] * m_n for n in abundance)
    atoms_per_barn_cm = rho * n_avogadro / mean_mass * 1e-24
    sig_t = sum(abundance[n] * atoms_per_barn_cm * micro_t[n] for n in abundance)
    sig_a = sum(abundance[n] * atoms_per_barn_cm * micro_a[n] for n in abundance)
    return sig_t, sig_a


def test_d_iron_transmission_14mev(report, note):
    from mcslab.config_ce import CERunConfig, MonoEnergetic, run_ce
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource

    sig_t, sig_a = raw_iron_macroscopic(E14)
    cfg = CERunConfig(SlabGeometry([0.0, D_LENGTH_CM], [cm.iron()]), BeamSource(),
                      MonoEnergetic(E14), LIB, n_batches=D_BATCHES,
                      histories_per_batch=D_PER_BATCH, seed=D_SEED)
    res = run_ce(cfg)
    assert res.lost == 0 and res.max_draws < STRIDE

    expected = math.exp(-sig_t * D_LENGTH_CM)
    mean, se = T.batch_stats(res.leakage_right)
    report(f"T(14.1 MeV, Fe {D_LENGTH_CM:g} cm), Sig_t={sig_t:.6f}", mean, expected, se,
           res.max_draws)
    assert abs(mean - expected) <= 3.0 * se, (
        f"transmission {mean:.6f} vs {expected:.6f} ({(mean - expected) / se:+.2f} SE)")

    # exact: every history either collides or is transmitted; none reflect
    n = float(D_PER_BATCH)
    coll = res.region_sums[:, T.COLLISION, 0]
    assert np.array_equal(coll + res.surface_sums[:, T.POS, -1], np.full(D_BATCHES, n))
    assert not res.surface_sums[:, T.NEG, 0].any()
    # each collision scores Sigma_a/Sigma_t (energy fixed): ratio check
    ratio = res.region_sums[:, T.ABSORPTION, 0].sum() / coll.sum()
    note(f"(d) absorption per collision {ratio:.12g} vs raw-HDF5 Sigma_a/Sigma_t "
         f"{sig_a / sig_t:.12g}")
    assert math.isclose(ratio, sig_a / sig_t, rel_tol=1e-12)
