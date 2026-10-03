"""Analytic validation of the Phase 1 transport engine.

Statistical checks require |measured - expected| <= 3 standard errors, with
the standard error from batch statistics. "Exact" checks are integer
identities on raw weight sums (weights are exactly 1.0, so the sums are exact
integers in float64) and must hold with no tolerance.

Seeds are fixed and were chosen before the first run. Do NOT change a seed
or an expected value to make a check pass (docs/development.md). With 35
statistical checks at 3 sigma (two-sided p = 0.27% each), the chance that at
least one genuine fluctuation exceeds 3 sigma is roughly 9% if they were
independent; that would be a finding to report, not a reason to reseed.
"""
import math

import numpy as np
import pytest

from mcslab import tallies as T
from mcslab.config import RunConfig, run
from mcslab.geometry import SlabGeometry
from mcslab.materials import VOID, Material
from mcslab.rng import STRIDE
from mcslab.sources import BeamSource, IsotropicPlaneSource

N_BATCHES = 100
N_PER_BATCH = 10_000
N_SIGMA = 3.0


def check(report, quantity, batches, expected, max_draws):
    """3-sigma check of a batch-estimated scalar against an exact value."""
    mean, se = T.batch_stats(batches)
    report(quantity, mean, expected, se, max_draws)
    assert abs(mean - expected) <= N_SIGMA * se, (
        f"{quantity}: {mean:.7g} vs expected {expected:.7g} "
        f"({(mean - expected) / se:+.2f} SE)")


def check_same(report, quantity, batches_1, batches_2, max_draws):
    """3-sigma check that two INDEPENDENT runs estimate the same value."""
    m1, s1 = T.batch_stats(batches_1)
    m2, s2 = T.batch_stats(batches_2)
    se = math.hypot(s1, s2)
    report(quantity + " (diff)", m1 - m2, 0.0, se, max_draws)
    assert abs(m1 - m2) <= N_SIGMA * se, (
        f"{quantity}: {m1:.7g} vs {m2:.7g} ({(m1 - m2) / se:+.2f} combined SE)")


def check_exact(report, quantity, value, expected, max_draws):
    report(quantity, value, expected, 0.0, max_draws)
    assert value == expected, f"{quantity}: {value!r} != {expected!r}"


def assert_healthy(res):
    assert res.lost == 0
    assert res.max_draws < STRIDE


def cfg(geom, source, seed, n_batches=N_BATCHES, n_per_batch=N_PER_BATCH):
    return RunConfig(geom, source, n_batches=n_batches,
                     histories_per_batch=n_per_batch, seed=seed)


# ---------------------------------------------------------------------------
# (a) Pure absorber slab, beam source: transmission = exp(-Sigma_t L)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [0.5, 2.0, 5.0])
def test_a_pure_absorber_transmission(report, length):
    # All three lengths share seed 1001, so history i draws the same flight
    # distance in each: the three transmission estimates are strongly
    # correlated (a common fluctuation shows up in all of them).
    sigma = 1.0
    absorber = Material("absorber", sigma_s=0.0, sigma_a=sigma)
    res = run(cfg(SlabGeometry.uniform(absorber, 0.0, length), BeamSource(), seed=1001))
    assert_healthy(res)
    check(report, f"transmission, L={length} mfp", res.leakage_right,
          math.exp(-sigma * length), res.max_draws)
    # exact: nothing reflects, and every history is absorbed or transmitted
    n = float(N_PER_BATCH)
    check_exact(report, "left leakage (raw)", res.surface_sums[:, T.NEG, 0].sum(), 0.0,
                res.max_draws)
    per_batch = res.region_sums[:, T.ABSORPTION, 0] + res.surface_sums[:, T.POS, -1]
    check_exact(report, "absorbed + transmitted == n, all batches",
                float(np.all(per_batch == n)), 1.0, res.max_draws)


def test_a_absorber_void_absorber_transmission(report):
    """Void gap between two absorbers: T = exp(-(S1 L1 + S2 L2)); the
    beam's track length in the void is (gap width) * exp(-S1 L1)."""
    a1 = Material("a1", 0.0, 1.0)
    a2 = Material("a2", 0.0, 0.6)
    geom = SlabGeometry([0.0, 1.0, 3.0, 4.5], [a1, VOID, a2])
    res = run(cfg(geom, BeamSource(), seed=1002))
    assert_healthy(res)
    check(report, "transmission", res.leakage_right, math.exp(-(1.0 * 1.0 + 0.6 * 1.5)),
          res.max_draws)
    check(report, "track length in void", res.track_length[:, 1], 2.0 * math.exp(-1.0),
          res.max_draws)


# ---------------------------------------------------------------------------
# (b) Effectively infinite medium: <collisions> = St/Sa, <track length> = 1/Sa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma_t,sigma_a,half_width,seed", [
    (1.0, 0.1, 100.0, 2001),     # c = 0.9, diffusion length ~1.9 cm
    (2.0, 0.5, 50.0, 2002),      # c = 0.75
])
def test_b_infinite_medium(report, sigma_t, sigma_a, half_width, seed):
    mat = Material("m", sigma_s=sigma_t - sigma_a, sigma_a=sigma_a)
    geom = SlabGeometry.uniform(mat, -half_width, half_width)
    res = run(cfg(geom, IsotropicPlaneSource(0.0), seed=seed))
    assert_healthy(res)
    # exact: "infinite" is verified, not assumed -- nothing may leak
    check_exact(report, "leakage (raw, both sides)", res.surface_sums[:, :, [0, -1]].sum(),
                0.0, res.max_draws)
    check(report, f"collisions/history (St={sigma_t},Sa={sigma_a})",
          res.collisions.sum(axis=1), sigma_t / sigma_a, res.max_draws)
    check(report, "track length/history (TL)", res.track_length.sum(axis=1),
          1.0 / sigma_a, res.max_draws)
    # Single material: the collision-estimator track length is exactly
    # collisions / Sigma_t, so this is not independent of the collision check.
    check(report, "track length/history (coll)",
          res.region_batches(T.COLL_ESTIMATOR).sum(axis=1), 1.0 / sigma_a, res.max_draws)


# ---------------------------------------------------------------------------
# (c) Pure scatterer: left + right leakage = 1 exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source,length,seed", [
    (BeamSource(), 5.0, 3001),
    (IsotropicPlaneSource(3.7), 20.0, 3002),
], ids=["beam_5mfp", "iso_20mfp"])
def test_c_pure_scatterer_conservation(report, source, length, seed):
    scat = Material("scatterer", sigma_s=1.0, sigma_a=0.0)
    res = run(cfg(SlabGeometry.uniform(scat, 0.0, length), source, seed=seed))
    assert_healthy(res)
    n = float(N_PER_BATCH)
    leaked = res.surface_sums[:, T.NEG, 0] + res.surface_sums[:, T.POS, -1]
    check_exact(report, "left+right leakage == n, all batches",
                float(np.all(leaked == n)), 1.0, res.max_draws)
    check_exact(report, "left+right leakage per history",
                leaked.sum() / (n * N_BATCHES), 1.0, res.max_draws)
    check_exact(report, "absorptions (raw)", res.region_sums[:, T.ABSORPTION].sum(), 0.0,
                res.max_draws)


# ---------------------------------------------------------------------------
# (d) Track-length and collision flux estimators agree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source,seed", [
    (IsotropicPlaneSource(1.0), 4001),
    (BeamSource(), 4002),
], ids=["iso", "beam"])
def test_d_flux_estimators_agree(report, source, seed):
    """Both estimators come from the same histories, so they are correlated;
    the SE of the difference is taken from the per-batch differences, which
    accounts for that correlation."""
    m1 = Material("m1", sigma_s=0.8, sigma_a=0.2)
    m2 = Material("m2", sigma_s=0.45, sigma_a=0.05)
    m3 = Material("m3", sigma_s=0.5, sigma_a=1.5)
    geom = SlabGeometry([0.0, 2.0, 5.0, 6.0], [m1, m2, m3])
    res = run(cfg(geom, source, seed=seed))
    assert_healthy(res)
    diff = res.flux_tl - res.flux_coll
    for r in range(geom.n_regions):
        check(report, f"flux TL - coll, region {r}", diff[:, r], 0.0, res.max_draws)


# ---------------------------------------------------------------------------
# (e) Splitting a uniform slab into regions changes nothing (boundary crossing)
# ---------------------------------------------------------------------------
def test_e_split_slab_matches_unsplit(report):
    """Split runs use different random numbers (resampling at each crossing),
    so they are compared statistically, with independent seeds."""
    mat = Material("m", sigma_s=0.7, sigma_a=0.3)
    one = run(cfg(SlabGeometry.uniform(mat, 0.0, 6.0), BeamSource(), seed=5001))
    three = run(cfg(SlabGeometry([0.0, 1.0, 2.5, 6.0], [mat] * 3), BeamSource(), seed=5002))
    sixty = run(cfg(SlabGeometry.uniform(mat, 0.0, 6.0, 60), BeamSource(), seed=5003))
    for res in (one, three, sixty):
        assert_healthy(res)
    md = max(one.max_draws, three.max_draws, sixty.max_draws)

    def totals(res):
        return {
            "left leakage": res.leakage_left,
            "right leakage": res.leakage_right,
            "collisions": res.collisions.sum(axis=1),
            "absorptions": res.absorptions.sum(axis=1),
            "track length": res.track_length.sum(axis=1),
        }
    t1, t3, t60 = totals(one), totals(three), totals(sixty)
    for key in t1:
        check_same(report, f"{key}: 1 vs 3 regions", t1[key], t3[key], md)
        check_same(report, f"{key}: 1 vs 60 regions", t1[key], t60[key], md)

    # region-level: aggregate the 60 fine regions onto the 3 coarse ones
    coarse = [(0, 10), (10, 25), (25, 60)]
    for r, (i0, i1) in enumerate(coarse):
        check_same(report, f"track length, coarse region {r}",
                   three.track_length[:, r], sixty.track_length[:, i0:i1].sum(axis=1), md)
    # interface currents at x = 1.0 and x = 2.5, both directions
    for s3, s60 in ((1, 10), (2, 25)):
        check_same(report, f"current +mu at surface {s3}",
                   three.current_pos[:, s3], sixty.current_pos[:, s60], md)
        check_same(report, f"current -mu at surface {s3}",
                   three.current_neg[:, s3], sixty.current_neg[:, s60], md)


def test_e_split_pure_absorber_transmission(report):
    """Beam through a pure absorber split into 10 unequal regions."""
    sigma = 0.8
    absorber = Material("absorber", sigma_s=0.0, sigma_a=sigma)
    bounds = [0.0, 0.1, 0.35, 0.4, 1.0, 1.7, 2.0, 2.05, 3.1, 3.3, 4.0]
    res = run(cfg(SlabGeometry(bounds, [absorber] * 10), BeamSource(), seed=5004))
    assert_healthy(res)
    check(report, "transmission, 10 regions", res.leakage_right,
          math.exp(-sigma * 4.0), res.max_draws)
