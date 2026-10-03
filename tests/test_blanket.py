"""Part B of Phase 3: Li-6 enrichment (approved D18) and the reflective
plasma-side boundary (approved D17).

Enrichment: enriched FLiBe keeps the molar (formula-unit, hence atom)
density of natural-Li FLiBe at the Janz density; only the Li isotopic
split changes. The natural-Li path (li6_fraction=None) is unchanged bit for
bit. AWRs are those of the ENDF/B-VIII.0 files (no data needed here).

Reflection (seeds fixed in the plan): exact checks in a void slab (seed
20261037) and on W | FLiBe | Fe (seed 20261038): the reflection draws no
random number, flips u exactly, keeps E and x = 0, is not counted as
leakage, and keeps the neutron balance exact. One statistical test (4
checks, seeds 20261033 and 20261034): a reflective slab equals the
mirrored vacuum slab.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

from mcslab import ce_materials as cm
from mcslab import synthetic as S
from mcslab import tallies as T
from mcslab.ce_materials import VOID_CE
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.nucdata import Library
from mcslab.rng import RNG_SIZE, init_history, prn
from mcslab.sources import BeamSource, IsotropicPlaneSource
from mcslab.tracks import EV_REFLECT, TF_E, TF_U, TF_X, TI_EVENT, TI_PID, TI_REGION

AWR = {"Li6": 5.961817, "Li7": 6.955732, "Be9": 8.93478, "F19": 18.835}


def test_natural_path_is_unchanged():
    """flibe() with natural Li evaluates exactly the pre-Phase-3 formula."""
    for T in (900.0, 973.0):
        mat = cm.flibe(temperature_K=T)
        assert mat.reference_atom_fractions is None
        rho = 2.413 - 4.884e-4 * T
        fracs = (("Li6", 2 * 0.0759 / 7), ("Li7", 2 * 0.9241 / 7), ("Be9", 1 / 7),
                 ("F19", 4 / 7))
        assert mat.atom_fractions == fracs
        mean_mass = sum(f * AWR[n] * cm.NEUTRON_MASS_U for n, f in fracs)
        n_total = rho * cm.N_AVOGADRO / mean_mass * cm.BARN_CM2
        assert mat.number_densities(AWR) == tuple((n, f * n_total) for n, f in fracs)
        assert mat.mass_density(AWR) == mat.density_g_cm3


@pytest.mark.parametrize("li6", [0.0, 0.2, 0.4, 0.6, 0.9, 1.0])
def test_enriched_flibe_holds_molar_density(li6):
    nat = dict(cm.flibe(temperature_K=900.0).number_densities(AWR))
    mat = cm.flibe(li6, temperature_K=900.0)
    enr = dict(mat.number_densities(AWR))
    # same formula units per cm^3: Be and F unchanged bit for bit, Li total unchanged
    assert enr["Be9"] == nat["Be9"] and enr["F19"] == nat["F19"]
    li_n, li_e = nat["Li6"] + nat["Li7"], enr["Li6"] + enr["Li7"]
    assert math.isclose(li_e, li_n, rel_tol=1e-14)
    assert math.isclose(sum(enr.values()), sum(nat.values()), rel_tol=1e-14)
    assert math.isclose(enr["Li6"], li6 * li_e, rel_tol=1e-14, abs_tol=0.0)
    assert math.isclose(li_e / enr["Be9"], 2.0, rel_tol=1e-12)
    # the mass density follows the molar mass
    m_nat = 2 * (0.0759 * AWR["Li6"] + 0.9241 * AWR["Li7"]) + AWR["Be9"] + 4 * AWR["F19"]
    m_enr = 2 * (li6 * AWR["Li6"] + (1 - li6) * AWR["Li7"]) + AWR["Be9"] + 4 * AWR["F19"]
    assert math.isclose(mat.mass_density(AWR), mat.density_g_cm3 * m_enr / m_nat,
                        rel_tol=1e-14)
    if li6 == 0.9:
        assert 0.982 < mat.mass_density(AWR) / mat.density_g_cm3 < 0.984   # -1.7%


def test_reference_fractions_are_validated():
    with pytest.raises(ValueError):
        cm.CEMaterial("x", 1.0, (("A", 1.0),), reference_atom_fractions=(("B", 1.0),))
    with pytest.raises(ValueError):
        cm.CEMaterial("x", 1.0, (("A", 1.0),), reference_atom_fractions=(("A", 0.5),))
    with pytest.raises(ValueError):
        cm.flibe(1.2)


# ---------------------------------------------------------------------------
# Reflective plasma side (approved D17)
# ---------------------------------------------------------------------------
VOID_SEED = 20261037
BALANCE_SEED = 20261038
SYM_SEEDS = (20261033, 20261034)
E0 = 14.1e6


def _library():
    try:
        return Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def _quiet(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg)


def _reflections_in_tracks(tracks):
    """Every EV_REFLECT event with the event before it of the same particle."""
    out = []
    for h in range(tracks.f.shape[0]):
        f, i = tracks.events(h)
        for j in np.flatnonzero(i[:, TI_EVENT] == EV_REFLECT):
            prev = max(q for q in range(j) if i[q, TI_PID] == i[j, TI_PID])
            out.append((f[prev], f[j], i[j]))
    return out


def test_reflection_in_void_is_exact():
    """Void slab [0, 1], isotropic source at 0.5, reflective left side: a
    neutron with mu < 0 flies 0.5/|mu| to the wall, is reflected and flies
    1/|mu| out of the right side; one with mu > 0 flies 0.5/mu. The source
    direction is each history's only random number, replayed here."""
    nb, nh = 4, 250
    cfg = KinRunConfig(SlabGeometry([0.0, 1.0], [VOID_CE]), IsotropicPlaneSource(0.5),
                       MonoEnergetic(1.0e6), S.SyntheticLibrary({}), n_batches=nb,
                       histories_per_batch=nh, seed=VOID_SEED, reflect_left=True,
                       n_track=nb * nh, track_capacity=8, depth_bins=(4,))
    res = run_kin(cfg)
    assert res.max_draws == 1                       # the reflection drew nothing
    assert res.balance()["residual"] == 0.0
    assert res.count(T.K_LEAK_RIGHT) == nb * nh and res.count(T.K_LEAK_LEFT) == 0
    assert not res.surface_sums[:, T.NEG, 0].any()
    rng = np.zeros(RNG_SIZE, np.uint64)
    for b in range(nb):
        n_neg, tl = 0, 0.0
        for h in range(b * nh, (b + 1) * nh):
            init_history(rng, np.uint64(VOID_SEED), np.uint64(h))
            mu = 2.0 * prn(rng) - 1.0
            n_neg += mu < 0.0
            tl += 0.5 / mu if mu > 0.0 else 1.5 / -mu
        assert res.reflected_weight[b] == float(n_neg)
        assert math.isclose(res.region_sums[b, T.TRACK_LENGTH, 0], tl, rel_tol=1e-12)
        assert math.isclose(res.mesh_flux[b, :, T.EST_TL].sum(), tl, rel_tol=1e-12)
    refl = _reflections_in_tracks(res.tracks)
    assert len(refl) == res.reflected_weight.sum()
    for before, after, info in refl:
        assert after[TF_X] == 0.0 and after[TF_U] == -before[TF_U] and after[TF_U] > 0.0
        assert after[TF_E] == before[TF_E] and info[TI_REGION] == 0


def test_reflective_balance_on_real_materials():
    lib = _library()
    flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    geom = SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()])
    nb, nh = 4, 100
    cfg = KinRunConfig(geom, BeamSource(), MonoEnergetic(E0), lib, nb, nh, BALANCE_SEED,
                       reflect_left=True, n_track=nb * nh, track_capacity=3000)
    res = _quiet(cfg)
    bal = res.balance()
    assert bal["residual"] == 0.0 and bal["leak_left"] == 0.0 and bal["lost"] == 0.0
    assert res.count(T.K_LEAK_LEFT) == 0 and not res.surface_sums[:, T.NEG, 0].any()
    assert not res.tracks.truncated.any()
    refl = _reflections_in_tracks(res.tracks)
    assert len(refl) == res.reflected_weight.sum() > 0
    for before, after, info in refl:
        assert after[TF_X] == 0.0 and after[TF_U] == -before[TF_U] and after[TF_U] > 0.0
        assert after[TF_E] == before[TF_E] and info[TI_REGION] == 0
    # the same run with vacuum leaks to the left and reflects nothing
    vac = _quiet(dataclasses.replace(cfg, reflect_left=False, n_track=0))
    assert vac.count(T.K_LEAK_LEFT) > 0 and not vac.reflected_weight.any()


def _check_same(report, quantity, a, b, res):
    m1, s1 = T.batch_stats(a)
    m2, s2 = T.batch_stats(b)
    se = math.hypot(s1, s2)
    n_se = report(quantity + " (A - B)", m1 - m2, 0.0, se, res.max_draws,
                  cutoff=float(res.cutoff.sum(axis=1).mean()))
    return None if abs(m1 - m2) <= 3.0 * se else f"{quantity}: {m1:.7g} vs {m2:.7g} " \
                                                  f"({n_se:+.2f} combined SE)"


def test_reflection_equals_mirrored_slab(report):
    """Folding argument: a symmetric problem on [-L, L] (vacuum, isotropic
    source at 0) folded onto [0, L] is the reflective problem on [0, L]
    with the same source. So A (reflective, FLiBe [0, 20] cm) and B
    (vacuum, FLiBe [-20, 20] cm) have equal expected flux, absorption,
    right leakage (A) = total leakage (B), and tritium production.
    Independent runs: combined SE. 4 statistical checks."""
    lib = _library()
    flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    a = _quiet(KinRunConfig(SlabGeometry([0.0, 20.0], [flibe]), IsotropicPlaneSource(0.0),
                            MonoEnergetic(E0), lib, 100, 1000, SYM_SEEDS[0],
                            reflect_left=True, depth_bins=(1,)))
    b = _quiet(KinRunConfig(SlabGeometry([-20.0, 0.0, 20.0], [flibe, flibe]),
                            IsotropicPlaneSource(0.0), MonoEnergetic(E0), lib, 100, 1000,
                            SYM_SEEDS[1], depth_bins=(1, 1)))
    for res in (a, b):
        assert res.balance()["residual"] == 0.0 and res.lost == 0
    assert a.reflected_weight.sum() > 0 and a.count(T.K_LEAK_LEFT) == 0
    fails = [
        _check_same(report, "flux (track length)", a.track_length.sum(axis=1),
                    b.track_length.sum(axis=1), a),
        _check_same(report, "absorption (analog)", a.absorptions.sum(axis=1),
                    b.absorptions.sum(axis=1), a),
        _check_same(report, "leakage", a.leakage_right, b.leakage_left + b.leakage_right, a),
        _check_same(report, "tritium (track length)",
                    a.tally_batches(T.R_H3, T.EST_TL).sum(axis=1),
                    b.tally_batches(T.R_H3, T.EST_TL).sum(axis=1), a),
    ]
    fails = [f for f in fails if f]
    assert not fails, "\n".join(fails)
