"""Part B of Phase 3: Li-6 enrichment (approved D18).

Enriched FLiBe keeps the molar (formula-unit, hence atom) density of
natural-Li FLiBe at the Janz density; only the Li isotopic split changes.
The natural-Li path (li6_fraction=None) is unchanged bit for bit. AWRs are
those of the ENDF/B-VIII.0 files (no data needed here).
"""
import math

import pytest

from mcslab import ce_materials as cm

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
