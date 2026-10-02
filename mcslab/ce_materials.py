"""Continuous-energy materials: composition + density -> number densities.

Sources for every number used here:

[CODATA]  E. Tiesinga et al., "CODATA recommended values of the fundamental
          physical constants: 2018", Rev. Mod. Phys. 93, 025010 (2021).
          m_n = 1.00866491595 u. N_A = 6.02214076e23 /mol is exact (SI 2019).
[IUPAC]   J. Meija et al., "Isotopic compositions of the elements 2013 (IUPAC
          Technical Report)", Pure Appl. Chem. 88, 293-306 (2016), Table 1,
          "representative isotopic composition" (atom fractions).
[CRC]     W. M. Haynes (ed.), CRC Handbook of Chemistry and Physics, 97th ed.
          (CRC Press, 2016), sec. 4, "Physical Constants of Inorganic
          Compounds": densities at room temperature.
[JANZ]    G. J. Janz, "Thermodynamic and transport properties for molten salts:
          correlation equations for critically evaluated density, surface
          tension, electrical conductance, and viscosity data", J. Phys. Chem.
          Ref. Data 17, Suppl. 2 (1988). Recommended for FLiBe in R. R.
          Romatoski and L. W. Hu, "Fluoride salt coolant properties for
          nuclear reactor applications: A review", Ann. Nucl. Energy 109,
          635-647 (2017):  rho [kg/m^3] = 2413 - 0.4884 T[K].

Nuclide masses are AWR x m_n, where AWR is the atomic_weight_ratio in each
data file (atomic mass / neutron mass). This keeps the masses consistent with
the cross-section evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

N_AVOGADRO = 6.02214076e23          # 1/mol, exact [CODATA]
NEUTRON_MASS_U = 1.00866491595      # u [CODATA]
BARN_CM2 = 1.0e-24

# Natural isotopic composition, atom fractions [IUPAC]. Each sums to 1.
NATURAL: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "Li": (("Li6", 0.0759), ("Li7", 0.9241)),
    "Be": (("Be9", 1.0),),
    "F": (("F19", 1.0),),
    "Fe": (("Fe54", 0.05845), ("Fe56", 0.91754), ("Fe57", 0.02119), ("Fe58", 0.00282)),
    "W": (("W180", 0.0012), ("W182", 0.2650), ("W183", 0.1431), ("W184", 0.3064),
          ("W186", 0.2843)),
}

DENSITY_W = 19.3        # g/cm^3 [CRC]
DENSITY_FE = 7.874      # g/cm^3 [CRC]
FLIBE_DEFAULT_T = 973.0  # K, a typical blanket/loop operating temperature


def flibe_density(temperature_K: float) -> float:
    """FLiBe (2LiF-BeF2) density in g/cm^3 from the Janz correlation [JANZ],
    rho = 2.413 - 4.884e-4 T. It is a fit to liquid-salt data. FLiBe melts
    near 732 K, so this is an extrapolation below the melt and far above
    the fitted temperature range."""
    return 2.413 - 4.884e-4 * temperature_K


@dataclass(frozen=True)
class CEMaterial:
    """A material as (nuclide, atom fraction) pairs plus a mass density.
    Atom fractions are per atom of material and must sum to 1.

    temperature: K, used by the kinematic driver to select the data
    temperature (OpenMC's NEAREST rule) and hence kT for free-gas
    scattering. None means the run's default (293.6 K). It does not change
    the density: set that explicitly (e.g. flibe(temperature_K=...)). The
    Phase 2a driver refuses a material whose temperature selects data other
    than the run's single temperature."""
    name: str
    density_g_cm3: float
    atom_fractions: Tuple[Tuple[str, float], ...]
    temperature: Optional[float] = None

    def __post_init__(self):
        if self.density_g_cm3 < 0.0:
            raise ValueError(f"{self.name}: negative density")
        if self.temperature is not None and not self.temperature > 0.0:
            raise ValueError(f"{self.name}: temperature must be > 0 K")
        total = sum(f for _, f in self.atom_fractions)
        if self.atom_fractions and abs(total - 1.0) > 1e-12:
            raise ValueError(f"{self.name}: atom fractions sum to {total}, not 1")
        names = [n for n, _ in self.atom_fractions]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: repeated nuclide")

    @property
    def nuclide_names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in self.atom_fractions)

    def number_densities(self, awr: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
        """-> ((nuclide, atoms per barn-cm), ...), in the material's order.

        mean atomic mass  M = sum_i f_i AWR_i m_n            (g/mol)
        atom density      N = rho N_A / M                    (1/cm^3)
        N_i = f_i N x 1e-24                                  (1/(barn cm))
        """
        if not self.atom_fractions:
            return ()
        mean_mass = sum(f * awr[n] * NEUTRON_MASS_U for n, f in self.atom_fractions)
        n_total = self.density_g_cm3 * N_AVOGADRO / mean_mass * BARN_CM2
        return tuple((n, f * n_total) for n, f in self.atom_fractions)


VOID_CE = CEMaterial("void", 0.0, ())


def element(symbol: str, enrichment: Optional[Dict[str, float]] = None):
    """Isotopic atom fractions of an element: natural [IUPAC], or `enrichment`
    (nuclide -> atom fraction, validated to sum to 1)."""
    if enrichment is None:
        return NATURAL[symbol]
    comp = tuple(enrichment.items())
    if abs(sum(f for _, f in comp) - 1.0) > 1e-12:
        raise ValueError(f"{symbol} isotopic fractions must sum to 1")
    return comp


def _compound(name, density, formula):
    """formula: ((atoms per formula unit, isotopic composition), ...)"""
    atoms = sum(k for k, _ in formula)
    fracs = []
    for k, iso in formula:
        for nuc, f in iso:
            fracs.append((nuc, k * f / atoms))
    return CEMaterial(name, density, tuple(fracs))


def tungsten(density: float = DENSITY_W) -> CEMaterial:
    """Natural tungsten [IUPAC], 19.3 g/cm^3 [CRC]."""
    return CEMaterial("W", density, element("W"))


def iron(density: float = DENSITY_FE) -> CEMaterial:
    """Natural iron [IUPAC], 7.874 g/cm^3 [CRC]."""
    return CEMaterial("Fe", density, element("Fe"))


def flibe(li6_fraction: Optional[float] = None,
          temperature_K: float = FLIBE_DEFAULT_T) -> CEMaterial:
    """FLiBe, 2LiF-BeF2 = Li2BeF4 (7 atoms per formula unit).

    li6_fraction: Li-6 atom fraction in the lithium. None means natural
                  lithium (0.0759) [IUPAC].
    temperature_K: sets only the density (Janz correlation [JANZ]). The cross
                  sections stay at the temperature of the nuclear data (294 K).
    """
    if li6_fraction is not None and not 0.0 <= li6_fraction <= 1.0:
        raise ValueError("li6_fraction must be in [0, 1]")
    li = element("Li") if li6_fraction is None else element(
        "Li", {"Li6": li6_fraction, "Li7": 1.0 - li6_fraction})
    return _compound("FLiBe", flibe_density(temperature_K),
                     ((2, li), (1, element("Be")), (4, element("F"))))
