"""Reader for OpenMC HDF5 incident-neutron data (format version 3.x), h5py only.

The `openmc` package is deliberately not used. Layout read here (one nuclide
per file, e.g. neutron/Fe56.h5):

    /                      attrs filetype = "data_neutron", version = [3, x]
    /<name>                attrs Z, A, metastable, atomic_weight_ratio
    /<name>/energy/<T>     pointwise energy grid (eV), shared by all reactions
    /<name>/kTs/<T>        kT (eV)
    /<name>/reactions/reaction_<MT>
                           attrs mt, label, Q_value (eV), center_of_mass, redundant
        <T>/xs             cross section (b) on energy[threshold_idx:]
                           attr threshold_idx
        product_<k>        attrs particle, emission_mode, n_distribution
            yield          attr type (Polynomial | Tabulated1D)
            distribution_<j>  attr type (uncorrelated | correlated | kalbach-mann | nbody)
    /<name>/urr/<T>        unresolved-resonance probability tables (if any)

Interpolation: a reaction's `xs` dataset carries no interpolation law. It
holds the NJOY-reconstructed pointwise (ACE ESZ-style) cross section, which
is linearised to be interpolated lin-lin on the energy grid (ACE format
manual, LA-UR-19-29016, sec. on the ESZ block; OpenMC HDF5 format spec,
"Incident Neutron Data"). Any attribute other than threshold_idx on an
`xs` dataset would mean a different convention, so the reader refuses it
instead of guessing.

The library is a directory plus a label. Nothing here is specific to
ENDF/B-VIII.0, so another OpenMC HDF5 library (e.g. FENDL-3.2) is read the
same way.

Secondary distributions (Phase 2b). With `distributions=True` the reader
also reads the outgoing-neutron laws themselves, not only their
descriptions. The layout follows OpenMC's readers (OpenMC v0.16.0,
src/reaction_product.cpp, src/secondary_uncorrelated.cpp,
src/distribution_angle.cpp, src/distribution_energy.cpp):

    product_0/yield                 Polynomial (coefficients) or Tabulated1D
    product_0/distribution_<j>      attr type
        applicability               Tabulated1D, read only when n_distribution > 1
        angle/energy                incident energies (non-decreasing)
        angle/mu                    rows x, p, c; attrs offsets, interpolation
        energy                      attr type; "level" has threshold, mass_ratio

Only what mcslab implements is parsed. A law it does not implement (yet)
is kept as UnsupportedLaw, and the kinematic driver refuses a problem in
which such a reaction can occur. Anything unexpected inside a law it does
parse (an unknown attribute or key, an interpolation code other than
histogram or lin-lin, a malformed table) raises instead of being guessed.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

DATA_ENV = "MCSLAB_DATA"
SUPPORTED_MAJOR_VERSION = 3
DEFAULT_TEMPERATURE = "294K"


def data_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """Library root: `path` if given, else $MCSLAB_DATA. The root is the
    directory that contains neutron/ (e.g. ~/nuclear_data/endfb-viii.0-hdf5)."""
    if path is None:
        path = os.environ.get(DATA_ENV)
        if not path:
            raise FileNotFoundError(
                f"no nuclear data directory: set {DATA_ENV} or pass data_dir= "
                "(see scripts/fetch_data.sh)")
    root = Path(path).expanduser()
    if not (root / "neutron").is_dir():
        raise FileNotFoundError(f"{root} has no neutron/ subdirectory")
    return root


def temperature_key(temperature: Union[str, int, float]) -> str:
    """294 -> "294K"; "294K" passes through. OpenMC names temperature groups
    by the rounded temperature (293.6 K is stored as "294K")."""
    if isinstance(temperature, str):
        return temperature if temperature.endswith("K") else f"{temperature}K"
    return f"{int(round(temperature))}K"


def _str(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HISTOGRAM = 1        # ENDF interpolation codes accepted inside laws
LIN_LIN = 2
_LAW_INTERPOLATIONS = (HISTOGRAM, LIN_LIN)


@dataclass(frozen=True)
class Tabular:
    """A tabulated PDF p and CDF c on points x (OpenMC `Tabular`), with
    interpolation HISTOGRAM or LIN_LIN. Stored as read; normalisation by
    c[-1] happens at packing, as in OpenMC's Tabular::init."""
    x: np.ndarray = field(repr=False)
    p: np.ndarray = field(repr=False)
    c: np.ndarray = field(repr=False)
    interpolation: int


@dataclass(frozen=True)
class Tabulated1D:
    """A single-region tabulated function y(x), histogram or lin-lin."""
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    interpolation: int


@dataclass(frozen=True)
class AngleDistribution:
    """mu tables at incident energies (OpenMC `AngleDistribution`). The
    energies are non-decreasing; a repeated energy marks a discontinuity."""
    energy: np.ndarray = field(repr=False)
    mu: Tuple[Tabular, ...] = field(repr=False)


@dataclass(frozen=True)
class LevelInelastic:
    """Discrete level: the stored parameters, E_cm = mass_ratio (E - threshold)
    in OpenMC. mcslab computes E_cm from AWR and Q instead (see
    docs/deviations_from_openmc.md); these are kept for comparison."""
    threshold: float     # eV
    mass_ratio: float


@dataclass(frozen=True)
class UncorrelatedAngleEnergy:
    angle: Optional[AngleDistribution]   # None: isotropic
    energy: Optional[LevelInelastic]     # None: no energy law (elastic)


@dataclass(frozen=True)
class UnsupportedLaw:
    """A secondary law mcslab does not implement yet (kept, never guessed)."""
    description: str


@dataclass(frozen=True)
class Product:
    """Outgoing particle of a reaction. The descriptive fields are always
    read. The laws themselves are read only for neutrons when the nuclide is
    loaded with distributions=True (laws_read is then True)."""
    particle: str
    emission_mode: str
    yield_type: str
    yield_constant: Optional[float]   # Polynomial yield of degree 0, else None
    distributions: Tuple[str, ...]    # e.g. "uncorrelated(angle=tabular, energy=level)"
    laws_read: bool = False
    yield_coefficients: Tuple[float, ...] = ()      # Polynomial yield
    yield_table: Optional[Tabulated1D] = None       # Tabulated1D yield
    laws: Tuple[object, ...] = ()                   # one per distribution
    applicability: Tuple[Tabulated1D, ...] = ()     # only when len(laws) > 1


@dataclass(frozen=True)
class Reaction:
    mt: int
    label: str
    q_value: float                    # eV
    center_of_mass: bool
    redundant: bool
    threshold_idx: int
    xs: np.ndarray = field(repr=False)   # barns, on energy[threshold_idx:]
    products: Tuple[Product, ...] = ()

    @property
    def neutron_products(self) -> Tuple[Product, ...]:
        return tuple(p for p in self.products if p.particle == "neutron")


@dataclass(frozen=True)
class Nuclide:
    name: str
    Z: int
    A: int
    metastable: int
    awr: float                        # atomic mass / neutron mass
    temperature: str
    kT: float                         # eV
    energy: np.ndarray = field(repr=False)
    reactions: Dict[int, Reaction] = field(repr=False)
    has_urr: bool = False
    path: str = ""
    library: str = ""

    def threshold_energy(self, mt: int) -> float:
        """First grid energy at which reaction `mt` is tabulated."""
        return float(self.energy[self.reactions[mt].threshold_idx])

    @property
    def partial_mts(self) -> Tuple[int, ...]:
        """Non-redundant reactions, in increasing MT order. Their sum is the
        total cross section (MT 1 is not stored in OpenMC HDF5 files)."""
        return tuple(mt for mt in sorted(self.reactions)
                     if not self.reactions[mt].redundant)

    def summed_on_grid(self, mts) -> np.ndarray:
        """Sum of the given reactions on this nuclide's grid, accumulated in
        the order given. Reactions are zero below their threshold index."""
        out = np.zeros_like(self.energy)
        for mt in mts:
            rx = self.reactions[mt]
            out[rx.threshold_idx:] += rx.xs
        return out

    def total_on_grid(self) -> np.ndarray:
        return self.summed_on_grid(self.partial_mts)

    @property
    def absorption_mts(self) -> Tuple[int, ...]:
        """Non-redundant reactions with no outgoing neutron (capture and
        charged-particle channels), taken from the product data rather
        than from an MT list."""
        return tuple(mt for mt in self.partial_mts
                     if not self.reactions[mt].neutron_products)

    def absorption_on_grid(self) -> np.ndarray:
        return self.summed_on_grid(self.absorption_mts)


def _describe_distribution(group) -> str:
    kind = _str(group.attrs["type"])
    if kind != "uncorrelated":
        return kind
    angle = "tabular" if "angle" in group else "isotropic"
    energy = _str(group["energy"].attrs["type"]) if "energy" in group else "none"
    return f"uncorrelated(angle={angle}, energy={energy})"


def _refuse(obj, what):
    raise ValueError(f"{obj.file.filename}:{obj.name}: {what}")


def _check_attrs(obj, expected):
    got = set(obj.attrs)
    if got != set(expected):
        _refuse(obj, f"attributes {sorted(got)}, expected {sorted(expected)}")


def _check_table(obj, x, p, c, interp, lo=-np.inf, hi=np.inf):
    if interp not in _LAW_INTERPOLATIONS:
        _refuse(obj, f"interpolation code {interp} (only histogram/lin-lin)")
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(p).all()
                          and np.isfinite(c).all()):
        _refuse(obj, "table with < 2 points or non-finite values")
    if (np.diff(x) <= 0.0).any() or x[0] < lo or x[-1] > hi:
        _refuse(obj, f"table abscissae not strictly increasing within [{lo}, {hi}]")
    if (p < 0.0).any() or (np.diff(c) < 0.0).any() or not c[-1] > 0.0:
        _refuse(obj, "negative PDF, decreasing CDF, or CDF ending at 0")


def _read_tabulated1d(ds) -> Tabulated1D:
    _check_attrs(ds, ("type", "breakpoints", "interpolation"))
    if _str(ds.attrs["type"]) != "Tabulated1D":
        _refuse(ds, f"type {_str(ds.attrs['type'])}, expected Tabulated1D")
    data = np.array(ds[()], dtype=np.float64)
    bp = np.asarray(ds.attrs["breakpoints"]).ravel()
    interp = np.asarray(ds.attrs["interpolation"]).ravel()
    if data.ndim != 2 or data.shape[0] != 2:
        _refuse(ds, f"shape {data.shape}, expected (2, n)")
    n = data.shape[1]
    if bp.size != 1 or int(bp[0]) != n or int(interp[0]) not in _LAW_INTERPOLATIONS:
        _refuse(ds, f"breakpoints {bp} / interpolation {interp}: only one "
                    "histogram or lin-lin region is supported")
    x, y = data
    if n < 1 or (np.diff(x) < 0.0).any() or not np.isfinite(data).all():
        _refuse(ds, "abscissae decreasing or non-finite values")
    return Tabulated1D(x, y, int(interp[0]))


def _read_angle(group) -> AngleDistribution:
    if set(group) != {"energy", "mu"} or group.attrs.keys():
        _refuse(group, f"unexpected contents {sorted(group)}")
    energy = np.array(group["energy"][()], dtype=np.float64)
    if group["energy"].attrs.keys():
        _refuse(group["energy"], "unexpected attributes")
    # A repeated incident energy is a discontinuity (the Fe and W elastic
    # data repeat one point at the evaluation's high-energy join). OpenMC's
    # lower_bound_index then uses the left table from below and the right
    # table from above. A repeat at the first point would divide by zero.
    if (np.diff(energy) < 0.0).any() or (energy.size > 1 and energy[1] == energy[0]):
        _refuse(group["energy"], "incident energies decreasing or repeated at the start")
    ds = group["mu"]
    _check_attrs(ds, ("offsets", "interpolation"))
    data = np.array(ds[()], dtype=np.float64)
    offsets = np.asarray(ds.attrs["offsets"], dtype=np.int64)
    interps = np.asarray(ds.attrs["interpolation"], dtype=np.int64)
    if data.ndim != 2 or data.shape[0] != 3 or offsets.size != energy.size \
            or interps.size != energy.size:
        _refuse(ds, "mu table layout does not match the incident energies")
    ends = np.append(offsets[1:], data.shape[1])
    tables = []
    for j0, j1, it in zip(offsets, ends, interps):
        x, p, c = data[0, j0:j1], data[1, j0:j1], data[2, j0:j1]
        _check_table(ds, x, p, c, int(it), -1.0, 1.0)
        tables.append(Tabular(x, p, c, int(it)))
    return AngleDistribution(energy, tuple(tables))


def _read_law(group):
    kind = _str(group.attrs["type"])
    if kind != "uncorrelated":
        return UnsupportedLaw(kind)
    _check_attrs(group, ("type",))
    extra = set(group) - {"angle", "energy", "applicability"}
    if extra:
        _refuse(group, f"unexpected contents {sorted(extra)}")
    angle = _read_angle(group["angle"]) if "angle" in group else None
    energy = None
    if "energy" in group:
        eg = group["energy"]
        etype = _str(eg.attrs["type"])
        if etype != "level":
            return UnsupportedLaw(f"uncorrelated(energy={etype})")
        _check_attrs(eg, ("type", "threshold", "mass_ratio"))
        if isinstance(eg, h5py.Group) and len(eg):
            _refuse(eg, "level law with sub-datasets")
        energy = LevelInelastic(float(eg.attrs["threshold"]), float(eg.attrs["mass_ratio"]))
    return UncorrelatedAngleEnergy(angle, energy)


def _read_product(group, laws=False) -> Product:
    y = group["yield"]
    ytype = _str(y.attrs["type"])
    yconst = None
    if ytype == "Polynomial":
        coef = np.asarray(y[()], dtype=np.float64).ravel()
        if coef.size == 1:
            yconst = float(coef[0])
    n_dist = int(group.attrs["n_distribution"])
    dists = tuple(_describe_distribution(group[f"distribution_{j}"])
                  for j in range(n_dist))
    particle = _str(group.attrs["particle"])
    base = (particle, _str(group.attrs["emission_mode"]), ytype, yconst, dists)
    if not laws or particle != "neutron":
        return Product(*base)

    if ytype == "Polynomial":
        _check_attrs(y, ("type",))
        ycoef = tuple(float(v) for v in np.asarray(y[()], dtype=np.float64).ravel())
        ytab = None
    else:
        ycoef, ytab = (), _read_tabulated1d(y)
    groups = [group[f"distribution_{j}"] for j in range(n_dist)]
    # As in OpenMC (ReactionProduct constructor), applicability is read only
    # when there is more than one distribution to choose from.
    app = tuple(_read_tabulated1d(g["applicability"]) for g in groups) if n_dist > 1 else ()
    return Product(*base, laws_read=True, yield_coefficients=ycoef, yield_table=ytab,
                   laws=tuple(_read_law(g) for g in groups), applicability=app)


def read_nuclide(path: Union[str, Path], temperature=DEFAULT_TEMPERATURE,
                 library: str = "", distributions: bool = False) -> Nuclide:
    """Read one OpenMC HDF5 incident-neutron file at one temperature. With
    distributions=True, the outgoing-neutron laws of non-redundant reactions
    are read too (see the module docstring)."""
    path = Path(path)
    T = temperature_key(temperature)
    with h5py.File(path, "r") as f:
        filetype = _str(f.attrs.get("filetype", b""))
        version = tuple(int(v) for v in f.attrs.get("version", ()))
        if filetype != "data_neutron":
            raise ValueError(f"{path}: filetype {filetype!r}, expected 'data_neutron'")
        if not version or version[0] != SUPPORTED_MAJOR_VERSION:
            raise ValueError(f"{path}: format version {version} unsupported "
                             f"(need {SUPPORTED_MAJOR_VERSION}.x)")
        (name,) = list(f)
        g = f[name]
        if T not in g["energy"]:
            raise KeyError(f"{path}: no {T} data (have {sorted(g['energy'])})")
        energy = np.array(g["energy"][T], dtype=np.float64)
        if g["energy"][T].attrs.keys():
            raise ValueError(f"{path}: unexpected attributes on energy/{T}")

        reactions = {}
        for key in g["reactions"]:
            rg = g["reactions"][key]
            mt = int(rg.attrs["mt"])
            if T not in rg:
                raise KeyError(f"{path}: reaction {mt} has no {T} data")
            ds = rg[T]["xs"]
            extra = set(ds.attrs) - {"threshold_idx"}
            if extra:
                raise ValueError(f"{path}: MT {mt} xs has unsupported attributes "
                                 f"{sorted(extra)} (interpolation law?)")
            thr = int(ds.attrs["threshold_idx"])
            xs = np.array(ds[()], dtype=np.float64)
            if thr < 0 or thr + xs.size != energy.size:
                raise ValueError(f"{path}: MT {mt}: threshold_idx {thr} + "
                                 f"{xs.size} values != {energy.size} grid points")
            n_prod = sum(1 for k in rg if k.startswith("product_"))
            laws = distributions and not bool(rg.attrs["redundant"])
            products = tuple(_read_product(rg[f"product_{k}"], laws) for k in range(n_prod))
            if mt in reactions:
                raise ValueError(f"{path}: MT {mt} appears twice")
            reactions[mt] = Reaction(
                mt=mt, label=_str(rg.attrs["label"]), q_value=float(rg.attrs["Q_value"]),
                center_of_mass=bool(rg.attrs["center_of_mass"]),
                redundant=bool(rg.attrs["redundant"]), threshold_idx=thr,
                xs=xs, products=products)

        return Nuclide(
            name=name, Z=int(g.attrs["Z"]), A=int(g.attrs["A"]),
            metastable=int(g.attrs["metastable"]),
            awr=float(g.attrs["atomic_weight_ratio"]), temperature=T,
            kT=float(g["kTs"][T][()]), energy=energy, reactions=reactions,
            has_urr=("urr" in g and T in g["urr"]), path=str(path), library=library)


@dataclass(frozen=True)
class Library:
    """An OpenMC HDF5 data library on disk: its root directory plus a label
    ("ENDF/B-VIII.0", "FENDL-3.2", ...) carried into results and reports."""
    root: Path
    label: str

    @classmethod
    def open(cls, path=None, label: str = "ENDF/B-VIII.0") -> "Library":
        return cls(data_dir(path), label)

    def nuclide_path(self, name: str) -> Path:
        p = self.root / "neutron" / f"{name}.h5"
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found (run scripts/fetch_data.sh)")
        return p

    def load(self, name: str, temperature=DEFAULT_TEMPERATURE,
             distributions: bool = False) -> Nuclide:
        return read_nuclide(self.nuclide_path(name), temperature, self.label,
                            distributions)
