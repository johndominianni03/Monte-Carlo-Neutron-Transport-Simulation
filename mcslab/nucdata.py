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


@dataclass(frozen=True)
class Product:
    """Outgoing particle of a reaction (metadata only; no sampling yet)."""
    particle: str
    emission_mode: str
    yield_type: str
    yield_constant: Optional[float]   # Polynomial yield of degree 0, else None
    distributions: Tuple[str, ...]    # e.g. "uncorrelated(angle=tabular, energy=level)"


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


def _read_product(group) -> Product:
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
    return Product(_str(group.attrs["particle"]), _str(group.attrs["emission_mode"]),
                   ytype, yconst, dists)


def read_nuclide(path: Union[str, Path], temperature=DEFAULT_TEMPERATURE,
                 library: str = "") -> Nuclide:
    """Read one OpenMC HDF5 incident-neutron file at one temperature."""
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
            products = tuple(_read_product(rg[f"product_{k}"]) for k in range(n_prod))
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

    def load(self, name: str, temperature=DEFAULT_TEMPERATURE) -> Nuclide:
        return read_nuclide(self.nuclide_path(name), temperature, self.label)
