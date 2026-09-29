"""Run configuration and Python driver for continuous-energy problems.

Mirrors config.py: Python objects (materials, library, spectra) stop here.
The kernels receive only the plain arrays built by xs.pack.
"""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from . import tallies as T
from .config import Results
from .geometry import SlabGeometry
from .nucdata import DEFAULT_TEMPERATURE, Library, temperature_key
from .rng import STRIDE, validate_seed
from .sources import BeamSource, IsotropicPlaneSource
from .transport_ce import SRC_E_LOG_UNIFORM, SRC_E_MONO, run_batches_ce
from .xs import pack

Source = Union[BeamSource, IsotropicPlaneSource]


@dataclass(frozen=True)
class MonoEnergetic:
    energy_eV: float

    def pack(self):
        return SRC_E_MONO, float(self.energy_eV), float(self.energy_eV)


@dataclass(frozen=True)
class LogUniform:
    """Source energies with density proportional to 1/E on [e_lo, e_hi)."""
    e_lo: float
    e_hi: float

    def __post_init__(self):
        if not 0.0 < self.e_lo < self.e_hi:
            raise ValueError("need 0 < e_lo < e_hi")

    def pack(self):
        return SRC_E_LOG_UNIFORM, float(self.e_lo), float(self.e_hi)


Spectrum = Union[MonoEnergetic, LogUniform]


@dataclass(frozen=True)
class CERunConfig:
    """geometry.region_materials are ce_materials.CEMaterial objects."""
    geometry: SlabGeometry
    source: Source
    spectrum: Spectrum
    library: Library
    n_batches: int = 100
    histories_per_batch: int = 10_000
    seed: int = 1
    temperature: str = DEFAULT_TEMPERATURE

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValueError("n_batches must be >= 2 for batch statistics")
        if self.histories_per_batch < 1:
            raise ValueError("histories_per_batch must be >= 1")
        validate_seed(self.seed)

    @property
    def n_histories(self) -> int:
        return self.n_batches * self.histories_per_batch


@functools.lru_cache(maxsize=None)
def _load(library: Library, name: str, temperature: str):
    return library.load(name, temperature)


def pack_problem(config: CERunConfig):
    """-> (bounds, mat_of_region, PackedXS, {nuclide: Nuclide})."""
    g = config.geometry
    unique, mat_of_region = [], []
    for m in g.region_materials:
        if m not in unique:
            unique.append(m)
        mat_of_region.append(unique.index(m))
    T_key = temperature_key(config.temperature)
    names = sorted({n for m in unique for n in m.nuclide_names})
    nuclides = {n: _load(config.library, n, T_key) for n in names}
    awr = {n: nuc.awr for n, nuc in nuclides.items()}
    packed = pack([m.number_densities(awr) for m in unique], nuclides)
    return (np.asarray(g.bounds, dtype=np.float64),
            np.asarray(mat_of_region, dtype=np.int64), packed, nuclides)


def run_ce(config: CERunConfig, batch_range: Optional[Tuple[int, int]] = None,
           out: Optional[Results] = None) -> Results:
    """Run a continuous-energy problem (first collision ends each history).
    `batch_range` behaves as in config.run."""
    g = config.geometry
    bounds, mat_of_region, p, _ = pack_problem(config)
    src_type, src_x, src_region = config.source.pack(g)
    e_type, e_lo, e_hi = config.spectrum.pack()
    if not (p.e_min <= e_lo and e_hi <= p.e_max):
        raise ValueError(f"source energies [{e_lo}, {e_hi}] eV fall outside the "
                         f"common data range [{p.e_min}, {p.e_max}] eV")

    if out is None:
        out = Results(config, *T.allocate(config.n_batches, g.n_regions))
    b0, b1 = (0, config.n_batches) if batch_range is None else batch_range
    if not 0 <= b0 <= b1 <= config.n_batches:
        raise ValueError(f"bad batch range {batch_range}")

    run_batches_ce(b0, b1, config.histories_per_batch, validate_seed(config.seed),
                   bounds, mat_of_region,
                   p.egrid, p.e_off, p.tot, p.absn, p.m_off, p.mat_nuc, p.mat_dens,
                   src_type, src_x, src_region, e_type, e_lo, e_hi,
                   out.region_sums, out.surface_sums, out.diagnostics)

    if b1 > b0:
        worst = int(out.diagnostics[b0:b1, T.DIAG_MAX_DRAWS].max())
        if worst > STRIDE:
            warnings.warn(f"a history used {worst} random numbers > STRIDE={STRIDE}; "
                          "streams of neighbouring histories overlap")
    return out

