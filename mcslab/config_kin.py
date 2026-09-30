"""Run configuration and Python driver for continuous-energy transport with
kinematics (Phase 2b). Python objects stop here; the kernels in
transport_kin.py receive plain arrays built by xs.pack and
collision.pack_physics.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from . import tallies as T
from .collision import pack_physics
from .config import Results
from .config_ce import LogUniform, MonoEnergetic
from .geometry import SlabGeometry
from .nucdata import DEFAULT_TEMPERATURE, temperature_key
from .rng import STRIDE, validate_seed
from .sources import BeamSource, IsotropicPlaneSource
from .transport_kin import run_batches_kin
from .xs import pack

Source = Union[BeamSource, IsotropicPlaneSource]
Spectrum = Union[MonoEnergetic, LogUniform]


class EnergyCutoffWarning(UserWarning):
    """Issued when the energy cutoff killed weight (approved deviation D5:
    never silent)."""


@dataclass(frozen=True)
class KinRunConfig:
    """A kinematic transport problem.

    geometry.region_materials: anything with `nuclide_names` and
        `number_densities(awr)`, e.g. ce_materials.CEMaterial or
        synthetic.SyntheticMaterial.
    library: nucdata.Library or synthetic.SyntheticLibrary.
    energy_cutoff: eV. None means the largest grid minimum of the nuclides
        used (1e-5 eV for ENDF/B-VIII.0), below which no lookup is possible.
    energy_edges: increasing bin edges (eV) for the spectrum tally. None
        means one bin [energy_cutoff, common grid maximum).
    """
    geometry: SlabGeometry
    source: Source
    spectrum: Spectrum
    library: object
    n_batches: int = 100
    histories_per_batch: int = 10_000
    seed: int = 1
    temperature: str = DEFAULT_TEMPERATURE
    energy_cutoff: Optional[float] = None
    energy_edges: Optional[Tuple[float, ...]] = None
    bank_capacity: int = 10_000     # OpenMC's default max_secondaries

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValueError("n_batches must be >= 2 for batch statistics")
        if self.histories_per_batch < 1:
            raise ValueError("histories_per_batch must be >= 1")
        if self.bank_capacity < 1:
            raise ValueError("bank_capacity must be >= 1")
        validate_seed(self.seed)

    @property
    def n_histories(self) -> int:
        return self.n_batches * self.histories_per_batch


@dataclass
class KinResults(Results):
    """Results plus the kinematic kernel's extra tallies. `diagnostics` is
    the counts array (tallies.K_*); its first two columns are max draws and
    lost, so Results.max_draws / .lost apply unchanged."""
    spectrum: np.ndarray = None          # (B, N_SPEC, n_regions, n_ebins)
    cutoff_weight: np.ndarray = None     # (B, n_regions)
    chan_events: np.ndarray = None       # (B, n_channels) int64
    chan_created: np.ndarray = None      # (B, n_channels) int64
    channel_labels: Tuple[Tuple[str, int], ...] = ()
    energy_edges: np.ndarray = None
    energy_cutoff: float = 0.0

    @property
    def counts(self) -> np.ndarray:
        return self.diagnostics

    def count(self, k: int) -> int:
        return int(self.diagnostics[:, k].sum())

    @property
    def cutoff(self):
        """(B, n_regions) weight killed by the energy cutoff per source particle."""
        return self.cutoff_weight / self._n

    def spectrum_batches(self, estimator: int) -> np.ndarray:
        """(B, n_regions, n_ebins) per source particle: the flux integrated
        over the region width and the energy bin (cm), not divided by either."""
        return self.spectrum[:, estimator] / self._n

    def balance(self) -> dict:
        """Exact integer neutron balance from the raw sums (weights are 1).
        Every entry is an exact integer in float64; `residual` must be 0."""
        src = float(self.count(T.K_SOURCE))
        created = float(self.count(T.K_CREATED))
        absorbed = float(self.region_sums[:, T.ABSORPTION, :].sum())
        left = float(self.surface_sums[:, T.NEG, 0].sum())
        right = float(self.surface_sums[:, T.POS, -1].sum())
        cut = float(self.cutoff_weight.sum())
        lost = float(self.count(T.K_LOST))
        return {"source": src, "created": created, "absorbed": absorbed,
                "leak_left": left, "leak_right": right, "cutoff": cut, "lost": lost,
                "residual": src + created - (absorbed + left + right + cut + lost)}

    def channel_counts(self, nuclide: str, mt: int) -> Tuple[int, int]:
        """(events, secondaries created) summed over batches for one channel."""
        c = self.channel_labels.index((nuclide, mt))
        return int(self.chan_events[:, c].sum()), int(self.chan_created[:, c].sum())


_CACHE = {}


def _load(library, name: str, temperature: str):
    key = (id(library), name, temperature)
    if key not in _CACHE:
        _CACHE[key] = (library, library.load(name, temperature, distributions=True))
    return _CACHE[key][1]


def pack_problem(config: KinRunConfig):
    """-> (bounds, mat_of_region, PackedXS, PackedPhysics, {name: Nuclide})."""
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
    phys = pack_physics(packed.nuclide_names, nuclides)
    return (np.asarray(g.bounds, dtype=np.float64),
            np.asarray(mat_of_region, dtype=np.int64), packed, phys, nuclides)


def _energy_cutoff(config, nuclides) -> float:
    floor = max((float(nuc.energy[0]) for nuc in nuclides.values()), default=0.0)
    if config.energy_cutoff is None:
        return floor
    e_cut = float(config.energy_cutoff)
    if not e_cut >= floor:
        raise ValueError(f"energy_cutoff {e_cut} eV is below the data grid minimum "
                         f"{floor} eV (lookups there are refused)")
    return e_cut


def _refuse_unsupported(phys, e_hi):
    """Part 1 energies never increase (target at rest), so a channel whose xs
    is positive only above the highest source energy can never be sampled."""
    reach = [(reason, phys.first_positive[c]) for c, reason in phys.unsupported
             if phys.first_positive[c] < e_hi]
    if reach:
        lines = "\n  ".join(f"{r} (xs > 0 above {e:.6g} eV)" for r, e in reach)
        raise NotImplementedError(
            f"the source reaches reactions whose laws are not implemented yet "
            f"(max source energy {e_hi:.6g} eV):\n  {lines}")


def run_kin(config: KinRunConfig, batch_range: Optional[Tuple[int, int]] = None,
            out: Optional[KinResults] = None) -> KinResults:
    """Run a kinematic CE problem. `batch_range=(b0, b1)` runs only those
    batches into `out`; the rows it writes are bit-identical to the same rows
    from a full run."""
    g = config.geometry
    bounds, mat_of_region, p, phys, nuclides = pack_problem(config)
    src_type, src_x, src_region = config.source.pack(g)
    e_type, e_lo, e_hi = config.spectrum.pack()
    if not (p.e_min <= e_lo and e_hi <= p.e_max):
        raise ValueError(f"source energies [{e_lo}, {e_hi}] eV fall outside the "
                         f"common data range [{p.e_min}, {p.e_max}] eV")
    e_cut = _energy_cutoff(config, nuclides)
    _refuse_unsupported(phys, e_hi)
    edges = (np.array([e_cut, p.e_max], dtype=np.float64) if config.energy_edges is None
             else np.asarray(config.energy_edges, dtype=np.float64))
    if edges.ndim != 1 or edges.size < 2 or (np.diff(edges) <= 0.0).any():
        raise ValueError("energy_edges must be strictly increasing, at least 2 values")

    n_ch = phys.ch_int.shape[0]
    if out is None:
        arrays = T.allocate_kin(config.n_batches, g.n_regions, edges.size - 1, n_ch)
        out = KinResults(config, arrays[0], arrays[1], arrays[2], spectrum=arrays[3],
                         cutoff_weight=arrays[4], chan_events=arrays[5],
                         chan_created=arrays[6], channel_labels=phys.labels,
                         energy_edges=edges, energy_cutoff=e_cut)
    b0, b1 = (0, config.n_batches) if batch_range is None else batch_range
    if not 0 <= b0 <= b1 <= config.n_batches:
        raise ValueError(f"bad batch range {batch_range}")

    run_batches_kin(b0, b1, config.histories_per_batch, validate_seed(config.seed),
                    bounds, mat_of_region,
                    p.egrid, p.e_off, p.tot, p.absn, p.m_off, p.mat_nuc, p.mat_dens,
                    phys.nuc_awr, phys.ch_off, phys.ch_int, phys.ch_q, phys.chxs,
                    phys.ip, phys.fp,
                    src_type, src_x, src_region, e_type, e_lo, e_hi,
                    e_cut, edges, int(config.bank_capacity),
                    out.region_sums, out.surface_sums, out.spectrum, out.cutoff_weight,
                    out.diagnostics, out.chan_events, out.chan_created)

    if b1 > b0:
        worst = int(out.diagnostics[b0:b1, T.K_MAX_DRAWS].max())
        if worst > STRIDE:
            warnings.warn(f"a history family used {worst} random numbers > "
                          f"STRIDE={STRIDE}; streams of neighbouring histories overlap")
        killed = float(out.cutoff_weight[b0:b1].sum())
        if killed > 0.0:
            n = (b1 - b0) * config.histories_per_batch
            warnings.warn(f"energy cutoff {e_cut:.6g} eV killed weight {killed:.17g} "
                          f"({killed / n:.6g} per source particle); see "
                          "KinResults.cutoff", EnergyCutoffWarning)
    return out
