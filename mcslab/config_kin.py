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
from .depth_mesh import DepthMesh, make_mesh
from .geometry import SlabGeometry
from .nucdata import DEFAULT_TEMPERATURE_K, TEMPERATURE_TOLERANCE, temperature_key
from .responses import RESPONSE_MTS, pack_responses
from .rng import STRIDE, validate_seed
from .sources import BeamSource, IsotropicPlaneSource
from .tracks import TF_NCOL, TI_NCOL, Tracks
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
    temperature: K, for materials without their own `temperature`
        (OpenMC's default 293.6 K). Each material's temperature selects the
        data temperature of its nuclides by OpenMC's NEAREST rule within
        temperature_tolerance (nucdata.nearest_temperature); that group's
        kT is used for free-gas scattering. A string (e.g. "294K") names a
        data group directly.
    free_gas_threshold: target at rest iff E >= threshold * kT and awr > 1
        (OpenMC's free_gas_threshold, default 400).
    free_gas: False disables target motion everywhere (a diagnostic switch,
        mcslab only; it reproduces Part 1 exactly).
    n_track: record the events of histories 0 .. n_track - 1 (see
        mcslab/tracks.py). Recording draws no random number and touches no
        tally.
    track_capacity: events kept per recorded history (more are counted as
        truncated).
    depth_bins: uniform depth bins per region for the response tallies
        (Phase 3; mcslab/depth_mesh.py, mcslab/responses.py). None (the
        default) turns them off. They draw no random number and change no
        other output.
    reflect_left: specular reflection at the left boundary (the plasma
        side; Phase 3 Part B, approved D17) instead of vacuum. No random
        number is drawn; the reflected weight is in
        KinResults.reflected_weight. Default False (vacuum).
    """
    geometry: SlabGeometry
    source: Source
    spectrum: Spectrum
    library: object
    n_batches: int = 100
    histories_per_batch: int = 10_000
    seed: int = 1
    temperature: Union[float, str] = DEFAULT_TEMPERATURE_K
    energy_cutoff: Optional[float] = None
    energy_edges: Optional[Tuple[float, ...]] = None
    bank_capacity: int = 10_000     # OpenMC's default max_secondaries
    temperature_tolerance: float = TEMPERATURE_TOLERANCE
    free_gas_threshold: float = 400.0
    free_gas: bool = True
    n_track: int = 0
    track_capacity: int = 4000
    depth_bins: Optional[Tuple[int, ...]] = None
    reflect_left: bool = False

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValueError("n_batches must be >= 2 for batch statistics")
        if self.histories_per_batch < 1:
            raise ValueError("histories_per_batch must be >= 1")
        if self.bank_capacity < 1:
            raise ValueError("bank_capacity must be >= 1")
        if self.n_track < 0 or self.track_capacity < 1:
            raise ValueError("n_track must be >= 0 and track_capacity >= 1")
        if self.depth_bins is not None and (
                len(self.depth_bins) != self.geometry.n_regions
                or any(int(n) < 1 for n in self.depth_bins)):
            raise ValueError("depth_bins needs one count >= 1 per region")
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
    zero_yield_weight: np.ndarray = None  # (B, n_regions) ended by multiplicity 0
    chan_zero: np.ndarray = None         # (B, n_channels) int64 zero-yield events
    tracks: Optional[Tracks] = None      # recorded events, if n_track > 0
    channel_labels: Tuple[Tuple[str, int], ...] = ()
    energy_edges: np.ndarray = None
    energy_cutoff: float = 0.0
    # Phase 3 response tallies (tallies.py layout); zero depth bins when off
    tally: np.ndarray = None             # (B, n_bins, n_nuclides + 1, N_RESP, N_EST)
    mesh_flux: np.ndarray = None         # (B, n_bins, N_EST)
    mesh: Optional[DepthMesh] = None
    tally_nuclides: Tuple[str, ...] = ()  # packed nuclide labels, then "total"
    response_present: np.ndarray = None  # (n_nuclides, N_RESP) bool: data exist
    region_atom_density: np.ndarray = None  # (n_regions,) atoms / (b cm)
    reflected_weight: np.ndarray = None  # (B,) weight reflected at the left boundary

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
        zero = float(self.zero_yield_weight.sum())
        lost = float(self.count(T.K_LOST))
        return {"source": src, "created": created, "absorbed": absorbed,
                "leak_left": left, "leak_right": right, "cutoff": cut, "zero_yield": zero,
                "lost": lost,
                "residual": src + created - (absorbed + left + right + cut + zero + lost)}

    def channel_counts(self, nuclide: str, mt: int) -> Tuple[int, int]:
        """(events, secondaries created) summed over batches for one channel."""
        c = self.channel_labels.index((nuclide, mt))
        return int(self.chan_events[:, c].sum()), int(self.chan_created[:, c].sum())

    def tally_batches(self, response: int, estimator: int,
                      nuclide: Optional[str] = None) -> np.ndarray:
        """(B, n_bins) per source particle: one response (tallies.R_*) and
        estimator (tallies.EST_*), for one nuclide label or, with None, the
        material total. Track length: integrated over the bin width (e.g.
        eV of heating in the bin per source neutron per cm^2 of wall)."""
        k = len(self.tally_nuclides) - 1 if nuclide is None else \
            self.tally_nuclides.index(nuclide)
        return self.tally[:, :, k, response, estimator] / self._n

    def mesh_flux_batches(self, estimator: int) -> np.ndarray:
        """(B, n_bins) per source particle, integrated over the bin width
        (cm); divide by mesh.bin_width for the bin-averaged flux."""
        return self.mesh_flux[:, :, estimator] / self._n

    def channel_zero(self, nuclide: str, mt: int) -> int:
        """Zero-yield events of one channel, summed over batches."""
        return int(self.chan_zero[:, self.channel_labels.index((nuclide, mt))].sum())


_CACHE = {}


def _load(library, name: str, temperature: str):
    key = (id(library), name, temperature)
    if key not in _CACHE:
        _CACHE[key] = (library, library.load(name, temperature, distributions=True))
    return _CACHE[key][1]


def data_temperature(config: KinRunConfig, material, name: str) -> str:
    """Data temperature group of nuclide `name` in `material`: the
    material's own temperature, else the run's, by OpenMC's NEAREST rule (a
    string names the group directly)."""
    T = getattr(material, "temperature", None)
    T = config.temperature if T is None else T
    if isinstance(T, str):
        return temperature_key(T)
    return config.library.select_temperature(name, float(T), config.temperature_tolerance)


def pack_problem(config: KinRunConfig):
    """-> (bounds, mat_of_region, PackedXS, PackedPhysics, {label: Nuclide}).

    Each (nuclide, data temperature) pair is packed as its own entry, with
    its own grid, cross sections and kT. Its label is the nuclide name, or
    "name@group" (e.g. "Li7@900K") when the problem uses that nuclide at
    more than one data temperature."""
    g = config.geometry
    unique, mat_of_region = [], []
    for m in g.region_materials:
        if m not in unique:
            unique.append(m)
        mat_of_region.append(unique.index(m))
    keys = [[(n, data_temperature(config, m, n)) for n in m.nuclide_names] for m in unique]
    groups = {}
    for ent in keys:
        for n, key in ent:
            groups.setdefault(n, set()).add(key)

    def label(n, key):
        return n if len(groups[n]) == 1 else f"{n}@{key}"

    nuclides = {}
    for ent in keys:
        for n, key in ent:
            if label(n, key) not in nuclides:
                nuclides[label(n, key)] = _load(config.library, n, key)
    compositions = []
    for m, ent in zip(unique, keys):
        lab = dict((n, label(n, key)) for n, key in ent)
        awr = {n: nuclides[lab[n]].awr for n in lab}
        compositions.append(tuple((lab[n], d) for n, d in m.number_densities(awr)))
    packed = pack(compositions, nuclides)
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


def _refuse_unsupported(phys):
    """Refuse any problem whose nuclides have a channel with a law mcslab
    does not implement, whether or not the source can reach it: with target
    motion, energies are no longer monotone, so reachability cannot be
    bounded by the source energy."""
    if phys.unsupported:
        lines = "\n  ".join(reason for _, reason in phys.unsupported)
        raise NotImplementedError(
            f"the problem's nuclides have reactions whose laws mcslab does not "
            f"implement:\n  {lines}")


def _track_args(out):
    """(n_track, f, i, n, truncated) for the kernel; empty arrays of the
    right rank when nothing is recorded."""
    t = out.tracks
    if t is None:
        return (0, np.zeros((0, 1, TF_NCOL)), np.zeros((0, 1, TI_NCOL), dtype=np.int64),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    return t.f.shape[0], t.f, t.i, t.n, t.truncated


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
    _refuse_unsupported(phys)
    edges = (np.array([e_cut, p.e_max], dtype=np.float64) if config.energy_edges is None
             else np.asarray(config.energy_edges, dtype=np.float64))
    if edges.ndim != 1 or edges.size < 2 or (np.diff(edges) <= 0.0).any():
        raise ValueError("energy_edges must be strictly increasing, at least 2 values")

    n_ch = phys.ch_int.shape[0]
    n_nuc = len(p.nuclide_names)
    if config.depth_bins is not None:
        mesh = make_mesh(g.bounds, config.depth_bins)
        resp = pack_responses(p.nuclide_names, nuclides)
        rxs, roff, rthr = resp.rxs, resp.roff, resp.rthr
    else:
        mesh = None
        rxs = np.zeros(1, dtype=np.float64)
        roff = np.zeros((n_nuc, len(RESPONSE_MTS)), dtype=np.int64)
        rthr = np.zeros((n_nuc, len(RESPONSE_MTS)), dtype=np.int64)
    m_edges = mesh.edges if mesh is not None else np.zeros(0, dtype=np.float64)
    m_eoff = mesh.eoff if mesh is not None else np.zeros(1, dtype=np.int64)
    m_boff = mesh.boff if mesh is not None else np.zeros(1, dtype=np.int64)
    if out is None:
        arrays = T.allocate_kin(config.n_batches, g.n_regions, edges.size - 1, n_ch)
        tally, mesh_flux = T.allocate_tally(config.n_batches, mesh.n_bins if mesh is not None else 0,
                                            n_nuc)
        dens = np.array([p.mat_dens[p.m_off[m]:p.m_off[m + 1]].sum() for m in mat_of_region])
        n_track = min(int(config.n_track), config.n_histories)
        cap = int(config.track_capacity) if n_track else 1
        tracks = Tracks(np.zeros((n_track, cap, TF_NCOL)),
                        np.zeros((n_track, cap, TI_NCOL), dtype=np.int64),
                        np.zeros(n_track, dtype=np.int64), np.zeros(n_track, dtype=np.int64),
                        bounds.copy(), tuple(m.name for m in g.region_materials), e_cut)
        out = KinResults(config, arrays[0], arrays[1], arrays[2], spectrum=arrays[3],
                         cutoff_weight=arrays[4], chan_events=arrays[5],
                         chan_created=arrays[6], zero_yield_weight=arrays[7],
                         chan_zero=arrays[8], channel_labels=phys.labels,
                         energy_edges=edges, energy_cutoff=e_cut,
                         tracks=tracks if n_track else None,
                         tally=tally, mesh_flux=mesh_flux, mesh=mesh,
                         tally_nuclides=tuple(p.nuclide_names) + ("total",),
                         response_present=(resp.present if mesh is not None else
                                           np.zeros((n_nuc, T.N_RESP), dtype=bool)),
                         region_atom_density=dens,
                         reflected_weight=np.zeros(config.n_batches, dtype=np.float64))
    b0, b1 = (0, config.n_batches) if batch_range is None else batch_range
    if not 0 <= b0 <= b1 <= config.n_batches:
        raise ValueError(f"bad batch range {batch_range}")

    run_batches_kin(b0, b1, config.histories_per_batch, validate_seed(config.seed),
                    bounds, mat_of_region,
                    p.egrid, p.e_off, p.tot, p.absn, p.m_off, p.mat_nuc, p.mat_dens,
                    phys.nuc_awr, phys.nuc_kT, float(config.free_gas_threshold),
                    bool(config.free_gas),
                    phys.ch_off, phys.ch_int, phys.ch_q, phys.chxs,
                    phys.ip, phys.fp,
                    src_type, src_x, src_region, e_type, e_lo, e_hi,
                    e_cut, edges, int(config.bank_capacity),
                    out.region_sums, out.surface_sums, out.spectrum, out.cutoff_weight,
                    out.diagnostics, out.chan_events, out.chan_created,
                    out.zero_yield_weight, out.chan_zero,
                    *_track_args(out),
                    rxs, roff, rthr, m_edges, m_eoff, m_boff, out.tally, out.mesh_flux,
                    bool(config.reflect_left), out.reflected_weight)

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
