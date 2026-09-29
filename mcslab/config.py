"""Run configuration and the Python driver around the transport kernels."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from . import tallies as T
from .geometry import SlabGeometry
from .rng import STRIDE, validate_seed
from .sources import BeamSource, IsotropicPlaneSource
from .transport import run_batches

Source = Union[BeamSource, IsotropicPlaneSource]


@dataclass(frozen=True)
class RunConfig:
    geometry: SlabGeometry
    source: Source
    n_batches: int = 100
    histories_per_batch: int = 10_000
    seed: int = 1

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValueError("n_batches must be >= 2 for batch statistics")
        if self.histories_per_batch < 1:
            raise ValueError("histories_per_batch must be >= 1")
        validate_seed(self.seed)

    @property
    def n_histories(self) -> int:
        return self.n_batches * self.histories_per_batch


@dataclass
class Results:
    """Raw per-batch tallies plus normalised views.

    Every *_batches property returns per-source-particle estimates with the
    batch axis first, ready for tallies.batch_stats.
    """
    config: RunConfig
    region_sums: np.ndarray      # (B, N_REGION_SCORES, n_regions)
    surface_sums: np.ndarray     # (B, 2, n_regions + 1)
    diagnostics: np.ndarray      # (B, N_DIAG) int64

    @property
    def _n(self):
        return float(self.config.histories_per_batch)

    def region_batches(self, score: int) -> np.ndarray:
        return self.region_sums[:, score, :] / self._n

    @property
    def collisions(self):
        return self.region_batches(T.COLLISION)

    @property
    def absorptions(self):
        return self.region_batches(T.ABSORPTION)

    @property
    def track_length(self):
        return self.region_batches(T.TRACK_LENGTH)

    @property
    def flux_tl(self):
        """Track-length estimator of the region-averaged scalar flux
        (per source particle per cm^2 of slab face)."""
        return self.region_batches(T.TRACK_LENGTH) / self.config.geometry.widths

    @property
    def flux_coll(self):
        """Collision estimator of the region-averaged scalar flux. Identically
        zero in void regions, where it is not a valid estimator."""
        return self.region_batches(T.COLL_ESTIMATOR) / self.config.geometry.widths

    @property
    def current_neg(self):
        """(B, n_regions + 1) crossings per source particle with mu < 0."""
        return self.surface_sums[:, T.NEG, :] / self._n

    @property
    def current_pos(self):
        """(B, n_regions + 1) crossings per source particle with mu > 0."""
        return self.surface_sums[:, T.POS, :] / self._n

    @property
    def leakage_left(self):
        return self.current_neg[:, 0]

    @property
    def leakage_right(self):
        return self.current_pos[:, -1]

    @property
    def max_draws(self) -> int:
        return int(self.diagnostics[:, T.DIAG_MAX_DRAWS].max())

    @property
    def lost(self) -> int:
        return int(self.diagnostics[:, T.DIAG_LOST].sum())

    @staticmethod
    def stats(batches) -> Tuple[np.ndarray, np.ndarray]:
        return T.batch_stats(batches)


def run(config: RunConfig, batch_range: Optional[Tuple[int, int]] = None,
        out: Optional[Results] = None) -> Results:
    """Run the problem. `batch_range=(b0, b1)` runs only those batches into
    `out` (allocated if None); the rows it writes are bit-identical to the
    same rows from a full run."""
    g = config.geometry
    bounds, mat_of_region, sig_t, _sig_s, sig_a = g.pack()
    src_type, src_x, src_region = config.source.pack(g)

    if out is None:
        out = Results(config, *T.allocate(config.n_batches, g.n_regions))
    b0, b1 = (0, config.n_batches) if batch_range is None else batch_range
    if not 0 <= b0 <= b1 <= config.n_batches:
        raise ValueError(f"bad batch range {batch_range}")

    run_batches(b0, b1, config.histories_per_batch, validate_seed(config.seed),
                bounds, mat_of_region, sig_t, sig_a,
                src_type, src_x, src_region,
                out.region_sums, out.surface_sums, out.diagnostics)

    if b1 > b0:
        worst = int(out.diagnostics[b0:b1, T.DIAG_MAX_DRAWS].max())
        if worst > STRIDE:
            warnings.warn(f"a history used {worst} random numbers > STRIDE={STRIDE}; "
                          "streams of neighbouring histories overlap")
    return out
