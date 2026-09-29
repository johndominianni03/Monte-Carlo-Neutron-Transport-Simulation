"""Bit-identity of results under different ways of splitting the batch loop.

This is the property that lets run_batches' loop become prange later: a
batch's tally row must not depend on which call (or thread) produced it, nor
on the order batches are run in.
"""
import numpy as np

from mcslab.config import Results, RunConfig, run
from mcslab.geometry import SlabGeometry
from mcslab.materials import VOID, Material
from mcslab.sources import BeamSource, IsotropicPlaneSource
from mcslab import tallies as T


def _config(source):
    a = Material("a", sigma_s=0.6, sigma_a=0.4)
    s = Material("s", sigma_s=1.0, sigma_a=0.02)
    geom = SlabGeometry([0.0, 0.7, 1.0, 3.2, 4.0], [a, VOID, s, a])
    return RunConfig(geom, source, n_batches=12, histories_per_batch=500, seed=77)


def _assert_bit_identical(r1, r2):
    for name in ("region_sums", "surface_sums", "diagnostics"):
        a, b = getattr(r1, name), getattr(r2, name)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), name


def _split_run(cfg, ranges):
    out = Results(cfg, *T.allocate(cfg.n_batches, cfg.geometry.n_regions))
    for rng in ranges:
        run(cfg, batch_range=rng, out=out)
    return out


def test_repeat_runs_bit_identical():
    for src in (BeamSource(), IsotropicPlaneSource(1.0)):
        cfg = _config(src)
        _assert_bit_identical(run(cfg), run(cfg))


def test_split_and_reordered_batches_bit_identical():
    for src in (BeamSource(), IsotropicPlaneSource(2.1)):
        cfg = _config(src)
        full = run(cfg)
        _assert_bit_identical(full, _split_run(cfg, [(0, 5), (5, 6), (6, 12)]))
        _assert_bit_identical(full, _split_run(cfg, [(9, 12), (0, 3), (3, 9)]))
        _assert_bit_identical(full, _split_run(cfg, [(b, b + 1) for b in reversed(range(12))]))


def test_stale_output_rows_are_overwritten():
    """A batch zeroes its own row first, so garbage in `out` cannot leak in."""
    cfg = _config(BeamSource())
    full = run(cfg)
    dirty = Results(cfg, *T.allocate(cfg.n_batches, cfg.geometry.n_regions))
    dirty.region_sums[:] = 123.0
    dirty.surface_sums[:] = -7.0
    run(cfg, out=dirty)
    _assert_bit_identical(full, dirty)


def test_different_seed_changes_results():
    cfg = _config(BeamSource())
    other = RunConfig(cfg.geometry, cfg.source, cfg.n_batches, cfg.histories_per_batch, seed=78)
    assert run(cfg).region_sums.tobytes() != run(other).region_sums.tobytes()
