"""Bit-identity of the kinematic CE kernel under different ways of running
the batch loop, including histories with secondary neutrons (the bank
continues the parent's random-number stream, so a family depends only on
(master seed, history id)). Seed 20261015 was fixed in
docs/phase2b_plan.md. Synthetic data only, so this runs without MCSLAB_DATA.
"""
import warnings

import pytest

from mcslab import synthetic as S
from mcslab import tallies as T
from mcslab.ce_materials import VOID_CE
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.sources import BeamSource, IsotropicPlaneSource

SEED = 20261015
ARRAYS = ("region_sums", "surface_sums", "diagnostics", "spectrum", "cutoff_weight",
          "chan_events", "chan_created")


def _config(source, seed=SEED, **kw):
    x = S.nuclide("X", awr=9.0, elastic_b=2.0, capture_b=0.2,
                  reactions=[(51, -1.0e6, 0.5, 1), (16, -2.0e6, 0.4, 2),
                             (17, -4.0e6, 0.3, 3)])
    y = S.nuclide("Y", awr=56.0, elastic_b=3.0, capture_b=0.05,
                  reactions=[(51, -0.8e6, 1.0, 1)])
    lib = S.SyntheticLibrary({"X": x, "Y": y})
    a = S.SyntheticMaterial("a", (("X", 0.08), ("Y", 0.02)))
    b = S.SyntheticMaterial("b", (("Y", 0.09),))
    geom = SlabGeometry([0.0, 3.0, 4.0, 9.0, 12.0], [a, VOID_CE, b, a])
    kw.setdefault("energy_cutoff", 2.0e4)
    kw.setdefault("energy_edges", (2.0e4, 1.0e5, 1.0e6, 5.0e6, 1.5e7))
    return KinRunConfig(geom, source, MonoEnergetic(14.1e6), lib, n_batches=12,
                        histories_per_batch=300, seed=seed, **kw)


def _run(cfg, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg, **kw)


def _assert_bit_identical(r1, r2):
    for name in ARRAYS:
        a, b = getattr(r1, name), getattr(r2, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


def _split_run(cfg, ranges):
    out = None
    for rng in ranges:
        out = _run(cfg, batch_range=rng, out=out)
    return out


def test_families_have_secondaries_and_balance():
    """Guards the premise of this file: secondaries really are created."""
    res = _run(_config(IsotropicPlaneSource(1.5)))
    assert res.count(T.K_CREATED) > 0 and res.diagnostics[:, T.K_MAX_BANK].max() >= 2
    assert res.balance()["residual"] == 0.0


def test_repeat_runs_bit_identical():
    for src in (BeamSource(), IsotropicPlaneSource(1.0)):
        cfg = _config(src)
        _assert_bit_identical(_run(cfg), _run(cfg))


def test_split_and_reordered_batches_bit_identical():
    for src in (BeamSource(), IsotropicPlaneSource(10.5)):
        cfg = _config(src)
        full = _run(cfg)
        _assert_bit_identical(full, _split_run(cfg, [(0, 5), (5, 6), (6, 12)]))
        _assert_bit_identical(full, _split_run(cfg, [(9, 12), (0, 3), (3, 9)]))
        _assert_bit_identical(full, _split_run(cfg, [(b, b + 1) for b in reversed(range(12))]))


def test_stale_output_rows_are_overwritten():
    cfg = _config(BeamSource())
    full = _run(cfg)
    dirty = _run(cfg)
    for name in ARRAYS:
        getattr(dirty, name)[:] = 7
    _run(cfg, out=dirty)
    _assert_bit_identical(full, dirty)


def test_bank_capacity_does_not_change_results():
    cfg = _config(BeamSource())
    full = _run(cfg)
    need = int(full.diagnostics[:, T.K_MAX_BANK].max())
    _assert_bit_identical(full, _run(_config(BeamSource(), bank_capacity=need)))


def test_bank_overflow_raises_instead_of_dropping():
    with pytest.raises(RuntimeError, match="bank overflow"):
        _run(_config(BeamSource(), bank_capacity=1))


def test_different_seed_changes_results():
    a = _run(_config(BeamSource()))
    b = _run(_config(BeamSource(), seed=SEED + 1))
    assert a.region_sums.tobytes() != b.region_sums.tobytes()
