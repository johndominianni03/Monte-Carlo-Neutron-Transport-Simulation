"""Bit-identity of the Phase 3 response tallies (plan check 4).

- Tallies on or off: problem D7 (seed 20261023) with the D7 depth mesh
  reproduces all 9 arrays of tests/reference_kin byte for byte, so the
  tallies draw no random number and change no other output.
- The tally arrays are byte-identical for a full run, split, chunked and
  reversed batch ranges, a reused output object with garbage rows, track
  recording on or off, and a fresh process with an empty Numba cache
  (NUMBA_CACHE_DIR in a new temporary directory, so every kernel is
  compiled from scratch). Small problem: D7 geometry, 12 x 200, seed
  20261032.
"""
import dataclasses
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import test_regression_kin as RK                            # noqa: E402
from mcslab import ce_materials as cm                       # noqa: E402
from mcslab.config_ce import MonoEnergetic                  # noqa: E402
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin  # noqa: E402
from mcslab.geometry import SlabGeometry                    # noqa: E402
from mcslab.nucdata import Library                          # noqa: E402
from mcslab.sources import BeamSource                       # noqa: E402

D7_BINS = (10, 20, 20)
SMALL_SEED = 20261032
TALLY_ARRAYS = ("tally", "mesh_flux")
ALL_ARRAYS = RK.ARRAYS + TALLY_ARRAYS


def _library():
    try:
        return Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def small_config(library, **kw):
    flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    geom = SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()])
    base = dict(n_batches=12, histories_per_batch=200, seed=SMALL_SEED,
                energy_edges=RK.EDGES, depth_bins=D7_BINS)
    base.update(kw)
    return KinRunConfig(geom, BeamSource(), MonoEnergetic(14.1e6), library, **base)


def run_quiet(cfg, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg, **kw)


def arrays(res, names=ALL_ARRAYS):
    return {a: getattr(res, a) for a in names}


def assert_identical(a, b, names=ALL_ARRAYS):
    for name in names:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert x.tobytes() == y.tobytes(), name


def test_d7_transport_unchanged_with_tallies_on():
    lib = _library()
    cfg = dataclasses.replace(RK.reference_problems(lib)["kin_d7"], depth_bins=D7_BINS)
    reference, _ = RK.load_reference()
    res = run_quiet(cfg)
    for a in RK.ARRAYS:
        ref = reference[f"kin_d7__{a}"]
        cur = getattr(res, a)
        assert ref.dtype == cur.dtype and ref.shape == cur.shape, a
        assert ref.tobytes() == cur.tobytes(), f"{a} changed with tallies on"
    assert res.tally.shape == (20, 50, 14, 6, 4) and res.mesh_flux.shape == (20, 50, 4)
    assert res.tally.any() and res.balance()["residual"] == 0.0


def test_tallies_off_gives_empty_arrays_and_same_transport():
    lib = _library()
    on = run_quiet(small_config(lib))
    off = run_quiet(small_config(lib, depth_bins=None))
    assert off.tally.shape[1] == 0 and off.mesh_flux.shape[1] == 0
    assert_identical(on, off, RK.ARRAYS)


def test_split_chunked_and_reversed_batches_bit_identical():
    lib = _library()
    cfg = small_config(lib)
    full = run_quiet(cfg)
    for ranges in ([(0, 5), (5, 6), (6, 12)],
                   [(0, 4), (4, 8), (8, 12)],
                   [(b, b + 1) for b in reversed(range(12))]):
        out = None
        for rng in ranges:
            out = run_quiet(cfg, batch_range=rng, out=out)
        assert_identical(full, out)


def test_stale_rows_are_overwritten():
    lib = _library()
    cfg = small_config(lib)
    full = run_quiet(cfg)
    dirty = run_quiet(cfg, batch_range=(0, 0))
    dirty.tally[:] = 7.0
    dirty.mesh_flux[:] = -3.0
    run_quiet(cfg, out=dirty)
    assert_identical(full, dirty)


def test_track_recording_changes_no_tally():
    lib = _library()
    plain = run_quiet(small_config(lib))
    rec = run_quiet(small_config(lib, n_track=60, track_capacity=50))
    assert rec.tracks.truncated.any()        # truncation exercised too
    assert_identical(plain, rec)


_CHILD = r"""
import sys, numpy as np
sys.path.insert(0, {repo!r}); sys.path.insert(0, {here!r})
import test_tally_reproducibility as M
res = M.run_quiet(M.small_config(M.Library.open()))
np.savez({out!r}, **M.arrays(res))
"""


def test_fresh_numba_cache_bit_identical(tmp_path):
    lib = _library()
    res = run_quiet(small_config(lib))
    cache = tmp_path / "numba_cache"
    cache.mkdir()
    out = tmp_path / "child.npz"
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache))
    proc = subprocess.run([sys.executable, "-c",
                           _CHILD.format(repo=REPO, here=HERE, out=str(out))],
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    compiled = [p for p in cache.rglob("*") if p.suffix in (".nbi", ".nbc")]
    assert compiled, "the child did not compile into the fresh cache"
    with np.load(out) as child:
        for name, arr in arrays(res).items():
            assert child[name].dtype == arr.dtype and child[name].shape == arr.shape, name
            assert child[name].tobytes() == arr.tobytes(), name
