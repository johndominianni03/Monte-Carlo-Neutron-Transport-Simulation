"""Track recording (Phase 2b Part 2): recording consumes no random number
and touches no tally, and the recorded tracks are self-consistent.

Seed 20261024 was fixed in docs/phase2b_plan.md before this test existed.
Every check here is exact (bit identity, integer identities) or round-off
with a stated bound; there is no statistical check.

The problem is D7 (tests/test_regression_kin.py) cut to 4 x 500 histories,
all of them recorded. What is covered:
  - recording on vs off, and with a slot range so small that it truncates:
    every tally and count array (max draws included) is byte-identical;
  - the recorded tracks do not depend on how the batches are split;
  - each particle starts with a source / born event and ends with exactly
    one terminal event; time never decreases along it; x stays in the
    slab; E is constant along each flight; and x(t) follows the flight
    (|dx - u v(E) dt| <= 1e-8 cm);
  - summed over all recorded histories, the events match the kernel's own
    counters exactly: collisions, absorptions, leaks, cutoffs, zero-yield
    events and created secondaries.
It does not test the animation script (scripts/animate_tracks.py).
"""
import dataclasses
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mcslab import tallies as T                              # noqa: E402
from mcslab import tracks as TR                              # noqa: E402
from mcslab.transport_kin import speed                       # noqa: E402

SEED_TRACKS = 20261024


def _config(**kw):
    import test_regression_kin as R
    from mcslab.nucdata import Library
    try:
        lib = Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")
    cfg = R.reference_problems(lib)["kin_d7"]
    return dataclasses.replace(cfg, n_batches=4, histories_per_batch=500, seed=SEED_TRACKS,
                               **kw), R.ARRAYS


def _run(cfg, **kw):
    import warnings
    from mcslab.config_kin import EnergyCutoffWarning, run_kin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg, **kw)


def _same(a, b, arrays):
    for name in arrays:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert x.tobytes() == y.tobytes(), name


def test_recording_changes_no_tally(note):
    cfg, arrays = _config()
    off = _run(cfg)
    on = _run(dataclasses.replace(cfg, n_track=cfg.n_histories))
    _same(off, on, arrays)
    tiny = _run(dataclasses.replace(cfg, n_track=cfg.n_histories, track_capacity=5))
    _same(off, tiny, arrays)
    assert on.tracks.truncated.sum() == 0 and tiny.tracks.truncated.sum() > 0
    note(f"(tracks) recording on / off / truncated: all {len(arrays)} tally and count "
         f"arrays byte-identical (max draws {off.max_draws} in all three); "
         f"{int(on.tracks.n.sum())} events recorded, {int(tiny.tracks.truncated.sum())} "
         "dropped by the 5-event slots")


def test_tracks_independent_of_batch_split():
    cfg, _ = _config(n_track=2000)
    full = _run(cfg)
    split = None
    for rng in [(2, 4), (0, 1), (1, 2)]:
        split = _run(cfg, batch_range=rng, out=split)
    for name in ("f", "i", "n", "truncated"):
        assert getattr(full.tracks, name).tobytes() == getattr(split.tracks, name).tobytes()


def test_tracks_are_consistent(note):
    cfg, _ = _config(n_track=2000)
    res = _run(cfg)
    tr = res.tracks
    assert tr.truncated.sum() == 0
    lo, hi = float(tr.bounds[0]), float(tr.bounds[-1])
    totals = dict.fromkeys(("collision", "absorb", "leak", "cutoff", "zero_yield",
                            "created", "born_below"), 0)
    worst_dx = 0.0
    for h in range(tr.f.shape[0]):
        parts = tr.particles(h)
        assert sorted(parts) == list(range(len(parts)))          # ids 0 .. n-1
        totals["created"] += len(parts) - 1
        for pid, (f, i) in parts.items():
            ev = i[:, TR.TI_EVENT]
            assert ev[0] == (TR.EV_SOURCE if pid == 0 else TR.EV_BORN), (h, pid)
            assert (i[:, TR.TI_PARENT] == (-1 if pid == 0 else i[0, TR.TI_PARENT])).all()
            term = np.isin(ev, TR.TERMINAL)
            assert term.sum() == 1 and term[-1], (h, pid, ev)
            t, x, E, u = f[:, TR.TF_T], f[:, TR.TF_X], f[:, TR.TF_E], f[:, TR.TF_U]
            assert (np.diff(t) >= 0.0).all(), (h, pid)
            assert (x >= lo).all() and (x <= hi).all(), (h, pid)
            for j in range(1, ev.size):
                if ev[j] in (TR.EV_CUTOFF, TR.EV_ZERO_YIELD) and ev[j - 1] == TR.EV_COLLISION:
                    # ends at the collision it follows
                    assert t[j] == t[j - 1] and x[j] == x[j - 1] and E[j] == E[j - 1]
                    continue
                if ev[j] == TR.EV_CUTOFF and ev[j - 1] == TR.EV_BORN:
                    continue                                       # born below the cutoff
                # a flight from event j-1 at energy E[j-1], direction u[j-1]
                if ev[j] != TR.EV_COLLISION:
                    assert E[j] == E[j - 1], (h, pid, j)
                dx = x[j] - x[j - 1]
                worst_dx = max(worst_dx, abs(dx - u[j - 1] * speed(E[j - 1]) * (t[j] - t[j - 1])))
            totals["collision"] += int((ev == TR.EV_COLLISION).sum())
            totals["absorb"] += int((ev == TR.EV_ABSORB).sum())
            totals["leak"] += int((ev == TR.EV_LEAK).sum())
            totals["cutoff"] += int((ev == TR.EV_CUTOFF).sum())
            totals["zero_yield"] += int((ev == TR.EV_ZERO_YIELD).sum())
    note(f"(tracks) {tr.f.shape[0]} histories: worst |dx - u v(E) dt| along a flight = "
         f"{worst_dx:.1e} cm (bound 1e-8)")
    assert worst_dx <= 1e-8
    # the recorded events against the kernel's own counters (all histories recorded)
    assert totals["collision"] + totals["absorb"] == res.count(T.K_COLLISIONS)
    assert totals["absorb"] == res.count(T.K_ABSORBED)
    assert totals["leak"] == res.count(T.K_LEAK_LEFT) + res.count(T.K_LEAK_RIGHT)
    assert totals["cutoff"] == res.count(T.K_CUTOFF)
    assert totals["zero_yield"] == res.count(T.K_ZERO_YIELD)
    assert totals["created"] == res.count(T.K_CREATED)
    assert res.count(T.K_CREATED) > 0 and res.count(T.K_FREE_GAS) > 0


def test_saved_tracks_round_trip(tmp_path):
    cfg, _ = _config(n_track=50)
    tr = _run(cfg).tracks
    path = tmp_path / "tracks.npz"
    tr.save(path)
    back = TR.load_tracks(path)
    for name in ("f", "i", "n", "truncated", "bounds"):
        assert getattr(back, name).tobytes() == getattr(tr, name).tobytes()
    assert back.region_names == tr.region_names and back.energy_cutoff == tr.energy_cutoff
    paths = TR.particle_paths(back, [0, 1])
    assert all(p[2].size == p[3].size == p[4].size for p in paths)
    assert not math.isnan(sum(float(p[3].sum()) for p in paths))
