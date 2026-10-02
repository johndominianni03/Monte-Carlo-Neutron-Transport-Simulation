#!/usr/bin/env python3
"""Record neutron tracks in the D7 slab and save them for animate_tracks.py.

    MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5 ./venv/bin/python scripts/record_tracks.py [out.npz]

Problem D7, as in tests/test_regression_kin.py: a 14.1 MeV beam at x = 0
on W 0.5 cm (293.6 K) | FLiBe 20 cm (natural Li, 900 K data, Janz
density at 900 K) | Fe 10 cm (293.6 K), seed 20261023. Histories 0 .. N-1
are run and all recorded. History i depends only on (seed, i), so these
are exactly the first N histories of the D7 reference run.

Writes outputs/tracks_d7.npz by default (outputs/ is not committed).
Recording draws no random number and touches no tally
(tests/test_tracks.py).
"""
import dataclasses
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mcslab import ce_materials as cm                        # noqa: E402
from mcslab.config_ce import MonoEnergetic                   # noqa: E402
from mcslab.config_kin import (EnergyCutoffWarning,          # noqa: E402
                               KinRunConfig, run_kin)
from mcslab.geometry import SlabGeometry                     # noqa: E402
from mcslab.nucdata import Library                           # noqa: E402
from mcslab.sources import BeamSource                        # noqa: E402

N_HISTORIES = 200
SEED = 20261023          # the D7 reference seed
OUT = os.path.join(REPO, "outputs", "tracks_d7.npz")


def main(argv):
    out = argv[1] if len(argv) > 1 else OUT
    flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    geom = SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()])
    cfg = KinRunConfig(geom, BeamSource(), MonoEnergetic(14.1e6), Library.open(),
                       n_batches=2, histories_per_batch=N_HISTORIES // 2, seed=SEED,
                       n_track=N_HISTORIES, track_capacity=4000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(cfg)
    tr = res.tracks
    tr.description = (f"D7 (W 0.5 / FLiBe 20 / Fe 10 cm, 14.1 MeV beam), histories "
                      f"0-{N_HISTORIES - 1}, seed {SEED}")
    if tr.truncated.sum():
        raise RuntimeError("track slots overflowed; raise track_capacity")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tr.save(out)
    b = res.balance()
    print(f"wrote {os.path.relpath(out, REPO)}: {N_HISTORIES} histories, "
          f"{int(tr.n.sum())} events, max {int(tr.n.max())} per history; "
          f"created {b['created']:.0f}, absorbed {b['absorbed']:.0f}, leaked "
          f"{b['leak_left'] + b['leak_right']:.0f}, cutoff {b['cutoff']:.0f}, "
          f"zero-yield {b['zero_yield']:.0f}; last event at "
          f"{float(np.max(tr.f[:, :, 3])):.3g} s")


if __name__ == "__main__":
    main(sys.argv)
