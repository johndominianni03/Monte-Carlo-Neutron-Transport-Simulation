"""Recorded particle tracks of the kinematic kernel: the saved format and
helpers to read it. Plain NumPy; no Numba, so plotting scripts can use it
without the transport code.

A recording (KinRunConfig.n_track > 0) holds, for each recorded history h:

    f[h, j] = (x cm, u, E eV, t s)      of its j-th event, j < n[h]
    i[h, j] = (particle id, parent id, event code, MT, region)
    truncated[h]                        events dropped because the slot
                                        range (track_capacity) was full

Event codes are transport_kin.EV_* (names in EVENT_NAMES). The primary is
particle 0 (parent -1); each secondary has the next id at its creation.
Between two consecutive events of a particle it flies in a straight line
at constant energy, so x(t) is linear between events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Event codes: a primary starts; a secondary starts (popped from the bank,
# or born below the cutoff); a scattering collision (state after it, MT of
# the channel); an interface crossing; absorbed (state at the collision);
# leaked (state on the outer surface); energy cutoff; zero-yield; lost.
EVENT_NAMES = ("source", "born", "collision", "surface", "absorb", "leak", "cutoff",
               "zero_yield", "lost")
EV_SOURCE, EV_BORN, EV_COLLISION, EV_SURFACE, EV_ABSORB, EV_LEAK, EV_CUTOFF, \
    EV_ZERO_YIELD, EV_LOST = range(9)
TERMINAL = (EV_ABSORB, EV_LEAK, EV_CUTOFF, EV_ZERO_YIELD, EV_LOST)
TF_X, TF_U, TF_E, TF_T, TF_NCOL = 0, 1, 2, 3, 4
TI_PID, TI_PARENT, TI_EVENT, TI_MT, TI_REGION, TI_NCOL = 0, 1, 2, 3, 4, 5


@dataclass
class Tracks:
    f: np.ndarray            # (n_track, capacity, 4) float64
    i: np.ndarray            # (n_track, capacity, 5) int64
    n: np.ndarray            # (n_track,) int64
    truncated: np.ndarray    # (n_track,) int64
    bounds: np.ndarray       # slab region boundaries (cm)
    region_names: Tuple[str, ...]
    energy_cutoff: float
    description: str = ""

    def save(self, path) -> None:
        np.savez_compressed(path, f=self.f, i=self.i, n=self.n, truncated=self.truncated,
                            bounds=self.bounds, region_names=np.array(self.region_names),
                            energy_cutoff=np.array(self.energy_cutoff),
                            description=np.array(self.description))

    def events(self, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """(f, i) rows of history h, in recording order."""
        return self.f[h, :self.n[h]], self.i[h, :self.n[h]]

    def particles(self, h: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """particle id -> (f, i) rows of that particle's events, in order."""
        f, i = self.events(h)
        out = {}
        for pid in np.unique(i[:, TI_PID]):
            sel = i[:, TI_PID] == pid
            out[int(pid)] = (f[sel], i[sel])
        return out


def load_tracks(path) -> Tracks:
    with np.load(path, allow_pickle=False) as d:
        return Tracks(d["f"], d["i"], d["n"], d["truncated"], d["bounds"],
                      tuple(str(s) for s in d["region_names"]), float(d["energy_cutoff"]),
                      str(d["description"]))


def particle_paths(tracks: Tracks, histories: List[int]):
    """[(history, pid, t, x, E, terminal event code)] for every particle of
    the given histories: the piecewise-linear x(t) and piecewise-constant
    E(t) between its recorded events."""
    out = []
    for h in histories:
        for pid, (f, i) in tracks.particles(h).items():
            out.append((h, pid, f[:, TF_T].copy(), f[:, TF_X].copy(), f[:, TF_E].copy(),
                        int(i[-1, TI_EVENT])))
    return out
