#!/usr/bin/env python3
"""Animate recorded neutron tracks: depth vs log10 energy over time.

    ./venv/bin/python scripts/animate_tracks.py [tracks.npz] [--families 40] [--frames 100]

Reads only a saved recording (scripts/record_tracks.py writes
outputs/tracks_d7.npz); it runs no transport. Writes
docs/figures/tracks.gif (matplotlib PillowWriter) and a static
docs/figures/tracks.png of the same tracks.

Each neutron is a dot at (depth x, log10 E). Between recorded events it
flies in a straight line at constant energy, so x is interpolated
linearly in time and E is held. Frames are log-spaced in time: a 14 MeV
neutron crosses the 30 cm slab in about 6e-9 s, while thermal neutrons live
until about 1e-4 s. Each dot trails its positions in the previous frames,
fading out. Primary neutrons and secondaries (from (n,2n)-type reactions)
are told apart by color and listed in the legend. Material regions are
labelled in text, so their tint carries no meaning on its own.

Python 3.9 compatible; matplotlib only.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib                                            # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402

from mcslab.tracks import load_tracks, particle_paths        # noqa: E402

# Reference palette (dataviz skill, references/palette.md): light chart
# surface, text inks, categorical slots 1 (blue) and 2 (orange).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
REGION_TINTS = ("#ecebe7", "#f5f4f1", "#e3e2dd")
TRAIL = 6                 # frames of trail behind each dot
FIG = os.path.join(REPO, "docs", "figures")


def state_at(t, x, E, tau):
    """(x, log10 E) of a particle at time tau, or None if it is not alive.
    Linear in time between events (straight flight), E held."""
    if tau < t[0] or tau > t[-1]:
        return None
    j = int(np.searchsorted(t, tau, side="right")) - 1
    j = min(j, t.size - 1)
    if j == t.size - 1:
        return x[j], np.log10(E[j])
    dt = t[j + 1] - t[j]
    xx = x[j] if dt <= 0.0 else x[j] + (x[j + 1] - x[j]) * (tau - t[j]) / dt
    return xx, np.log10(E[j])


def setup_axes(ax, tracks, y_lo):
    b = tracks.bounds
    for k, name in enumerate(tracks.region_names):
        ax.axvspan(b[k], b[k + 1], color=REGION_TINTS[k % len(REGION_TINTS)], lw=0, zorder=0)
        ax.text(0.5 * (b[k] + b[k + 1]), 7.55, name, ha="center", va="bottom",
                color=TEXT_2, fontsize=9, zorder=5)
    for xb in b[1:-1]:
        ax.axvline(xb, color="#c3c2b7", lw=0.8, zorder=1)
    ax.set_xlim(b[0] - 0.4, b[-1] + 0.4)
    ax.set_ylim(y_lo, 7.9)
    ax.set_xlabel("depth x (cm)", color=TEXT_2)
    ax.set_ylabel("log$_{10}$ E (eV)", color=TEXT_2)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0.5)
    # legend below the axes, right of the x label, so it covers no track
    ax.figure.legend(handles=[Line2D([], [], marker="o", ls="", color=PRIMARY, label="primary"),
                              Line2D([], [], marker="o", ls="", color=SECONDARY,
                                     label="secondary (n,2n)-type")],
                     loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=2, fontsize=8,
                     frameon=False, labelcolor=TEXT_2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="?", default=os.path.join(REPO, "outputs", "tracks_d7.npz"))
    ap.add_argument("--families", type=int, default=40, help="histories drawn (cap)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--t0", type=float, default=1e-10, help="first frame time (s)")
    args = ap.parse_args(argv)

    tracks = load_tracks(args.tracks)
    histories = list(range(min(args.families, tracks.f.shape[0])))
    paths = particle_paths(tracks, histories)
    t_end = max(float(p[2][-1]) for p in paths)
    # the energy axis spans the drawn neutrons (never below the cutoff)
    e_min = min(float(np.min(p[4])) for p in paths)
    y_lo = max(np.log10(tracks.energy_cutoff), np.floor(np.log10(e_min)) - 0.5)
    times = np.geomspace(args.t0, t_end * 1.05, args.frames)
    colors = [PRIMARY if pid == 0 else SECONDARY for _, pid, *_ in paths]
    pos = np.full((args.frames, len(paths), 2), np.nan)
    for f, tau in enumerate(times):
        for p, (_, _, t, x, E, _) in enumerate(paths):
            s = state_at(t, x, E, tau)
            if s is not None:
                pos[f, p] = s

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    setup_axes(ax, tracks, y_lo)
    ax.set_title(f"Neutrons in W | FLiBe | Fe, 14.1 MeV beam ({len(histories)} source "
                 "neutrons and their secondaries)", color=TEXT, fontsize=10, loc="left")
    trails = [ax.scatter([], [], s=10, lw=0, zorder=3) for _ in range(TRAIL)]
    dots = ax.scatter([], [], s=22, lw=0.6, edgecolors=SURFACE, zorder=4)
    clock = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom",
                    color=TEXT, fontsize=9, zorder=6)
    rgba = np.array([matplotlib.colors.to_rgba(c) for c in colors])
    fig.tight_layout()

    def draw(f):
        for k, sc in enumerate(trails):
            g = f - (k + 1)
            if g < 0:
                sc.set_offsets(np.empty((0, 2)))
                continue
            ok = ~np.isnan(pos[g, :, 0]) & ~np.isnan(pos[f, :, 0])
            sc.set_offsets(pos[g, ok])
            c = rgba[ok].copy()
            c[:, 3] = 0.45 * (1.0 - (k + 1) / (TRAIL + 1))
            sc.set_facecolors(c)
        ok = ~np.isnan(pos[f, :, 0])
        dots.set_offsets(pos[f, ok])
        dots.set_facecolors(rgba[ok])
        clock.set_text(f"t = {times[f]:.2e} s   neutrons in flight: {int(ok.sum())}")
        return trails + [dots, clock]

    os.makedirs(FIG, exist_ok=True)
    gif = os.path.join(FIG, "tracks.gif")
    anim = FuncAnimation(fig, draw, frames=args.frames, blit=False)
    anim.save(gif, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    # static figure: every particle's path in (x, log10 E)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    setup_axes(ax, tracks, y_lo)
    for (_, pid, t, x, E, _), col in zip(paths, colors):
        # flights are horizontal (E held), collisions vertical (E changes)
        xs = np.repeat(x, 2)[1:]
        es = np.repeat(np.log10(E), 2)[:-1]
        ax.plot(xs, es, color=col, lw=0.7, alpha=0.55, zorder=2)
    ax.set_title(f"Paths of {len(histories)} source neutrons and their secondaries "
                 "(W | FLiBe | Fe, 14.1 MeV)", color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    png = os.path.join(FIG, "tracks.png")
    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)
    for path in (gif, png):
        print(f"wrote {os.path.relpath(path, REPO)} ({os.path.getsize(path) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
