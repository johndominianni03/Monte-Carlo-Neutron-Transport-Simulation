#!/usr/bin/env python3
"""Phase 3 Part B: tritium per source neutron vs FLiBe thickness and Li-6
enrichment, with a reflective plasma side (approved D19).

Geometry: W 0.5 cm | FLiBe L | Fe 10 cm; the left (plasma) side reflects
specularly (KinRunConfig.reflect_left), the right side is vacuum. 14.1 MeV
beam at normal incidence into the W. FLiBe uses its 900 K data and the
Janz density at 900 K; enriched FLiBe holds the molar density of
natural-Li FLiBe (approved D18). W and Fe at 293.6 K.

    L        10, 20, 30, 50, 75, 100 cm
    Li-6     7.59 % (natural, the unchanged natural-Li path), 20, 40, 60, 90 %

30 runs of 20 batches x 5000 histories, ALL ON SEED 20261035 (common random
numbers): the curves are correlated with each other, so differences
between neighbouring points are smoother than their SEs suggest, and the
SEs of different points are not independent.

What it is: an idealised 1D upper bound. A reflective plane returns every
neutron that leaves the plasma side, and there is no structure, coolant,
ports or gaps. It is not a tritium breeding ratio of any real design.

Writes docs/blanket_sweep.csv and docs/figures/blanket_sweep.png.

    ./venv/bin/python scripts/blanket_sweep.py
"""
import dataclasses
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import numpy as np                                          # noqa: E402

from mcslab import ce_materials as cm                       # noqa: E402
from mcslab import tallies as T                             # noqa: E402
from mcslab.config_ce import MonoEnergetic                  # noqa: E402
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin  # noqa: E402
from mcslab.geometry import SlabGeometry                    # noqa: E402
from mcslab.nucdata import Library                          # noqa: E402
from mcslab.sources import BeamSource                       # noqa: E402

SEED = 20261035
N_BATCHES = 20
HISTORIES_PER_BATCH = 5000
THICKNESSES_CM = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0)
ENRICHMENTS = (None, 0.20, 0.40, 0.60, 0.90)          # None: natural Li (0.0759)
W_CM, FE_CM, T_FLIBE = 0.5, 10.0, 900.0
CSV_OUT = os.path.join(REPO, "docs", "blanket_sweep.csv")
FIG_OUT = os.path.join(REPO, "docs", "figures", "blanket_sweep.png")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
# ordinal blue ramp (reference data-viz palette, steps 250..650): light = low Li-6
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
MARKERS = ("o", "s", "^", "D", "v")

CSV_HEADER = f"""# Phase 3 Part B blanket sweep (mcslab, ENDF/B-VIII.0): tritium per source neutron.
# Geometry: W {W_CM} cm | FLiBe L | Fe {FE_CM} cm; REFLECTIVE plasma side, vacuum behind the Fe.
# 14.1 MeV beam at normal incidence. FLiBe: 900 K data, Janz density at 900 K, molar density of
# natural-Li FLiBe held for enriched Li. {N_BATCHES} batches x {HISTORIES_PER_BATCH} histories per point.
# ALL POINTS SHARE ONE SEED ({SEED}): the curves are correlated with each other (common random
# numbers); per-point SEs are not independent between points.
# Idealised 1D upper bound: no structure, coolant, ports or gaps; not the TBR of a real design.
# Columns: track-length estimator, mean and SE over batches, per source neutron.
"""
COLUMNS = ("flibe_cm", "li6_atom_percent", "tritium", "tritium_se", "tritium_li6",
           "tritium_li6_se", "tritium_li7", "tritium_li7_se", "tritium_be9", "tritium_be9_se",
           "tritium_f19", "tritium_f19_se", "tritium_fe", "tritium_fe_se", "leak_right",
           "leak_right_se", "reflected", "reflected_se", "cutoff", "zero_yield",
           "flibe_mass_density_g_cm3", "max_draws")


def run_point(lib, L, li6):
    flibe = dataclasses.replace(cm.flibe(li6, temperature_K=T_FLIBE), temperature=T_FLIBE)
    geom = SlabGeometry([0.0, W_CM, W_CM + L, W_CM + L + FE_CM],
                        [cm.tungsten(), flibe, cm.iron()])
    cfg = KinRunConfig(geom, BeamSource(), MonoEnergetic(14.1e6), lib, N_BATCHES,
                       HISTORIES_PER_BATCH, SEED, depth_bins=(1, 1, 1), reflect_left=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(cfg)
    secs = time.perf_counter() - t0
    assert res.balance()["residual"] == 0.0 and res.lost == 0
    awr = {n: lib.load(n, "900K").awr for n in flibe.nuclide_names}
    row = {"flibe_cm": L, "li6_atom_percent": 7.59 if li6 is None else 100.0 * li6}

    def put(key, batches):
        m, s = T.batch_stats(batches)
        row[key], row[key + "_se"] = float(m), float(s)
    put("tritium", res.tally_batches(T.R_H3, T.EST_TL).sum(axis=1))
    for nuc in ("Li6", "Li7", "Be9", "F19"):
        put(f"tritium_{nuc.lower()}", res.tally_batches(T.R_H3, T.EST_TL, nuc)[:, 1])
    put("tritium_fe", res.tally_batches(T.R_H3, T.EST_TL)[:, 2])
    put("leak_right", res.leakage_right)
    put("reflected", res.reflected_weight / HISTORIES_PER_BATCH)
    row["cutoff"] = float(res.cutoff_weight.sum() / cfg.n_histories)
    row["zero_yield"] = float(res.zero_yield_weight.sum() / cfg.n_histories)
    row["flibe_mass_density_g_cm3"] = flibe.mass_density(awr)
    row["max_draws"] = res.max_draws
    row["seconds"] = secs
    return row


def write_csv(rows, path):
    with open(path, "w") as fh:
        fh.write(CSV_HEADER)
        fh.write(",".join(COLUMNS) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.10g}" if isinstance(r[c], float) else str(r[c])
                              for c in COLUMNS) + "\n")


def figure(rows, path):
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(THICKNESSES_CM[0] - 7.5, 1.0, "1 triton per\nsource neutron", color=INK2,
            fontsize=8, va="center", ha="left")
    for k, li6 in enumerate(ENRICHMENTS):
        pct = 7.59 if li6 is None else 100.0 * li6
        pts = sorted((r for r in rows if r["li6_atom_percent"] == pct),
                     key=lambda r: r["flibe_cm"])
        x = np.array([r["flibe_cm"] for r in pts])
        y = np.array([r["tritium"] for r in pts])
        e = np.array([r["tritium_se"] for r in pts])
        label = f"{pct:g}% Li-6" + (" (natural)" if li6 is None else "")
        ax.plot(x, y, color=RAMP[k], lw=2.0, zorder=2)
        ax.errorbar(x, y, yerr=e, fmt=MARKERS[k], ms=5, color=RAMP[k], ecolor=RAMP[k],
                    elinewidth=1.0, capsize=0, lw=0, label=label, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xlim(0, THICKNESSES_CM[-1] + 5)
    ax.set_xlabel("FLiBe thickness L (cm)", color=INK2, fontsize=9.5)
    ax.set_ylabel("tritons per source neutron", color=INK2, fontsize=9.5)
    ax.set_title("Tritium production vs FLiBe thickness and Li-6 enrichment "
                 "(reflective plasma side)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    fig.text(0.01, 0.01,
             f"Idealised 1D upper bound: W {W_CM:g} cm | FLiBe L | Fe {FE_CM:g} cm with a "
             "reflective plane on the plasma side; no structure, coolant, ports or gaps.\n"
             "Not the tritium breeding ratio of a real design. All 30 points share one seed "
             f"({SEED}), so the curves are correlated with each other.\n14.1 MeV beam, "
             f"{N_BATCHES} x {HISTORIES_PER_BATCH} histories per point, error bars +-1 SE "
             "(smaller than the markers). ENDF/B-VIII.0, FLiBe at 900 K data and density.",
             color=INK2, fontsize=7.8, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def headline(rows):
    lines = []
    for li6 in ENRICHMENTS:
        pct = 7.59 if li6 is None else 100.0 * li6
        pts = sorted((r for r in rows if r["li6_atom_percent"] == pct),
                     key=lambda r: r["flibe_cm"])
        above = [r for r in pts if r["tritium"] > 1.0]
        if not above:
            lines.append(f"  {pct:g}% Li-6: never above 1 (max {max(r['tritium'] for r in pts):.4f})")
        else:
            r = above[0]
            lines.append(f"  {pct:g}% Li-6: first above 1 at L = {r['flibe_cm']:g} cm "
                         f"({r['tritium']:.4f} +- {r['tritium_se']:.4f}, "
                         f"{(r['tritium'] - 1.0) / r['tritium_se']:+.1f} SE above 1)")
    return "\n".join(lines)


def main():
    lib = Library.open()
    rows = []
    t0 = time.perf_counter()
    for li6 in ENRICHMENTS:
        for L in THICKNESSES_CM:
            row = run_point(lib, L, li6)
            rows.append(row)
            print(f"L = {L:5.0f} cm, Li-6 {row['li6_atom_percent']:5.2f}%: tritium "
                  f"{row['tritium']:.4f} +- {row['tritium_se']:.4f}  ({row['seconds']:.1f} s)")
    print(f"total {time.perf_counter() - t0:.0f} s")
    write_csv(rows, CSV_OUT)
    figure(rows, FIG_OUT)
    print("smallest L with tritium per source above 1:")
    print(headline(rows))
    print(f"wrote {os.path.relpath(CSV_OUT, REPO)} and {os.path.relpath(FIG_OUT, REPO)}")


if __name__ == "__main__":
    main()
