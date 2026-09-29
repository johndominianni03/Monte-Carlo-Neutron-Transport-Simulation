#!/usr/bin/env python3
"""Plot cross sections vs energy on log-log axes, saved as PNGs in docs/figures/.

    MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5 ./venv/bin/python scripts/plot_xs.py

Figures (1e-5 eV to 20 MeV, with a marker and value label at 14.1 MeV):
  fe56_total.png           Fe-56 total (sum of non-redundant reactions)
  tritium_production.png   Li-6 and Li-7 tritium production (MT 205)
  be9_n2n.png              Be-9 (n,2n) (MT 16)
  overview.png             all of the above as small multiples

Curves are drawn through the stored pointwise data (every grid point). The
14.1 MeV values come from mcslab.xs, the same lookup the kernel uses.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mcslab import xs                             # noqa: E402
from mcslab.nucdata import Library                # noqa: E402

OUT = os.path.join(REPO, "docs", "figures")
E_MIN, E_MAX, E14 = 1e-5, 20e6, 14.1e6

# Light-theme tokens (reference data-viz palette): surface, ink, axes, series.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]    # categorical slots 1-2, fixed order


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(E_MIN, E_MAX)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("Incident neutron energy (eV)", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.axvline(E14, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)


def curve(ax, energy, sigma, color, label=None):
    keep = (energy >= E_MIN) & (energy <= E_MAX) & (sigma > 0)
    ax.plot(energy[keep], sigma[keep], color=color, linewidth=1.4, label=label,
            solid_joinstyle="round", zorder=2)


def mark14(ax, value, color, text, offset=(-8, 8)):
    """Dot at the 14.1 MeV value plus a label on a surface-coloured backing
    (with a thin leader line when it sits away from the dot)."""
    ax.plot([E14], [value], "o", markersize=6, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    far = abs(offset[0]) > 12 or abs(offset[1]) > 12
    ax.annotate(text, (E14, value), xytext=offset, textcoords="offset points",
                ha="right", va="center", fontsize=8.5, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.95),
                arrowprops=(dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                 shrinkA=0, shrinkB=4) if far else None))


def full(nu, mt):
    return nu.summed_on_grid([mt])


def draw_fe56(ax, lib):
    fe = lib.load("Fe56")
    curve(ax, fe.energy, fe.total_on_grid(), SERIES[0])
    v = xs.lookup_total(fe, E14)
    mark14(ax, v, SERIES[0], f"{v:.3f} b at 14.1 MeV", offset=(-30, 70))
    style(ax, "Fe-56 total cross section", "Cross section (b)")


def draw_tritium(ax, lib, legend=True):
    li6, li7 = lib.load("Li6"), lib.load("Li7")
    curve(ax, li6.energy, full(li6, 205), SERIES[0], "Li-6 (n,t)  [MT 205 = MT 105]")
    curve(ax, li7.energy, full(li7, 205), SERIES[1], "Li-7 (n,Xt)  [MT 205]")
    v6, v7 = xs.lookup(li6, 205, E14), xs.lookup(li7, 205, E14)
    mark14(ax, v6, SERIES[0], f"Li-6: {v6 * 1e3:.1f} mb", offset=(-40, -22))
    mark14(ax, v7, SERIES[1], f"Li-7: {v7 * 1e3:.0f} mb", offset=(-40, 26))
    style(ax, "Tritium production", "Cross section (b)")
    if legend:
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)


def draw_be9(ax, lib):
    be = lib.load("Be9")
    curve(ax, be.energy, full(be, 16), SERIES[0])
    v = xs.lookup(be, 16, E14)
    mark14(ax, v, SERIES[0], f"{v:.3f} b at 14.1 MeV", offset=(-10, -14))
    style(ax, "Be-9 (n,2n)", "Cross section (b)")
    ax.set_xlim(1e5, E_MAX)          # zero below the 1.75 MeV threshold


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")


def main():
    lib = Library.open()
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    footer = f"{lib.label}, 294 K. Dashed line: 14.1 MeV."

    for name, draw in (("fe56_total.png", draw_fe56),
                       ("tritium_production.png", draw_tritium),
                       ("be9_n2n.png", draw_be9)):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        draw(ax, lib)
        fig.text(0.01, 0.01, footer, fontsize=7.5, color=MUTED)
        save(fig, name)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    draw_fe56(axes[0, 0], lib)
    draw_tritium(axes[0, 1], lib)
    draw_be9(axes[1, 0], lib)
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.0, 0.9, "Phase 2a cross sections", fontsize=11, color=INK,
            transform=ax.transAxes)
    ax.text(0.0, 0.8, footer + "\nCurves pass through every stored grid point.\n"
            "The Be-9 panel starts at 100 keV (threshold 1.75 MeV).",
            fontsize=9, color=INK_2, va="top", transform=ax.transAxes)
    save(fig, "overview.png")


if __name__ == "__main__":
    main()
