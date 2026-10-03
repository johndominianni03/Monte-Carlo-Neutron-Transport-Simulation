#!/usr/bin/env python3
"""Phase 3 results: run the tally reference problem(s), post-process them
(mcslab/postprocess.py), and write

    docs/phase3_results.json          per-layer results with inputs (deterministic)
    docs/figures/phase3_profiles.png  depth profiles with +-1 SE error bars

and print the README results table.

The problems are those of tests/test_regression_tally.py, so the numbers are
exactly those pinned by the tally references. Problem kin_d7 is D7 (seed
20261023, 20 x 5000, vacuum on both sides); kin_d7_reflect is the same slab
with a reflective plasma side (seed 20261035, Part B). Means and SE are over
the 20 batches of each. The run is also timed with the tallies off (wall clock, printed
only, never written to the JSON).

    ./venv/bin/python scripts/phase3_results.py
"""
import dataclasses
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import numpy as np                                          # noqa: E402

import test_regression_tally as RT                          # noqa: E402
from mcslab import postprocess as P                         # noqa: E402
from mcslab.config_kin import EnergyCutoffWarning, run_kin  # noqa: E402
from mcslab.nucdata import Library                          # noqa: E402

PROBLEMS = ("kin_d7", "kin_d7_reflect")
JSON_OUT = os.path.join(REPO, "docs", "phase3_results.json")
FIG_OUT = os.path.join(REPO, "docs", "figures", "phase3_profiles.png")

# Reference data-viz palette (light surface): categorical slots in fixed order
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")


def run_quiet(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg)


def fmt(e, digits=4):
    return f"{e['mean']:.{digits}g} ± {e['se']:.2g}"


def table(summary, label):
    L = summary["layers"]
    T = summary["totals"]
    rows = [
        ("NRT dpa per FPY, front bin", fmt(L["W"]["dpa_per_fpy_front"]), "",
         fmt(L["Fe"]["dpa_per_fpy_front"])),
        ("NRT dpa per FPY, layer average", fmt(L["W"]["dpa_per_fpy_avg"]), "",
         fmt(L["Fe"]["dpa_per_fpy_avg"])),
        ("He appm per FPY, front bin", fmt(L["W"]["he_appm_per_fpy_front"]), "",
         fmt(L["Fe"]["he_appm_per_fpy_front"])),
        ("He appm per FPY, layer average", fmt(L["W"]["he_appm_per_fpy_avg"]), "",
         fmt(L["Fe"]["he_appm_per_fpy_avg"])),
        ("He appm / dpa, front bin", fmt(L["W"]["he_appm_per_dpa_front"]), "",
         fmt(L["Fe"]["he_appm_per_dpa_front"])),
        ("heating W/cm^3, front bin, 301 (no photons) to 901 (local photons)",
         *(f"{fmt(L[n]['heating_301_W_cm3_front'], 3)} to "
           f"{fmt(L[n]['heating_901_W_cm3_front'], 3)}" for n in ("W", "FLiBe", "Fe"))),
        ("heating W/cm^3, layer average, 301 to 901",
         *(f"{fmt(L[n]['heating_301_W_cm3_avg'], 3)} to "
           f"{fmt(L[n]['heating_901_W_cm3_avg'], 3)}" for n in ("W", "FLiBe", "Fe"))),
        ("tritium per source neutron", "0 (no MT 205 data)",
         fmt(L["FLiBe"]["tritium_per_source"]), fmt(L["Fe"]["tritium_per_source"], 3)),
    ]
    lines = [f"**{label}** (1 MW/m^2 of 14.1 MeV neutrons = "
             f"{summary['inputs']['source_rate_n_cm2_s']:.4e} n/cm^2/s; 1 FPY = "
             f"{summary['inputs']['full_power_year_s']:.6g} s; E_d W 90 eV, Fe 40 eV)", "",
             "| quantity | W (0-0.5 cm) | FLiBe (0.5-20.5 cm) | Fe (20.5-30.5 cm) |",
             "|---|---|---|---|"]
    lines += [f"| {q} | {a} | {b} | {c} |" for q, a, b, c in rows]
    by = L["FLiBe"]["tritium_per_source_by_nuclide"]
    lines += ["", "Tritium per source neutron in FLiBe by nuclide: " +
              ", ".join(f"{n} {fmt(by[n])}" for n in ("Li6", "Li7", "Be9", "F19")) + "; "
              f"total over all layers {fmt(T['tritium_per_source'])}.",
              f"Total heating / 14.1 MeV: MT 301 {fmt(T['heating_301_fraction_of_source_energy'])}"
              f", MT 901 {fmt(T['heating_901_fraction_of_source_energy'])}."]
    return "\n".join(lines)


def _bands(ax, edges, region, names):
    for r, name in enumerate(names):
        sel = np.flatnonzero(region == r)
        if r % 2 == 0:
            ax.axvspan(edges[sel[0]], edges[sel[-1] + 1], color=BAND, zorder=0, lw=0)


def _layer_label(ax, x0, x1, name):
    ax.text(0.5 * (x0 + x1), 0.96, name, transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=INK2, fontsize=8.5)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _series(ax, x, mean, se, k, label=None):
    ok = np.isfinite(mean)
    ax.errorbar(x[ok], mean[ok], yerr=se[ok], fmt=MARKERS[k], ms=3.5, color=SERIES[k],
                ecolor=SERIES[k], elinewidth=1.0, capsize=0, lw=0, label=label, zorder=3)
    idx = np.flatnonzero(ok)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        ax.plot(x[run], mean[run], color=SERIES[k], lw=1.6, zorder=2)


def _pair(fig, gs_w, gs_f, x, mean, se, edges, region, names, title, ylabel):
    """W and Fe sub-panels of one quantity, sharing the y axis."""
    aw = fig.add_subplot(gs_w)
    af = fig.add_subplot(gs_f, sharey=aw)
    for ax, name in ((aw, "W"), (af, "Fe")):
        _style(ax)
        sel = np.flatnonzero(region == names.index(name))
        _series(ax, x[sel], mean[sel], se[sel], 0)
        ax.set_xlim(edges[sel[0]], edges[sel[-1] + 1])
        _layer_label(ax, edges[sel[0]], edges[sel[-1] + 1], name)
        ax.set_xlabel("depth x (cm)", color=INK2, fontsize=9)
    plt.setp(af.get_yticklabels(), visible=False)
    aw.set_ylim(0, 1.12 * np.nanmax(mean + se))
    aw.set_ylabel(ylabel, color=INK2, fontsize=9)
    aw.set_title(title, loc="left", color=INK, fontsize=10.5)
    return aw, af


def figure(res, summary, path):
    from mcslab import tallies as T
    pr = P.profiles(res)
    edges, region, names = pr["edges_cm"], pr["region"], list(pr["region_names"])
    x = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 2, 0.35, 1, 2], height_ratios=[1, 1.15],
                          hspace=0.42, wspace=0.12)
    _pair(fig, gs[0, 0], gs[0, 1], x, *pr["dpa_per_fpy"], edges, region, names,
          "NRT dpa per full-power year", "dpa / FPY")
    _pair(fig, gs[0, 3], gs[0, 4], x, *pr["he_appm_per_fpy"], edges, region, names,
          "He production, appm per full-power year", "appm / FPY")

    c = fig.add_subplot(gs[1, 0:2])
    _style(c)
    _bands(c, edges, region, names)
    for k, (key, lab) in enumerate((("heating_301_W_cm3", "MT 301 (photon energy excluded)"),
                                    ("heating_901_W_cm3",
                                     "MT 901 (photons deposited locally)"))):
        _series(c, x, *pr[key], k, label=lab)
    c.set_yscale("log")
    c.set_xlim(edges[0], edges[-1])
    for r, name in enumerate(names):
        sel = np.flatnonzero(region == r)
        x0, x1 = edges[sel[0]], edges[sel[-1] + 1]
        if x1 - x0 > 2:
            _layer_label(c, x0, x1, name)
        else:
            c.text(x1 + 0.25, 0.96, name, transform=c.get_xaxis_transform(), ha="left",
                   va="top", color=INK2, fontsize=8.5)
    c.set_xlabel("depth x (cm)", color=INK2, fontsize=9)
    c.set_ylabel("W/cm$^3$", color=INK2, fontsize=9)
    c.set_title("Heating at 1 MW/m$^2$ (the deposited value lies between the two)",
                loc="left", color=INK, fontsize=10.5)
    c.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)

    d = fig.add_subplot(gs[1, 3:5])
    _style(d)
    fl = np.flatnonzero(region == names.index("FLiBe"))
    xf = x[fl]
    width = res.mesh.bin_width[fl]
    for k, nuc in enumerate(("Li6", "Li7", "F19", "Be9")):
        m, s = T.batch_stats(res.tally_batches(T.R_H3, T.EST_TL, nuc)[:, fl] / width)
        _series(d, xf, m, s, k)
        d.text(xf[-1] + 0.3, m[-1], nuc, color=INK2, fontsize=8.5, va="center")
    d.set_yscale("log")
    d.set_xlim(edges[fl[0]], edges[fl[-1] + 1] + 1.8)
    d.set_xlabel("depth x (cm), FLiBe layer", color=INK2, fontsize=9)
    d.set_ylabel("tritons / (source neutron cm)", color=INK2, fontsize=9)
    d.set_title("Tritium production by nuclide (W has no MT 205 data)", loc="left",
                color=INK, fontsize=10.5)

    inp = summary["inputs"]
    fig.suptitle("Problem D7: W 0.5 | FLiBe 20 | Fe 10 cm, 14.1 MeV beam at normal "
                 "incidence, scaled to 1 MW/m$^2$, ENDF/B-VIII.0", color=INK, fontsize=11.5,
                 x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.012,
             f"Track-length estimator, {inp['n_batches']} batches x "
             f"{inp['histories_per_batch']} histories (seed {inp['seed']}); error bars are "
             "+-1 SE, mostly smaller than the markers. dpa: NRT on the MT 444 damage energy, "
             "E_d 90 eV (W), 40 eV (Fe).\nVacuum on both sides, so 56% of the source leaks "
             "back out of the plasma side; no photon transport; 1D slab. dpa and He are not "
             "given for the liquid FLiBe.", color=INK2, fontsize=8, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.12)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    lib = Library.open()
    problems = RT.reference_problems(lib)
    out = {}
    for name in PROBLEMS:
        cfg = problems[name][0]
        run_quiet(cfg)                         # warm (kernels, data)
        t0 = time.perf_counter()
        res = run_quiet(cfg)
        t_on = time.perf_counter() - t0
        t0 = time.perf_counter()
        run_quiet(dataclasses.replace(cfg, depth_bins=None))
        t_off = time.perf_counter() - t0
        assert res.balance()["residual"] == 0.0
        summary = P.summarize(res)
        summary["problem"] = name
        out[name] = summary
        print(f"{name}: run_kin with tallies {t_on:.2f} s, without {t_off:.2f} s "
              f"(warm, incl. packing): overhead {100 * (t_on / t_off - 1):+.0f}%")
        print(table(summary, name))
        print()
        if name == "kin_d7":
            figure(res, summary, FIG_OUT)
            print(f"wrote {os.path.relpath(FIG_OUT, REPO)}")
    P.write_json(out, JSON_OUT)
    print(f"wrote {os.path.relpath(JSON_OUT, REPO)}")


if __name__ == "__main__":
    main()
