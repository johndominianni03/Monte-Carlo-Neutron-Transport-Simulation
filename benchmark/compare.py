#!/usr/bin/env python3
"""Phase 4 comparison of mcslab and OpenMC 0.16.0 (docs/phase4_plan.md).

Applies the pre-declared checks to the committed result pairs in
benchmark/results/. Needs numpy only (matplotlib only with --figures); it
imports neither openmc, mcslab nor h5py, so the comparison is reproducible
without OpenMC installed.

    ./venv/bin/python benchmark/compare.py            # tables, results/comparison.json
    ./venv/bin/python benchmark/compare.py --figures  # also docs/figures/phase4_*.png

Rules (fixed before any benchmark run):
- z = (mcslab - OpenMC) / sqrt(SE_m^2 + SE_o^2), means and SEs over 100
  batches in each code; a primary check fails at |z| > 3.
- Primary checks (82), layer-integrated: flux; absorption, heating (301),
  heating-local (901), damage-energy (444), He4-production (207) and
  H3-production (205) of the material total, except where no nuclide of the
  layer has the data (exact zero checks instead) and P3's Fe tritium
  (diagnostic only, D38); FLiBe tritium by nuclide; leakage right, and left
  unless the left side reflects (then exactly 0, an exact check).
- Exact checks: identical inputs (problem file, data, densities, edges),
  declared settings, exact zeros, exact neutron balance, nothing lost.
- Diagnostics (no pass/fail): per-bin z for flux and every score of each
  nuclide present and the total, the spectrum of each layer, the
  uncollided flux against each other and the first-flight formula, P3's Fe
  tritium, OpenMC's mesh sums against its cell tallies, and the
  default-settings runs of D37.
- Added after the runs, diagnostics only: a source energy lying exactly on
  a spectrum edge (P4, 1 MeV) puts the uncollided neutrons one bin apart in
  the two codes (edge conventions differ); those two bins are reported
  separately and merged, not in the per-bin z statistics. Bins where a
  code scored no uncollided event cannot be compared with the
  first-flight formula and are counted instead.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark import jsonio                                 # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
PROBLEMS_JSON = os.path.join(HERE, "problems.json")
CHECKSUMS = os.path.join(REPO, "scripts", "checksums", "endfb-viii.0.sha256")
COMPARISON_JSON = os.path.join(RESULTS_DIR, "comparison.json")

PROBLEMS = ("P1", "P2", "P3", "P4")
Z_LIMIT = 3.0
OPENMC_VERSION = "0.16.0"
OPENMC_COMMIT = "617d35a5063c57796b43428bc401e627d2011046"
# order of the primary scores in tables
SCORES = ("absorption", "heating", "heating-local", "damage-energy", "He4-production",
          "H3-production")
SCORE_LABEL = {"absorption": "absorption", "heating": "heating 301",
               "heating-local": "heating-local 901", "damage-energy": "damage-energy 444",
               "He4-production": "He4 207", "H3-production": "H3 205"}
# D38: P3's Fe tritium has about 70 contributing histories; diagnostic only
DIAGNOSTIC_ONLY = {("P3", "Fe", "H3-production")}
# the weakest primary checks (flagged, still primary)
WEAKEST = {("P3", "Fe")}
EXPECTED_PRIMARY = {"P1": 26, "P2": 25, "P3": 24, "P4": 7}
D37 = {"P1": "ptables_on", "P4": "cutoff_0"}


# ---------------------------------------------------------------- loading
def problems():
    return jsonio.read(PROBLEMS_JSON)


def result_path(code, name, variant="main"):
    suffix = "" if variant == "main" else f"_{variant}"
    return os.path.join(RESULTS_DIR, f"{code}_{name}{suffix}.json")


def load(code, name, variant="main"):
    path = result_path(code, name, variant)
    return jsonio.read(path) if os.path.isfile(path) else None


def pinned_checksums():
    pins = {}
    with open(CHECKSUMS) as fh:
        for line in fh:
            digest, path = line.split()
            pins[path] = digest
    return pins


# ---------------------------------------------------------------- statistics
def zscore(m, sm, o, so):
    """(m - o) / sqrt(sm^2 + so^2); 0 when both SEs are 0 and the means
    agree exactly, infinite when they do not."""
    se = math.sqrt(sm * sm + so * so)
    if se == 0.0:
        return 0.0 if m == o else math.copysign(math.inf, m - o)
    return (m - o) / se


def zarray(m, sm, o, so):
    m, sm, o, so = (np.asarray(a, dtype=np.float64) for a in (m, sm, o, so))
    se = np.sqrt(sm * sm + so * so)
    out = np.zeros_like(m)
    pos = se > 0.0
    out[pos] = (m[pos] - o[pos]) / se[pos]
    bad = ~pos & (m != o)
    out[bad] = np.copysign(np.inf, (m - o)[bad])
    return out


def structurally_zero(m_doc, layer, score, nuclide):
    """No data for this response: the nuclide (or every nuclide of the
    layer, for the total) lacks the MT, so both codes score exactly 0."""
    absent = set(m_doc["response_absent"])
    if nuclide != "total":
        return f"{nuclide} {score}" in absent
    return all(f"{n['name']} {score}" in absent for n in layer["nuclides"])


# ---------------------------------------------------------------- checks
def check_row(label, m, o, weakest=False):
    return {"check": label, "mcslab": list(m), "openmc": list(o),
            "z": zscore(m[0], m[1], o[0], o[1]), "weakest": weakest}


def primary_checks(spec, name, m_doc, o_doc):
    """The pre-declared primary checks of one problem, in a fixed order."""
    rows = []
    for layer in spec["layers"]:
        ln = layer["name"]
        weak = (name, ln) in WEAKEST
        rows.append(check_row(f"{ln} flux", m_doc["layers"][ln]["flux"],
                              o_doc["layers"][ln]["flux"], weak))
        for score in SCORES:
            if structurally_zero(m_doc, layer, score, "total") or \
                    (name, ln, score) in DIAGNOSTIC_ONLY:
                continue
            rows.append(check_row(f"{ln} {SCORE_LABEL[score]}",
                                  m_doc["layers"][ln]["scores"][score]["total"],
                                  o_doc["layers"][ln]["scores"][score]["total"], weak))
        if ln == "FLiBe":
            for nuc in layer["nuclides"]:
                n = nuc["name"]
                rows.append(check_row(f"FLiBe H3 205 {n}",
                                      m_doc["layers"][ln]["scores"]["H3-production"][n],
                                      o_doc["layers"][ln]["scores"]["H3-production"][n]))
    if not spec["reflect_left"]:
        rows.append(check_row("leakage left", m_doc["leakage"]["left"],
                              o_doc["leakage"]["left"]))
    rows.append(check_row("leakage right", m_doc["leakage"]["right"],
                          o_doc["leakage"]["right"]))
    return rows


def exact_checks(problem_file, problems_sha, name, m_doc, o_doc, variant="main"):
    """(label, passed, detail) for the deterministic checks of one pair."""
    spec = problem_file["problems"][name]
    out = []

    def add(label, ok, detail=""):
        out.append((label, bool(ok), detail))

    # inputs
    add("problem file sha256 (mcslab)", m_doc["problems_sha256"] == problems_sha)
    add("problem file sha256 (OpenMC)", o_doc["problems_sha256"] == problems_sha)
    pins = pinned_checksums()
    data = problem_file["data_files"]
    add("data sha256 = pins", all(pins.get(k) == v for k, v in data.items()))
    add("OpenMC data sha256 = problem file", o_doc["provenance"]["data_files"] == data)
    add("OpenMC index = the problem's nuclides",
        o_doc["provenance"]["cross_sections_index"] ==
        sorted(k.split("/")[-1][:-3] for k in data))
    rb = o_doc["readback"]
    add("atom densities bit-identical",
        all(rb["atoms_per_barn_cm"][ly["name"]] ==
            {n["name"]: n["atoms_per_barn_cm"] for n in ly["nuclides"]}
            for ly in spec["layers"]))
    add("mesh edges bit-identical", rb["edges"]["mesh_x_cm"] == spec["mesh_edges_cm"])
    add("spectrum edges bit-identical",
        rb["edges"]["spectrum_eV"] == problem_file["spectrum_edges_eV"])
    # settings
    seeds = spec["seeds"]
    if variant == "main":
        o_seed, hist, ptables, cutoff = (seeds["openmc"], spec["histories_per_batch"], False,
                                         spec["energy_cutoff_eV"])
        m_seed = seeds["mcslab"]
    elif variant == "protocol":
        o_seed, hist, ptables, cutoff = (seeds["openmc_protocol"],
                                         spec["protocol_histories_per_batch"], False,
                                         spec["energy_cutoff_eV"])
        m_seed = seeds["mcslab_protocol"]
    else:
        diag = spec["openmc_diagnostics"][variant]
        o_seed, hist = diag["seed"], spec["histories_per_batch"]
        ptables = bool(diag.get("ptables", False))
        cutoff = diag.get("energy_cutoff_eV", spec["energy_cutoff_eV"])
        m_seed = seeds["mcslab"]
    s = o_doc["settings"]
    sp = s["statepoint"]
    add("OpenMC version and commit",
        o_doc["provenance"]["openmc_version"] == OPENMC_VERSION and sp["version"] ==
        OPENMC_VERSION and o_doc["provenance"]["openmc_commit"] == OPENMC_COMMIT and
        any(OPENMC_COMMIT in ln for ln in o_doc["provenance"]["openmc_version_text"]))
    add("OpenMC seed, batches, histories",
        s["seed"] == sp["seed"] == o_seed and
        s["n_batches"] == sp["n_batches"] == sp["n_realizations"] == spec["n_batches"] and
        s["histories_per_batch"] == sp["n_particles"] == hist)
    add("OpenMC settings as declared",
        s["run_mode"] == sp["run_mode"] == "fixed source" and s["threads"] == 1 and
        s["photon_transport"] is False and s["ptables"] is ptables and
        s["survival_biasing"] is False and s["event_based"] is False and
        s["energy_cutoff_eV"] == cutoff and s["free_gas_threshold"] == 400.0 and
        s["temperature"] == {"method": "nearest", "tolerance": 10.0, "default": 293.6,
                             "multipole": False} and s["estimator"] == "tracklength")
    ms = m_doc["settings"]
    add("mcslab seed, batches, histories, cutoff",
        ms["seed"] == m_seed and ms["n_batches"] == spec["n_batches"] and
        ms["histories_per_batch"] == (spec["protocol_histories_per_batch"]
                                      if variant == "protocol" else spec["histories_per_batch"])
        and ms["energy_cutoff_eV"] == spec["energy_cutoff_eV"] and
        ms["reflect_left"] == spec["reflect_left"] and ms["free_gas_threshold"] == 400.0)
    # exact zeros
    zeros = []
    for layer in spec["layers"]:
        ln = layer["name"]
        for score in SCORES:
            for nuc in [n["name"] for n in layer["nuclides"]] + ["total"]:
                if structurally_zero(m_doc, layer, score, nuc):
                    zeros.append(m_doc["layers"][ln]["scores"][score][nuc] == [0.0, 0.0] and
                                 o_doc["layers"][ln]["scores"][score][nuc] == [0.0, 0.0])
    add("no-data responses exactly 0 in both codes", all(zeros), f"{len(zeros)} values")
    if spec["reflect_left"]:
        add("left leakage exactly 0 in both codes (reflective)",
            m_doc["leakage"]["left"] == [0.0, 0.0] and o_doc["leakage"]["left"] == [0.0, 0.0])
    # bookkeeping
    b = m_doc["bookkeeping"]
    add("mcslab balance residual exactly 0, nothing lost",
        b["balance"]["residual"] == 0.0 and b["lost"] == 0 and b["balance"]["lost"] == 0.0)
    add("mcslab max draws <= STRIDE", b["max_draws"] <= b["stride"],
        f"{b['max_draws']} / {b['stride']}")
    ob = o_doc["bookkeeping"]
    add("OpenMC: no lost particle, no warning",
        ob["n_lost_messages"] == 0 and not ob["warnings"])
    return out


# ---------------------------------------------------------------- diagnostics
def first_flight(spec):
    """Expected uncollided track length per bin per source neutron:
    (exp(-tau_a) - exp(-tau_b)) / Sigma_t, Sigma_t at the source energy."""
    edges = np.asarray(spec["mesh_edges_cm"])
    sig = np.repeat([ly["sigma_t_source_per_cm"] for ly in spec["layers"]],
                    [ly["depth_bins"] for ly in spec["layers"]])
    tau = np.concatenate([[0.0], np.cumsum(sig * np.diff(edges))])
    return (np.exp(-tau[:-1]) - np.exp(-tau[1:])) / sig


def zstats(z):
    """Summary of finite z values; non-finite ones (both SEs 0, means
    different) are counted, never averaged."""
    z = np.asarray(z, dtype=np.float64)
    bad = int((~np.isfinite(z)).sum())
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n": 0, "n_nonfinite": bad}
    a = np.abs(z)
    return {"n": int(z.size), "frac_gt_2": float((a > 2.0).mean()),
            "frac_gt_3": float((a > 3.0).mean()), "max_abs": float(a.max()),
            "mean": float(z.mean()), "std": float(z.std()), "n_nonfinite": bad}


def source_edge(spec, spectrum_edges):
    """Index k of an interior spectrum edge equal to the source energy, or
    None. OpenMC's EnergyFilter puts an energy on an edge into the bin
    below it (lower_bound_index, src/tallies/filter_energy.cpp), mcslab
    into the bin above (transport_kin.energy_bin), so the uncollided source
    neutrons of such a problem land in bins k - 1 (OpenMC) and k (mcslab)."""
    e0 = spec["source"]["energy_eV"]
    hits = [k for k in range(1, len(spectrum_edges) - 1) if spectrum_edges[k] == e0]
    return hits[0] if hits else None


def edge_bins(spec, spectrum_edges, m_doc, o_doc):
    """The two spectrum bins at a source energy on an edge, per layer: each
    bin's z and the z of their sum (SE assuming independent bins).
    Diagnostic only; added after the P4 run showed the effect."""
    k = source_edge(spec, spectrum_edges)
    if k is None:
        return None
    out = {"edge_index": k, "edge_eV": spectrum_edges[k], "layers": {}}
    for layer in spec["layers"]:
        a, b = m_doc["spectrum"][layer["name"]], o_doc["spectrum"][layer["name"]]
        rows = {str(g): {"mcslab": [a["mean"][g], a["se"][g]], "openmc": [b["mean"][g], b["se"][g]],
                         "z": zscore(a["mean"][g], a["se"][g], b["mean"][g], b["se"][g])}
                for g in (k - 1, k)}
        ms = a["mean"][k - 1] + a["mean"][k]
        os_ = b["mean"][k - 1] + b["mean"][k]
        rows["merged"] = {"mcslab": [ms, math.hypot(a["se"][k - 1], a["se"][k])],
                          "openmc": [os_, math.hypot(b["se"][k - 1], b["se"][k])]}
        rows["merged"]["z"] = zscore(*rows["merged"]["mcslab"], *rows["merged"]["openmc"])
        out["layers"][layer["name"]] = rows
    return out


def per_bin(spec, name, m_doc, o_doc, spectrum_edges=None):
    """z per bin for every per-bin quantity, by category, plus labels of
    the extreme values and of bins where exactly one code is 0. The two
    spectrum bins at a source energy lying on an edge are left out (see
    edge_bins)."""
    cats = {"flux": [], "scores": [], "spectrum": [], "uncollided": []}
    labels = {k: [] for k in cats}
    one_sided = []
    k_edge = None if spectrum_edges is None else source_edge(spec, spectrum_edges)

    def add(cat, label, m, o):
        mm, ms, om, os_ = (np.asarray(v, dtype=np.float64)
                           for v in (m["mean"], m["se"], o["mean"], o["se"]))
        keep = ~((mm == 0.0) & (om == 0.0))
        if cat == "spectrum" and k_edge is not None:
            keep[[k_edge - 1, k_edge]] = False
        for i in np.flatnonzero(keep & ((mm == 0.0) ^ (om == 0.0))):
            one_sided.append(f"{label} bin {i}: mcslab {mm[i]:.3e}, OpenMC {om[i]:.3e}")
        z = zarray(mm, ms, om, os_)
        for i in np.flatnonzero(keep):
            cats[cat].append(z[i])
            labels[cat].append(f"{label} bin {i}")

    for layer in spec["layers"]:
        ln = layer["name"]
        add("flux", f"{ln} flux", m_doc["bins"][ln]["flux"], o_doc["bins"][ln]["flux"])
        for score in SCORES:
            for nuc in [n["name"] for n in layer["nuclides"]] + ["total"]:
                if structurally_zero(m_doc, layer, score, nuc):
                    continue
                add("scores", f"{ln} {score} {nuc}", m_doc["bins"][ln]["scores"][score][nuc],
                    o_doc["bins"][ln]["scores"][score][nuc])
        add("spectrum", f"{ln} spectrum", m_doc["spectrum"][ln], o_doc["spectrum"][ln])
        add("uncollided", f"{ln} uncollided", m_doc["uncollided"][ln],
            o_doc["uncollided"][ln])
    out = {"by_category": {}, "one_sided_zero_bins": one_sided}
    all_z, all_l = [], []
    for cat in cats:
        z = np.asarray(cats[cat])
        st = zstats(z)
        if z.size:
            st["max_at"] = labels[cat][int(np.argmax(np.abs(z)))]
        out["by_category"][cat] = st
        all_z += cats[cat]
        all_l += labels[cat]
    st = zstats(all_z)
    if all_z:
        st["max_at"] = all_l[int(np.argmax(np.abs(all_z)))]
    out["all"] = st
    return out, np.asarray(all_z)


def uncollided(spec, m_doc, o_doc):
    """Both codes' uncollided flux per bin against the first-flight formula
    (z with each code's own SE), and OpenMC's n_collision = 0 flux of
    banked secondaries (deviation 16)."""
    an = first_flight(spec)
    names = [ly["name"] for ly in spec["layers"]]
    cat = {}
    for code, doc in (("mcslab", m_doc), ("openmc", o_doc)):
        m = np.concatenate([doc["uncollided"][n]["mean"] for n in names])
        s = np.concatenate([doc["uncollided"][n]["se"] for n in names])
        # a bin with no event (SE 0) cannot be compared with the formula;
        # it is counted with the track length the formula expects there
        scored = s > 0.0
        cat[code] = zstats(zarray(m[scored], s[scored], an[scored], np.zeros(scored.sum())))
        cat[code]["bins_without_events"] = int((~scored).sum())
        cat[code]["expected_in_those_bins_per_source"] = float(an[~scored].sum())
    sec = {n: float(np.sum(o_doc["uncollided_secondaries"][n]["mean"])) for n in names}
    prim = {n: float(np.sum(o_doc["uncollided"][n]["mean"])) for n in names}
    return {"vs_first_flight": cat,
            "openmc_secondary_first_flight_per_layer": sec,
            "openmc_primary_uncollided_per_layer": prim}


def mesh_vs_cell(spec, o_doc):
    """Largest relative difference between OpenMC's mesh-bin sum and its
    cell tally per layer, over flux and the totals of every score
    (round-off and slivers only)."""
    worst = 0.0
    for layer in spec["layers"]:
        ln = layer["name"]
        pairs = [(o_doc["bins"][ln]["flux"]["mean"], o_doc["layers"][ln]["flux"][0])]
        pairs += [(o_doc["bins"][ln]["scores"][s]["total"]["mean"],
                   o_doc["layers"][ln]["scores"][s]["total"][0]) for s in SCORES]
        for bins, cell in pairs:
            if cell != 0.0:
                worst = max(worst, abs(math.fsum(bins) - cell) / abs(cell))
    return worst


def default_settings(spec, name, m_doc, base, diag):
    """D37: change of each primary quantity when OpenMC runs with one
    default setting, in percent and in SE (two independent OpenMC runs),
    and the z distribution of that run against mcslab."""
    rows = []
    for b, d in zip(primary_checks(spec, name, m_doc, base),
                    primary_checks(spec, name, m_doc, diag)):
        (bm, bs), (dm, ds) = b["openmc"], d["openmc"]
        rows.append({"check": b["check"], "base": [bm, bs], "variant": [dm, ds],
                     "change_percent": 100.0 * (dm - bm) / bm if bm != 0.0 else None,
                     "change_se": zscore(dm, ds, bm, bs), "z_vs_mcslab": d["z"]})
    pct = [abs(r["change_percent"]) for r in rows if r["change_percent"] is not None]
    k = int(np.argmax([abs(r["change_se"]) for r in rows]))
    return {"rows": rows, "max_abs_change_percent": max(pct),
            "max_abs_change_se": abs(rows[k]["change_se"]), "max_change_se_at": rows[k]["check"],
            "z_vs_mcslab": zstats([r["z_vs_mcslab"] for r in rows]),
            "n_beyond_3_vs_mcslab": int(sum(abs(r["z_vs_mcslab"]) > Z_LIMIT for r in rows))}


# ---------------------------------------------------------------- driver
def compare_problem(problem_file, problems_sha, name):
    spec = problem_file["problems"][name]
    m_doc, o_doc = load("mcslab", name), load("openmc", name)
    primary = primary_checks(spec, name, m_doc, o_doc)
    exact = exact_checks(problem_file, problems_sha, name, m_doc, o_doc)
    bins, _ = per_bin(spec, name, m_doc, o_doc, problem_file["spectrum_edges_eV"])
    diag_only = [check_row(f"{ln} {SCORE_LABEL[s]} (diagnostic only, D38)",
                           m_doc["layers"][ln]["scores"][s]["total"],
                           o_doc["layers"][ln]["scores"][s]["total"])
                 for (p, ln, s) in sorted(DIAGNOSTIC_ONLY) if p == name]
    out = {
        "title": spec["title"],
        "primary": primary,
        "n_primary": len(primary),
        "n_primary_failed": sum(abs(r["z"]) > Z_LIMIT for r in primary),
        "max_abs_z": max(abs(r["z"]) for r in primary),
        "exact": [{"check": lab, "passed": ok, "detail": det} for lab, ok, det in exact],
        "diagnostic_only": diag_only,
        "per_bin": bins,
        "spectrum_source_edge_bins": edge_bins(spec, problem_file["spectrum_edges_eV"],
                                               m_doc, o_doc),
        "uncollided": uncollided(spec, m_doc, o_doc),
        "openmc_mesh_vs_cell_max_relative": mesh_vs_cell(spec, o_doc),
        "openmc_mesh_slivers_max_relative": o_doc["mesh_slivers_max_relative"],
        "bookkeeping": {
            "mcslab_cutoff_weight_per_source": m_doc["bookkeeping"]["cutoff_weight_per_source"],
            "mcslab_zero_yield_weight_per_source":
                m_doc["bookkeeping"]["zero_yield_weight_per_source"],
            "mcslab_reflected_weight_per_source":
                m_doc["bookkeeping"]["reflected_weight_per_source"],
            "mcslab_max_draws": m_doc["bookkeeping"]["max_draws"]},
    }
    if name in D37 and load("openmc", name, D37[name]) is not None:
        diag = load("openmc", name, D37[name])
        out["default_settings"] = {
            "variant": D37[name],
            "exact": [{"check": lab, "passed": ok, "detail": det} for lab, ok, det in
                      exact_checks(problem_file, problems_sha, name, m_doc, diag, D37[name])],
            **default_settings(spec, name, m_doc, o_doc, diag)}
    m_p, o_p = load("mcslab", name, "protocol"), load("openmc", name, "protocol")
    if m_p is not None and o_p is not None:
        out["protocol"] = {
            "primary": primary_checks(spec, name, m_p, o_p),
            "exact": [{"check": lab, "passed": ok, "detail": det} for lab, ok, det in
                      exact_checks(problem_file, problems_sha, name, m_p, o_p, "protocol")]}
    return out


def compare_all():
    pf = problems()
    sha = jsonio.file_sha256(PROBLEMS_JSON)
    per = {name: compare_problem(pf, sha, name) for name in PROBLEMS}
    prim = [r for p in per.values() for r in p["primary"]]
    return {
        "description": "Phase 4 comparison of mcslab and OpenMC 0.16.0 "
                       "(benchmark/compare.py, docs/phase4_plan.md).",
        "problems_sha256": sha,
        "z_limit": Z_LIMIT,
        "n_primary": len(prim),
        "n_primary_failed": sum(abs(r["z"]) > Z_LIMIT for r in prim),
        "max_abs_z": max(abs(r["z"]) for r in prim),
        "max_abs_z_at": max(((abs(r["z"]), f"{n} {r['check']}") for n, p in per.items()
                             for r in p["primary"]))[1],
        "exact_all_passed": all(e["passed"] for p in per.values() for e in p["exact"]),
        "problems": per,
    }


# ---------------------------------------------------------------- figures
# Reference data-viz palette (light surface), as the Phase 3 figures:
# categorical slots 1 and 2 for the two codes, ink for references.
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#f0efec"
SERIES = ("#2a78d6", "#eb6834")
FIG_DIR = os.path.join(REPO, "docs", "figures")
RATIO_PANELS = (("flux", "flux"),) + tuple((s, SCORE_LABEL[s]) for s in SCORES)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _layer_bands(ax, spec, x_of_layer):
    for i, (layer, (x0, x1)) in enumerate(zip(spec["layers"], x_of_layer)):
        if i % 2 == 0:
            ax.axvspan(x0, x1, color=BAND, zorder=0, lw=0)
        ax.text(0.5 * (x0 + x1), 0.97, layer["name"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK2, fontsize=7.5, zorder=4)


def _ratio(m, sm, o, so):
    m, sm, o, so = (np.asarray(v, dtype=np.float64) for v in (m, sm, o, so))
    ok = (m != 0.0) & (o != 0.0)
    r = np.full_like(m, np.nan)
    e = np.full_like(m, np.nan)
    r[ok] = m[ok] / o[ok]
    e[ok] = r[ok] * np.sqrt((sm[ok] / m[ok]) ** 2 + (so[ok] / o[ok]) ** 2)
    return r, e


def figure_ratio(pf, name, m_doc, o_doc, path):
    """mcslab / OpenMC per depth bin for the flux and every score of the
    material total, +-1 SE. x is the bin index, so every bin has the same
    width whatever its thickness; layers are banded. Points beyond the
    y range are drawn as triangles on its edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = pf["problems"][name]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), facecolor=SURFACE)
    axes = axes.ravel()
    counts = [ly["depth_bins"] for ly in spec["layers"]]
    starts = np.concatenate([[0], np.cumsum(counts)])
    for k, (key, label) in enumerate(RATIO_PANELS):
        ax = axes[k]
        _style(ax)
        _layer_bands(ax, spec, [(starts[i] - 0.5, starts[i + 1] - 0.5)
                                for i in range(len(counts))])
        ax.axhline(1.0, color=INK2, lw=1.0, zorder=1)
        xs, rs, es = [], [], []
        for i, layer in enumerate(spec["layers"]):
            ln = layer["name"]
            if key == "flux":
                mb, ob = m_doc["bins"][ln]["flux"], o_doc["bins"][ln]["flux"]
            elif structurally_zero(m_doc, layer, key, "total"):
                continue
            else:
                mb = m_doc["bins"][ln]["scores"][key]["total"]
                ob = o_doc["bins"][ln]["scores"][key]["total"]
            r, e = _ratio(mb["mean"], mb["se"], ob["mean"], ob["se"])
            xs.append(np.arange(starts[i], starts[i + 1]))
            rs.append(r)
            es.append(e)
        if not xs:
            ax.set_title(label, loc="left", color=INK, fontsize=9.5, pad=4)
            ax.text(0.5, 0.5, "no data for any nuclide of this problem\n(both codes score "
                    "exactly 0)", transform=ax.transAxes, ha="center", va="center",
                    color=INK2, fontsize=8.5)
            ax.set_yticks([])
            ax.set_xlim(-0.5, starts[-1] - 0.5)
            continue
        x, r, e = np.concatenate(xs), np.concatenate(rs), np.concatenate(es)
        ok = np.isfinite(r)
        dev = np.abs(r[ok] - 1.0) + e[ok]
        lim = max(0.005, 1.15 * float(np.quantile(dev, 0.95))) if dev.size else 0.05
        inside = ok & (np.abs(r - 1.0) <= lim)
        ax.errorbar(x[inside], r[inside], yerr=e[inside], fmt="o", ms=3, color=SERIES[0],
                    ecolor=SERIES[0], elinewidth=0.9, capsize=0, lw=0, zorder=3)
        hi, lo = ok & (r > 1.0 + lim), ok & (r < 1.0 - lim)
        ax.plot(x[hi], np.full(hi.sum(), 1.0 + lim), "^", ms=4, color=SERIES[0], clip_on=False)
        ax.plot(x[lo], np.full(lo.sum(), 1.0 - lim), "v", ms=4, color=SERIES[0], clip_on=False)
        ax.set_ylim(1.0 - lim, 1.0 + lim)
        ax.set_xlim(-0.5, starts[-1] - 0.5)
        missing = [ly["name"] for ly in spec["layers"]
                   if key != "flux" and structurally_zero(m_doc, ly, key, "total")]
        note = f"  (no data in {', '.join(missing)})" if missing else ""
        ax.set_title(f"{label}{note}", loc="left", color=INK, fontsize=9.5, pad=4)
        ax.set_ylabel("mcslab / OpenMC", color=INK2, fontsize=8)
    for ax in axes[len(RATIO_PANELS):]:
        ax.axis("off")
    axes[-1].text(0.02, 0.9, "How to read: each point is mcslab / OpenMC for one depth bin,\n"
                  "+-1 SE of the ratio. Neighbouring bins share histories, so a\n"
                  "run-wide fluctuation (e.g. in the fast flux at depth)\n"
                  "moves a whole layer together. Pass/fail is judged only on\n"
                  "the layer-integrated primary checks (benchmark/compare.py).\n"
                  "Triangles: points beyond the y range, drawn on its edge.",
                  transform=axes[-1].transAxes, ha="left", va="top", color=INK2, fontsize=8.5)
    bins = ", ".join(f"{ly['name']} {ly['depth_bins']} x "
                     f"{(ly['x_cm'][1] - ly['x_cm'][0]) / ly['depth_bins']:g} cm"
                     for ly in spec["layers"])
    for ax in axes[len(RATIO_PANELS) - 2:len(RATIO_PANELS)]:
        ax.set_xlabel(f"depth bin ({bins})", color=INK2, fontsize=8)
    fig.suptitle(f"{name}: {spec['title']}\nmcslab / OpenMC per depth bin, material total, "
                 f"+-1 SE ({spec['n_batches']} x {spec['histories_per_batch']:,} "
                 "histories per code)", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_spectra(pf, name, m_doc, o_doc, path):
    """Flux per unit lethargy in each layer (both codes) and the ratio
    with +-1 SE, on the 246-bin grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = pf["problems"][name]
    edges = np.asarray(pf["spectrum_edges_eV"])
    du = np.log(edges[1:] / edges[:-1])
    n = len(spec["layers"])
    # energy axis: a decade below the lowest bin with flux in either code
    lowest = min(edges[np.flatnonzero(np.asarray(doc["spectrum"][ly["name"]]["mean"]) > 0)[0]]
                 for doc in (m_doc, o_doc) for ly in spec["layers"])
    xlim = (lowest / 10.0, edges[-1])
    fig, axes = plt.subplots(2, n, figsize=(max(4.2 * n + 0.6, 8.0), 6.4), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for j, layer in enumerate(spec["layers"]):
        ln = layer["name"]
        top, bot = axes[0, j], axes[1, j]
        _style(top)
        _style(bot)
        for k, (code, doc, ls) in enumerate((("mcslab", m_doc, "-"), ("OpenMC", o_doc, "--"))):
            y = np.asarray(doc["spectrum"][ln]["mean"]) / du
            top.stairs(np.where(y > 0, y, np.nan), edges, color=SERIES[k], lw=1.4, ls=ls,
                       label=code)
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_xlim(*xlim)
        top.set_title(f"{ln} ({layer['x_cm'][0]:g}-{layer['x_cm'][1]:g} cm)", loc="left",
                      color=INK, fontsize=9.5)
        if j == 0:
            top.set_ylabel("flux per unit lethargy\n(cm per source neutron)", color=INK2,
                           fontsize=8)
            top.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
        doc_m, doc_o = m_doc["spectrum"][ln], o_doc["spectrum"][ln]
        r, e = _ratio(doc_m["mean"], doc_m["se"], doc_o["mean"], doc_o["se"])
        c = np.sqrt(edges[:-1] * edges[1:])
        ok = np.isfinite(r) & (e < 0.2)
        bot.axhline(1.0, color=INK2, lw=1.0)
        bot.errorbar(c[ok], r[ok], yerr=e[ok], fmt="o", ms=2.2, color=SERIES[0],
                     ecolor=SERIES[0], elinewidth=0.7, capsize=0, lw=0)
        bot.set_xscale("log")
        bot.set_xlim(top.get_xlim())
        bot.set_ylim(0.8, 1.2)
        bot.set_xlabel("neutron energy (eV)", color=INK2, fontsize=8)
        if j == 0:
            bot.set_ylabel("mcslab / OpenMC\n(bins with SE < 20%)", color=INK2, fontsize=8)
    fig.suptitle(f"{name}: {spec['title']}\nTrack-length flux spectrum per layer, "
                 "246 bins (20 per decade)", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_z(pf, summary, path):
    """Per problem: histogram of the per-bin diagnostic z (flux, scores,
    spectrum, uncollided) against N(0,1), with the primary checks' z as
    ticks below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(13, 3.6), facecolor=SURFACE,
                             sharey=False)
    grid = np.linspace(-5, 5, 401)
    for ax, name in zip(axes, PROBLEMS):
        _style(ax)
        spec = pf["problems"][name]
        _, z = per_bin(spec, name, load("mcslab", name), load("openmc", name),
                       pf["spectrum_edges_eV"])
        z = z[np.isfinite(z)]
        clipped = np.clip(z, -5.0, 5.0)
        ax.hist(clipped, bins=np.linspace(-5, 5, 41), density=True, color=SERIES[0],
                alpha=0.85, edgecolor=SURFACE, lw=0.6)
        ax.plot(grid, np.exp(-0.5 * grid ** 2) / math.sqrt(2 * math.pi), color=INK, lw=1.4,
                label="N(0, 1)")
        prim = np.array([r["z"] for r in summary["problems"][name]["primary"]])
        ax.plot(np.clip(prim, -5, 5), np.full(prim.size, -0.02), "|", color=SERIES[1], ms=9,
                mew=1.4, clip_on=False, label="primary checks")
        a = summary["problems"][name]["per_bin"]["all"]
        ax.text(0.98, 0.97, f"{a['n']} bins\n|z|>2: {100 * a['frac_gt_2']:.1f}%\n"
                f"|z|>3: {100 * a['frac_gt_3']:.2f}%\nmax {a['max_abs']:.1f}",
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=7.5)
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.04, None)
        ax.set_xlabel("z (|z| > 5 shown at the edge)", color=INK2, fontsize=8)
    axes[0].set_ylabel("density", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Per-bin diagnostics, mcslab vs OpenMC: z distribution (bins are "
                 "correlated; descriptive only)", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def make_figures(summary):
    pf = problems()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = []
    for name in PROBLEMS:
        m_doc, o_doc = load("mcslab", name), load("openmc", name)
        for kind, fn in (("ratio", figure_ratio), ("spectra", figure_spectra)):
            path = os.path.join(FIG_DIR, f"phase4_{kind}_{name}.png")
            fn(pf, name, m_doc, o_doc, path)
            out.append(path)
    path = os.path.join(FIG_DIR, "phase4_z.png")
    figure_z(pf, summary, path)
    out.append(path)
    for p in out:
        print(f"wrote {os.path.relpath(p, REPO)}")


def fmt(pair, digits=5):
    return f"{pair[0]:.{digits}g} ± {pair[1]:.2g}"


def tables(summary):
    lines = []
    for name, p in summary["problems"].items():
        lines += ["", f"**{name}: {p['title']}**", "",
                  "| check | mcslab | OpenMC | mcslab / OpenMC | z |", "|---|---|---|---|---|"]
        for r in p["primary"] + p["diagnostic_only"]:
            ratio = r["mcslab"][0] / r["openmc"][0] if r["openmc"][0] != 0.0 else float("nan")
            flag = " (weakest)" if r["weakest"] else ""
            lines.append(f"| {r['check']}{flag} | {fmt(r['mcslab'])} | {fmt(r['openmc'])} | "
                         f"{ratio:.5f} | {r['z']:+.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figures", action="store_true",
                    help="also write docs/figures/phase4_*.png (needs matplotlib)")
    args = ap.parse_args()
    summary = compare_all()
    sha = jsonio.write(summary, COMPARISON_JSON)
    print(tables(summary))
    print(f"\n{summary['n_primary']} primary checks, {summary['n_primary_failed']} beyond "
          f"{Z_LIMIT} SE; largest |z| {summary['max_abs_z']:.2f} ({summary['max_abs_z_at']}); "
          f"exact checks {'all pass' if summary['exact_all_passed'] else 'FAIL'}")
    for name, p in summary["problems"].items():
        a = p["per_bin"]["all"]
        print(f"{name} per-bin diagnostics: n {a['n']}, |z|>2 {a['frac_gt_2']:.4f}, "
              f"|z|>3 {a['frac_gt_3']:.4f}, max {a['max_abs']:.2f} ({a.get('max_at')})")
        if "default_settings" in p:
            d = p["default_settings"]
            print(f"{name} {d['variant']}: max change {d['max_abs_change_percent']:.3f}% / "
                  f"{d['max_abs_change_se']:.2f} SE ({d['max_change_se_at']}); "
                  f"{d['n_beyond_3_vs_mcslab']} beyond 3 SE against mcslab")
    print(f"wrote {os.path.relpath(COMPARISON_JSON, REPO)} (sha256 {sha[:12]})")
    if args.figures:
        make_figures(summary)


if __name__ == "__main__":
    main()
