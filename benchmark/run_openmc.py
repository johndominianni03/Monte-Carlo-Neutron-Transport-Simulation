#!/usr/bin/env python3
"""Phase 4 benchmark, OpenMC side (docs/phase4_plan.md, D24-D30, D33, D36, D37).

Runs only in OpenMC's own environment (OpenMC 0.16.0, Python 3.13). It
never imports mcslab and reads only benchmark/problems.json.

    MCSLAB_DATA=... <openmc-env>/bin/python benchmark/run_openmc.py run \
        [--problem P1 ...] [--variant main|protocol|ptables_on|cutoff_0] [--workdir DIR]
    MCSLAB_DATA=... <openmc-env>/bin/python benchmark/run_openmc.py dev [--workdir DIR]

run  Builds each problem's model from the problem file, runs the OpenMC
     executable on one thread in a run directory outside the repository
     (default ~/mcslab_openmc_runs/phase4/<problem>_<variant>_<seed>/), and
     writes benchmark/results/openmc_<P>[_<variant>].json. Variants: main
     (100 x 10,000 on the declared seed), protocol (the failure-protocol
     run, 100 x 40,000 on the second seed), and the default-settings
     diagnostics of D37 declared per problem (ptables_on for P1, cutoff_0
     for P4).
dev  Development check: P1 at 2 x 1000 on seed 20261199; nothing is written
     to the repository.

Before running, every data file's sha256 is checked against the problem
file and scripts/checksums/endfb-viii.0.sha256, and a 14-entry
cross_sections.xml is written into the run root (D24). OpenMC finds it
through OPENMC_CROSS_SECTIONS, so its absolute path never enters model.xml
or the results. The results contain no paths, dates or timings: on the
same OpenMC build a rerun reproduces them byte for byte (one thread, D30).

Model (D26, D27, D33):
- x-planes at the problem bounds; reflective y and z planes at +-1000 cm
  (an infinite slab in effect); vacuum on the right.
- A void buffer cell between x = -1 cm (vacuum) and x = 0. OpenMC's
  source-site check (Source::satisfies_spatial_constraints,
  src/source.cpp) locates the start point with a fixed direction (0, 0, 1),
  which puts a point on the x = 0 plane on its negative side; without a
  cell there every start is rejected. The particle itself is then located
  with its own direction, +x (Particle::event_calculate_xs, Surface::sense),
  so it starts in the first layer at exactly x = 0. x = 0 is an internal
  surface (vacuum problems) or reflective, in which case the buffer is
  never entered.
- Materials: the problem file's per-nuclide atom densities (atom/b-cm,
  set_density("sum")) and temperatures; no cell temperatures, so cells take
  the material's (assign_temperatures, src/geometry_aux.cpp).
- Tallies, all track-length (set explicitly): mesh flux; mesh x 6 scores x
  nuclides + total; cell flux; cell x 6 scores x nuclides + total; cell x
  246-bin energy flux; surface current on both x boundaries (analog);
  mesh x CollisionFilter([0]) x energy {[0, E0(1 - 1e-9)), E0 +- 1e-9}.
  OpenMC 0.16.0 gives neutrons banked by (n,2n)-type reactions
  n_collision = 0 (deviation 16), so only the band at the source energy
  is the uncollided flux of source neutrons; the other bin is reported
  separately.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import h5py                                                  # noqa: E402
import numpy as np                                           # noqa: E402
import openmc                                                # noqa: E402

from benchmark import jsonio                                 # noqa: E402

PROBLEMS_JSON = os.path.join(HERE, "problems.json")
RESULTS_DIR = os.path.join(HERE, "results")
CHECKSUMS = os.path.join(REPO, "scripts", "checksums", "endfb-viii.0.sha256")
DEFAULT_WORKDIR = os.path.join("~", "mcslab_openmc_runs", "phase4")

OPENMC_VERSION = "0.16.0"
OPENMC_COMMIT = "617d35a5063c57796b43428bc401e627d2011046"   # the v0.16.0 tag
DEV_SEED = 20261199
K_BOLTZMANN = 8.617333262e-5                                  # eV/K, as OpenMC
TEMPERATURE = {"method": "nearest", "tolerance": 10.0, "default": 293.6,
               "multipole": False}
FREE_GAS_THRESHOLD = 400.0
# statepoint labels of the response scores (src/reaction.cpp)
STATEPOINT_SCORE = {"H3-production": "(n,Xt)", "He4-production": "(n,Xa)"}


# ---------------------------------------------------------------- setup
def openmc_build():
    """(executable, `openmc --version` lines); refuses another version or
    commit."""
    if openmc.__version__ != OPENMC_VERSION:
        sys.exit(f"OpenMC {openmc.__version__} found; the benchmark is fixed to "
                 f"{OPENMC_VERSION}")
    exe = os.path.join(os.path.dirname(sys.executable), "openmc")
    if not os.path.isfile(exe):
        exe = shutil.which("openmc")
    if exe is None:
        sys.exit("openmc executable not found next to this Python or on PATH")
    text = subprocess.run([exe, "--version"], capture_output=True, text=True,
                          check=True).stdout
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if not any(OPENMC_COMMIT in ln for ln in lines):
        sys.exit(f"openmc executable is not commit {OPENMC_COMMIT}:\n{text}")
    return exe, lines


def rosetta():
    """True when this (x86_64) process runs translated on Apple silicon."""
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return None
    return {"1": True, "0": False}.get(out)


def pinned_checksums():
    pins = {}
    with open(CHECKSUMS) as fh:
        for line in fh:
            digest, path = line.split()
            pins[path] = digest
    return pins


def data_root(problems):
    """$MCSLAB_DATA, with every file the problems use checked against the
    problem file and the repository's pins."""
    root = os.environ.get("MCSLAB_DATA")
    if not root:
        sys.exit("set MCSLAB_DATA (see scripts/fetch_data.sh)")
    root = os.path.expanduser(root)
    pins = pinned_checksums()
    for rel, digest in problems["data_files"].items():
        actual = jsonio.file_sha256(os.path.join(root, rel))
        if actual != digest or pins.get(rel) != digest:
            sys.exit(f"{rel}: sha256 {actual} differs from the problem file or the pins")
    return root


def write_index(root, problems, workdir):
    """The 14-entry cross_sections.xml (D24), in the run root."""
    lib = openmc.data.DataLibrary()
    for rel in sorted(problems["data_files"]):
        lib.register_file(os.path.join(root, rel))
    path = os.path.join(workdir, "cross_sections.xml")
    lib.export_to_xml(path)
    return path, sorted(m for entry in lib.libraries for m in entry["materials"])


def check_temperatures(root, spec):
    """Each nuclide's data temperature by OpenMC's NEAREST rule (src/nuclide.cpp:
    available temperatures round(kT / k_B), nearest within the tolerance)
    must be the problem file's, with the same kT."""
    for layer in spec["layers"]:
        for nuc in layer["nuclides"]:
            with h5py.File(os.path.join(root, "neutron", nuc["name"] + ".h5"), "r") as f:
                kts = {k: float(f[nuc["name"]]["kTs"][k][()]) for k in f[nuc["name"]]["kTs"]}
            avail = sorted((math.floor(kt / K_BOLTZMANN + 0.5), k) for k, kt in kts.items())
            t_actual, key = min(avail, key=lambda a: abs(a[0] - layer["temperature_K"]))
            if not abs(t_actual - layer["temperature_K"]) < TEMPERATURE["tolerance"]:
                sys.exit(f"{nuc['name']}: no data within the tolerance")
            if key != nuc["data_temperature"] or kts[key] != nuc["kT_eV"]:
                sys.exit(f"{nuc['name']}: OpenMC would use {key} (kT {kts[key]}), the "
                         f"problem file says {nuc['data_temperature']} ({nuc['kT_eV']})")


# ---------------------------------------------------------------- model
def variant_setup(spec, variant):
    """(seed, histories per batch, ptables, energy cutoff or None)."""
    seeds = spec["seeds"]
    if variant == "main":
        return seeds["openmc"], spec["histories_per_batch"], False, spec["energy_cutoff_eV"]
    if variant == "protocol":
        return (seeds["openmc_protocol"], spec["protocol_histories_per_batch"], False,
                spec["energy_cutoff_eV"])
    diag = spec["openmc_diagnostics"].get(variant)
    if diag is None:
        raise ValueError(f"variant {variant} is not declared for this problem")
    cutoff = diag.get("energy_cutoff_eV", spec["energy_cutoff_eV"])
    return (diag["seed"], spec["histories_per_batch"], bool(diag.get("ptables", False)),
            None if cutoff == 0.0 else cutoff)


def build_model(problems, spec, seed, histories, ptables, cutoff, n_batches=None):
    consts = problems["openmc_model"]
    half, buf = consts["lateral_half_width_cm"], consts["buffer_cm"]
    bounds = spec["bounds_cm"]

    mats = []
    for layer in spec["layers"]:
        mat = openmc.Material(name=layer["name"], temperature=layer["temperature_K"])
        for nuc in layer["nuclides"]:
            mat.add_nuclide(nuc["name"], nuc["atoms_per_barn_cm"], "ao")
        mat.set_density("sum")
        mats.append(mat)

    xs = [openmc.XPlane(x0=b) for b in bounds]
    xs[0].boundary_type = "reflective" if spec["reflect_left"] else "transmission"
    xs[-1].boundary_type = "vacuum"
    x_buffer = openmc.XPlane(x0=bounds[0] - buf, boundary_type="vacuum")
    lateral = (+openmc.YPlane(-half, boundary_type="reflective")
               & -openmc.YPlane(half, boundary_type="reflective")
               & +openmc.ZPlane(-half, boundary_type="reflective")
               & -openmc.ZPlane(half, boundary_type="reflective"))
    cells = [openmc.Cell(name=layer["name"], fill=mat, region=+xs[i] & -xs[i + 1] & lateral)
             for i, (layer, mat) in enumerate(zip(spec["layers"], mats))]
    buffer_cell = openmc.Cell(name="buffer", fill=None, region=+x_buffer & -xs[0] & lateral)

    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.batches = spec["n_batches"] if n_batches is None else n_batches
    s.particles = histories
    s.seed = seed
    s.photon_transport = False
    s.ptables = ptables
    s.survival_biasing = False
    s.temperature = dict(TEMPERATURE)
    s.free_gas_threshold = FREE_GAS_THRESHOLD
    s.event_based = False
    if cutoff is not None:
        s.cutoff = {"energy_neutron": cutoff}
    src = spec["source"]
    s.source = openmc.IndependentSource(
        space=openmc.stats.Point(tuple(src["position_cm"])),
        angle=openmc.stats.Monodirectional(tuple(src["direction"])),
        energy=openmc.stats.Discrete([src["energy_eV"]], [1.0]), strength=1.0)
    s.output = {"tallies": False, "summary": True}

    mesh = openmc.RectilinearMesh(name="depth")
    mesh.x_grid = spec["mesh_edges_cm"]
    mesh.y_grid = [-half, half]
    mesh.z_grid = [-half, half]
    mf = openmc.MeshFilter(mesh)
    cf = openmc.CellFilter(cells)
    nuclides = [n["name"] for layer in spec["layers"] for n in layer["nuclides"]] + ["total"]
    e0 = src["energy_eV"]
    band = consts["uncollided_band_rel"]

    def tally(name, filters, scores, nucs=None, estimator="tracklength"):
        t = openmc.Tally(name=name)
        t.filters = filters
        t.scores = scores
        if nucs:
            t.nuclides = nucs
        if estimator:
            t.estimator = estimator
        return t

    tallies = openmc.Tallies([
        tally("mesh_flux", [mf], ["flux"]),
        tally("mesh_scores", [mf], list(problems["responses"]), nuclides),
        tally("cell_flux", [cf], ["flux"]),
        tally("cell_scores", [cf], list(problems["responses"]), nuclides),
        tally("cell_spectrum", [cf, openmc.EnergyFilter(problems["spectrum_edges_eV"])],
              ["flux"]),
        tally("leakage", [openmc.SurfaceFilter([xs[0], xs[-1]])], ["current"],
              estimator=None),
        tally("uncollided", [mf, openmc.CollisionFilter([0]),
                             openmc.EnergyFilter([0.0, e0 * (1.0 - band), e0 * (1.0 + band)])],
              ["flux"]),
    ])
    return openmc.Model(geometry=openmc.Geometry(cells + [buffer_cell]),
                        materials=openmc.Materials(mats), settings=s, tallies=tallies)


# ---------------------------------------------------------------- run
def run_openmc(exe, model, run_dir, index):
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    model.export_to_model_xml(os.path.join(run_dir, "model.xml"))
    env = dict(os.environ, OPENMC_CROSS_SECTIONS=index)
    proc = subprocess.run([exe, "-s", "1"], cwd=run_dir, env=env, capture_output=True,
                          text=True)
    with open(os.path.join(run_dir, "openmc.log"), "w") as fh:
        fh.write(proc.stdout)
        fh.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(f"openmc failed ({proc.returncode}); see {run_dir}/openmc.log")
    with open(os.path.join(run_dir, "model.xml")) as fh:
        model_xml = fh.read()
    return proc.stdout + proc.stderr, model_xml


def score_index(t, score):
    return list(t.scores).index(STATEPOINT_SCORE.get(score, score))


def pair(mean, se):
    return [float(mean), float(se)]


def arrays(mean, se):
    return {"mean": [float(v) for v in mean], "se": [float(v) for v in se]}


def extract(problems, spec, run_dir, n_batches):
    """Layer, per-bin, uncollided, spectrum and leakage results plus the
    read-back inputs, in the mcslab results schema."""
    edges = np.asarray(spec["mesh_edges_cm"])
    n_bins = edges.size - 1
    bin_layer = np.concatenate([np.full(layer["depth_bins"], i)
                                for i, layer in enumerate(spec["layers"])])
    n_e = len(problems["spectrum_edges_eV"]) - 1
    layers, bins, unc, unc_sec, spectrum, slivers = {}, {}, {}, {}, {}, {}
    with openmc.StatePoint(os.path.join(run_dir, f"statepoint.{n_batches}.h5")) as sp:
        mesh_flux = sp.get_tally(name="mesh_flux")
        mesh_scores = sp.get_tally(name="mesh_scores")
        cell_flux = sp.get_tally(name="cell_flux")
        cell_scores = sp.get_tally(name="cell_scores")
        cell_spec = sp.get_tally(name="cell_spectrum")
        leak = sp.get_tally(name="leakage")
        uncoll = sp.get_tally(name="uncollided")
        for t in (mesh_flux, mesh_scores, cell_flux, cell_scores, cell_spec, uncoll):
            if t.estimator != "tracklength":
                sys.exit(f"tally {t.name} used the {t.estimator} estimator")
        nucs = list(mesh_scores.nuclides)
        um, us = uncoll.mean.reshape(n_bins, 2), uncoll.std_dev.reshape(n_bins, 2)
        sm = cell_spec.mean.reshape(len(spec["layers"]), n_e)
        ss = cell_spec.std_dev.reshape(len(spec["layers"]), n_e)
        for i, layer in enumerate(spec["layers"]):
            name = layer["name"]
            sel = bin_layer == i
            present = [n["name"] for n in layer["nuclides"]] + ["total"]
            lay = {"flux": pair(cell_flux.mean.ravel()[i], cell_flux.std_dev.ravel()[i]),
                   "scores": {}}
            per = {"flux": arrays(mesh_flux.mean.ravel()[sel], mesh_flux.std_dev.ravel()[sel]),
                   "scores": {}}
            for score in problems["responses"]:
                lay["scores"][score], per["scores"][score] = {}, {}
                for nuc in present:
                    k, s = nucs.index(nuc), score_index(cell_scores, score)
                    lay["scores"][score][nuc] = pair(cell_scores.mean[i, k, s],
                                                     cell_scores.std_dev[i, k, s])
                    k, s = nucs.index(nuc), score_index(mesh_scores, score)
                    per["scores"][score][nuc] = arrays(mesh_scores.mean[sel, k, s],
                                                       mesh_scores.std_dev[sel, k, s])
            layers[name], bins[name] = lay, per
            unc[name] = arrays(um[sel, 1], us[sel, 1])
            unc_sec[name] = arrays(um[sel, 0], us[sel, 0])
            spectrum[name] = arrays(sm[i], ss[i])
        # round-off slivers: a nuclide's mesh score outside its own layer,
        # relative to its largest value inside (deviation 14, Phase 4 finding)
        for score in problems["responses"]:
            worst = 0.0
            for i, layer in enumerate(spec["layers"]):
                for n in layer["nuclides"]:
                    v = np.abs(mesh_scores.mean[:, nucs.index(n["name"]),
                                                score_index(mesh_scores, score)])
                    inside = v[bin_layer == i].max()
                    if inside > 0.0:
                        worst = max(worst, float(v[bin_layer != i].max(initial=0.0) / inside))
            slivers[score] = worst
        lm, ls = leak.mean.ravel(), leak.std_dev.ravel()
        leakage = {"left": pair(0.0 - lm[0], ls[0]), "right": pair(lm[1], ls[1])}
        mesh = sp.meshes[mesh_flux.find_filter(openmc.MeshFilter).mesh.id]
        readback_edges = {
            "mesh_x_cm": [float(x) for x in mesh.x_grid],
            "spectrum_eV": [float(x) for x in cell_spec.find_filter(openmc.EnergyFilter).values],
        }
        sp_info = {"version": ".".join(str(int(v)) for v in sp.version), "seed": int(sp.seed),
                   "n_batches": int(sp.n_batches), "n_particles": int(sp.n_particles),
                   "run_mode": str(sp.run_mode),
                   "n_realizations": int(mesh_flux.num_realizations)}
    densities = {}
    with h5py.File(os.path.join(run_dir, "summary.h5"), "r") as f:
        for key in f["materials"]:
            g = f["materials"][key]
            name = g["name"][()].decode()
            densities[name] = {n.decode(): float(d) for n, d in
                               zip(g["nuclides"][()], g["nuclide_densities"][()])}
    return (layers, bins, unc, unc_sec, spectrum, leakage, slivers, readback_edges,
            sp_info, densities)


def log_summary(log, redact):
    """Counts of warnings and lost-particle messages; warning text with the
    local paths replaced."""
    warnings_ = [ln.strip() for ln in log.splitlines() if "WARNING" in ln]
    for path, label in redact:
        warnings_ = [w.replace(path, label) for w in warnings_]
    lost = [ln for ln in log.splitlines() if re.search(r"\blost\b", ln, re.IGNORECASE)]
    return {"warnings": warnings_, "n_lost_messages": len(lost)}


def results_path(name, variant):
    suffix = "" if variant == "main" else f"_{variant}"
    return os.path.join(RESULTS_DIR, f"openmc_{name}{suffix}.json")


def run_one(exe, version_lines, problems, problems_sha, name, variant, workdir, index,
            index_entries, root, n_batches=None, seed=None, histories=None):
    spec = problems["problems"][name]
    check_temperatures(root, spec)
    v_seed, v_hist, ptables, cutoff = variant_setup(spec, variant)
    seed = v_seed if seed is None else seed
    histories = v_hist if histories is None else histories
    n_batches = spec["n_batches"] if n_batches is None else n_batches
    model = build_model(problems, spec, seed, histories, ptables, cutoff, n_batches)
    run_dir = os.path.join(workdir, f"{name}_{variant}_{seed}")
    log, model_xml = run_openmc(exe, model, run_dir, index)
    redact = [(workdir, "$RUN"), (root, "$MCSLAB_DATA"), (os.path.expanduser("~"), "~")]
    for path, _ in redact:
        if path in model_xml:
            sys.exit(f"model.xml contains a local path ({path})")
    (layers, bins, unc, unc_sec, spectrum, leakage, slivers, readback_edges, sp_info,
     densities) = extract(problems, spec, run_dir, n_batches)
    return {
        "code": "openmc",
        "problem": name,
        "variant": variant,
        "problems_sha256": problems_sha,
        "settings": {
            "seed": seed, "n_batches": n_batches, "histories_per_batch": histories,
            "run_mode": "fixed source", "threads": 1, "event_based": False,
            "photon_transport": False, "ptables": ptables, "survival_biasing": False,
            "weight_windows": False, "resonance_scattering": False,
            "energy_cutoff_eV": 0.0 if cutoff is None else cutoff,
            "temperature": dict(TEMPERATURE), "free_gas_threshold": FREE_GAS_THRESHOLD,
            "source_strength": 1.0, "estimator": "tracklength",
            "statepoint": sp_info},
        "bookkeeping": log_summary(log, redact),
        "leakage": leakage,
        "layers": layers,
        "bins": bins,
        "uncollided": unc,
        "uncollided_secondaries": unc_sec,
        "spectrum": spectrum,
        "readback": {"atoms_per_barn_cm": densities, "edges": readback_edges},
        "mesh_slivers_max_relative": slivers,
        "provenance": {
            "openmc_version": openmc.__version__,
            "openmc_commit": OPENMC_COMMIT,
            "openmc_version_text": version_lines,
            "python": platform.python_version(),
            "numpy": np.__version__, "h5py": h5py.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "rosetta_translated": rosetta(),
            "library": problems["library"],
            "data_files": problems["data_files"],
            "cross_sections_index": index_entries,
            "model_xml": model_xml,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("run", "dev"))
    ap.add_argument("--problem", nargs="*")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    args = ap.parse_args()

    exe, version_lines = openmc_build()
    problems = jsonio.read(PROBLEMS_JSON)
    problems_sha = jsonio.file_sha256(PROBLEMS_JSON)
    root = data_root(problems)
    workdir = os.path.realpath(os.path.expanduser(args.workdir))
    if workdir == REPO or workdir.startswith(REPO + os.sep):
        sys.exit("the run directory must be outside the repository")
    os.makedirs(workdir, exist_ok=True)
    index, entries = write_index(root, problems, workdir)

    if args.command == "dev":
        doc = run_one(exe, version_lines, problems, problems_sha, "P1", "main",
                      os.path.join(workdir, "dev"), index, entries, root,
                      n_batches=2, seed=DEV_SEED, histories=1000)
        os.makedirs(os.path.join(workdir, "dev"), exist_ok=True)
        out = os.path.join(workdir, "dev", "openmc_P1_dev.json")
        jsonio.write(doc, out)
        print(f"dev check written to {out}")
        print(json.dumps({k: doc[k] for k in ("leakage", "bookkeeping")}, indent=1))
        return

    names = args.problem or sorted(problems["problems"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in names:
        doc = run_one(exe, version_lines, problems, problems_sha, name, args.variant,
                      workdir, index, entries, root)
        path = results_path(name, args.variant)
        sha = jsonio.write(doc, path)
        print(f"{name} {args.variant}: lost messages "
              f"{doc['bookkeeping']['n_lost_messages']}, warnings "
              f"{len(doc['bookkeeping']['warnings'])} -> {os.path.relpath(path, REPO)} "
              f"(sha256 {sha[:12]})")


if __name__ == "__main__":
    main()
