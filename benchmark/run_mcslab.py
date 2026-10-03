#!/usr/bin/env python3
"""Phase 4 benchmark, mcslab side (docs/phase4_plan.md, D31-D36, D39).

Runs in mcslab's venv and never imports openmc.

    MCSLAB_DATA=... ./venv/bin/python benchmark/run_mcslab.py export
    MCSLAB_DATA=... ./venv/bin/python benchmark/run_mcslab.py run [--problem P1 ...] [--protocol]
    MCSLAB_DATA=... ./venv/bin/python benchmark/run_mcslab.py check --problem P1

export  Writes benchmark/problems.json, the only input of the OpenMC side:
        bounds; the kernel's own atom densities (PackedXS.mat_dens from
        pack_problem, checked bit for bit against
        CEMaterial.number_densities); data temperatures and kT; depth-mesh
        and spectrum edges; source; energy cutoff; Sigma_t at the source
        energy from raw h5py (for the first-flight formula); sizes; seeds;
        the sha256 of every data file (checked against
        scripts/checksums/endfb-viii.0.sha256).
run     Runs problems on their pre-declared seeds and writes
        benchmark/results/mcslab_<P>.json. --protocol runs the
        failure-protocol size (100 x 40,000) on the second seed and writes
        mcslab_<P>_protocol.json. Refuses to run if the problem file no
        longer matches this script.
check   Reruns one problem and compares everything but the provenance with
        the committed results file, byte for byte.

Results are per source neutron, means and standard errors over the 100
batches (tallies.batch_stats), track-length estimators throughout. Layer
values are per-batch sums over the layer's depth bins. Per-bin values are
kept for the nuclides present in each layer and the material total.
"""
import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import h5py                                                  # noqa: E402
import llvmlite                                              # noqa: E402
import numba                                                 # noqa: E402
import numpy as np                                           # noqa: E402
import platform                                              # noqa: E402

from benchmark import jsonio                                 # noqa: E402
from mcslab import ce_materials as cm                        # noqa: E402
from mcslab import tallies as T                              # noqa: E402
from mcslab.ce_materials import CEMaterial                   # noqa: E402
from mcslab.config_ce import MonoEnergetic                   # noqa: E402
from mcslab.config_kin import (EnergyCutoffWarning, KinRunConfig,  # noqa: E402
                               _energy_cutoff, data_temperature, pack_problem, run_kin)
from mcslab.depth_mesh import make_mesh                      # noqa: E402
from mcslab.geometry import SlabGeometry                     # noqa: E402
from mcslab.nucdata import Library, file_sha256              # noqa: E402
from mcslab.rng import STRIDE                                # noqa: E402
from mcslab.sources import BeamSource                        # noqa: E402

PROBLEMS_JSON = os.path.join(HERE, "problems.json")
RESULTS_DIR = os.path.join(HERE, "results")
CHECKSUMS = os.path.join(REPO, "scripts", "checksums", "endfb-viii.0.sha256")

E_FUSION = 14.1e6
T_FLIBE = 900.0
# Phase 3 spectrum grid (D12): edges 10^(k/20) eV, k = -100 .. 146 (246 bins)
SPECTRUM_EDGES = tuple(float(10.0 ** (k / 20.0)) for k in range(-100, 147))
N_BATCHES = 100
HISTORIES = 10_000
PROTOCOL_HISTORIES = 40_000

# OpenMC model constants (D26, D33), recorded in the problem file
LATERAL_HALF_WIDTH_CM = 1000.0
BUFFER_CM = 1.0
UNCOLLIDED_BAND_REL = 1e-9

# Problems, sizes and seeds as fixed in docs/phase4_plan.md (D31, D32, seeds)
PROBLEMS = {
    "P1": dict(title="D7: W 0.5 | FLiBe 20 | Fe 10 cm, vacuum on both sides",
               bounds=(0.0, 0.5, 20.5, 30.5), materials="fusion", depth_bins=(10, 20, 20),
               reflect_left=False, energy_eV=E_FUSION,
               seeds=dict(mcslab=20261040, openmc=20261100,
                          mcslab_protocol=20261044, openmc_protocol=20261140),
               openmc_diagnostics={"ptables_on": {"seed": 20261180, "ptables": True}}),
    "P2": dict(title="D7 with a reflective plasma side",
               bounds=(0.0, 0.5, 20.5, 30.5), materials="fusion", depth_bins=(10, 20, 20),
               reflect_left=True, energy_eV=E_FUSION,
               seeds=dict(mcslab=20261041, openmc=20261110,
                          mcslab_protocol=20261045, openmc_protocol=20261150),
               openmc_diagnostics={}),
    "P3": dict(title="W 0.5 | FLiBe 100 | Fe 10 cm, reflective plasma side",
               bounds=(0.0, 0.5, 100.5, 110.5), materials="fusion", depth_bins=(10, 50, 20),
               reflect_left=True, energy_eV=E_FUSION,
               seeds=dict(mcslab=20261042, openmc=20261120,
                          mcslab_protocol=20261046, openmc_protocol=20261160),
               openmc_diagnostics={}),
    "P4": dict(title="H-1 slab, 0.0708 g/cm^3, 30 cm, 1 MeV beam, vacuum on both sides",
               bounds=(0.0, 30.0), materials="hydrogen", depth_bins=(30,),
               reflect_left=False, energy_eV=1.0e6,
               seeds=dict(mcslab=20261043, openmc=20261130,
                          mcslab_protocol=20261047, openmc_protocol=20261170),
               openmc_diagnostics={"cutoff_0": {"seed": 20261190, "energy_cutoff_eV": 0.0}}),
}


def materials(kind):
    """D7's materials (tests/test_regression_kin.py) or the README hydrogen
    slab's (scripts/hydrogen_cutoff.py)."""
    if kind == "fusion":
        flibe = dataclasses.replace(cm.flibe(temperature_K=T_FLIBE), temperature=T_FLIBE)
        return [cm.tungsten(), flibe, cm.iron()]
    if kind == "hydrogen":
        return [CEMaterial("H", 0.0708, (("H1", 1.0),))]
    raise ValueError(kind)


def config(library, name, variant="main"):
    p = PROBLEMS[name]
    geom = SlabGeometry(list(p["bounds"]), materials(p["materials"]))
    protocol = variant == "protocol"
    return KinRunConfig(geom, BeamSource(), MonoEnergetic(p["energy_eV"]), library,
                        n_batches=N_BATCHES,
                        histories_per_batch=PROTOCOL_HISTORIES if protocol else HISTORIES,
                        seed=p["seeds"]["mcslab_protocol" if protocol else "mcslab"],
                        energy_edges=SPECTRUM_EDGES, depth_bins=p["depth_bins"],
                        reflect_left=p["reflect_left"])


# ---------------------------------------------------------------- export
def raw_sigma_t(path, nuclide, temperature, energy):
    """Microscopic total cross section (b) at `energy` straight from the
    HDF5 file: the sum of the non-redundant reactions on the nuclide's grid,
    interpolated linearly (independent of mcslab's reader)."""
    with h5py.File(path, "r") as f:
        g = f[nuclide]
        e = g["energy"][temperature][()]
        total = np.zeros_like(e)
        for key in g["reactions"]:
            rx = g["reactions"][key]
            if int(rx.attrs["redundant"]):
                continue
            ds = rx[temperature]["xs"]
            i0 = int(ds.attrs["threshold_idx"])
            xs = ds[()]
            total[i0:i0 + xs.size] += xs
    return float(np.interp(energy, e, total))


def pinned_checksums():
    pins = {}
    with open(CHECKSUMS) as fh:
        for line in fh:
            digest, path = line.split()
            pins[path] = digest
    return pins


def describe(library, name):
    cfg = config(library, name)
    p = PROBLEMS[name]
    bounds, mat_of_region, packed, _phys, nuclides = pack_problem(cfg)
    mesh = make_mesh(cfg.geometry.bounds, cfg.depth_bins)
    layers = []
    for r, mat in enumerate(cfg.geometry.region_materials):
        m = int(mat_of_region[r])
        sl = slice(int(packed.m_off[m]), int(packed.m_off[m + 1]))
        labels = [packed.nuclide_names[k] for k in packed.mat_nuc[sl]]
        dens = [float(d) for d in packed.mat_dens[sl]]
        expect = mat.number_densities({n: nuclides[n].awr for n in labels})
        if [n for n, _ in expect] != labels or [d for _, d in expect] != dens:
            raise RuntimeError(f"{name}/{mat.name}: packed densities differ from "
                               "CEMaterial.number_densities")
        temperature = mat.temperature if mat.temperature is not None else cfg.temperature
        entries, sigma = [], 0.0
        for n, d in zip(labels, dens):
            key = data_temperature(cfg, mat, n)
            s = raw_sigma_t(library.nuclide_path(n), n, key, p["energy_eV"])
            entries.append({"name": n, "atoms_per_barn_cm": d, "data_temperature": key,
                            "kT_eV": float(nuclides[n].kT), "sigma_t_source_barn": s})
            sigma += d * s
        layers.append({"name": mat.name, "x_cm": [float(bounds[r]), float(bounds[r + 1])],
                       "temperature_K": float(temperature),
                       "depth_bins": int(cfg.depth_bins[r]),
                       "sigma_t_source_per_cm": sigma, "nuclides": entries})
    return {
        "title": p["title"],
        "bounds_cm": [float(b) for b in bounds],
        "reflect_left": bool(p["reflect_left"]),
        "layers": layers,
        "mesh_edges_cm": mesh.bin_edges.tolist(),
        "source": {"energy_eV": float(p["energy_eV"]),
                   "position_cm": [float(bounds[0]), 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        "energy_cutoff_eV": _energy_cutoff(cfg, nuclides),
        "n_batches": N_BATCHES,
        "histories_per_batch": HISTORIES,
        "protocol_histories_per_batch": PROTOCOL_HISTORIES,
        "seeds": dict(p["seeds"]),
        "openmc_diagnostics": p["openmc_diagnostics"],
    }


def problem_file(library):
    names = sorted({n for p in PROBLEMS.values() for m in materials(p["materials"])
                    for n in m.nuclide_names})
    data = {f"neutron/{n}.h5": file_sha256(library.nuclide_path(n)) for n in names}
    pins = pinned_checksums()
    bad = [k for k, v in data.items() if pins.get(k) != v]
    if bad:
        raise RuntimeError(f"data files differ from {CHECKSUMS}: {bad}")
    return {
        "description": "Phase 4 benchmark problems (docs/phase4_plan.md). Written by "
                       "benchmark/run_mcslab.py export; the only input of "
                       "benchmark/run_openmc.py.",
        "library": library.label,
        "data_files": data,
        "responses": list(T.RESPONSE_NAMES),
        "spectrum_edges_eV": list(SPECTRUM_EDGES),
        "openmc_model": {"lateral_half_width_cm": LATERAL_HALF_WIDTH_CM,
                         "buffer_cm": BUFFER_CM,
                         "uncollided_band_rel": UNCOLLIDED_BAND_REL,
                         "threads": 1},
        "problems": {name: describe(library, name) for name in PROBLEMS},
    }


# ---------------------------------------------------------------- run
def pair(batches):
    m, s = T.batch_stats(batches)
    return [float(m), float(s)]


def arrays(batches):
    m, s = T.batch_stats(batches)
    return {"mean": m.tolist(), "se": s.tolist()}


def code_state():
    """Commit, whether mcslab/ or the benchmark scripts differ from it, and
    a content hash of exactly those sources."""
    files = sorted(os.path.join("mcslab", f) for f in os.listdir(os.path.join(REPO, "mcslab"))
                   if f.endswith(".py"))
    files += [os.path.join("benchmark", f) for f in ("__init__.py", "jsonio.py",
                                                     "run_mcslab.py")]
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        with open(os.path.join(REPO, f), "rb") as fh:
            h.update(fh.read())

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "HEAD") or None
    dirty = bool(git("status", "--porcelain", "--untracked-files=all", "--", *files))
    return {"git_commit": commit, "code_dirty": dirty, "code_sha256": h.hexdigest(),
            "code_files": files}


def run_problem(library, name, variant, spec, problems_sha):
    cfg = config(library, name, variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(cfg)
    if res.balance()["residual"] != 0.0 or res.lost != 0:
        raise RuntimeError(f"{name}: balance residual {res.balance()['residual']}, "
                           f"lost {res.lost}")
    reg = res.mesh.bin_region
    flux = res.mesh_flux_batches(T.EST_TL)
    unc = res.mesh_flux_batches(T.EST_TL_UNC)
    spec_b = res.spectrum_batches(T.SPEC_TL)
    layers, bins, uncollided, spectrum = {}, {}, {}, {}
    for r, layer in enumerate(spec["layers"]):
        sel = reg == r
        present = [n["name"] for n in layer["nuclides"]] + ["total"]
        lay = {"flux": pair(flux[:, sel].sum(axis=1)), "scores": {}}
        per = {"flux": arrays(flux[:, sel]), "scores": {}}
        for s, score in enumerate(T.RESPONSE_NAMES):
            lay["scores"][score], per["scores"][score] = {}, {}
            for nuc in present:
                tb = res.tally_batches(s, T.EST_TL, None if nuc == "total" else nuc)[:, sel]
                lay["scores"][score][nuc] = pair(tb.sum(axis=1))
                per["scores"][score][nuc] = arrays(tb)
        layers[layer["name"]] = lay
        bins[layer["name"]] = per
        uncollided[layer["name"]] = arrays(unc[:, sel])
        spectrum[layer["name"]] = arrays(spec_b[:, r, :])
    n = float(cfg.n_histories)
    return {
        "code": "mcslab",
        "problem": name,
        "variant": variant,
        "problems_sha256": problems_sha,
        "settings": {
            "seed": cfg.seed, "n_batches": cfg.n_batches,
            "histories_per_batch": cfg.histories_per_batch,
            "energy_cutoff_eV": float(res.energy_cutoff),
            "free_gas_threshold": float(cfg.free_gas_threshold), "free_gas": cfg.free_gas,
            "temperature_default_K": float(cfg.temperature),
            "temperature_tolerance_K": float(cfg.temperature_tolerance),
            "reflect_left": cfg.reflect_left, "depth_bins": list(cfg.depth_bins),
            "bank_capacity": cfg.bank_capacity, "source": repr(cfg.source),
            "spectrum": repr(cfg.spectrum), "estimator": "tracklength"},
        "bookkeeping": {
            "balance": {k: float(v) for k, v in res.balance().items()},
            "lost": res.lost, "max_draws": res.max_draws, "stride": STRIDE,
            "cutoff_weight_per_source": float(res.cutoff_weight.sum()) / n,
            "zero_yield_weight_per_source": float(res.zero_yield_weight.sum()) / n,
            "reflected_weight_per_source": pair(res.reflected_weight / cfg.histories_per_batch)},
        "response_absent": [f"{res.tally_nuclides[k]} {T.RESPONSE_NAMES[s]}"
                            for k, s in zip(*np.nonzero(~res.response_present))],
        "leakage": {"left": pair(res.leakage_left), "right": pair(res.leakage_right)},
        "layers": layers,
        "bins": bins,
        "uncollided": uncollided,
        "spectrum": spectrum,
        "provenance": {
            **code_state(),
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "numba": numba.__version__, "llvmlite": llvmlite.__version__,
                            "h5py": h5py.__version__, "platform": platform.platform(),
                            "machine": platform.machine()},
            "library": library.label,
        },
    }


def results_path(name, variant):
    suffix = "" if variant == "main" else f"_{variant}"
    return os.path.join(RESULTS_DIR, f"mcslab_{name}{suffix}.json")


def load_problems(library):
    """The committed problem file, refused if this script would now write
    a different one."""
    if not os.path.isfile(PROBLEMS_JSON):
        sys.exit("no benchmark/problems.json: run `export` first")
    with open(PROBLEMS_JSON, encoding="utf-8") as fh:
        text = fh.read()
    if jsonio.dumps(problem_file(library)) != text:
        sys.exit("benchmark/problems.json is out of date with run_mcslab.py")
    return jsonio.read(PROBLEMS_JSON), jsonio.file_sha256(PROBLEMS_JSON)


def strip(doc):
    return {k: v for k, v in doc.items() if k != "provenance"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("export", "run", "check"))
    ap.add_argument("--problem", nargs="*", default=sorted(PROBLEMS))
    ap.add_argument("--protocol", action="store_true",
                    help="failure-protocol run (100 x 40,000, second seed)")
    args = ap.parse_args()
    library = Library.open()

    if args.command == "export":
        sha = jsonio.write(problem_file(library), PROBLEMS_JSON)
        print(f"wrote {os.path.relpath(PROBLEMS_JSON, REPO)} (sha256 {sha})")
        return

    problems, problems_sha = load_problems(library)
    variant = "protocol" if args.protocol else "main"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.problem:
        t0 = time.perf_counter()
        doc = run_problem(library, name, variant, problems["problems"][name], problems_sha)
        secs = time.perf_counter() - t0
        path = results_path(name, variant)
        if args.command == "run":
            jsonio.write(doc, path)
            b = doc["bookkeeping"]
            print(f"{name} {variant}: {secs:.1f} s, residual {b['balance']['residual']}, "
                  f"cutoff/src {b['cutoff_weight_per_source']:.3g}, max draws "
                  f"{b['max_draws']} (STRIDE {STRIDE}) -> {os.path.relpath(path, REPO)}")
        else:
            same = jsonio.dumps(strip(doc)) == jsonio.dumps(strip(jsonio.read(path)))
            print(f"{name} {variant}: {'[OK] identical' if same else '[FAIL] differs'} "
                  f"({secs:.1f} s)")
            if not same:
                sys.exit(1)


if __name__ == "__main__":
    main()
