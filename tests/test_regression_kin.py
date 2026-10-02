#!/usr/bin/env python3
"""Bit-identical regression test for the kinematic CE kernel (Phase 2b, test h).

Same rules and CLI as tests/test_regression.py (read its docstring), with
its own reference in tests/reference_kin/, so the Phase 1 and Phase 2a
references are never touched by kinematic-kernel work:

    ./venv/bin/python tests/test_regression_kin.py record --reason "why"
    ./venv/bin/python tests/test_regression_kin.py compare [--loose]

As in the CE harness, the manifest pins the sha256 of every HDF5 file
used. Different data are a setup problem (exit 2), not a pass or a fail.
The test skips under pytest when MCSLAB_DATA is not set.

Problem D7 (approved; temperatures per approved P1): a 14.1 MeV beam at
x = 0 on
    W 0.5 cm (293.6 K) | FLiBe 20 cm (natural Li; 900 K data, Janz density
    at 900 K) | Fe 10 cm (293.6 K),
20 batches x 5000 histories, seed 20261023, default energy cutoff
(1e-5 eV), spectrum in 40 log bins from 1e-5 eV to 20 MeV. At 14.1 MeV it
reaches every law in the data (ACE law 61 in CM and lab, the F19
applicability mixture, ACE law 4, levels, tabulated yields), free-gas
scattering at two temperatures, the secondary bank, zero-yield events and
the energy cutoff.
"""
import argparse
import dataclasses
import datetime
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np                                        # noqa: E402

import test_regression as base                            # noqa: E402
from mcslab import ce_materials as cm                     # noqa: E402
from mcslab.config_ce import MonoEnergetic                # noqa: E402
from mcslab.config_kin import (EnergyCutoffWarning,       # noqa: E402
                               KinRunConfig, data_temperature, run_kin)
from mcslab.geometry import SlabGeometry                  # noqa: E402
from mcslab.nucdata import Library, file_sha256           # noqa: E402
from mcslab.sources import BeamSource                     # noqa: E402

REFERENCE_DIR = os.path.join(HERE, "reference_kin")
REFERENCE_NPZ = os.path.join(REFERENCE_DIR, "reference.npz")
MANIFEST = os.path.join(REFERENCE_DIR, "manifest.json")

# ---- Fixed reference problem (D7). Changing any of this invalidates the reference.
SEED = 20261023
N_BATCHES = 20
HISTORIES_PER_BATCH = 5000
E_SOURCE = 14.1e6
FLIBE_T = 900.0
EDGES = tuple(float(e) for e in np.exp(np.linspace(np.log(1.0e-5), np.log(2.0e7), 41)))
ARRAYS = ("region_sums", "surface_sums", "diagnostics", "spectrum", "cutoff_weight",
          "zero_yield_weight", "chan_events", "chan_created", "chan_zero")


def reference_problems(library):
    flibe = dataclasses.replace(cm.flibe(temperature_K=FLIBE_T), temperature=FLIBE_T)
    geom = SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()])
    return {"kin_d7": KinRunConfig(geom, BeamSource(), MonoEnergetic(E_SOURCE), library,
                                   N_BATCHES, HISTORIES_PER_BATCH, SEED,
                                   energy_edges=EDGES)}


def describe(cfg):
    awr = {}
    mats = []
    for m in cfg.geometry.region_materials:
        for n in m.nuclide_names:
            if n not in awr:
                awr[n] = cfg.library.load(n, data_temperature(cfg, m, n)).awr
        mats.append({"name": m.name, "density_g_cm3": m.density_g_cm3,
                     "temperature_K": m.temperature if m.temperature is not None
                     else cfg.temperature,
                     "data_temperature": {n: data_temperature(cfg, m, n)
                                          for n in m.nuclide_names},
                     "atoms_per_barn_cm": dict(m.number_densities(awr))})
    return {
        "bounds_cm": list(cfg.geometry.bounds),
        "materials": mats,
        "source": repr(cfg.source),
        "spectrum": repr(cfg.spectrum),
        "library": cfg.library.label,
        "n_batches": cfg.n_batches,
        "histories_per_batch": cfg.histories_per_batch,
        "seed": cfg.seed,
        "energy_cutoff": cfg.energy_cutoff,
        "energy_edges": list(cfg.energy_edges),
        "free_gas": cfg.free_gas,
        "free_gas_threshold": cfg.free_gas_threshold,
        "temperature_tolerance": cfg.temperature_tolerance,
    }


def data_files(library):
    names = set()
    for cfg in reference_problems(library).values():
        for m in cfg.geometry.region_materials:
            names.update(m.nuclide_names)
    return {f"neutron/{n}.h5": file_sha256(library.nuclide_path(n)) for n in sorted(names)}


def run_reference(library):
    data = {}
    for name, cfg in reference_problems(library).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EnergyCutoffWarning)
            res = run_kin(cfg)
        for a in ARRAYS:
            data[f"{name}__{a}"] = getattr(res, a)
    return data


def load_reference():
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    with np.load(REFERENCE_NPZ, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    recorded = manifest.get("arrays", {})
    if set(recorded) != set(arrays):
        raise ValueError("manifest.json and reference.npz list different arrays")
    for k, arr in arrays.items():
        if recorded[k]["sha256"] != base.sha256(arr):
            raise ValueError(f"reference.npz array {k} does not match its manifest sha256")
    return arrays, manifest


def data_mismatches(manifest, library):
    recorded = manifest.get("data_files", {})
    now = data_files(library)
    return [f"{k}: reference {recorded.get(k)} vs current {now.get(k)}"
            for k in sorted(set(recorded) | set(now)) if recorded.get(k) != now.get(k)]


# =======================================================
# PYTEST ENTRY POINT
# =======================================================
def test_regression_kin_bit_identical():
    import pytest
    try:
        library = Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")
    assert os.path.isfile(REFERENCE_NPZ) and os.path.isfile(MANIFEST), (
        f"no reference in {REFERENCE_DIR}; run "
        "'./venv/bin/python tests/test_regression_kin.py record --reason ...' first")
    reference, manifest = load_reference()
    bad_data = data_mismatches(manifest, library)
    assert not bad_data, "nuclear data differ from the reference's:\n  " + "\n  ".join(bad_data)
    failures = base.compare(reference, run_reference(library), loose=False)
    if failures:
        report = ["kinematic regression (byte-exact) failed:"] + [f"  {f}" for f in failures]
        notes = base.environment_notes(manifest)
        if notes:
            report += ["  environment differs from the reference's:"]
            report += [f"    {n}" for n in notes]
        raise AssertionError("\n".join(report))


# =======================================================
# CLI
# =======================================================
def cmd_record(args):
    reason = args.reason.strip()
    if not reason:
        print("[ERROR] --reason must not be empty", file=sys.stderr)
        return 2
    library = Library.open()
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    history = []
    if os.path.isfile(MANIFEST):
        with open(MANIFEST) as fh:
            old = json.load(fh)
        history = old.get("history", [])
        history.append({k: old.get(k) for k in
                        ("reason", "recorded_utc", "git_commit", "git_dirty")})

    print(f"Running the kinematic reference problem (seed {SEED}, {N_BATCHES} batches x "
          f"{HISTORIES_PER_BATCH} histories, {library.label})...")
    data = run_reference(library)
    np.savez(REFERENCE_NPZ, **data)

    commit, dirty = base.git_state()
    manifest = {
        "reason": reason,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
                                .isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": base.environment(),
        "library": library.label,
        "data_files": data_files(library),
        "problems": {n: describe(c) for n, c in reference_problems(library).items()},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                       "sha256": base.sha256(v)} for k, v in sorted(data.items())},
        "history": history,
    }
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"\n[REFERENCE] wrote {os.path.relpath(REFERENCE_NPZ, REPO)} "
          f"({os.path.getsize(REFERENCE_NPZ) / 1024:.1f} KiB) and "
          f"{os.path.relpath(MANIFEST, REPO)}")
    for name, arr in sorted(data.items()):
        print(f"    {name:<30} {str(arr.shape):<16} {arr.dtype}")
    print(f"    reason: {reason}")
    return 0


def cmd_compare(args):
    if not (os.path.isfile(REFERENCE_NPZ) and os.path.isfile(MANIFEST)):
        print(f"[ERROR] no reference in {REFERENCE_DIR}", file=sys.stderr)
        return 2
    try:
        library = Library.open()
        reference, manifest = load_reference()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    bad_data = data_mismatches(manifest, library)
    if bad_data:
        print("[ERROR] nuclear data differ from the reference's:", file=sys.stderr)
        for line in bad_data:
            print(f"  {line}", file=sys.stderr)
        return 2

    mode = f"loose (rtol {base.LOOSE_RTOL:.0e})" if args.loose else "strict (byte-exact)"
    print(f"Comparing against {os.path.relpath(REFERENCE_NPZ, REPO)} [{mode}]")
    print(f"  reference reason: {manifest.get('reason')!r} "
          f"({manifest.get('recorded_utc')}, commit {str(manifest.get('git_commit'))[:10]})")
    print(f"  data: {manifest.get('library')}, {len(manifest['data_files'])} files, "
          "sha256 match")

    failures = base.compare(reference, run_reference(library), loose=args.loose)
    notes = base.environment_notes(manifest)
    if notes:
        print("\nEnvironment differs from the reference's:")
        for note in notes:
            print(f"  {note}")
    if not failures:
        print(f"\n[OK] all {len(reference)} arrays match, {mode}.")
        return 0
    print(f"\n[FAIL] {len(failures)} array(s) diverged:")
    for failure in failures:
        print(f"  {failure}")
    return 1


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Bit-identical regression test for the mcslab kinematic kernel (Phase 2b).",
        epilog="exit codes: 0 = pass, 1 = regression, 2 = no / inconsistent reference "
               "or different nuclear data")
    sub = parser.add_subparsers(dest="mode")
    p = sub.add_parser("record", help="write the reference arrays and manifest")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("compare", help="check current results against the reference")
    p.add_argument("--loose", action="store_true")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode, args.func, args.loose = "compare", cmd_compare, False
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_cli())
