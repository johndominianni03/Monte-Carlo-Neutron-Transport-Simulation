#!/usr/bin/env python3
"""Bit-identical regression test for the Phase 3 response tallies.

One reference file and manifest per problem, in
tests/reference_tally/<problem>/{reference.npz, manifest.json}, so adding a
problem never re-records another. Same rules as the other harnesses: a
legitimate change of results is re-recorded with a stated reason (kept in
the manifest history); `--loose` (rtol 1e-6) is a diagnostic only. The
manifest pins the sha256 of every HDF5 file used; different data are a
setup problem (exit 2).

    ./venv/bin/python tests/test_regression_tally.py record --problem kin_d7 --reason "why"
    ./venv/bin/python tests/test_regression_tally.py compare [--problem NAME] [--loose]

`record` refuses to overwrite an existing reference unless --overwrite is
given (each reference is meant to be recorded once).

Problem kin_d7: D7 exactly as tests/test_regression_kin.py defines it
(seed 20261023, 20 x 5000), with the depth mesh W 10 / FLiBe 20 / Fe 20
and the Phase 3 spectrum grid (20 bins per decade, 1e-5 eV to 19.95 MeV).
Pinned: tally, mesh_flux, spectrum. The other 8 transport arrays must hash
exactly as in tests/reference_kin (the spectrum differs only by its grid),
which ties this reference to the D7 one without duplicating it.

Problem kin_d7_reflect (Part B, approved D17/D19): the same slab, mesh and
grid with a reflective plasma side (reflect_left), seed 20261035,
20 x 5000. No other reference covers reflective transport, so all of its
transport arrays are pinned too, with tally, mesh_flux and
reflected_weight.
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
import test_regression_kin as RK                          # noqa: E402
from mcslab import tallies as T                           # noqa: E402
from mcslab.config_kin import EnergyCutoffWarning, run_kin  # noqa: E402
from mcslab.nucdata import Library                        # noqa: E402
from mcslab.responses import RESPONSE_MTS                 # noqa: E402

REFERENCE_ROOT = os.path.join(HERE, "reference_tally")
D7_BINS = (10, 20, 20)
# Phase 3 spectrum grid (approved D12): edges 10^(k/20) eV, k = -100 .. 146
PHASE3_EDGES = tuple(float(10.0 ** (k / 20.0)) for k in range(-100, 147))
TALLY_ARRAYS = ("tally", "mesh_flux", "spectrum")
CROSSCHECK_ARRAYS = tuple(a for a in RK.ARRAYS if a != "spectrum")
REFLECT_SEED = 20261035
REFLECT_ARRAYS = RK.ARRAYS + ("tally", "mesh_flux", "reflected_weight")


def reference_problems(library):
    d7 = RK.reference_problems(library)["kin_d7"]
    return {
        "kin_d7": (dataclasses.replace(d7, depth_bins=D7_BINS, energy_edges=PHASE3_EDGES),
                   TALLY_ARRAYS, CROSSCHECK_ARRAYS),
        "kin_d7_reflect": (dataclasses.replace(d7, depth_bins=D7_BINS,
                                               energy_edges=PHASE3_EDGES, reflect_left=True,
                                               seed=REFLECT_SEED),
                           REFLECT_ARRAYS, ()),
    }


def paths(problem):
    d = os.path.join(REFERENCE_ROOT, problem)
    return d, os.path.join(d, "reference.npz"), os.path.join(d, "manifest.json")


def describe(cfg, res):
    out = RK.describe(cfg)
    out.update({
        "depth_bins": list(cfg.depth_bins),
        "mesh_bin_edges_cm": res.mesh.bin_edges.tolist(),
        "responses": list(T.RESPONSE_NAMES),
        "response_mts": list(RESPONSE_MTS) + ["absorption"],
        "estimators": list(T.ESTIMATOR_NAMES),
        "tally_nuclides": list(res.tally_nuclides),
        "response_absent": [f"{res.tally_nuclides[k]} {T.RESPONSE_NAMES[s]}"
                            for k, s in zip(*np.nonzero(~res.response_present))],
        "reflect_left": bool(getattr(cfg, "reflect_left", False)),
    })
    return out


def data_files(library, cfg):
    names = sorted({n for m in cfg.geometry.region_materials for n in m.nuclide_names})
    return {f"neutron/{n}.h5": RK.file_sha256(library.nuclide_path(n)) for n in names}


def run_problem(library, problem):
    cfg, pinned, crosscheck = reference_problems(library)[problem]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(cfg)
    return cfg, res, {a: getattr(res, a) for a in pinned}, \
        {a: base.sha256(getattr(res, a)) for a in crosscheck}


def crosscheck_failures(hashes):
    """Arrays whose sha256 differs from tests/reference_kin's manifest."""
    if not hashes:
        return []
    with open(RK.MANIFEST) as fh:
        kin = json.load(fh)["arrays"]
    return [f"{a}: {h} vs reference_kin {kin[f'kin_d7__{a}']['sha256']}"
            for a, h in hashes.items() if kin[f"kin_d7__{a}"]["sha256"] != h]


def load_reference(problem):
    _, npz, man = paths(problem)
    with open(man) as fh:
        manifest = json.load(fh)
    with np.load(npz, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    recorded = manifest.get("arrays", {})
    if set(recorded) != set(arrays):
        raise ValueError(f"{problem}: manifest.json and reference.npz list different arrays")
    for k, arr in arrays.items():
        if recorded[k]["sha256"] != base.sha256(arr):
            raise ValueError(f"{problem}: array {k} does not match its manifest sha256")
    return arrays, manifest


def data_mismatches(manifest, library, cfg):
    recorded = manifest.get("data_files", {})
    now = data_files(library, cfg)
    return [f"{k}: reference {recorded.get(k)} vs current {now.get(k)}"
            for k in sorted(set(recorded) | set(now)) if recorded.get(k) != now.get(k)]


# =======================================================
# PYTEST ENTRY POINT
# =======================================================
def test_regression_tally_bit_identical():
    import pytest
    try:
        library = Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")
    for problem in reference_problems(library):
        _, npz, man = paths(problem)
        assert os.path.isfile(npz) and os.path.isfile(man), (
            f"no reference for {problem}; run './venv/bin/python "
            f"tests/test_regression_tally.py record --problem {problem} --reason ...'")
        reference, manifest = load_reference(problem)
        cfg = reference_problems(library)[problem][0]
        bad = data_mismatches(manifest, library, cfg)
        assert not bad, f"{problem}: nuclear data differ:\n  " + "\n  ".join(bad)
        _, _, current, hashes = run_problem(library, problem)
        failures = base.compare(reference, current, loose=False)
        failures += crosscheck_failures(hashes)
        if failures:
            report = [f"{problem}: tally regression (byte-exact) failed:"]
            report += [f"  {f}" for f in failures]
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
    problems = reference_problems(library)
    if args.problem not in problems:
        print(f"[ERROR] unknown problem {args.problem!r} (have {sorted(problems)})",
              file=sys.stderr)
        return 2
    d, npz, man = paths(args.problem)
    history = []
    if os.path.isfile(man):
        if not args.overwrite:
            print(f"[ERROR] {os.path.relpath(man, REPO)} exists; references are recorded "
                  "once (pass --overwrite for a legitimate change of results)",
                  file=sys.stderr)
            return 2
        with open(man) as fh:
            old = json.load(fh)
        history = old.get("history", [])
        history.append({k: old.get(k) for k in
                        ("reason", "recorded_utc", "git_commit", "git_dirty")})
    print(f"Running tally problem {args.problem} ({library.label})...")
    cfg, res, data, hashes = run_problem(library, args.problem)
    bad = crosscheck_failures(hashes)
    if bad:
        print("[ERROR] transport arrays differ from tests/reference_kin:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        return 1
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(npz, **data)
    commit, dirty = base.git_state()
    manifest = {
        "reason": reason,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
                                .isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": base.environment(),
        "library": library.label,
        "data_files": data_files(library, cfg),
        "problem": describe(cfg, res),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                       "sha256": base.sha256(v)} for k, v in sorted(data.items())},
        "transport_crosscheck": {
            "reference": "tests/reference_kin (kin_d7)" if hashes else None,
            "sha256": hashes,
        },
        "history": history,
    }
    with open(man, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"\n[REFERENCE] wrote {os.path.relpath(npz, REPO)} "
          f"({os.path.getsize(npz) / 1024:.1f} KiB) and {os.path.relpath(man, REPO)}")
    for name, arr in sorted(data.items()):
        print(f"    {name:<12} {str(arr.shape):<24} {arr.dtype}")
    if hashes:
        print(f"    transport arrays match tests/reference_kin: {', '.join(hashes)}")
    print(f"    reason: {reason}")
    return 0


def cmd_compare(args):
    try:
        library = Library.open()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    problems = list(reference_problems(library)) if args.problem is None else [args.problem]
    mode = f"loose (rtol {base.LOOSE_RTOL:.0e})" if args.loose else "strict (byte-exact)"
    status = 0
    for problem in problems:
        _, npz, man = paths(problem)
        if not (os.path.isfile(npz) and os.path.isfile(man)):
            print(f"[ERROR] no reference for {problem}", file=sys.stderr)
            return 2
        try:
            reference, manifest = load_reference(problem)
        except (ValueError, KeyError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        cfg = reference_problems(library)[problem][0]
        bad = data_mismatches(manifest, library, cfg)
        if bad:
            print(f"[ERROR] {problem}: nuclear data differ from the reference's:",
                  file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 2
        print(f"Comparing {problem} against {os.path.relpath(npz, REPO)} [{mode}]")
        print(f"  reference reason: {manifest.get('reason')!r} "
              f"({manifest.get('recorded_utc')}, commit "
              f"{str(manifest.get('git_commit'))[:10]})")
        _, _, current, hashes = run_problem(library, problem)
        failures = base.compare(reference, current, loose=args.loose)
        failures += crosscheck_failures(hashes)
        notes = base.environment_notes(manifest)
        if notes:
            print("  environment differs from the reference's:")
            for note in notes:
                print(f"    {note}")
        if failures:
            print(f"[FAIL] {problem}: {len(failures)} array(s) diverged:")
            for failure in failures:
                print(f"  {failure}")
            status = 1
        else:
            extra = (f"; transport arrays match tests/reference_kin" if hashes else "")
            print(f"[OK] {problem}: all {len(reference)} arrays match, {mode}{extra}.")
    return status


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Bit-identical regression test for the mcslab response tallies (Phase 3).",
        epilog="exit codes: 0 = pass, 1 = regression, 2 = no / inconsistent reference "
               "or different nuclear data")
    sub = parser.add_subparsers(dest="mode")
    p = sub.add_parser("record", help="write one problem's reference arrays and manifest")
    p.add_argument("--problem", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("compare", help="check current results against the references")
    p.add_argument("--problem", default=None)
    p.add_argument("--loose", action="store_true")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode, args.func, args.loose, args.problem = "compare", cmd_compare, False, None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_cli())
