#!/usr/bin/env python3
"""Bit-identical regression test for the transport engine.

Runs a fixed reference problem (two sources on a 4-region slab with a void
gap) and compares every raw per-batch tally array against the committed
reference in tests/reference/.

    ./venv/bin/python tests/test_regression.py record --reason "why"   # write the reference
    ./venv/bin/python tests/test_regression.py compare                 # byte-exact
    ./venv/bin/python tests/test_regression.py compare --loose         # rtol 1e-6

pytest runs test_regression_bit_identical (strict, byte-exact).

Exit codes: 0 pass, 1 a regression, 2 setup problem (no reference yet, or
the reference files are inconsistent with each other).

------------------------------------------------------------------
Rules -- read before regenerating the reference
------------------------------------------------------------------
* Strict mode compares dtype, shape and raw bytes. A refactor must pass it.
* --loose (rtol 1e-6, atol 0; integer arrays still exact) is a diagnostic
  for "did only the last bits move?". It is not a pass criterion.
* `record` requires --reason. The reason, date, git commit and library
  versions are written to manifest.json, and every earlier entry is kept in
  its "history" list, so each regeneration leaves an audit trail.
* The arrays are the raw per-batch weight sums, not means: any change in
  random-number consumption, event order or summation order shows up.
* Library versions and platform are recorded and reported on mismatch but
  never compared -- "numba was upgraded" should be a diagnosable failure.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import llvmlite                                          # noqa: E402
import numba                                             # noqa: E402
import numpy as np                                       # noqa: E402

from mcslab.config import RunConfig, run                 # noqa: E402
from mcslab.geometry import SlabGeometry                 # noqa: E402
from mcslab.materials import VOID, Material              # noqa: E402
from mcslab.sources import BeamSource, IsotropicPlaneSource  # noqa: E402

REFERENCE_DIR = os.path.join(HERE, "reference")
REFERENCE_NPZ = os.path.join(REFERENCE_DIR, "reference.npz")
MANIFEST = os.path.join(REFERENCE_DIR, "manifest.json")

LOOSE_RTOL = 1e-6

# ---- Fixed reference problem. Changing any of this invalidates the reference.
SEED = 20260929
N_BATCHES = 20
HISTORIES_PER_BATCH = 5000


def reference_problems():
    """name -> RunConfig. Exercises absorption, scattering, a void region,
    several interfaces and both source types."""
    wall = Material("wall", sigma_s=0.6, sigma_a=0.9)
    moderator = Material("moderator", sigma_s=1.2, sigma_a=0.03)
    breeder = Material("breeder", sigma_s=0.4, sigma_a=1.1)
    geom = SlabGeometry([0.0, 0.8, 1.3, 4.1, 5.0], [wall, VOID, moderator, breeder])
    return {
        "beam": RunConfig(geom, BeamSource(), N_BATCHES, HISTORIES_PER_BATCH, SEED),
        "iso": RunConfig(geom, IsotropicPlaneSource(2.35), N_BATCHES,
                         HISTORIES_PER_BATCH, SEED + 1),
    }


def describe(cfg):
    g = cfg.geometry
    return {
        "bounds_cm": list(g.bounds),
        "materials": [{"name": m.name, "sigma_t": m.sigma_t, "sigma_s": m.sigma_s,
                       "sigma_a": m.sigma_a} for m in g.region_materials],
        "source": repr(cfg.source),
        "n_batches": cfg.n_batches,
        "histories_per_batch": cfg.histories_per_batch,
        "seed": cfg.seed,
    }


# =======================================================
# THE RUN
# =======================================================
def run_reference():
    """-> flat dict of every raw array, ready for np.savez."""
    data = {}
    for name, cfg in reference_problems().items():
        res = run(cfg)
        data[f"{name}__region_sums"] = res.region_sums
        data[f"{name}__surface_sums"] = res.surface_sums
        data[f"{name}__diagnostics"] = res.diagnostics
    return data


def environment():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD") or None
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except OSError:
        commit, dirty = None, None
    return commit, dirty


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# =======================================================
# COMPARISON
# =======================================================
def compare_arrays(name, ref, cur, loose):
    """Return a failure description, or None when the arrays agree."""
    if ref.dtype != cur.dtype:
        return f"{name}: DTYPE {ref.dtype} -> {cur.dtype}"
    if ref.shape != cur.shape:
        return f"{name}: SHAPE {ref.shape} -> {cur.shape}"
    if ref.tobytes() == cur.tobytes():
        return None

    differ = np.ravel(ref != cur)
    first = np.flatnonzero(differ)[:5]
    detail = (f"{int(differ.sum())} of {ref.size} values differ "
              f"(first flat indices {[int(i) for i in first]}; "
              f"ref {np.ravel(ref)[first]} -> cur {np.ravel(cur)[first]})")

    if ref.dtype.kind in "biu" or not loose:
        if ref.dtype.kind == "f":
            g, c = ref.astype(np.float64), cur.astype(np.float64)
            diff = np.abs(g - c)
            nz = g != 0.0
            max_rel = float((diff[nz] / np.abs(g[nz])).max()) if nz.any() else 0.0
            detail += f"; max|abs| {float(diff.max()):.3e}, max|rel| {max_rel:.3e}"
        return f"{name}: {detail}"

    if np.allclose(cur, ref, rtol=LOOSE_RTOL, atol=0.0, equal_nan=True):
        return None
    return f"{name}: exceeds rtol {LOOSE_RTOL:.0e}: {detail}"


def compare(reference, current, loose):
    failures = []
    for missing in sorted(set(reference) - set(current)):
        failures.append(f"{missing}: in the reference, not produced now")
    for added in sorted(set(current) - set(reference)):
        failures.append(f"{added}: produced now, absent from the reference")
    for name in sorted(set(reference) & set(current)):
        result = compare_arrays(name, reference[name], current[name], loose)
        if result:
            failures.append(result)
    return failures


def load_reference():
    """-> (arrays, manifest). Raises FileNotFoundError / ValueError on a
    missing or internally inconsistent reference."""
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    with np.load(REFERENCE_NPZ, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    recorded = manifest.get("arrays", {})
    if set(recorded) != set(arrays):
        raise ValueError("manifest.json and reference.npz list different arrays")
    for k, arr in arrays.items():
        if recorded[k]["sha256"] != sha256(arr):
            raise ValueError(f"reference.npz array {k} does not match its manifest sha256")
    return arrays, manifest


def environment_notes(manifest):
    recorded = manifest.get("environment", {})
    now = environment()
    return [f"{k}: reference {recorded.get(k)!r} vs current {now[k]!r}"
            for k in now if recorded.get(k) != now[k]]


# =======================================================
# PYTEST ENTRY POINT
# =======================================================
def test_regression_bit_identical():
    assert os.path.isfile(REFERENCE_NPZ) and os.path.isfile(MANIFEST), (
        f"no reference in {REFERENCE_DIR}; run "
        "'./venv/bin/python tests/test_regression.py record --reason ...' first")
    reference, manifest = load_reference()
    failures = compare(reference, run_reference(), loose=False)
    if failures:
        report = ["Regression (byte-exact) failed:"] + [f"  {f}" for f in failures]
        notes = environment_notes(manifest)
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
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    history = []
    if os.path.isfile(MANIFEST):
        with open(MANIFEST) as fh:
            old = json.load(fh)
        history = old.get("history", [])
        history.append({k: old.get(k) for k in
                        ("reason", "recorded_utc", "git_commit", "git_dirty")})

    print(f"Running the reference problems (seed {SEED}, {N_BATCHES} batches x "
          f"{HISTORIES_PER_BATCH} histories each)...")
    data = run_reference()
    np.savez(REFERENCE_NPZ, **data)

    commit, dirty = git_state()
    manifest = {
        "reason": reason,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
                                .isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": environment(),
        "problems": {n: describe(c) for n, c in reference_problems().items()},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": sha256(v)}
                   for k, v in sorted(data.items())},
        "history": history,
    }
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    size_kb = os.path.getsize(REFERENCE_NPZ) / 1024
    print(f"\n[REFERENCE] wrote {os.path.relpath(REFERENCE_NPZ, REPO)} ({size_kb:.1f} KiB)"
          f" and {os.path.relpath(MANIFEST, REPO)}")
    for name, arr in sorted(data.items()):
        print(f"    {name:<26} {str(arr.shape):<14} {arr.dtype}")
    print(f"    reason: {reason}")
    return 0


def cmd_compare(args):
    if not (os.path.isfile(REFERENCE_NPZ) and os.path.isfile(MANIFEST)):
        print(f"[ERROR] no reference in {REFERENCE_DIR}", file=sys.stderr)
        print("[ERROR] run 'tests/test_regression.py record --reason ...' first.",
              file=sys.stderr)
        return 2
    try:
        reference, manifest = load_reference()
    except (ValueError, KeyError) as exc:
        print(f"[ERROR] reference is inconsistent: {exc}", file=sys.stderr)
        return 2

    mode = f"loose (rtol {LOOSE_RTOL:.0e})" if args.loose else "strict (byte-exact)"
    print(f"Comparing against {os.path.relpath(REFERENCE_NPZ, REPO)} [{mode}]")
    print(f"  reference reason: {manifest.get('reason')!r} "
          f"({manifest.get('recorded_utc')}, commit {str(manifest.get('git_commit'))[:10]})")

    failures = compare(reference, run_reference(), loose=args.loose)

    notes = environment_notes(manifest)
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
    if not args.loose:
        print("\n  To see whether only the last bits moved, retry with --loose. "
              "A legitimate change needs 'record --reason \"...\"'.")
    return 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bit-identical regression test for the mcslab transport engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 = pass, 1 = regression, 2 = no / inconsistent reference",
    )
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("record", help="write the reference arrays and manifest")
    p.add_argument("--reason", required=True,
                   help="why the reference is being (re)generated; stored in manifest.json")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("compare", help="check current results against the reference")
    p.add_argument("--loose", action="store_true",
                   help=f"float arrays within rtol {LOOSE_RTOL:.0e} (diagnostic only)")
    p.set_defaults(func=cmd_compare)
    return parser


def main_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode, args.func, args.loose = "compare", cmd_compare, False
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_cli())
