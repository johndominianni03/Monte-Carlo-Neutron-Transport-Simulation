"""Phase 4: mcslab against OpenMC 0.16.0 on the committed result pairs
(docs/phase4_plan.md; benchmark/compare.py holds the rules).

No transport is run and neither openmc nor nuclear data is needed: the
tests read benchmark/problems.json and benchmark/results/*.json, which
benchmark/run_mcslab.py and benchmark/run_openmc.py wrote with the
pre-declared seeds (mcslab 20261040-20261043, OpenMC 20261100-20261130,
100 x 10,000 histories each).

What is checked:
- exact: identical inputs (problem file, data sha256 against
  scripts/checksums, atom densities and mesh and spectrum edges bit for
  bit), the declared settings, exact zeros (responses without data, left
  leakage with a reflective side), mcslab's exact balance, nothing lost
  in either code, no local path or user name in a committed file, and
  that the comparison needs numpy only;
- statistical: the 82 primary checks, |z| <= 3 with
  z = (mcslab - OpenMC) / sqrt(SE_m^2 + SE_o^2). The chance that at least
  one fails by luck is at most 19.9% (22.1% with the Student-t rate).

What is not: the per-bin, spectrum and uncollided diagnostics and the
default-settings runs (reported by compare.py, no pass/fail); agreement
with experiment (both codes read the same data).

PROTOCOL_EXCEPTIONS is the named table of the failure protocol: a primary
check that exceeded 3 SE on the main pair and passed on the 4x protocol
pair (100 x 40,000, second seed pair) is listed with both z values and the
reason. Both values are recomputed from the committed files here, so the
table cannot drift. A check failing on both pairs is never listed: it is a
real difference and stops the work.
"""
import getpass
import os
import re
import subprocess
import sys

import pytest

from benchmark import compare as C
from benchmark import jsonio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (problem, check) -> {"z": main pair, "z_protocol": protocol pair, "reason": ...}
PROTOCOL_EXCEPTIONS = {}


@pytest.fixture(scope="module")
def problem_file():
    return C.problems(), jsonio.file_sha256(C.PROBLEMS_JSON)


def pair(name, variant="main"):
    m, o = C.load("mcslab", name, variant), C.load("openmc", name, variant)
    assert m is not None and o is not None, f"missing results for {name} ({variant})"
    return m, o


@pytest.mark.parametrize("name", C.PROBLEMS)
def test_exact_checks(name, problem_file):
    pf, sha = problem_file
    m, o = pair(name)
    failed = [f"{lab} {det}" for lab, ok, det in C.exact_checks(pf, sha, name, m, o) if not ok]
    assert not failed, failed


@pytest.mark.parametrize("name", sorted(C.D37))
def test_default_settings_runs_are_as_declared(name, problem_file):
    """The D37 runs differ from the main OpenMC run in exactly one declared
    setting (their results are diagnostics, not checks)."""
    pf, sha = problem_file
    m = C.load("mcslab", name)
    d = C.load("openmc", name, C.D37[name])
    assert d is not None, f"missing D37 run {name} {C.D37[name]}"
    failed = [lab for lab, ok, _ in C.exact_checks(pf, sha, name, m, d, C.D37[name]) if not ok]
    assert not failed, failed


@pytest.mark.parametrize("name", C.PROBLEMS)
def test_primary_checks(name, problem_file, report):
    pf, _ = problem_file
    m, o = pair(name)
    rows = C.primary_checks(pf["problems"][name], name, m, o)
    assert len(rows) == C.EXPECTED_PRIMARY[name]
    failed = []
    for r in rows:
        report(f"{name} {r['check']}", r["mcslab"][0], r["openmc"][0],
               (r["mcslab"][1] ** 2 + r["openmc"][1] ** 2) ** 0.5)
        if abs(r["z"]) > C.Z_LIMIT and (name, r["check"]) not in PROTOCOL_EXCEPTIONS:
            failed.append(f"{r['check']}: z = {r['z']:+.2f}")
    assert not failed, failed


def test_primary_count():
    assert sum(C.EXPECTED_PRIMARY.values()) == 82


def test_protocol_exceptions(problem_file):
    """Each listed exception: z beyond 3 on the main pair, within 3 on the
    protocol pair, both as recorded (passes trivially when the table is
    empty)."""
    pf, _ = problem_file
    for (name, check), entry in sorted(PROTOCOL_EXCEPTIONS.items()):
        assert entry["reason"]
        spec = pf["problems"][name]
        main = {r["check"]: r["z"] for r in C.primary_checks(spec, name, *pair(name))}
        prot = {r["check"]: r["z"] for r in
                C.primary_checks(spec, name, *pair(name, "protocol"))}
        assert abs(main[check]) > C.Z_LIMIT and main[check] == entry["z"]
        assert abs(prot[check]) <= C.Z_LIMIT and prot[check] == entry["z_protocol"]


def committed_benchmark_files():
    out = subprocess.run(["git", "ls-files", "benchmark", "docs/phase4_plan.md",
                          "tests/test_benchmark_openmc.py"], cwd=REPO, capture_output=True,
                         text=True).stdout.split()
    extra = [os.path.relpath(os.path.join(C.RESULTS_DIR, f), REPO)
             for f in os.listdir(C.RESULTS_DIR)] if os.path.isdir(C.RESULTS_DIR) else []
    return sorted(set(out) | set(extra))


def test_no_local_paths_or_user_names():
    # the roots are joined at run time so this file does not match itself
    roots = ("Users", "home", "private", "var/folders")
    pattern = re.compile("|".join(re.escape("/" + r + "/") for r in roots) +
                         r"|[A-Za-z]:\\\\")
    user = getpass.getuser()
    found = []
    for rel in committed_benchmark_files():
        with open(os.path.join(REPO, rel), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if pattern.search(text) or (len(user) >= 4 and user in text):
            found.append(rel)
    assert not found, found


def test_compare_needs_numpy_only():
    code = ("import sys; sys.path.insert(0, %r); import benchmark.compare; "
            "print([m for m in ('openmc', 'mcslab', 'h5py', 'numba', 'matplotlib') "
            "if m in sys.modules])" % REPO)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]", out
