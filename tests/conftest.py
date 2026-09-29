"""Shared fixtures: JIT warm-up and a validation summary table.

Tests call the `report` fixture to record each checked quantity; the table
(measured, expected, distance in standard errors, test runtime, max random
numbers per history vs STRIDE) prints at the end of the pytest run.
"""
import math

import pytest

from mcslab.rng import STRIDE

_RECORDS = []
_DURATIONS = {}


@pytest.fixture(scope="session", autouse=True)
def _warm_jit():
    """Load/compile the kernels once, so per-test runtimes measure transport."""
    from mcslab.config import RunConfig, run
    from mcslab.geometry import SlabGeometry
    from mcslab.materials import Material
    from mcslab.sources import BeamSource, IsotropicPlaneSource
    g = SlabGeometry.uniform(Material("w", 0.5, 0.5), 0.0, 1.0)
    for src in (BeamSource(), IsotropicPlaneSource(0.5)):
        run(RunConfig(g, src, n_batches=2, histories_per_batch=10))


@pytest.fixture
def report(request):
    test = request.node.name

    def _record(quantity, measured, expected, se, max_draws=None):
        measured, expected, se = float(measured), float(expected), float(se)
        if se > 0.0:
            n_se = (measured - expected) / se
        else:
            n_se = 0.0 if measured == expected else math.inf
        _RECORDS.append((test, quantity, measured, expected, se, n_se, max_draws))
        return n_se
    return _record


def pytest_runtest_logreport(report):
    if report.when == "call":
        _DURATIONS[report.nodeid.split("::")[-1]] = report.duration


def pytest_terminal_summary(terminalreporter):
    if not _RECORDS:
        return
    tr = terminalreporter
    tr.section("physics validation summary")
    tr.write_line(f"{'test':<46}{'quantity':<34}{'measured':>14}{'expected':>14}"
                  f"{'SE':>11}{'n_SE':>8}{'time s':>8}{'max_draws':>11}")
    last = None
    for test, q, m, e, se, n_se, md in _RECORDS:
        head = test if test != last else ""
        t = f"{_DURATIONS[test]:.2f}" if test != last and test in _DURATIONS else ""
        mds = str(md) if md is not None and test != last else ""
        se_s = f"{se:.3e}" if se > 0 else "exact"
        tr.write_line(f"{head:<46}{q:<34}{m:>14.7g}{e:>14.7g}{se_s:>11}"
                      f"{n_se:>+8.2f}{t:>8}{mds:>11}")
        last = test
    worst = max(abs(r[5]) for r in _RECORDS)
    tr.write_line(f"{len(_RECORDS)} checks; largest |n_SE| = {worst:.2f}; "
                  f"STRIDE = {STRIDE}")
