"""Validation of the Phase 3 response tallies on problem D7 (plan checks 2,
3, 4 and 5, plus the reported diagnostics).

One validation run, fixed in the plan before this file existed: D7 (W 0.5
| FLiBe 20 (900 K data and Janz density) | Fe 10 cm, 14.1 MeV beam at
x = 0), depth bins W 10 / FLiBe 20 / Fe 20, 100 batches x 2000 histories,
seed 20261030. Statistical checks are |mean - expected| <= 3 SE with the
batch SE. Same-run comparisons (checks 3 and 4) use the per-batch paired
difference, which accounts for the correlation (as Phase 1 test d does).

Check 2, first flight. A source neutron reaches depth x uncollided with
probability exp(-tau(x)), tau the optical depth at 14.1 MeV, so its
expected track length in a bin [a, b] inside one material is
    (exp(-tau_a) - exp(-tau_b)) / Sigma_t,
and its expected first-collision score w / Sigma_t summed over a layer has
the same value for that layer. Sigma_t and N_k sigma_s are computed here
from the raw HDF5 files (h5py + np.interp) with inline number densities,
independently of mcslab.nucdata / xs / ce_materials, as in the Phase 2a
transmission test. Uncollided neutrons all fly at 14.1 MeV, so the
uncollided response divided by the uncollided flux must equal
N_k sigma_s(14.1 MeV) to round-off (deterministic, 1e-11 relative).
Check 3: track length vs collision estimator per layer for the flux and
every response, and for FLiBe tritium per nuclide.
Check 4: track-length absorption per layer vs the analog count of
histories ended by a *sampled absorption* (region_sums[ABSORPTION]).
Zero-yield and energy-cutoff kills are separate tallies and are not
included.
Check 5 (exact): W has no MT 205, so W tritium is exactly 0 in every
estimator; the summed depth-mesh track length and collision estimator of
each layer equal the region tallies to 1e-11 relative.

Statistical checks here: 50 + 3 + 24 + 3 = 80. Failure protocol (plan): a
check beyond 3 SE stops the work; one diagnostic is then run on seed
20261039 with 400 x 2000 histories, and both results are reported.
"""
import dataclasses
import math
import os
import time
import warnings

import h5py
import numpy as np
import pytest

from mcslab import ce_materials as cm
from mcslab import tallies as T
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import EnergyCutoffWarning, KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.nucdata import Library
from mcslab.rng import STRIDE
from mcslab.sources import BeamSource

SEED = 20261030
N_BATCHES = 100
N_PER_BATCH = 2000
BOUNDS = (0.0, 0.5, 20.5, 30.5)
BINS = (10, 20, 20)
E0 = 14.1e6
N_SIGMA = 3.0
RATIO_BOUND = 1e-11          # uncollided response / flux vs raw N sigma
CONSERVATION_BOUND = 1e-11   # summed mesh tallies vs region tallies
LAYERS = ("W", "FLiBe", "Fe")
RESP_MTS = (301, 901, 444, 205, 207)

# Inline material definitions (independent of ce_materials): mass density
# (g/cm^3), data temperature and atom fractions. W, Fe: CRC densities and
# IUPAC abundances; FLiBe: Janz density at 900 K, Li2BeF4 with natural Li.
FLIBE_RHO = 2.413 - 4.884e-4 * 900.0
RAW_MATERIALS = (
    (19.3, "294K", {"W180": 0.0012, "W182": 0.2650, "W183": 0.1431, "W184": 0.3064,
                    "W186": 0.2843}),
    (FLIBE_RHO, "900K", {"Li6": 2 * 0.0759 / 7, "Li7": 2 * 0.9241 / 7, "Be9": 1 / 7,
                         "F19": 4 / 7}),
    (7.874, "294K", {"Fe54": 0.05845, "Fe56": 0.91754, "Fe57": 0.02119,
                     "Fe58": 0.00282}),
)
N_AVOGADRO, M_NEUTRON_U = 6.02214076e23, 1.00866491595


def _library():
    try:
        return Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def validation_config(library, seed=SEED, n_batches=N_BATCHES, n_per_batch=N_PER_BATCH,
                      depth_bins=BINS):
    flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
    geom = SlabGeometry(BOUNDS, [cm.tungsten(), flibe, cm.iron()])
    return KinRunConfig(geom, BeamSource(), MonoEnergetic(E0), library, n_batches,
                        n_per_batch, seed, depth_bins=depth_bins)


def run_quiet(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        return run_kin(cfg)


@pytest.fixture(scope="module")
def val():
    lib = _library()
    t0 = time.perf_counter()
    res = run_quiet(validation_config(lib))
    elapsed = time.perf_counter() - t0
    assert res.lost == 0 and res.max_draws < STRIDE
    assert res.balance()["residual"] == 0.0
    return lib, res, elapsed


def raw_layers(lib):
    """Per layer: (Sigma_t, {nuclide: (N, {mt: sigma}, sigma_a)}) at 14.1 MeV
    from the raw files. sigma_a sums the non-redundant reactions without an
    outgoing neutron."""
    out = []
    for rho, T_key, fracs in RAW_MATERIALS:
        micro, awr = {}, {}
        for name in fracs:
            with h5py.File(os.path.join(str(lib.root), "neutron", name + ".h5"), "r") as f:
                g = f[name]
                awr[name] = float(g.attrs["atomic_weight_ratio"])
                grid = g["energy"][T_key][()]
                st = sa = 0.0
                resp = {}
                for key in g["reactions"]:
                    r = g["reactions"][key]
                    d = r[T_key]["xs"]
                    i0 = int(d.attrs["threshold_idx"])
                    val = 0.0 if E0 < grid[i0] else float(np.interp(E0, grid[i0:], d[()]))
                    mt = int(r.attrs["mt"])
                    if mt in RESP_MTS:
                        resp[mt] = val
                    if r.attrs["redundant"]:
                        continue
                    st += val
                    if not any(r[p].attrs["particle"] == b"neutron"
                               for p in r if p.startswith("product_")):
                        sa += val
                micro[name] = (st, resp, sa)
        mean_mass = sum(fracs[n] * awr[n] * M_NEUTRON_U for n in fracs)
        n_total = rho * N_AVOGADRO / mean_mass * 1e-24
        nucs = {n: (fracs[n] * n_total, micro[n][1], micro[n][2]) for n in fracs}
        sig_t = sum(fracs[n] * n_total * micro[n][0] for n in fracs)
        out.append((sig_t, nucs))
    return out


def _check(report, res, quantity, batches, expected, failures):
    mean, se = T.batch_stats(batches)
    n_se = report(quantity, mean, expected, se, res.max_draws,
                  cutoff=float(res.cutoff.sum(axis=1).mean()))
    if not abs(mean - expected) <= N_SIGMA * se:
        failures.append(f"{quantity}: {mean:.7g} vs {expected:.7g} ({n_se:+.2f} SE)")


def _layer_bins(res, r):
    return np.flatnonzero(res.mesh.bin_region == r)


# ---------------------------------------------------------------------------
# check 2: first flight
# ---------------------------------------------------------------------------
def test_first_flight_tracklength_per_bin(val, report):
    lib, res, _ = val
    layers = raw_layers(lib)
    edges = res.mesh.bin_edges
    tl = res.mesh_flux_batches(T.EST_TL_UNC)
    tau0 = 0.0
    failures = []
    for r, (sig_t, _) in enumerate(layers):
        for b in _layer_bins(res, r):
            ta = tau0 + sig_t * (edges[b] - BOUNDS[r])
            tb = tau0 + sig_t * (edges[b + 1] - BOUNDS[r])
            expected = (math.exp(-ta) - math.exp(-tb)) / sig_t
            _check(report, res, f"TL_unc {LAYERS[r]} bin {b} [{edges[b]:.2f},"
                   f"{edges[b + 1]:.2f}]", tl[:, b], expected, failures)
        tau0 += sig_t * (BOUNDS[r + 1] - BOUNDS[r])
    assert not failures, "\n".join(failures)


def test_first_collision_per_layer(val, report):
    lib, res, _ = val
    layers = raw_layers(lib)
    coll = res.mesh_flux_batches(T.EST_COLL_UNC)
    tau0 = 0.0
    failures = []
    for r, (sig_t, _) in enumerate(layers):
        tau1 = tau0 + sig_t * (BOUNDS[r + 1] - BOUNDS[r])
        expected = (math.exp(-tau0) - math.exp(-tau1)) / sig_t
        _check(report, res, f"COLL_first {LAYERS[r]} layer",
               coll[:, _layer_bins(res, r)].sum(axis=1), expected, failures)
        tau0 = tau1
    assert not failures, "\n".join(failures)


def test_uncollided_response_ratios(val, note):
    """Deterministic: every uncollided score is (flux) x N_k sigma_s(E0)."""
    lib, res, _ = val
    layers = raw_layers(lib)
    labels = res.tally_nuclides
    tal = res.tally.sum(axis=0)
    flx = res.mesh_flux.sum(axis=0)
    worst = 0.0
    n_checked = 0
    for est in (T.EST_TL_UNC, T.EST_COLL_UNC):
        for r, (_, nucs) in enumerate(layers):
            for b in _layer_bins(res, r):
                assert flx[b, est] > 0.0, (b, est)
                for k, label in enumerate(labels[:-1]):
                    for s in range(T.N_RESP):
                        v = tal[b, k, s, est]
                        if label not in nucs:
                            assert v == 0.0, (b, label, s)
                            continue
                        N, resp, sig_a = nucs[label]
                        if s == T.R_ABSORPTION:
                            want = N * sig_a
                        else:
                            mt = RESP_MTS[s]
                            if mt not in resp:
                                assert v == 0.0 and not res.response_present[k, s]
                                continue
                            want = N * resp[mt]
                        err = abs(v / flx[b, est] - want) / want
                        worst = max(worst, err)
                        n_checked += 1
                        assert err <= RATIO_BOUND, (b, label, s, est, v / flx[b, est], want)
                for s in range(T.N_RESP):
                    want = sum((n[0] * n[2]) if s == T.R_ABSORPTION else
                               n[0] * n[1].get(RESP_MTS[s], 0.0) for n in nucs.values())
                    v = tal[b, -1, s, est]
                    if want == 0.0:
                        assert v == 0.0
                        continue
                    err = abs(v / flx[b, est] - want) / want
                    worst = max(worst, err)
                    n_checked += 1
                    assert err <= RATIO_BOUND, (b, "total", s, est)
    note(f"(check 2) uncollided response / flux vs raw N sigma(14.1 MeV): {n_checked} "
         f"ratios, worst relative difference {worst:.2e} (bound {RATIO_BOUND:g})")


# ---------------------------------------------------------------------------
# check 5: exact identities
# ---------------------------------------------------------------------------
def test_exact_w_tritium_and_conservation(val, report, note):
    lib, res, _ = val
    w_bins = _layer_bins(res, 0)
    assert not res.tally[:, w_bins, :, T.R_H3, :].any()
    report("W tritium, all bins / estimators (no MT 205)",
           float(np.abs(res.tally[:, w_bins, :, T.R_H3, :]).sum()), 0.0, 0.0, res.max_draws,
           cutoff=float(res.cutoff.sum(axis=1).mean()))
    worst = 0.0
    for r in range(3):
        bins = _layer_bins(res, r)
        for est, score in ((T.EST_TL, T.TRACK_LENGTH), (T.EST_COLL, T.COLL_ESTIMATOR)):
            mesh = res.mesh_flux[:, bins, est].sum(axis=1)
            region = res.region_sums[:, score, r]
            err = float((np.abs(mesh - region) / region).max())
            worst = max(worst, err)
            assert err <= CONSERVATION_BOUND, (LAYERS[r], est, err)
    note(f"(check 5) summed depth-mesh TL / collision estimator vs region tallies, per "
         f"layer and batch: worst relative difference {worst:.2e} "
         f"(bound {CONSERVATION_BOUND:g})")


# ---------------------------------------------------------------------------
# check 3: track length vs collision estimator
# ---------------------------------------------------------------------------
def test_estimator_agreement(val, report):
    lib, res, _ = val
    failures = []
    for r in range(3):
        bins = _layer_bins(res, r)
        d = (res.mesh_flux_batches(T.EST_TL)[:, bins].sum(axis=1)
             - res.mesh_flux_batches(T.EST_COLL)[:, bins].sum(axis=1))
        _check(report, res, f"TL - coll: flux, {LAYERS[r]}", d, 0.0, failures)
        for s, name in enumerate(T.RESPONSE_NAMES):
            if r == 0 and s == T.R_H3:
                continue        # exactly 0 in both (W has no MT 205): exact check above
            d = (res.tally_batches(s, T.EST_TL)[:, bins].sum(axis=1)
                 - res.tally_batches(s, T.EST_COLL)[:, bins].sum(axis=1))
            _check(report, res, f"TL - coll: {name}, {LAYERS[r]}", d, 0.0, failures)
    bins = _layer_bins(res, 1)
    for nuc in ("Li6", "Li7", "Be9", "F19"):
        d = (res.tally_batches(T.R_H3, T.EST_TL, nuc)[:, bins].sum(axis=1)
             - res.tally_batches(T.R_H3, T.EST_COLL, nuc)[:, bins].sum(axis=1))
        _check(report, res, f"TL - coll: H3 {nuc}, FLiBe", d, 0.0, failures)
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# check 4: track-length absorption vs sampled analog absorptions
# ---------------------------------------------------------------------------
def test_tracklength_absorption_vs_analog(val, report):
    lib, res, _ = val
    failures = []
    analog = res.absorptions        # histories ended by a sampled absorption only
    for r in range(3):
        bins = _layer_bins(res, r)
        tl = res.tally_batches(T.R_ABSORPTION, T.EST_TL)[:, bins].sum(axis=1)
        _check(report, res, f"TL absorption - analog, {LAYERS[r]}", tl - analog[:, r], 0.0,
               failures)
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# diagnostics (reported only)
# ---------------------------------------------------------------------------
def test_diagnostics_reported(val, note):
    lib, res, elapsed = val
    # Li-6 205 vs 105, Li-7 205 vs sum of 52-82, from the raw 900 K files
    for name, comp in (("Li6", [105]), ("Li7", list(range(52, 83)))):
        with h5py.File(os.path.join(str(lib.root), "neutron", name + ".h5"), "r") as f:
            g = f[name]
            grid = g["energy"]["900K"][()]

            def full(mt):
                d = g["reactions"][f"reaction_{mt:03d}"]["900K"]["xs"]
                out = np.zeros_like(grid)
                out[int(d.attrs["threshold_idx"]):] = d[()]
                return out
            s205 = full(205)
            present = [m for m in comp if f"reaction_{m:03d}" in g["reactions"]]
            diff = np.abs(s205 - sum(full(m) for m in present))
            rel = diff[s205 > 0] / s205[s205 > 0]
            note(f"(diag) {name} sigma_205 vs sum of MT {present[0]}-{present[-1]} at 900 K: "
                 f"max |diff| {diff.max():.3e} b, max relative {rel.max():.3e}")
    # per-nuclide sums vs the in-kernel material total
    tal = res.tally.sum(axis=0)
    parts = tal[:, :-1].sum(axis=1)
    tot = tal[:, -1]
    nz = tot != 0.0
    note(f"(diag) sum over nuclides vs in-kernel total (all bins, responses, estimators): "
         f"max relative difference {float((np.abs(parts - tot)[nz] / np.abs(tot[nz])).max()):.2e}")
    # analog Li-7 MT 52-82 events vs track-length Li-7 tritium in FLiBe
    li7 = [c for c, (n, mt) in enumerate(res.channel_labels) if n == "Li7" and 52 <= mt <= 82]
    analog = res.chan_events[:, li7].sum(axis=1) / res._n
    tl = res.tally_batches(T.R_H3, T.EST_TL, "Li7")[:, _layer_bins(res, 1)].sum(axis=1)
    m_a, _ = T.batch_stats(analog)
    m_t, _ = T.batch_stats(tl)
    _, se = T.batch_stats(tl - analog)
    note(f"(diag) Li-7 tritium per source: track length {m_t:.6g}, analog MT 52-82 events "
         f"{m_a:.6g}, paired difference {(m_t - m_a) / se:+.2f} SE (not a pass/fail check)")
    # runtime overhead: the same run with the tallies off and on again, both
    # warm (the fixture's run also paid for loading the data and kernels)
    times = {}
    for label, bins in (("off", None), ("on", BINS)):
        t0 = time.perf_counter()
        run_quiet(validation_config(lib, depth_bins=bins))
        times[label] = time.perf_counter() - t0
    note(f"(diag) validation run (2e5 histories, warm, incl. packing): tallies on "
         f"{times['on']:.2f} s, off {times['off']:.2f} s, overhead "
         f"{100 * (times['on'] / times['off'] - 1):+.0f}% (first, cold call: {elapsed:.2f} s)")
