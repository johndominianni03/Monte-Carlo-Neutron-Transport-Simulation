"""Phase 2b Part 2: secondary-neutron laws on real ENDF/B-VIII.0 data.

Seeds were fixed in docs/phase2b_plan.md before any of these tests existed
(docs/development.md: no seed-shopping). Acceptance: 3 SE for means; p >= 0.0027 for
chi-square tests; "exact" = integer identities or round-off with a stated
bound.

What each test covers, and what it doesn't:
  reader: every non-redundant neutron reaction of the 14 nuclides is read
       as the law that scripts/law_inventory.py (h5py only, independent of
       mcslab.nucdata) finds for it, and malformed or unimplemented laws
       are refused or flagged explicitly, never skipped. It does not test
       sampling.
  (f)  test_f_replay_every_new_law: all 56 reactions with a Part 2 law (ACE
       law 61 in CM and lab, the F19 applicability mixture, ACE law 4 with
       tabular angles, tabulated yields) at the first tabulated incident
       energy, the tabulated energy nearest 14.1 MeV and the midpoint of the
       pair bracketing 14.1 MeV, 2000 events each. Each event's draws are
       replayed in plain Python on raw h5py arrays (independent index
       rules): the sampled E_out is the inverse of the chosen table's stored
       CDF after the scaled interpolation (<= 1e-12 E_in, and the forward
       CDF matches the draw to <= 1e-9); mu is the inverse of the angle
       table the replayed rule selects (<= 1e-9); collision.inelastic_scatter
       gives the CM -> lab transform of the same values (<= 1e-12) and the
       multiplicity of approved deviation D2 exactly.
       test_f_law_joint_distribution: the joint (E_out, mu) of five laws
       (Fe56 MT 91 and W184 MT 16: law 61 CM lin-lin; Be9 MT 16: law 61 lab
       histogram; F19 MT 16: two laws + applicability; Li7 MT 16: law 4 +
       angle) at an on-grid and a midpoint incident energy: chi-square
       against probabilities built from the PDFs p alone (exact integration,
       the stored c never read), modelling the stochastic table choice, the
       scaled interpolation, the angle-table rule and the applicability
       mixture. It does not test the evaluated data themselves, or laws in
       transport (that is (g), the balance tests and the D7 regression).
  packing: every reaction packs as inventoried, nothing unsupported; a
       problem whose nuclide has an unimplemented law is refused even below
       that reaction's threshold.
  (g)  test_g_mt5_multiplicity: MT 5 of Fe56 and W184 at energies where
       y(E) lies in (0,1), (1,2) and (2,3), on and between yield-table
       points: n is floor(y) or floor(y) + 1 (exact) and its mean is y(E)
       at 3 SE (approved deviation D2). test_g_zero_and_integer_yields_
       draw_nothing: integer yields (including y = 0) make no extra draw.
       test_g_kernel_multiplicity_and_balance: through the kernel on real
       W | Fe (30 MeV) and FLiBe (14.1 MeV) slabs, every constant-yield
       channel creates exactly (y - 1) secondaries per event, zero-yield
       events come only from MT 5, and the per-batch balance including the
       zero-yield term is exact. These runs use a 1 MeV energy cutoff, so
       they say nothing about low-energy transport.
"""
import math
import os
import shutil
import sys

import h5py
import numpy as np
import pytest
from numba import njit

from mcslab import collision as C
from mcslab import distributions as D
from mcslab import nucdata as N
from mcslab.rng import RNG_SIZE, init_history, prn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import law_inventory  # noqa: E402

KIND = {"angle-only": type(None), "level": N.LevelInelastic,
        "continuous": N.ContinuousTabular}


def _library():
    try:
        return N.Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
def test_reader_matches_law_inventory(note):
    lib = _library()
    rows = law_inventory.scan(str(lib.root))
    nucs = {}
    for row in rows:
        name, mt = row["nuclide"], row["mt"]
        if name not in nucs:
            nucs[name] = lib.load(name, distributions=True)
        rx = nucs[name].reactions[mt]
        assert not rx.redundant and rx.center_of_mass == row["cm"], (name, mt)
        prod = rx.products[0]
        assert prod.particle == "neutron" and prod.laws_read, (name, mt)
        assert len(prod.laws) == len(row["distributions"]), (name, mt)
        assert len(prod.applicability) == (len(prod.laws) if len(prod.laws) > 1 else 0)
        y = row["yield"]
        if y["type"] == "Polynomial":
            assert prod.yield_table is None and prod.yield_coefficients == y["coefficients"]
        else:
            assert np.array_equal(prod.yield_table.x, y["x"])
            assert np.array_equal(prod.yield_table.y, y["y"])
        for law, d in zip(prod.laws, row["distributions"]):
            if d["kind"] == "correlated":
                assert isinstance(law, N.CorrelatedAngleEnergy), (name, mt)
                assert np.array_equal(law.energy, d["incident_energies"])
                assert sum(len(m) for m in law.mu) == sum(t.x.size for t in law.tables)
            else:
                assert isinstance(law, N.UncorrelatedAngleEnergy), (name, mt)
                assert isinstance(law.energy, KIND[d["kind"]]), (name, mt, d["kind"])
                assert (law.angle is None) == (d["angle"] is None)
                if d["kind"] == "continuous":
                    assert np.array_equal(law.energy.energy, d["incident_energies"])
    note(f"(reader) {len(rows)} non-redundant neutron reactions of {len(nucs)} nuclides "
         "read as the h5py inventory lists them; 0 UnsupportedLaw")


def _copy(lib, tmp_path, name):
    dst = tmp_path / f"{name}.h5"
    shutil.copyfile(lib.nuclide_path(name), dst)
    return dst


def _law(path, mt, j=0):
    nuc = N.read_nuclide(path, distributions=True)
    return nuc.reactions[mt].products[0].laws[j]


BE16 = "Be9/reactions/reaction_016/product_0/distribution_0"
LI16 = "Li7/reactions/reaction_016/product_0/distribution_0"


def _edit_attr(path, obj, attr, fn):
    with h5py.File(path, "r+") as f:
        a = np.array(f[obj].attrs[attr])
        f[obj].attrs[attr] = fn(a)


def test_reader_flags_unimplemented_laws(tmp_path):
    """Unimplemented features become an explicit UnsupportedLaw (the driver
    then refuses the problem; tests in this file check that too)."""
    lib = _library()

    def set_first(value):
        def fn(a):
            a = a.copy()
            a.flat[0] = value
            return a
        return fn

    cases = [
        ("Be9", BE16 + "/energy_out", "n_discrete_lines", set_first(1), "discrete lines"),
        ("Be9", BE16 + "/energy", "interpolation",
         lambda a: np.array([[a[0, 0]], [1]]), "histogram incident"),
        ("Li7", LI16 + "/energy/distribution", "n_discrete_lines", set_first(2),
         "discrete lines"),
        ("Li7", LI16 + "/energy", "type", lambda a: np.bytes_(b"evaporation"),
         "uncorrelated(energy=evaporation)"),
        ("Be9", BE16, "type", lambda a: np.bytes_(b"kalbach-mann"), "kalbach-mann"),
    ]
    for name, obj, attr, fn, text in cases:
        path = _copy(lib, tmp_path, name)
        _edit_attr(path, obj, attr, fn)
        law = _law(path, 16)
        assert isinstance(law, N.UnsupportedLaw), (obj, attr)
        assert text in law.description, law.description


def test_reader_refuses_malformed_laws(tmp_path):
    """Anything unexpected inside a law that is parsed raises."""
    lib = _library()

    def code(value):
        def fn(a):
            a = a.copy()
            a.flat[0] = value
            return a
        return fn

    path = _copy(lib, tmp_path, "Be9")
    _edit_attr(path, BE16 + "/energy_out", "interpolation", code(5))
    with pytest.raises(ValueError, match="interpolation code 5"):
        _law(path, 16)

    path = _copy(lib, tmp_path, "Be9")
    with h5py.File(path, "r+") as f:
        f[BE16 + "/energy_out"].attrs["unexpected"] = 1
    with pytest.raises(ValueError, match="attributes"):
        _law(path, 16)

    path = _copy(lib, tmp_path, "Be9")
    _edit_attr(path, BE16 + "/energy", "interpolation",
               lambda a: np.array([[5, a[0, 0]], [2, 2]]))
    with pytest.raises(ValueError, match="regions"):
        _law(path, 16)

    # a repeated outgoing energy that carries probability mass
    path = _copy(lib, tmp_path, "Be9")
    with h5py.File(path, "r+") as f:
        ds = f[BE16 + "/energy_out"]
        n0 = int(ds.attrs["offsets"][1])          # first table: columns 0 .. n0-1
        j = int(np.flatnonzero(np.diff(ds[2, :n0]) > 0.0)[0])
        ds[0, j + 1] = ds[0, j]
    with pytest.raises(ValueError, match="repeated with probability mass"):
        _law(path, 16)

    # an angle table outside [-1, 1]
    path = _copy(lib, tmp_path, "Be9")
    with h5py.File(path, "r+") as f:
        f[BE16 + "/mu"][0, 0] = -1.5
    with pytest.raises(ValueError, match="strictly increasing"):
        _law(path, 16)


# ---------------------------------------------------------------------------
# (f) the Part 2 laws: per-event replay and statistical tests
# ---------------------------------------------------------------------------
# Seed fixed in docs/phase2b_plan.md before this test existed.
SEED_F = 20261020
N_REPLAY = 2000
N_STAT = 200_000
E14 = 14.1e6
NUCS13 = ["Fe54", "Fe56", "Fe57", "Fe58", "W180", "W182", "W183", "W184", "W186",
          "Li6", "Li7", "Be9", "F19"]


def raw_product(lib, nuclide, mt):
    """The neutron product of one reaction straight from the HDF5 file
    (h5py): {"cm", "yield": (x, y) or float, "dists": [...], "app": [...]}.
    Tables are (x, p, c, interp) tuples, as stored."""
    with h5py.File(lib.nuclide_path(nuclide), "r") as f:
        r = f[f"{nuclide}/reactions/reaction_{mt:03d}"]
        p = r["product_0"]
        y = p["yield"]
        yld = (float(np.ravel(y[()])[0]) if y.attrs["type"] == b"Polynomial"
               else (y[()][0].copy(), y[()][1].copy(), int(np.ravel(y.attrs["interpolation"])[0])))
        dists, apps = [], []
        nd = int(p.attrs["n_distribution"])
        for j in range(nd):
            d = p[f"distribution_{j}"]
            if nd > 1:
                a = d["applicability"]
                apps.append((a[()][0].copy(), a[()][1].copy(), int(np.ravel(a.attrs["interpolation"])[0])))
            if d.attrs["type"] == b"correlated":
                eo = d["energy_out"]
                data = eo[()]
                offs = list(eo.attrs["offsets"]) + [data.shape[1]]
                ints = eo.attrs["interpolation"]
                mu = d["mu"][()]
                moff = np.round(data[4]).astype(int)
                mend = np.append(moff[1:], mu.shape[1])
                mint = np.round(data[3]).astype(int)
                tables, angles = [], []
                for t in range(len(offs) - 1):
                    a0, a1 = offs[t], offs[t + 1]
                    tables.append((data[0, a0:a1].copy(), data[1, a0:a1].copy(),
                                   data[2, a0:a1].copy(), int(ints[t])))
                    angles.append([(mu[0, moff[k]:mend[k]].copy(), mu[1, moff[k]:mend[k]].copy(),
                                    mu[2, moff[k]:mend[k]].copy(), int(mint[k]))
                                   for k in range(a0, a1)])
                dists.append({"kind": "correlated", "energy": d["energy"][()].copy(),
                              "tables": tables, "mu": angles})
            else:
                eg = d["energy"]
                dist = eg["distribution"]
                data = dist[()]
                offs = list(dist.attrs["offsets"]) + [data.shape[1]]
                ints = dist.attrs["interpolation"]
                tables = [(data[0, offs[t]:offs[t + 1]].copy(), data[1, offs[t]:offs[t + 1]].copy(),
                           data[2, offs[t]:offs[t + 1]].copy(), int(ints[t]))
                          for t in range(len(offs) - 1)]
                ag = d["angle"]
                amu = ag["mu"]
                adata = amu[()]
                aoffs = list(amu.attrs["offsets"]) + [adata.shape[1]]
                aints = amu.attrs["interpolation"]
                atabs = [(adata[0, aoffs[t]:aoffs[t + 1]].copy(), adata[1, aoffs[t]:aoffs[t + 1]].copy(),
                          adata[2, aoffs[t]:aoffs[t + 1]].copy(), int(aints[t]))
                         for t in range(len(aoffs) - 1)]
                dists.append({"kind": "continuous", "energy": eg["energy"][()].copy(),
                              "tables": tables, "angle": (ag["energy"][()].copy(), atabs)})
        return {"cm": bool(r.attrs["center_of_mass"]), "yield": yld, "dists": dists, "app": apps}


# ---- plain-Python transcriptions of the index rules (for the replay) ------
def py_lower_bound_index(a, v):
    """OpenMC lower_bound_index: 0 if a[0] == v, else (first index with
    a >= v) - 1."""
    if a[0] == v:
        return 0
    return int(np.searchsorted(a, v, side="left")) - 1


def py_energy_index(a, E):
    """OpenMC get_energy_index."""
    i, f = 0, 0.0
    if E >= a[0]:
        i = py_lower_bound_index(a, E)
        if i + 1 < a.size:
            f = (E - a[i]) / (a[i + 1] - a[i])
    return i, f


def py_cont_index(a, E):
    """ContinuousTabular::sample's own bracketing."""
    if E < a[0]:
        return 0, 0.0
    if E > a[-1]:
        return a.size - 2, 1.0
    i = py_lower_bound_index(a, E)
    return i, (E - a[i]) / (a[i + 1] - a[i])


def py_tab1d(x, y, interp, v):
    if v < x[0]:
        return y[0]
    if v > x[-1]:
        return y[-1]
    i = py_lower_bound_index(x, v)
    if interp == 1:
        return y[i]
    return y[i] + (v - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i])


def forward_cdf(tab, X, k, normalise):
    """The stored-CDF map of bin k at X: c_k + p_k d (+ m d^2 / 2). With
    normalise, p and c are divided by c[-1] (Tabular::init)."""
    x, p, c, interp = tab
    s = c[-1] if normalise else 1.0
    d = X - x[k]
    if interp == 1:
        return (c[k] + p[k] * d) / s
    m = (p[k + 1] - p[k]) / (x[k + 1] - x[k])
    return (c[k] + p[k] * d + 0.5 * m * d * d) / s


def tabular_bin(tab, v, normalise):
    """Tabular::sample_unbiased's bin: first k >= 1 with c_k >= v, minus 1."""
    c = tab[2] / (tab[2][-1] if normalise else 1.0)
    return int(np.searchsorted(c[1:], v, side="left"))


def eout_search(c, r1):
    """-> (k, c_k, c_k1), the outgoing bin search of OpenMC's law-4 / law-61
    samplers, written independently: k = number of c[j+1] <= r1 for
    j = 0 .. n-3; c_k1 = c[k+1] when the search breaks, c[k] when it runs to
    the last bin, +inf for 2-point tables (secondary_correlated.cpp)."""
    n = c.size
    k = int(np.searchsorted(c[1:n - 1], r1, side="right"))
    if n == 2:
        return 0, c[0], np.inf
    if k <= n - 3:
        return k, c[k], c[k + 1]
    return k, c[k], c[k]


def eout_inverse(tab, k, r1):
    x, p, c, interp = tab
    if interp == 1:
        return x[k] + (r1 - c[k]) / p[k] if p[k] > 0.0 else x[k]
    if x[k + 1] == x[k]:
        return x[k]
    m = (p[k + 1] - p[k]) / (x[k + 1] - x[k])
    if m == 0.0:
        return x[k] + (r1 - c[k]) / p[k]
    return x[k] + (math.sqrt(max(0.0, p[k] * p[k] + 2.0 * m * (r1 - c[k]))) - p[k]) / m


def scale_params(tables, i, r):
    """(E_1, E_K) of the scaled interpolation between tables i and i+1."""
    a, b = tables[i][0], tables[i + 1][0]
    return a[0] + r * (b[0] - a[0]), a[-1] + r * (b[-1] - a[-1])


@njit(cache=True)
def _draws(seed, n, k):
    """The first k draws of the streams of histories 0 .. n-1."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    out = np.empty((n, k))
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        for j in range(k):
            out[s, j] = prn(rng)
    return out


@njit(cache=True)
def _sample_product(ip, fp, prod, E, awr, q, seed, n):
    """n (E_out, mu) samples in the law's frame; sample s uses the stream of
    history s. Also the most draws any sample used."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    e = np.empty(n)
    m = np.empty(n)
    md = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        e[s], m[s] = D.sample_product(ip, fp, prod, E, awr, q, rng)
        md = max(md, int(rng[1]))
    return e, m, md


@njit(cache=True)
def _inelastic(ip, fp, prod, E, awr, q, cm, seed, n):
    """collision.inelastic_scatter on the same streams: (E_lab, mu_lab, n_out)."""
    rng = np.zeros(RNG_SIZE, np.uint64)
    e = np.empty(n)
    m = np.empty(n)
    k = np.empty(n, np.int64)
    md = 0
    for s in range(n):
        init_history(rng, seed, np.uint64(s))
        E2, u, v, w, mu, n_out = C.inelastic_scatter(E, 0.6, 0.8, 0.0, awr, q, cm, ip, fp,
                                                     prod, rng)
        e[s] = E2
        m[s] = mu
        k[s] = n_out
        md = max(md, int(rng[1]))
    return e, m, k, md


def replay_event(raw, E, d, e_s, mu_s):
    """Replays one law-frame event from its draws d. -> (|F_E - r1|,
    |E_pred - e_s| / E, |F_mu - xi_mu|, number of draws used)."""
    pos = 0
    j = 0
    if len(raw["dists"]) > 1:
        c, prob = d[0], 0.0
        j = len(raw["dists"]) - 1
        for jj, (ax, ay, ai) in enumerate(raw["app"]):
            prob += py_tab1d(ax, ay, ai, E)
            if c <= prob:
                j = jj
                break
        pos = 1
    law = raw["dists"][j]
    mu_err = 0.0
    if law["kind"] == "continuous":
        ae, atabs = law["angle"]
        ia, ra = py_energy_index(ae, E)
        t = atabs[ia + 1] if ra > d[pos] else atabs[ia]
        mu_err = abs(forward_cdf(t, mu_s, tabular_bin(t, d[pos + 1], True), True) - d[pos + 1])
        pos += 2
        i, r = py_cont_index(law["energy"], E)
    else:
        i, r = py_energy_index(law["energy"], E)
    tables = law["tables"]
    l = i + 1 if r > d[pos] else i
    r1 = d[pos + 1]
    pos += 2
    tab = tables[l]
    k, c_k, c_k1 = eout_search(tab[2], r1)
    E_1, E_K = scale_params(tables, i, r)
    x = tab[0]
    to_final = (E_K - E_1) / (x[-1] - x[0])
    e_pred = E_1 + (eout_inverse(tab, k, r1) - x[0]) * to_final
    e_unscaled = x[0] + (e_s - E_1) / to_final
    f_err = abs(forward_cdf(tab, e_unscaled, k, False) - r1)
    if law["kind"] == "correlated":
        a = k if (r1 - c_k < c_k1 - r1 or tab[3] == 1) else k + 1
        mt = law["mu"][l][a]
        mu_err = abs(forward_cdf(mt, mu_s, tabular_bin(mt, d[pos], True), True) - d[pos])
        pos += 1
    return f_err, abs(e_pred - e_s) / E, mu_err, pos


def law_energies(raw):
    """(first tabulated incident energy, the tabulated energy nearest
    14.1 MeV, the midpoint of the pair bracketing 14.1 MeV)."""
    e = raw["dists"][0]["energy"]
    j = int(np.argmin(np.abs(e - E14)))
    jm = int(np.searchsorted(e, E14, side="right")) - 1
    jm = min(max(jm, 0), e.size - 2)
    return float(e[0]), float(e[j]), 0.5 * float(e[jm] + e[jm + 1])


def _packed_real(names):
    lib = _library()
    nucs = {n: lib.load(n, distributions=True) for n in names}
    return lib, C.pack_physics(list(nucs), nucs)


def test_f_replay_every_new_law(note):
    """Exact (round-off) replay of every event of every Part 2 law: the
    sampled E_out is the inverse of the chosen table's stored CDF at the
    replayed draw, after the scaled interpolation; mu is the inverse of the
    angle table the replayed rule selects; CM -> lab and the multiplicity
    follow from the same draws."""
    lib, phys = _packed_real(NUCS13)
    cases = [(c, name, mt) for c, (name, mt) in enumerate(phys.labels)
             if mt != 2 and not 51 <= mt <= 90]
    assert len(cases) == 56
    seed = np.uint64(SEED_F)
    d_all = _draws(seed, N_REPLAY, 7)
    worst = {"F_E": 0.0, "E": 0.0, "F_mu": 0.0, "lab": 0.0}
    n_events = 0
    max_draws = 0
    for c, name, mt in cases:
        raw = raw_product(lib, name, mt)
        k_nuc = int(phys.ch_int[c, C.CH_NUC])
        awr, q, cm = float(phys.nuc_awr[k_nuc]), float(phys.ch_q[c]), int(phys.ch_int[c, C.CH_CM])
        prod = int(phys.ch_int[c, C.CH_PROD])
        assert cm == int(raw["cm"])
        for E in law_energies(raw):
            e_s, mu_s, md = _sample_product(phys.ip, phys.fp, prod, E, awr, q, seed, N_REPLAY)
            e_l, mu_l, n_out, md2 = _inelastic(phys.ip, phys.fp, prod, E, awr, q, cm, seed, N_REPLAY)
            max_draws = max(max_draws, md, md2)
            for s in range(N_REPLAY):
                f_e, e_err, f_mu, used = replay_event(raw, E, d_all[s], e_s[s], mu_s[s])
                worst["F_E"] = max(worst["F_E"], f_e)
                worst["E"] = max(worst["E"], e_err)
                worst["F_mu"] = max(worst["F_mu"], f_mu)
                # CM -> lab (OpenMC inelastic_scatter) from the law-frame values
                if cm:
                    ap1 = awr + 1.0
                    el = e_s[s] + (E + 2.0 * mu_s[s] * ap1 * math.sqrt(E * e_s[s])) / (ap1 * ap1)
                    ml = mu_s[s] * math.sqrt(e_s[s] / el) + math.sqrt(E / el) / ap1
                else:
                    el, ml = e_s[s], mu_s[s]
                ml = math.copysign(1.0, ml) if abs(ml) > 1.0 else ml
                worst["lab"] = max(worst["lab"], abs(el - e_l[s]) / E, abs(ml - mu_l[s]))
                # multiplicity (D2): the draw after phi decides a non-integer yield
                y = (raw["yield"] if isinstance(raw["yield"], float)
                     else py_tab1d(*raw["yield"], E))
                fl = math.floor(y)
                expect = int(fl) if fl == y else int(fl) + (1 if d_all[s, used + 1] < y - fl else 0)
                assert n_out[s] == expect, (name, mt, E, s, y, n_out[s])
            n_events += N_REPLAY
    note(f"(f) replay: {len(cases)} reactions x 3 energies x {N_REPLAY} events = {n_events}; "
         f"worst |F_E(E_out) - r1| = {worst['F_E']:.1e}, |E_out - replay| / E_in = "
         f"{worst['E']:.1e}, |F_mu(mu) - xi| = {worst['F_mu']:.1e} (bounds 1e-9, 1e-12, 1e-9); "
         f"CM->lab vs replay {worst['lab']:.1e} (bound 1e-12); multiplicity exact; "
         f"max draws {max_draws}")
    assert worst["F_E"] <= 1e-9 and worst["F_mu"] <= 1e-9
    assert worst["E"] <= 1e-12 and worst["lab"] <= 1e-12


# ---- statistical (f): reference built from the PDFs p only ----------------
def p_only_cdf(tab, X):
    """CDF of a tabulated PDF at X (array), from p alone: exact integral of
    the piecewise-constant or piecewise-linear p, normalised by its own
    integral. The stored c is never read."""
    x, p, _, interp = tab
    X = np.atleast_1d(np.asarray(X, dtype=np.float64))
    dx = np.diff(x)
    if interp == 1:
        full = p[:-1] * dx
    else:
        full = 0.5 * (p[:-1] + p[1:]) * dx
    cum = np.concatenate([[0.0], np.cumsum(full)])
    out = np.zeros(X.size)
    for j, v in enumerate(X):
        if v <= x[0]:
            continue
        if v >= x[-1]:
            out[j] = cum[-1]
            continue
        k = int(np.searchsorted(x, v, side="right")) - 1
        d = v - x[k]
        if interp == 1:
            out[j] = cum[k] + p[k] * d
        else:
            m = (p[k + 1] - p[k]) / (x[k + 1] - x[k])
            out[j] = cum[k] + p[k] * d + 0.5 * m * d * d
    return out / cum[-1]


class Pieces:
    """Outgoing-energy pieces of a law at one incident energy, from p only:
    each piece is part of one table interval, with its weight (mixture
    weight / the table's own integral), the linear map to the final
    (scaled) energy, and the angle table it is sampled with."""

    def __init__(self):
        self.cols = {k: [] for k in ("t0", "t1", "xk", "pk", "mk", "w", "x0", "E1", "g", "angle")}

    def add(self, **kw):
        for k, v in kw.items():
            self.cols[k].append(v)

    def finish(self):
        self.a = {k: np.asarray(v, dtype=np.float64) for k, v in self.cols.items()}
        return self

    def _mass(self, lo, hi):
        a = self.a
        lo = np.clip(lo, a["t0"], a["t1"])
        hi = np.clip(hi, a["t0"], a["t1"])
        return a["pk"] * (hi - lo) + 0.5 * a["mk"] * ((hi - a["xk"]) ** 2 - (lo - a["xk"]) ** 2)

    def _t(self, B):
        a = self.a
        return a["x0"] + (B - a["E1"]) / a["g"]

    def mass_below(self, B):
        """Weighted mass of every piece below final energy B."""
        return self.a["w"] * self._mass(self.a["t0"], self._t(B))

    def e_range(self):
        a = self.a
        f0 = a["E1"] + (a["t0"] - a["x0"]) * a["g"]
        f1 = a["E1"] + (a["t1"] - a["x0"]) * a["g"]
        return float(f0.min()), float(f1.max())


def add_table_pieces(pc, tab, weight, E_1, E_K, angle_ids, correlated):
    """Pieces of one outgoing-energy table. For a correlated lin-lin table,
    interval k < n-2 is split at its own (p-only) mass midpoint: the lower
    half takes angle table k and the upper half k+1. The last interval
    (n-2) takes table n-1, because OpenMC's search ends there with
    c_k1 == c_k (a 2-point table keeps table 0, c_k1 = +inf). Histogram
    E_out always takes table k."""
    x, p, _, interp = tab
    n = x.size
    full = (p[:-1] * np.diff(x)) if interp == 1 else 0.5 * (p[:-1] + p[1:]) * np.diff(x)
    w = weight / full.sum()
    g = (E_K - E_1) / (x[-1] - x[0])
    for k in range(n - 1):
        if x[k + 1] == x[k]:
            continue
        mk = 0.0 if interp == 1 else (p[k + 1] - p[k]) / (x[k + 1] - x[k])
        base = dict(xk=x[k], pk=p[k], mk=mk, w=w, x0=x[0], E1=E_1, g=g)
        if not correlated:
            pc.add(t0=x[k], t1=x[k + 1], angle=-1, **base)
        elif interp == 1:
            pc.add(t0=x[k], t1=x[k + 1], angle=angle_ids[k], **base)
        elif n == 2:
            pc.add(t0=x[k], t1=x[k + 1], angle=angle_ids[0], **base)
        elif k == n - 2:
            pc.add(t0=x[k], t1=x[k + 1], angle=angle_ids[n - 1], **base)
        else:
            M = full[k]
            disc = p[k] * p[k] + mk * M
            delta = M / (p[k] + math.sqrt(disc)) if p[k] + math.sqrt(max(disc, 0.0)) > 0 else 0.0
            split = x[k] + delta
            pc.add(t0=x[k], t1=split, angle=angle_ids[k], **base)
            pc.add(t0=split, t1=x[k + 1], angle=angle_ids[k + 1], **base)


def law_reference(raw, E):
    """-> (pieces, angle tables, mixture of angle tables for law 4 or None).
    Mixtures: applicability (first law with c <= cumulative sum, the last
    law takes the rest), the incident tables i / i+1 with weights 1-r / r,
    and the scaled interpolation, all as in the samplers."""
    pc = Pieces()
    angle_tabs = []
    law4_angle = None
    n_d = len(raw["dists"])
    weights = []
    cum = 0.0
    for j in range(n_d):
        if n_d == 1:
            weights.append(1.0)
            break
        if j < n_d - 1:
            a = py_tab1d(*raw["app"][j], E)
            weights.append(max(0.0, min(cum + a, 1.0) - cum))
            cum = min(cum + a, 1.0)
        else:
            weights.append(1.0 - cum)
    for law, wj in zip(raw["dists"], weights):
        if wj == 0.0:
            continue
        tables = law["tables"]
        if law["kind"] == "continuous":
            i, r = py_cont_index(law["energy"], E)
            ae, atabs = law["angle"]
            ia, ra = py_energy_index(ae, E)
            law4_angle = [(1.0 - ra, atabs[ia])] + ([(ra, atabs[ia + 1])] if ra > 0.0 else [])
        else:
            i, r = py_energy_index(law["energy"], E)
        E_1, E_K = scale_params(tables, i, r)
        for l, wl in ((i, 1.0 - r), (i + 1, r)):
            if wl == 0.0:
                continue
            ids = None
            if law["kind"] == "correlated":
                ids = list(range(len(angle_tabs), len(angle_tabs) + len(law["mu"][l])))
                angle_tabs.extend(law["mu"][l])
            add_table_pieces(pc, tables[l], wj * wl, E_1, E_K, ids, law["kind"] == "correlated")
    return pc.finish(), angle_tabs, law4_angle


def bisect_edges(cdf, lo, hi, k):
    edges = [lo]
    for j in range(1, k):
        a, b = lo, hi
        for _ in range(100):
            mid = 0.5 * (a + b)
            if cdf(mid) < j / k:
                a = mid
            else:
                b = mid
        edges.append(0.5 * (a + b))
    edges.append(hi)
    return np.array(edges)


def joint_probabilities(raw, E, k_e=8, k_mu=5):
    """-> (E edges, mu edges, P[k_e, k_mu]) from p only."""
    pc, angle_tabs, law4_angle = law_reference(raw, E)
    lo, hi = pc.e_range()
    e_edges = bisect_edges(lambda B: float(pc.mass_below(B).sum()), lo, hi, k_e)
    mass_e = np.array([pc.mass_below(b) for b in e_edges])        # (k_e+1, n_pieces)
    piece_mass_bins = np.diff(mass_e, axis=0)                       # (k_e, n_pieces)
    if law4_angle is not None:
        mu_cdf = lambda X: sum(w * p_only_cdf(t, X)[0] for w, t in law4_angle)  # noqa: E731
        mu_edges = bisect_edges(mu_cdf, -1.0, 1.0, k_mu)
        p_mu = np.diff([mu_cdf(v) for v in mu_edges])
        P = piece_mass_bins.sum(axis=1)[:, None] * p_mu[None, :]
        return e_edges, mu_edges, P
    ang = pc.a["angle"].astype(int)
    W = np.zeros(len(angle_tabs))
    np.add.at(W, ang, piece_mass_bins.sum(axis=0))
    used = np.flatnonzero(W > 0.0)
    mu_cdf = lambda X: float(sum(W[a] * p_only_cdf(angle_tabs[a], X)[0] for a in used))  # noqa: E731
    mu_edges = bisect_edges(mu_cdf, -1.0, 1.0, k_mu)
    P_mu = np.zeros((len(angle_tabs), k_mu))
    for a in used:
        P_mu[a] = np.diff(p_only_cdf(angle_tabs[a], mu_edges))
    # einsum, not @: numpy's Accelerate BLAS raises spurious floating-point
    # warnings in matmul on this platform
    P = np.einsum("ep,pm->em", piece_mass_bins, P_mu[ang])
    assert np.isfinite(P).all()
    return e_edges, mu_edges, P


def merge_sparse_cells(counts, expected, minimum=5.0):
    """Within each row, merge adjacent cells left to right until each group
    expects >= minimum; a short last group joins the previous one. ->
    (observed, expected) of the merged cells, flattened."""
    obs, exp = [], []
    for crow, erow in zip(counts, expected):
        groups, cur = [], []
        for j in range(erow.size):
            cur.append(j)
            if erow[cur].sum() >= minimum:
                groups.append(cur)
                cur = []
        if cur:
            if groups:
                groups[-1] = groups[-1] + cur
            else:
                groups.append(cur)
        for g in groups:
            obs.append(crow[g].sum())
            exp.append(erow[g].sum())
    return np.array(obs), np.array(exp)


F_STAT_CASES = [
    # (nuclide, MT, description)
    ("Fe56", 91, "law 61, CM, lin-lin"),
    ("W184", 16, "law 61, CM, lin-lin"),
    ("Be9", 16, "law 61, lab, histogram E_out"),
    ("F19", 16, "law 61, lab, two laws + applicability"),
    ("Li7", 16, "law 4 + tabular angle, lab"),
]


@pytest.mark.parametrize("where", ["on-grid", "midpoint"])
@pytest.mark.parametrize("nuclide,mt,desc", F_STAT_CASES)
def test_f_law_joint_distribution(report, note, nuclide, mt, desc, where):
    """Joint (E_out, mu) of one law in its own frame, at the tabulated
    incident energy nearest 14.1 MeV or at the midpoint of the pair
    bracketing it: chi-square on 8 x 5 cells (equiprobable E bins x
    equiprobable mu bins of the marginals) against probabilities built from
    p alone (see law_reference). Validity rule, the same for every case:
    within an E row, adjacent mu cells are merged until each merged cell
    expects >= 5 events (only Be-9's forward-peaked high-energy corner
    needs it). The stored-CDF construction differs from the p-only one by
    at most the inventory bounds (1.1e-7 for E_out, 9.2e-7 for mu).

    mu in the law's frame is not clamped (OpenMC clamps after the CM -> lab
    transform). Where a stored mu CDF exceeds the integral of its PDF, the
    inverse can step past +-1 by about (CDF mismatch) / p; the test bounds
    the overshoot by 1e-6 and reports it."""
    from test_kinematics import chi2_check
    lib, phys = _packed_real([nuclide])
    raw = raw_product(lib, nuclide, mt)
    _, E_on, E_mid = law_energies(raw)
    E = E_on if where == "on-grid" else E_mid
    c = phys.labels.index((nuclide, mt))
    prod = int(phys.ch_int[c, C.CH_PROD])
    awr = float(phys.nuc_awr[0])
    e_edges, mu_edges, P = joint_probabilities(raw, E)
    assert abs(P.sum() - 1.0) < 1e-12, P.sum()
    expected = N_STAT * P
    e_s, mu_s, md = _sample_product(phys.ip, phys.fp, prod, E, awr, float(phys.ch_q[c]),
                                    np.uint64(SEED_F), N_STAT)
    span = e_edges[-1] - e_edges[0]
    assert e_s.min() >= e_edges[0] - 1e-12 * span and e_s.max() <= e_edges[-1] + 1e-12 * span
    overshoot = max(float(np.abs(mu_s).max()) - 1.0, 0.0)
    note(f"(f) {nuclide} MT {mt} {where}: max |mu| - 1 in the law frame = {overshoot:.1e} "
         "(bound 1e-6)")
    assert overshoot <= 1e-6
    counts = np.histogram2d(np.clip(e_s, e_edges[0], e_edges[-1]), np.clip(mu_s, -1.0, 1.0),
                            bins=[e_edges, mu_edges])[0]
    obs, exp = merge_sparse_cells(counts, expected)
    assert exp.min() >= 5.0
    chi2_check(report, f"{nuclide} MT {mt} (E,mu) {where} {E:.6g} eV", obs, exp, md)


# ---------------------------------------------------------------------------
# packing: every reaction, as inventoried; unimplemented laws refused
# ---------------------------------------------------------------------------
def test_every_reaction_packs_as_inventoried(note):
    """Exact: every non-redundant neutron reaction of the 14 nuclides is a
    scatter channel with a packed law of the inventoried kind, and nothing
    is unsupported."""
    lib = _library()
    rows = law_inventory.scan(str(lib.root))
    names = list(dict.fromkeys(r["nuclide"] for r in rows))
    nucs = {n: lib.load(n, distributions=True) for n in names}
    phys = C.pack_physics(names, nucs)
    assert phys.unsupported == ()
    assert len(phys.labels) == len(rows)
    for row in rows:
        c = phys.labels.index((row["nuclide"], row["mt"]))
        prod = int(phys.ch_int[c, C.CH_PROD])
        assert prod >= 0 and int(phys.ch_int[c, C.CH_CM]) == int(row["cm"])
        ip = phys.ip
        assert ip[prod + 2] == len(row["distributions"])
        assert ip[prod] == (D.Y_CONST if row["yield"]["type"] == "Polynomial" else D.Y_TAB1)
        for j, d in enumerate(row["distributions"]):
            law = ip[prod + 4 + 2 * j]
            if d["kind"] == "correlated":
                assert ip[law] == D.LAW_CORRELATED
            else:
                expect = {"angle-only": D.E_NONE, "level": D.E_LEVEL,
                          "continuous": D.E_CONTINUOUS}[d["kind"]]
                assert ip[law] == D.LAW_UNCORRELATED and ip[law + 2] == expect
    note(f"(packing) {len(rows)} channels of {len(names)} nuclides packed as inventoried; "
         "0 unsupported")


def test_unimplemented_law_refused_even_if_unreachable(tmp_path):
    """A problem whose nuclide has an unimplemented law is refused, even when
    the source is far below that reaction's threshold (with target motion,
    reachability cannot be bounded by the source energy)."""
    from mcslab.ce_materials import CEMaterial
    from mcslab.config_ce import MonoEnergetic
    from mcslab.config_kin import KinRunConfig, run_kin
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource
    lib = _library()
    (tmp_path / "neutron").mkdir()
    path = tmp_path / "neutron" / "Be9.h5"
    shutil.copyfile(lib.nuclide_path("Be9"), path)
    _edit_attr(path, BE16 + "/energy_out", "n_discrete_lines",
               lambda a: np.where(np.arange(a.size) == 0, 1, a))
    edited = N.Library.open(tmp_path, label="edited")
    be = CEMaterial("Be", 1.85, (("Be9", 1.0),))
    cfg = KinRunConfig(SlabGeometry([0.0, 1.0], [be]), BeamSource(), MonoEnergetic(1.0e5),
                       edited, n_batches=2, histories_per_batch=10)
    with pytest.raises(NotImplementedError, match="Be9 MT 16.*discrete lines"):
        run_kin(cfg)


# ---------------------------------------------------------------------------
# (g) multiplicities
# ---------------------------------------------------------------------------
# Seed fixed in docs/phase2b_plan.md before this test existed.
SEED_G = 20261021
N_YIELD = 100_000
G_MT5 = [
    # (nuclide, E (eV), where y falls), y read from the raw HDF5 table
    ("Fe56", 14.1e6, "(0,1), between table points"),
    ("Fe56", 30.0e6, "(1,2), on a table point"),
    ("Fe56", 70.0e6, "(2,3), on a table point"),
    ("W184", 14.0e6, "(0,1), on a table point"),
    ("W184", 30.0e6, "(1,2), between table points"),
    ("W184", 32.0e6, "(2,3), on a table point"),
]


def raw_yield(lib, nuclide, mt, E):
    with h5py.File(lib.nuclide_path(nuclide), "r") as f:
        y = f[f"{nuclide}/reactions/reaction_{mt:03d}/product_0/yield"]
        assert int(np.ravel(y.attrs["interpolation"])[0]) == 2
        x, v = y[()]
    return float(np.interp(E, x, v))


@pytest.mark.parametrize("nuclide,E,where", G_MT5)
def test_g_mt5_multiplicity(report, note, nuclide, E, where):
    """MT 5 (energy-dependent yield, approved deviation D2). Per event the
    number of neutrons is floor(y) or floor(y) + 1 (exact), and its mean is
    y(E) at 3 SE, with y interpolated lin-lin in the test from the raw HDF5
    table (np.interp)."""
    lib, phys = _packed_real([nuclide])
    c = phys.labels.index((nuclide, 5))
    prod = int(phys.ch_int[c, C.CH_PROD])
    y = raw_yield(lib, nuclide, 5, E)
    lo, hi = (int(where[1]), int(where[3]))
    assert lo < y < hi, (y, where)
    _, _, n_out, md = _inelastic(phys.ip, phys.fp, prod, E, float(phys.nuc_awr[0]),
                                 float(phys.ch_q[c]), int(phys.ch_int[c, C.CH_CM]),
                                 np.uint64(SEED_G), N_YIELD)
    fl = math.floor(y)
    assert set(np.unique(n_out)) <= {fl, fl + 1}
    mean = float(n_out.mean())
    se = float(n_out.std(ddof=1) / math.sqrt(N_YIELD))
    report(f"{nuclide} MT 5 mean n at {E / 1e6:g} MeV", mean, y, se, md)
    assert abs(mean - y) <= 3.0 * se, (mean, y, se)


def test_g_zero_and_integer_yields_draw_nothing():
    """Exact: where y is an integer (MT 5 below 6 MeV: y = 0; MT 16: y = 2)
    no multiplicity draw is made: 3 law draws + 1 for phi."""
    lib, phys = _packed_real(["Fe56", "W184"])
    for name, mt, E, y in (("Fe56", 5, 5.0e6, 0), ("W184", 5, 5.0e6, 0),
                           ("Fe56", 16, 14.1e6, 2), ("W184", 16, 14.1e6, 2)):
        c = phys.labels.index((name, mt))
        k = int(phys.ch_int[c, C.CH_NUC])
        _, _, n_out, md = _inelastic(phys.ip, phys.fp, int(phys.ch_int[c, C.CH_PROD]), E,
                                     float(phys.nuc_awr[k]), float(phys.ch_q[c]),
                                     int(phys.ch_int[c, C.CH_CM]), np.uint64(SEED_G), 2000)
        assert np.all(n_out == y) and md == 4, (name, mt, md)


def _kernel_problem(kind):
    from mcslab import ce_materials as cm
    from mcslab.config_ce import MonoEnergetic
    from mcslab.config_kin import KinRunConfig
    from mcslab.geometry import SlabGeometry
    from mcslab.sources import BeamSource
    lib = _library()
    if kind == "W|Fe 30 MeV":
        geom = SlabGeometry([0.0, 2.0, 5.0], [cm.tungsten(), cm.iron()])
        E0 = 30.0e6
    else:
        geom = SlabGeometry([0.0, 10.0], [cm.flibe()])
        E0 = 14.1e6
    return KinRunConfig(geom, BeamSource(), MonoEnergetic(E0), lib, n_batches=20,
                        histories_per_batch=500, seed=SEED_G, energy_cutoff=1.0e6)


G_KERNEL = {
    "W|Fe 30 MeV": [("W184", 16), ("W184", 17), ("W184", 37), ("W184", 41), ("Fe56", 16)],
    "FLiBe 14.1 MeV": [("Be9", 16), ("Li6", 24), ("Li7", 16), ("Li7", 24), ("F19", 16)],
}


@pytest.mark.parametrize("kind", sorted(G_KERNEL))
def test_g_kernel_multiplicity_and_balance(report, kind):
    """Exact, through the kernel on real data (energy cutoff 1 MeV to keep
    the runs short): every channel with a constant integer yield y creates
    exactly (y - 1) secondaries per event and never a zero-yield event;
    zero-yield events come only from MT 5; per batch,
    sources + created == absorbed + leaks + cutoff + zero_yield, each term
    from a tally and equal to its int64 counter."""
    import warnings
    from mcslab import tallies as T
    from mcslab.config_kin import EnergyCutoffWarning, run_kin
    cfg = _kernel_problem(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EnergyCutoffWarning)
        res = run_kin(cfg)
    from mcslab.rng import STRIDE
    assert res.lost == 0 and res.max_draws < STRIDE
    nucs = {}
    lib = _library()
    for c, (name, mt) in enumerate(res.channel_labels):
        if name not in nucs:
            nucs[name] = lib.load(name, distributions=True)
        prod = nucs[name].reactions[mt].products[0]
        ev, made = res.channel_counts(name, mt)
        zero = res.channel_zero(name, mt)
        if prod.yield_table is None:
            y = int(prod.yield_coefficients[0])
            assert made == (y - 1) * ev and zero == 0, (name, mt, ev, made, zero)
        else:
            assert mt == 5
    for name, mt in G_KERNEL[kind]:
        assert res.channel_counts(name, mt)[0] > 0, (name, mt)
    counts = res.diagnostics
    src = counts[:, T.K_SOURCE].astype(np.float64)
    created = counts[:, T.K_CREATED].astype(np.float64)
    absorbed = res.region_sums[:, T.ABSORPTION, :].sum(axis=1)
    left, right = res.surface_sums[:, T.NEG, 0], res.surface_sums[:, T.POS, -1]
    cut = res.cutoff_weight.sum(axis=1)
    zero = res.zero_yield_weight.sum(axis=1)
    assert np.array_equal(zero, counts[:, T.K_ZERO_YIELD].astype(np.float64))
    assert np.array_equal(cut, counts[:, T.K_CUTOFF].astype(np.float64))
    assert np.array_equal(absorbed, counts[:, T.K_ABSORBED].astype(np.float64))
    assert int(res.chan_zero.sum()) == res.count(T.K_ZERO_YIELD)
    assert np.array_equal(src + created, absorbed + left + right + cut + zero)
    if kind.startswith("W"):
        assert res.count(T.K_ZERO_YIELD) > 0
    b = res.balance()
    assert b["residual"] == 0.0
    n = cfg.n_histories
    report("sources + created", b["source"] + b["created"],
           b["absorbed"] + b["leak_left"] + b["leak_right"] + b["cutoff"] + b["zero_yield"],
           0.0, res.max_draws, b["cutoff"] / n)
    report("  created per source", b["created"] / n, b["created"] / n, 0.0)
    report("  zero-yield per source", b["zero_yield"] / n, res.count(T.K_ZERO_YIELD) / n, 0.0)
