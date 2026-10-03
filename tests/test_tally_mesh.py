"""Deterministic checks of the Phase 3 depth mesh and response packing
(plan checks 1 and 2). No transport, no statistics.

Check 1, splitting (mcslab/depth_mesh.py):
  - hand-built segments with known pieces;
  - 1e5 random segments in the D7 mesh (W 10 x 0.05 cm, FLiBe 20 x 1 cm,
    Fe 20 x 0.5 cm) against exact rational arithmetic on the same float
    inputs (fractions.Fraction): every bin's length within 1e-12 d of the
    exact value, the pieces summing to d within 1e-12 d, no negative
    piece, every piece inside its layer. About 30% of the segments end on
    a layer boundary, with d computed as geometry.distance_to_boundary
    does; their last piece must lie in the layer's last bin in the
    direction of flight. Inputs come from a NumPy generator, seed 20261031.
Check 2, response packing (mcslab/responses.py), on the 13 D7 nuclides at
the D7 data temperatures:
  - the packed lookup (grid_locate + interp_at on the packed arrays) equals
    xs.lookup bit for bit for every response MT present, and is exactly 0
    for an absent one (MT 205 of the W isotopes);
  - against the raw HDF5 values (h5py + np.interp, independent of
    mcslab.nucdata / xs) to 1e-12 x the reaction's maximum;
  - absorption is the kernel's absn table, bit for bit;
  - refresh_responses gives N_j sigma for each material slot and their sum
    in the material's order.
  Energies: every grid point plus 2000 log-uniform energies per nuclide
  (NumPy generator, seed 20261036).

The bounds are fixed in the plan. If one is exceeded it is reported, not
loosened; the observed worst values are printed in the summary notes.
"""
import math
import os
from fractions import Fraction

import h5py
import numpy as np
import pytest

from mcslab import ce_materials as cm
from mcslab import nucdata as N
from mcslab import xs
from mcslab.depth_mesh import bin_of, make_mesh, segment_pieces, start_bin
from mcslab.geometry import distance_to_boundary
from mcslab.responses import RESPONSE_MTS, pack_responses, refresh_responses
from mcslab.tallies import N_RESP, R_ABSORPTION

D7_BOUNDS = (0.0, 0.5, 20.5, 30.5)
D7_BINS = (10, 20, 20)
SPLIT_SEED = 20261031
LOOKUP_SEED = 20261036
N_SEGMENTS = 100_000
SPLIT_BOUND = 1e-12          # relative to d (plan check 1)
LOOKUP_BOUND = 1e-12         # relative to the reaction's maximum (plan check 2)
D7_TEMPS = {"W180": "294K", "W182": "294K", "W183": "294K", "W184": "294K",
            "W186": "294K", "Li6": "900K", "Li7": "900K", "Be9": "900K", "F19": "900K",
            "Fe54": "294K", "Fe56": "294K", "Fe57": "294K", "Fe58": "294K"}


def _pieces(mesh, r, x0, u, d):
    pb = np.zeros(mesh.max_bins + 1, dtype=np.int64)
    pl = np.zeros(mesh.max_bins + 1, dtype=np.float64)
    e0 = mesh.eoff[r]
    n = mesh.counts[r]
    m = segment_pieces(mesh.edges, e0, n, x0, u, d, pb, pl)
    return pb[:m].copy(), pl[:m].copy()


# ---------------------------------------------------------------------------
# check 1: splitting
# ---------------------------------------------------------------------------
def test_mesh_edges_are_exact_and_uniform():
    mesh = make_mesh(D7_BOUNDS, D7_BINS)
    assert mesh.n_bins == 50
    be = mesh.bin_edges
    assert be.size == 51 and be[0] == 0.0 and be[-1] == 30.5
    for r in range(3):
        e = mesh.edges[mesh.eoff[r]:mesh.eoff[r + 1]]
        assert e[0] == D7_BOUNDS[r] and e[-1] == D7_BOUNDS[r + 1]
        assert np.allclose(np.diff(e), (D7_BOUNDS[r + 1] - D7_BOUNDS[r]) / D7_BINS[r],
                           rtol=1e-13, atol=0.0)
    assert list(mesh.bin_region) == [0] * 10 + [1] * 20 + [2] * 20
    with pytest.raises(ValueError):
        make_mesh(D7_BOUNDS, (10, 0, 20))
    with pytest.raises(ValueError):
        make_mesh(D7_BOUNDS, (10, 20))


def test_hand_built_segments():
    mesh = make_mesh(D7_BOUNDS, D7_BINS)
    w = mesh.edges[mesh.eoff[0]:mesh.eoff[1]]
    # beam across the whole W layer: s_k = e_k exactly, so pieces = diff(edges)
    d = distance_to_boundary(0.0, 1.0, 0.0, 0.5)
    pb, pl = _pieces(mesh, 0, 0.0, 1.0, d)
    assert list(pb) == list(range(10))
    assert np.array_equal(pl, np.diff(w))
    # collision inside W: 0.12 -> 0.27 along u = 0.5
    pb, pl = _pieces(mesh, 0, 0.12, 0.5, 0.3)
    assert list(pb) == [2, 3, 4, 5]
    assert np.allclose(pl, [0.06, 0.1, 0.1, 0.04], rtol=0.0, atol=1e-15)
    assert math.isclose(pl.sum(), 0.3, rel_tol=1e-15)
    # leftward from the right bound across the whole layer
    d = distance_to_boundary(0.5, -0.8, 0.0, 0.5)
    pb, pl = _pieces(mesh, 0, 0.5, -0.8, d)
    assert list(pb) == list(range(9, -1, -1))
    assert abs(pl.sum() - d) <= 1e-15 * d
    # start exactly on an internal edge: the bin being entered
    x = w[4]
    assert start_bin(mesh.edges, mesh.eoff[0], 10, x, 1.0) == 4
    assert start_bin(mesh.edges, mesh.eoff[0], 10, x, -1.0) == 3
    assert bin_of(mesh.edges, mesh.eoff[0], 10, x) == 4
    pb, pl = _pieces(mesh, 0, x, -1.0, 0.01)
    assert list(pb) == [3] and pl[0] == 0.01
    # u = 0 and zero-length segments: one piece, the whole length
    pb, pl = _pieces(mesh, 1, 7.3, 0.0, 2.5)
    assert list(pb) == [6] and pl[0] == 2.5
    pb, pl = _pieces(mesh, 1, 0.5, -1.0, -0.0)
    assert list(pb) == [0] and pl[0] == 0.0
    # FLiBe: start in bin 0 at its left bound, grazing direction stays in a bin
    pb, pl = _pieces(mesh, 1, 0.5, 1e-12, 3.0)
    assert list(pb) == [0] and pl[0] == 3.0


def _exact_lengths(edges, n, x0, u, d):
    """Per-bin lengths by exact rational arithmetic on the float inputs,
    with the same start-bin rule. Independent of depth_mesh."""
    E = [Fraction(float(e)) for e in edges]
    X, U, D = Fraction(x0), Fraction(u), Fraction(d)
    j = 0
    for k in range(n):
        if E[k] <= X:
            j = k
    if U < 0 and j > 0 and X == E[j]:
        j -= 1
    out = {}
    if U == 0 or D <= 0:
        out[j] = D
        return out
    done = Fraction(0)
    while True:
        last = (j == n - 1) if U > 0 else (j == 0)
        if not last:
            s = ((E[j + 1] if U > 0 else E[j]) - X) / U
            if s < D:
                out[j] = out.get(j, 0) + (s - done)
                done = s
                j += 1 if U > 0 else -1
                continue
        out[j] = out.get(j, 0) + (D - done)
        return out


def test_random_segments_against_exact_arithmetic(note):
    mesh = make_mesh(D7_BOUNDS, D7_BINS)
    rng = np.random.default_rng(SPLIT_SEED)
    worst_sum = worst_bin = 0.0
    n_boundary = 0
    for t in range(N_SEGMENTS):
        r = int(rng.integers(3))
        lo, hi = D7_BOUNDS[r], D7_BOUNDS[r + 1]
        edges = mesh.edges[mesh.eoff[r]:mesh.eoff[r + 1]]
        n = mesh.counts[r]
        x0 = float(edges[rng.integers(n + 1)]) if rng.random() < 0.1 \
            else float(lo + (hi - lo) * rng.random())
        kind = rng.random()
        if kind < 0.05:
            u = float(rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-12, -3))
        else:
            u = float(rng.uniform(-1.0, 1.0))
            if u == 0.0:
                u = 0.5
        if (u > 0 and x0 >= hi) or (u < 0 and x0 <= lo):
            u = -u
        d_bdy = distance_to_boundary(x0, u, lo, hi)
        if rng.random() < 0.3 and np.isfinite(d_bdy):
            d = d_bdy
            n_boundary += 1
            boundary = True
        else:
            d = float(min(d_bdy, 60.0) * rng.random())
            boundary = False
        pb, pl = _pieces(mesh, r, x0, u, d)
        assert (pl >= 0.0).all(), (r, x0, u, d, pl)
        assert ((pb >= 0) & (pb < n)).all()
        if boundary and d > 0.0:
            assert pb[-1] == (n - 1 if u > 0 else 0), (r, x0, u, d, pb)
        worst_sum = max(worst_sum, abs(pl.sum() - d) / d if d > 0 else 0.0)
        exact = _exact_lengths(edges, n, x0, u, d)
        got = {}
        for b, l in zip(pb, pl):
            got[int(b)] = got.get(int(b), 0.0) + float(l)
        if d > 0:
            for b in set(got) | set(exact):
                err = abs(Fraction(got.get(b, 0.0)) - exact.get(b, Fraction(0)))
                worst_bin = max(worst_bin, float(err / Fraction(d)))
    note(f"(check 1) {N_SEGMENTS} random segments ({n_boundary} ending on a layer "
         f"boundary): worst |sum(pieces) - d| / d = {worst_sum:.2e}, worst per-bin "
         f"|length - exact| / d = {worst_bin:.2e} (bound {SPLIT_BOUND:g})")
    assert worst_sum <= SPLIT_BOUND
    assert worst_bin <= SPLIT_BOUND


# ---------------------------------------------------------------------------
# check 2: response packing
# ---------------------------------------------------------------------------
def _library():
    try:
        return N.Library.open()
    except FileNotFoundError as exc:
        pytest.skip(f"nuclear data not available ({exc})")


def _raw(lib, name, T, mt):
    """(grid, threshold index, values) of one reaction straight from the file."""
    with h5py.File(os.path.join(str(lib.root), "neutron", name + ".h5"), "r") as f:
        g = f[name]
        grid = g["energy"][T][()]
        key = f"reaction_{mt:03d}"
        if key not in g["reactions"]:
            return grid, None, None
        d = g["reactions"][key][T]["xs"]
        return grid, int(d.attrs["threshold_idx"]), d[()]


@pytest.fixture(scope="module")
def d7_nuclides():
    lib = _library()
    return lib, {n: lib.load(n, T) for n, T in D7_TEMPS.items()}


def test_response_lookup_matches_reader_and_raw_files(d7_nuclides, note):
    lib, nucs = d7_nuclides
    names = tuple(D7_TEMPS)
    p = pack_responses(names, nucs)
    rng = np.random.default_rng(LOOKUP_SEED)
    worst = 0.0
    absent = []
    for k, name in enumerate(names):
        nuc = nucs[name]
        grid = nuc.energy
        n = grid.size
        energies = np.concatenate([grid, np.exp(rng.uniform(np.log(grid[0]),
                                                            np.log(grid[-1]), 2000))])
        absn = nuc.absorption_on_grid()
        for s, mt in enumerate(RESPONSE_MTS):
            raw_grid, raw_thr, raw_xs = _raw(lib, name, D7_TEMPS[name], mt)
            assert np.array_equal(raw_grid, grid)
            if raw_xs is None:
                assert not p.present[k, s] and mt not in nuc.reactions
                absent.append(f"{name} MT {mt}")
            else:
                assert p.present[k, s]
            scale = float(np.abs(raw_xs).max()) if raw_xs is not None else 0.0
            for E in energies:
                i, f = xs.grid_locate(grid, 0, n, E)
                v = xs.interp_at(i, f, n, p.rxs, p.roff[k, s], p.rthr[k, s])
                if raw_xs is None:
                    assert v == 0.0
                    continue
                assert v == xs.lookup(nuc, mt, E), (name, mt, E)
                ref = 0.0 if E < raw_grid[raw_thr] else float(
                    np.interp(E, raw_grid[raw_thr:], raw_xs))
                err = abs(v - ref) / scale
                worst = max(worst, err)
                assert err <= LOOKUP_BOUND, (name, mt, E, v, ref)
            # absorption from the kernel's absn table
        for E in energies[::50]:
            i, f = xs.grid_locate(grid, 0, n, E)
            assert xs.interp_at(i, f, n, absn, 0, 0) == xs.interp_xs(grid, 0, n, absn, 0,
                                                                     0, E)
    assert absent == [f"W{a} MT 205" for a in (180, 182, 183, 184, 186)]
    assert p.present[:, R_ABSORPTION].all()
    note(f"(check 2) response lookups: bit-identical to xs.lookup; worst |packed - raw "
         f"h5py| / max = {worst:.2e} (bound {LOOKUP_BOUND:g}); absent: {', '.join(absent)}")


def test_refresh_responses_per_material(d7_nuclides):
    lib, nucs = d7_nuclides
    flibe = cm.flibe(temperature_K=900.0)
    mats = [cm.tungsten(), flibe, cm.iron()]
    names = tuple(D7_TEMPS)
    comps = []
    for m in mats:
        awr = {n: nucs[n].awr for n in m.nuclide_names}
        comps.append(m.number_densities(awr))
    packed = xs.pack(comps, nucs)
    assert packed.nuclide_names == names
    p = pack_responses(names, nucs)
    ns = np.zeros((5, N_RESP))
    ntot = np.zeros(N_RESP)
    for m in range(3):
        j0, j1 = packed.m_off[m], packed.m_off[m + 1]
        for E in (1.0e-5, 0.0253, 2.0e3, 1.0e6, 14.1e6):
            ci = np.zeros(5, dtype=np.int64)
            cf = np.zeros(5)
            for j in range(j0, j1):
                k = packed.mat_nuc[j]
                g0, g1 = packed.e_off[k], packed.e_off[k + 1]
                ci[j - j0], cf[j - j0] = xs.grid_locate(packed.egrid, g0, g1 - g0, E)
            refresh_responses(m, packed.m_off, packed.mat_nuc, packed.mat_dens,
                              packed.e_off, packed.absn, p.rxs, p.roff, p.rthr, ci, cf,
                              ns, ntot)
            total = np.zeros(N_RESP)
            for j in range(j0, j1):
                k = packed.mat_nuc[j]
                nuc = nucs[names[k]]
                dens = packed.mat_dens[j]
                for s, mt in enumerate(RESPONSE_MTS):
                    want = dens * xs.lookup(nuc, mt, E) if mt in nuc.reactions else 0.0
                    assert ns[j - j0, s] == want, (names[k], mt, E)
                want_a = dens * xs.interp_xs(nuc.energy, 0, nuc.energy.size,
                                             nuc.absorption_on_grid(), 0, 0, E)
                assert ns[j - j0, R_ABSORPTION] == want_a
                total += ns[j - j0]
            assert np.array_equal(ntot, total)
