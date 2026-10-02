"""Phase 2b Part 2: secondary-neutron laws on real ENDF/B-VIII.0 data.

Seeds were fixed in docs/phase2b_plan.md before any of these tests existed
(project conventions: no seed-shopping). Acceptance: 3 SE for means; p >= 0.0027 for
chi-square tests; "exact" = integer identities or round-off with a stated
bound.

What each test covers, and what it doesn't:
  reader: every non-redundant neutron reaction of the 14 nuclides is read
       as the law that scripts/law_inventory.py (h5py only, independent of
       mcslab.nucdata) finds for it, and malformed or unimplemented laws
       are refused or flagged explicitly, never skipped. It does not test
       sampling.
"""
import os
import shutil
import sys

import h5py
import numpy as np
import pytest

from mcslab import nucdata as N

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
