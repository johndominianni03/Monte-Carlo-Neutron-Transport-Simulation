"""Unit tests for materials and slab geometry input handling."""
import numpy as np
import pytest

from mcslab.geometry import SlabGeometry, distance_to_boundary
from mcslab.materials import VOID, Material


def test_material_sigma_t_defaults_to_sum_and_is_checked():
    m = Material("m", sigma_s=0.7, sigma_a=0.3)
    assert m.sigma_t == 0.7 + 0.3
    Material("ok", 0.7, 0.3, sigma_t=1.0)
    with pytest.raises(ValueError):
        Material("bad", 0.7, 0.3, sigma_t=1.1)
    with pytest.raises(ValueError):
        Material("neg", -0.1, 0.3)
    assert VOID.is_void


def test_geometry_validation():
    m = Material("m", 1.0, 0.0)
    with pytest.raises(ValueError):
        SlabGeometry([0.0, 1.0, 1.0], [m, m])        # zero-width region
    with pytest.raises(ValueError):
        SlabGeometry([0.0, 1.0], [m, m])             # count mismatch


def test_pack_deduplicates_materials():
    a, b = Material("a", 1.0, 0.5), Material("b", 0.2, 0.0)
    g = SlabGeometry([0, 1, 2, 3], [a, b, a])
    bounds, mat_of_region, sig_t, sig_s, sig_a = g.pack()
    assert list(mat_of_region) == [0, 1, 0]
    assert list(sig_t) == [1.5, 0.2] and list(sig_a) == [0.5, 0.0]
    assert np.array_equal(g.widths, [1.0, 1.0, 1.0])


def test_region_of():
    m = Material("m", 1.0, 0.0)
    g = SlabGeometry([0, 1, 2], [m, m])
    assert g.region_of(0.0) == 0
    assert g.region_of(0.5) == 0
    assert g.region_of(1.0) == 1       # interface -> region on its right
    with pytest.raises(ValueError):
        g.region_of(2.0)


def test_distance_to_boundary():
    assert distance_to_boundary(0.25, 0.5, 0.0, 1.0) == 1.5
    assert distance_to_boundary(0.25, -0.5, 0.0, 1.0) == 0.5
    assert distance_to_boundary(0.25, 0.0, 0.0, 1.0) == np.inf
