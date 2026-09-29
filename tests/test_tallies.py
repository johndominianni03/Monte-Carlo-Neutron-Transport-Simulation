"""Unit tests for batch statistics."""
import numpy as np
import pytest

from mcslab.tallies import allocate, batch_stats


def test_batch_stats_matches_textbook_formula():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    mean, se = batch_stats(x)
    assert mean == 3.5
    assert np.isclose(se, np.std(x, ddof=1) / np.sqrt(x.size), rtol=1e-14)


def test_batch_stats_over_arrays_and_constant_input():
    x = np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (10, 1, 1))
    mean, se = batch_stats(x)
    assert np.array_equal(mean, [[1.0, 2.0], [3.0, 4.0]])
    assert np.all(se == 0.0)


def test_batch_stats_needs_two_batches():
    with pytest.raises(ValueError):
        batch_stats(np.array([1.0]))


def test_allocate_shapes():
    r, s, d = allocate(5, 3)
    assert r.shape == (5, 4, 3) and s.shape == (5, 2, 4) and d.shape == (5, 2)
