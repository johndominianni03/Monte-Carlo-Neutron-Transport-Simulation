"""Unit tests for the per-history LCG and its skip-ahead."""
import numpy as np
import pytest

from mcslab import rng as R

G = 2806196910506780709
C = 1
M = 2 ** 63


def py_step(s, n=1):
    """Exact big-integer reference."""
    for _ in range(n):
        s = (G * s + C) % M
    return s


def py_skip(s, n):
    """Closed form: s_n = G^n s + C (G^n - 1)/(G - 1)  (mod 2^63)."""
    gn = pow(G, n, M * (G - 1))
    return (gn * s + C * ((gn - 1) // (G - 1))) % M


@pytest.mark.parametrize("seed", [0, 1, 12345, 2 ** 63 - 1])
def test_step_matches_big_int(seed):
    s = np.uint64(seed)
    for k in range(1, 50):
        s = R.lcg_step(s)
        assert int(s) == py_step(seed, k)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 1000, R.STRIDE, 3 * R.STRIDE + 5])
def test_skip_ahead_matches_sequential(n):
    seed = 987654321
    assert int(R.skip_ahead(np.uint64(seed), np.uint64(n))) == py_step(seed, n)


@pytest.mark.parametrize("n", [10 ** 6, 10 ** 12 * R.STRIDE, 2 ** 62 + 17])
def test_skip_ahead_large_n_matches_closed_form(n):
    seed = 42
    assert int(R.skip_ahead(np.uint64(seed), np.uint64(n))) == py_skip(seed, n)


def test_history_seed_is_skip_of_stride():
    master = np.uint64(2024)
    for i in [0, 1, 5, 123456]:
        expected = py_skip(2024, i * R.STRIDE)
        assert int(R.history_seed(master, np.uint64(i))) == expected


def test_prn_range_and_draw_count():
    rng = np.zeros(R.RNG_SIZE, np.uint64)
    R.init_history(rng, np.uint64(7), np.uint64(3))
    xs = np.array([R.prn(rng) for _ in range(20000)])
    assert xs.min() >= 0.0 and xs.max() < 1.0
    assert int(rng[R.RNG_DRAWS]) == 20000
    # sanity: mean of U(0,1) within 4 sigma (sigma = sqrt(1/12/N))
    assert abs(xs.mean() - 0.5) < 4 * np.sqrt(1 / 12 / xs.size)


def test_prn_extremes_stay_in_unit_interval():
    """States near 2^63 must not round to 1.0; state 0 gives exactly 0.0."""
    rng = np.zeros(R.RNG_SIZE, np.uint64)
    # choose the state whose *successor* is 2^63 - 1
    target = M - 1
    ginv = pow(G, -1, M)
    rng[R.RNG_STATE] = np.uint64(((target - C) * ginv) % M)
    x = R.prn(rng)
    assert int(rng[R.RNG_STATE]) == target
    assert x < 1.0 and x == 1.0 - 2.0 ** -53


def test_validate_seed():
    assert R.validate_seed(5) == np.uint64(5)
    with pytest.raises(ValueError):
        R.validate_seed(-1)
    with pytest.raises(ValueError):
        R.validate_seed(2 ** 63)
