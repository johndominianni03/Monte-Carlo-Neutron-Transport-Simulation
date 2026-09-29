"""Reproducible per-history random number generator.

OpenMC-style 63-bit linear congruential generator

    s_{k+1} = (G * s_k + C) mod 2^63,   G = 2806196910506780709, C = 1

with O(log n) skip-ahead (F. B. Brown, "Random Number Generation with
Arbitrary Strides", Trans. Am. Nucl. Soc. 71, 202 (1994)).

History i starts at state skip_ahead(master_seed, i * STRIDE). Its stream is
therefore a function of (master_seed, i) only -- not of which batch, call or
thread ran it -- which is what makes results independent of how work is split.

Generator state lives in a small uint64 array so kernels can advance it in
place:  rng[0] = LCG state,  rng[1] = number of draws since the history began.

Numba pitfall: `uint64 op int64` is promoted to float64. Every integer
constant below is an explicit np.uint64, and callers must pass uint64 values.
"""
import numpy as np
from numba import njit

MULT = np.uint64(2806196910506780709)
INC = np.uint64(1)
MASK = np.uint64(0x7FFFFFFFFFFFFFFF)     # 2^63 - 1
STRIDE = 152917                          # random numbers reserved per history
_STRIDE_U = np.uint64(STRIDE)
_ONE = np.uint64(1)
_ZERO = np.uint64(0)
_SHIFT = np.uint64(10)                   # keep the top 53 of 63 state bits
_NORM53 = 2.0 ** -53

RNG_STATE = 0
RNG_DRAWS = 1
RNG_SIZE = 2


@njit(cache=True)
def lcg_step(state):
    """One LCG step. Arithmetic wraps mod 2^64; masking then gives mod 2^63
    exactly, since 2^63 divides 2^64."""
    return (MULT * state + INC) & MASK


@njit(cache=True)
def skip_ahead(state, n):
    """State after n LCG steps from `state`, in O(log n).

    Composes the affine map s -> G s + C with itself by repeated squaring:
    (G, C) o (G, C) = (G^2, C (G + 1)).
    """
    g = MULT
    c = INC
    g_new = _ONE
    c_new = _ZERO
    n = n & MASK
    while n > _ZERO:
        if n & _ONE:
            g_new = g_new * g
            c_new = c_new * g + c
        c = c * (g + _ONE)
        g = g * g
        n = n >> _ONE
    return (g_new * state + c_new) & MASK


@njit(cache=True)
def history_seed(master_seed, history):
    """Starting state for history number `history` (uint64 arguments)."""
    return skip_ahead(master_seed, history * _STRIDE_U)


@njit(cache=True)
def init_history(rng, master_seed, history):
    rng[RNG_STATE] = history_seed(master_seed, history)
    rng[RNG_DRAWS] = _ZERO


@njit(cache=True)
def prn(rng):
    """Advance the stream and return a uniform variate in [0, 1).

    Uses the top 53 bits of the 63-bit state, so the result is an exact
    multiple of 2^-53 in [0, 1 - 2^-53]. (Converting the full 63-bit state to
    float64 would round values near 2^63 up to exactly 1.0.) Consequently
    1 - xi is exact and lies in [2^-53, 1], so -log(1 - xi) is always finite.
    """
    s = lcg_step(rng[RNG_STATE])
    rng[RNG_STATE] = s
    rng[RNG_DRAWS] += _ONE
    return float(s >> _SHIFT) * _NORM53


def validate_seed(seed):
    """Python-side check; returns the seed as np.uint64."""
    seed = int(seed)
    if not 0 <= seed < 2 ** 63:
        raise ValueError(f"master seed must be in [0, 2^63), got {seed}")
    return np.uint64(seed)
