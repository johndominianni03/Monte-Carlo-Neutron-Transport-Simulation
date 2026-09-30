"""Continuous-energy cross-section lookup (Numba kernels plus packing).

Lookup rule, for a reaction tabulated on grid[thr:] of its nuclide's grid:

    i = largest index with grid[i] <= E          (binary search)
    E  < grid[thr]           -> 0.0 exactly      (below threshold)
    E == grid[n-1]           -> xs at the last point
    otherwise                -> xs_i + f (xs_{i+1} - xs_i),
                                f = (E - grid[i]) / (grid[i+1] - grid[i])

Interpolation is lin-lin, the law for these NJOY-linearised pointwise data
(see mcslab/nucdata.py). An energy exactly on the grid gives f = 0 and
returns the stored value bit for bit. For E in [grid[thr-1], grid[thr]) the
result is 0, which matches OpenMC. It is also continuous here, because every
reaction in the Phase 2a data is 0 at its first tabulated point (checked by
tests/test_nucdata.py).

Energies outside [grid[0], grid[-1]] (and NaN) raise ValueError. There is no
extrapolation.

For the kernels, nuclide data are packed CSR-style into plain arrays:
    egrid[e_off[k]:e_off[k+1]]      grid of nuclide k
    tot / absn (same layout)        total and absorption of nuclide k on its grid
    mat_nuc[m_off[m]:m_off[m+1]]    nuclide indices of material m
    mat_dens (same layout)          their number densities (atoms / barn-cm)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit


@njit(cache=True)
def find_index(grid, lo, hi, E):
    """Largest i in [lo, hi-1] with grid[i] <= E, given grid[lo] <= E and
    grid[lo:hi] strictly increasing."""
    a = lo
    b = hi - 1
    if grid[b] <= E:
        return b
    while b - a > 1:
        m = (a + b) >> 1
        if grid[m] <= E:
            a = m
        else:
            b = m
    return a


@njit(cache=True)
def interp_xs(grid, g0, n, xs, x0, thr, E):
    """Cross section at E for a reaction whose values xs[x0:x0 + n - thr] sit
    on grid[g0 + thr : g0 + n]. The nuclide grid is grid[g0 : g0 + n]."""
    if not (grid[g0] <= E <= grid[g0 + n - 1]):
        raise ValueError("energy outside the nuclide's energy grid")
    i = find_index(grid, g0, g0 + n, E) - g0
    if i < thr:
        return 0.0
    k = x0 + i - thr
    if i == n - 1:
        return xs[k]
    e_lo = grid[g0 + i]
    f = (E - e_lo) / (grid[g0 + i + 1] - e_lo)
    return xs[k] + f * (xs[k + 1] - xs[k])


@njit(cache=True)
def grid_locate(grid, g0, n, E):
    """(i, f) for E on the nuclide grid grid[g0 : g0 + n], exactly as
    interp_xs computes them. Pass the pair to interp_at so the search is done
    once per nuclide and energy. interp_at then returns bit-identical values
    to interp_xs."""
    if not (grid[g0] <= E <= grid[g0 + n - 1]):
        raise ValueError("energy outside the nuclide's energy grid")
    i = find_index(grid, g0, g0 + n, E) - g0
    if i == n - 1:
        return i, 0.0
    e_lo = grid[g0 + i]
    return i, (E - e_lo) / (grid[g0 + i + 1] - e_lo)


@njit(cache=True)
def interp_at(i, f, n, xs, x0, thr):
    """interp_xs for a precomputed (i, f) from grid_locate. The values
    xs[x0 : x0 + n - thr] sit on grid points thr .. n-1."""
    if i < thr:
        return 0.0
    k = x0 + i - thr
    if i == n - 1:
        return xs[k]
    return xs[k] + f * (xs[k + 1] - xs[k])


@njit(cache=True)
def interp_many(grid, xs, thr, energies):
    """Vectorised interp_xs for one reaction on its own nuclide grid."""
    n = grid.shape[0]
    out = np.empty(energies.shape[0], dtype=np.float64)
    for j in range(energies.shape[0]):
        out[j] = interp_xs(grid, 0, n, xs, 0, thr, energies[j])
    return out


@njit(cache=True)
def macro_xs(m, E, egrid, e_off, table, m_off, mat_nuc, mat_dens):
    """Macroscopic cross section (1/cm) of material m at E:
    sum_k N_k sigma_k(E), each nuclide interpolated on its own grid, summed
    in the material's nuclide order. `table` is tot or absn."""
    s = 0.0
    for j in range(m_off[m], m_off[m + 1]):
        k = mat_nuc[j]
        g0 = e_off[k]
        n = e_off[k + 1] - g0
        s += mat_dens[j] * interp_xs(egrid, g0, n, table, g0, 0, E)
    return s


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------
def lookup(nuclide, mt, E):
    """sigma_mt(E) in barns for a scalar or array E (Python convenience)."""
    rx = nuclide.reactions[mt]
    scalar = np.ndim(E) == 0
    e = np.atleast_1d(np.asarray(E, dtype=np.float64))
    out = interp_many(nuclide.energy, rx.xs, rx.threshold_idx, e)
    return float(out[0]) if scalar else out


def lookup_total(nuclide, E):
    """Total cross section (sum of non-redundant reactions) in barns, from the
    total summed on the nuclide grid."""
    scalar = np.ndim(E) == 0
    e = np.atleast_1d(np.asarray(E, dtype=np.float64))
    out = interp_many(nuclide.energy, nuclide.total_on_grid(), 0, e)
    return float(out[0]) if scalar else out


@dataclass(frozen=True)
class PackedXS:
    """Plain arrays for the kernels (see module docstring)."""
    nuclide_names: tuple
    egrid: np.ndarray
    e_off: np.ndarray
    tot: np.ndarray
    absn: np.ndarray
    m_off: np.ndarray
    mat_nuc: np.ndarray
    mat_dens: np.ndarray
    e_min: float          # every nuclide used is tabulated on [e_min, e_max]
    e_max: float


def pack(material_densities: Sequence, nuclides: dict) -> PackedXS:
    """material_densities[m] = [(nuclide name, atoms/barn-cm), ...] (empty for
    a void); nuclides maps name -> nucdata.Nuclide. Nuclides are packed in
    first-use order, which fixes every array layout."""
    names = []
    for comp in material_densities:
        for name, _ in comp:
            if name not in names:
                names.append(name)
    grids, tots, absns = [], [], []
    for name in names:
        nuc = nuclides[name]
        grids.append(nuc.energy)
        tots.append(nuc.total_on_grid())
        absns.append(nuc.absorption_on_grid())
    e_off = np.zeros(len(names) + 1, dtype=np.int64)
    e_off[1:] = np.cumsum([g.size for g in grids])

    m_off = np.zeros(len(material_densities) + 1, dtype=np.int64)
    m_off[1:] = np.cumsum([len(c) for c in material_densities])
    mat_nuc = np.array([names.index(n) for c in material_densities for n, _ in c],
                       dtype=np.int64)
    mat_dens = np.array([d for c in material_densities for _, d in c], dtype=np.float64)

    def cat(arrs):
        return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)

    e_min = max((float(g[0]) for g in grids), default=0.0)
    e_max = min((float(g[-1]) for g in grids), default=np.inf)
    return PackedXS(tuple(names), cat(grids), e_off, cat(tots), cat(absns),
                    m_off, mat_nuc, mat_dens, e_min, e_max)
