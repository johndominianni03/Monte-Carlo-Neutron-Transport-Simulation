"""Depth mesh for the response tallies (Phase 3): uniform bins inside each
slab region, laid over the geometry after the fact.

The mesh adds no surfaces and resamples no distances. A flight segment
(start x0, direction cosine u, length d) that the kernel has already
chosen is split across the bins of its region:

    s_k = (e_k - x0) / u        distance along the flight to bin edge e_k

Pieces are the differences of the s_k crossed, clipped to [0, d]. This is
the formula of geometry.distance_to_boundary, and a region's first and last
edges are its bounds exactly, so a segment that ends on a region boundary
ends exactly on the last edge it crosses. The start bin is chosen by the
direction: an edge belongs to the bin the particle is entering. A
collision is binned by its position, an internal edge going to the
right-hand bin, as OpenMC's mesh filter does for collision estimators
(src/tallies/filter_mesh.cpp, MeshFilter::get_all_bins).

Kernel layout (plain arrays):
    edges[eoff[r] : eoff[r+1]]   the n_r + 1 edges of region r
    boff[r]                      global index of region r's first bin
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit


@dataclass(frozen=True)
class DepthMesh:
    edges: np.ndarray        # concatenated per-region edges
    eoff: np.ndarray         # (n_regions + 1,) offsets into edges
    boff: np.ndarray         # (n_regions + 1,) global bin offsets
    counts: tuple            # bins per region

    @property
    def n_bins(self) -> int:
        return int(self.boff[-1])

    @property
    def max_bins(self) -> int:
        return max(self.counts)

    @property
    def bin_edges(self) -> np.ndarray:
        """(n_bins + 1,) increasing edges over the whole slab (shared
        region bounds appear once)."""
        parts = [self.edges[self.eoff[0]:self.eoff[1]]]
        for r in range(1, len(self.counts)):
            parts.append(self.edges[self.eoff[r] + 1:self.eoff[r + 1]])
        return np.concatenate(parts)

    @property
    def bin_region(self) -> np.ndarray:
        """(n_bins,) region index of every bin."""
        return np.repeat(np.arange(len(self.counts)), self.counts)

    @property
    def bin_width(self) -> np.ndarray:
        return np.diff(self.bin_edges)


def make_mesh(bounds: Sequence[float], counts: Sequence[int]) -> DepthMesh:
    """counts[r] uniform bins in region r = [bounds[r], bounds[r+1]]. The
    first and last edge of each region are its bounds exactly."""
    bounds = np.asarray(bounds, dtype=np.float64)
    counts = tuple(int(c) for c in counts)
    if len(counts) != bounds.size - 1:
        raise ValueError(f"{bounds.size - 1} regions but {len(counts)} bin counts")
    if any(c < 1 for c in counts):
        raise ValueError("every region needs at least one depth bin")
    parts = []
    for r, n in enumerate(counts):
        e = np.linspace(bounds[r], bounds[r + 1], n + 1)
        e[0] = bounds[r]
        e[-1] = bounds[r + 1]
        if (np.diff(e) <= 0.0).any():
            raise ValueError(f"region {r}: bins too narrow to be strictly increasing")
        parts.append(e)
    eoff = np.zeros(len(counts) + 1, dtype=np.int64)
    eoff[1:] = np.cumsum([n + 1 for n in counts])
    boff = np.zeros(len(counts) + 1, dtype=np.int64)
    boff[1:] = np.cumsum(counts)
    return DepthMesh(np.concatenate(parts), eoff, boff, counts)


@njit(cache=True)
def bin_of(edges, e0, n, x):
    """Local bin j in [0, n-1] of position x in a region whose n + 1 edges
    are edges[e0 : e0 + n + 1]: the largest j with edges[e0 + j] <= x,
    clamped (an internal edge belongs to the right-hand bin)."""
    a = 0
    b = n
    if x < edges[e0 + 1]:
        return 0
    if x >= edges[e0 + n - 1]:
        return n - 1
    # invariant: edges[e0 + a] <= x < edges[e0 + b]
    while b - a > 1:
        m = (a + b) >> 1
        if edges[e0 + m] <= x:
            a = m
        else:
            b = m
    return a


@njit(cache=True)
def start_bin(edges, e0, n, x, u):
    """Local bin in which a flight from x with direction cosine u starts.
    u >= 0: bin_of(x). u < 0: the smallest j with x <= edges[e0 + j + 1],
    so an edge belongs to the bin on its left (the one being entered)."""
    j = bin_of(edges, e0, n, x)
    if u < 0.0 and j > 0 and x == edges[e0 + j]:
        j -= 1
    return j


@njit(cache=True)
def segment_pieces(edges, e0, n, x0, u, d, pb, pl):
    """Split the flight (x0, u, length d) inside a region with n bins
    (edges[e0 : e0 + n + 1]). Writes local bin indices to pb[0:m] and
    lengths to pl[0:m] in flight order and returns m. The lengths are
    s_1 - 0, s_2 - s_1, ..., d - s_(m-1), so they telescope to d."""
    j = start_bin(edges, e0, n, x0, u)
    if u == 0.0 or not d > 0.0:
        pb[0] = j
        pl[0] = d
        return 1
    m = 0
    done = 0.0
    if u > 0.0:
        while True:
            if j == n - 1:
                break
            s = (edges[e0 + j + 1] - x0) / u
            if s >= d:
                break
            pb[m] = j
            pl[m] = s - done
            m += 1
            done = s
            j += 1
    else:
        while True:
            if j == 0:
                break
            s = (edges[e0 + j] - x0) / u
            if s >= d:
                break
            pb[m] = j
            pl[m] = s - done
            m += 1
            done = s
            j -= 1
    pb[m] = j
    pl[m] = d - done
    return m + 1
