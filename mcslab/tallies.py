"""Tally array layout and batch statistics.

Kernels accumulate raw weight sums, one row per batch:

    region_sums[b, score, r]    score in REGION_SCORES, r = region index
    surface_sums[b, dir, s]     dir in (NEG, POS), s = surface index (0..n_regions)
    diagnostics[b, k]           int64 per-batch diagnostics (DIAG_*)

Each batch writes only its own row, so rows are independent of how batches
are distributed over calls or threads. Normalisation (per source particle,
per unit width) happens afterwards in Python.
"""
import numpy as np

# region scores
COLLISION = 0        # sum of w at collisions
ABSORPTION = 1       # sum of w at absorptions
TRACK_LENGTH = 2     # sum of w * l  (track-length estimator numerator, cm)
COLL_ESTIMATOR = 3   # sum of w / Sigma_t at collisions (collision estimator numerator, cm)
N_REGION_SCORES = 4
REGION_SCORE_NAMES = ("collision", "absorption", "track_length", "coll_estimator")

# surface current directions
NEG = 0              # crossings with mu < 0
POS = 1              # crossings with mu > 0

# per-batch diagnostics
DIAG_MAX_DRAWS = 0   # max random numbers used by any single history in the batch
DIAG_LOST = 1        # histories terminated because they could not move (should be 0)
N_DIAG = 2


def allocate(n_batches, n_regions):
    region_sums = np.zeros((n_batches, N_REGION_SCORES, n_regions), dtype=np.float64)
    surface_sums = np.zeros((n_batches, 2, n_regions + 1), dtype=np.float64)
    diagnostics = np.zeros((n_batches, N_DIAG), dtype=np.int64)
    return region_sums, surface_sums, diagnostics


def batch_stats(x):
    """Mean and standard error over axis 0 (the batch axis).

    x[b] is the batch-b estimate (already normalised per source particle).
    mean = (1/B) sum_b x_b
    SE   = sqrt( sum_b (x_b - mean)^2 / (B (B - 1)) )
    Two-pass for numerical stability; the order is fixed, so it is deterministic.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 batches for a standard error")
    mean = x.sum(axis=0) / n
    dev = x - mean
    se = np.sqrt((dev * dev).sum(axis=0) / (n * (n - 1)))
    return mean, se
