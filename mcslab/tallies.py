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


# ---- Phase 2b (kinematic CE kernel) additions. region_sums and
# surface_sums keep the layout above. The kinematic kernel adds:
#   spectrum[b, est, r, g]   est in (SPEC_TL, SPEC_COLL); g = energy bin
#   cutoff_weight[b, r]      weight killed by the energy cutoff in region r
#   counts[b, k]             int64 per-batch counters (K_*); columns 0 and 1
#                            are DIAG_MAX_DRAWS and DIAG_LOST, so
#                            config.Results.max_draws / .lost work unchanged
#   chan_events[b, c]        int64 collisions that chose scatter channel c
#   chan_created[b, c]       int64 secondaries created by channel c
SPEC_TL = 0          # sum of w * l in the energy bin (track-length flux numerator)
SPEC_COLL = 1        # sum of w / Sigma_t at collisions in the bin
N_SPEC = 2

K_MAX_DRAWS = DIAG_MAX_DRAWS   # max draws by one history family (primary + secondaries)
K_LOST = DIAG_LOST
K_SOURCE = 2         # source particles started
K_CREATED = 3        # secondaries created by multiplication (banked or cut off)
K_ABSORBED = 4       # particles ended by absorption
K_LEAK_LEFT = 5      # particles leaked through the left vacuum boundary
K_LEAK_RIGHT = 6     # ... and the right one
K_CUTOFF = 7         # particles killed by the energy cutoff (incl. secondaries born below it)
K_COLLISIONS = 8     # collisions (every collision, absorbing or not)
K_ELASTIC = 9        # elastic scatters
K_INELASTIC = 10     # non-elastic scatters (any multiplicity)
K_MAX_BANK = 11      # largest secondary-bank occupancy seen in the batch
N_COUNTS = 12
COUNT_NAMES = ("max_draws", "lost", "source", "created", "absorbed", "leak_left",
               "leak_right", "cutoff", "collisions", "elastic", "inelastic", "max_bank")


def allocate(n_batches, n_regions):
    region_sums = np.zeros((n_batches, N_REGION_SCORES, n_regions), dtype=np.float64)
    surface_sums = np.zeros((n_batches, 2, n_regions + 1), dtype=np.float64)
    diagnostics = np.zeros((n_batches, N_DIAG), dtype=np.int64)
    return region_sums, surface_sums, diagnostics


def allocate_kin(n_batches, n_regions, n_ebins, n_channels):
    """-> region_sums, surface_sums, counts, spectrum, cutoff_weight,
    chan_events, chan_created, all zeroed."""
    region_sums, surface_sums, _ = allocate(n_batches, n_regions)
    counts = np.zeros((n_batches, N_COUNTS), dtype=np.int64)
    spectrum = np.zeros((n_batches, N_SPEC, n_regions, n_ebins), dtype=np.float64)
    cutoff_weight = np.zeros((n_batches, n_regions), dtype=np.float64)
    chan_events = np.zeros((n_batches, n_channels), dtype=np.int64)
    chan_created = np.zeros((n_batches, n_channels), dtype=np.int64)
    return (region_sums, surface_sums, counts, spectrum, cutoff_weight,
            chan_events, chan_created)


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
