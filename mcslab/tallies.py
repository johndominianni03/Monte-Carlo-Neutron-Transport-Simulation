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
#   zero_yield_weight[b, r]  weight ended by zero-yield events in region r
#                            (a sampled multiplicity of 0, approved P2)
#   chan_zero[b, c]          int64 zero-yield events of channel c
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
K_BORN_BELOW_CUTOFF = 12  # secondaries created below the cutoff (also in K_CUTOFF)
K_ZERO_YIELD = 13    # particles ended by a zero-yield event (multiplicity 0)
K_FREE_GAS = 14      # elastic scatters with free-gas target motion
N_COUNTS = 15
COUNT_NAMES = ("max_draws", "lost", "source", "created", "absorbed", "leak_left",
               "leak_right", "cutoff", "collisions", "elastic", "inelastic", "max_bank",
               "born_below_cutoff", "zero_yield", "free_gas")

# ---- Phase 3 (response tallies) additions. With KinRunConfig.depth_bins
# set, the kinematic kernel also fills
#   tally[b, i, k, s, e]   i = depth bin (mcslab/depth_mesh.py); k = packed
#                          nuclide, the last slot being the material total
#                          summed in the kernel; s in R_*; e in EST_*.
#                          Track-length estimators sum w l N sigma_s (cm x
#                          1/cm x unit of s); collision estimators sum
#                          w N sigma_s / Sigma_t at collisions.
#   mesh_flux[b, i, e]     sum of w l (EST_TL, EST_TL_UNC) or w / Sigma_t
#                          (EST_COLL, EST_COLL_UNC), in cm
# Responses (mcslab/responses.py; OpenMC score names in RESPONSE_NAMES):
R_HEATING = 0        # MT 301, eV
R_HEATING_LOCAL = 1  # MT 901, eV
R_DAMAGE = 2         # MT 444, eV of damage energy
R_H3 = 3             # MT 205, tritons
R_HE4 = 4            # MT 207, alphas
R_ABSORPTION = 5     # absorptions (disappearance reactions)
N_RESP = 6
RESPONSE_NAMES = ("heating", "heating-local", "damage-energy", "H3-production",
                  "He4-production", "absorption")
# Estimators. *_UNC score only source neutrons before their first
# collision: TL_UNC on their flights, COLL_UNC at that first collision
# (OpenMC CollisionFilter bin 0 with track length, bin 1 with collision).
EST_TL = 0
EST_COLL = 1
EST_TL_UNC = 2
EST_COLL_UNC = 3
N_EST = 4
ESTIMATOR_NAMES = ("tracklength", "collision", "tracklength_uncollided",
                   "collision_first")


def allocate(n_batches, n_regions):
    region_sums = np.zeros((n_batches, N_REGION_SCORES, n_regions), dtype=np.float64)
    surface_sums = np.zeros((n_batches, 2, n_regions + 1), dtype=np.float64)
    diagnostics = np.zeros((n_batches, N_DIAG), dtype=np.int64)
    return region_sums, surface_sums, diagnostics


def allocate_kin(n_batches, n_regions, n_ebins, n_channels):
    """-> region_sums, surface_sums, counts, spectrum, cutoff_weight,
    chan_events, chan_created, zero_yield_weight, chan_zero, all zeroed."""
    region_sums, surface_sums, _ = allocate(n_batches, n_regions)
    counts = np.zeros((n_batches, N_COUNTS), dtype=np.int64)
    spectrum = np.zeros((n_batches, N_SPEC, n_regions, n_ebins), dtype=np.float64)
    cutoff_weight = np.zeros((n_batches, n_regions), dtype=np.float64)
    chan_events = np.zeros((n_batches, n_channels), dtype=np.int64)
    chan_created = np.zeros((n_batches, n_channels), dtype=np.int64)
    zero_yield_weight = np.zeros((n_batches, n_regions), dtype=np.float64)
    chan_zero = np.zeros((n_batches, n_channels), dtype=np.int64)
    return (region_sums, surface_sums, counts, spectrum, cutoff_weight,
            chan_events, chan_created, zero_yield_weight, chan_zero)


def allocate_tally(n_batches, n_bins, n_nuclides):
    """-> tally (B, n_bins, n_nuclides + 1, N_RESP, N_EST) and mesh_flux
    (B, n_bins, N_EST), zeroed. n_bins = 0 gives empty arrays (tallies off)."""
    tally = np.zeros((n_batches, n_bins, n_nuclides + 1, N_RESP, N_EST), dtype=np.float64)
    mesh_flux = np.zeros((n_batches, n_bins, N_EST), dtype=np.float64)
    return tally, mesh_flux


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
