# Development rules

These rules govern every change to mcslab: how tests are written and
judged, how bit-identical results are protected, and what happens when a
statistical check fails. The phase plans in `docs/` record how each rule
was applied.

## Environment

- Python is always run from the project venv: `./venv/bin/python`.
  Tests: `./venv/bin/python -m pytest`.
- Dependencies are pinned in `requirements.txt`. Upgrading numpy, numba or
  llvmlite can change floating-point bits, so it is treated as a physics
  change (see "Regression references").
- Nuclear data live outside the repository (`MCSLAB_DATA`, fetched by
  `scripts/fetch_data.sh`) and are pinned by `scripts/checksums/*.sha256`.
  Without `MCSLAB_DATA`, the tests that need data are skipped, and the
  Phase 2a and later regression harnesses refuse to run.

## Analytic expectations

The expected values in `tests/test_physics.py` are exact results derived
from theory (for example `exp(-Sigma_t L)`, `Sigma_t/Sigma_a`,
`1/Sigma_a`). An analytic expectation is never edited to make a test
pass. If such a test fails, either the code or the test's setup is wrong,
and the failure report says which.

Round-off bounds ("exact to 1e-12") are stated in the test together with
the worst value observed, and are never loosened to make a test pass.

## Seed policy

- Every seed used by a statistical test is fixed in the phase plan before
  the test is written and before it is first run:
  - Phase 2b: `docs/phase2b_plan.md` (Part 1 and Part 2 sections);
  - Phase 3: `docs/phase3_plan.md`;
  - Phase 4: `docs/phase4_plan.md`.
- No seed-shopping. Changing a seed until a failing check passes is the
  same as editing its expectation, and is not done.
- History `i` is seeded as `skip_ahead(master_seed, i * STRIDE)` and
  depends on nothing else. Secondary neutrons continue their parent
  history's stream, so a history family depends only on (master seed,
  history id). The maximum number of draws per family is counted and
  checked against `STRIDE`.
- Every random number in a kernel comes from `mcslab.rng`; `np.random` is
  never used inside a kernel.

## Statistical checks

- Thresholds are fixed in the plan together with the seeds, before the
  test exists. A check passes at 3 sigma (two-sided), or, for chi-square
  and Hotelling tests, at p >= 0.0027, the same false-alarm rate.
- Chi-square cells expecting fewer than 5 events are merged by a rule
  fixed in the test (`merge_sparse_cells`), never by hand.
- There are 240 statistical checks:

  | Where | Checks |
  |---|---|
  | Phase 1 (`tests/test_physics.py`) | 35 |
  | `tests/test_nucdata.py` | 1 |
  | `tests/test_kinematics.py`: (a) 6, (a2) 4, (b) 4 | 14 |
  | `tests/test_laws.py`: (f) 10, (g) 6 | 16 |
  | `tests/test_freegas.py`: stationarity 4, kernel shape 4 | 8 |
  | `tests/test_tally_physics.py`: first flight per depth bin 50, first collision per layer 3, track length vs collision 24, track length vs analog absorption 3 | 80 |
  | `tests/test_blanket.py`: reflective vs mirrored slab | 4 |
  | `tests/test_benchmark_openmc.py`: mcslab vs OpenMC 0.16.0, layer-integrated (P1 26, P2 25, P3 24, P4 7) | 82 |

  At this false-alarm rate there is roughly a 48% chance that at least
  one of the 240 fails by chance in a full run (at most 20% for the 84
  Phase 3 checks alone, and 20% for the 82 Phase 4 checks).
- **A failure is reported, never fixed by changing a seed, a threshold
  or an expectation.** The work stops, the failure is reported and
  investigated (for example, a rerun with more histories *as a
  diagnosis*), and the outcome is documented next to the original
  result, which stays in the record.
- Phase 3 failure protocol (fixed in `docs/phase3_plan.md`): one
  diagnostic only, on seed 20261039 with 400 x 2000 histories. Both
  results are reported, and the original stays in the record.

## Benchmark failure protocol (Phase 4)

Fixed in `docs/phase4_plan.md` before the benchmark runs:

1. A failing primary check is rerun once, in both codes, at
   100 x 40,000 histories on that problem's declared second seed pair.
2. If it passes, both z values are recorded in the `PROTOCOL_EXCEPTIONS`
   table of `tests/test_benchmark_openmc.py` and in the README. A
   failure is never marked as an xfail without a record.
3. If it fails again, the work stops; no physics change is made before
   the cause is understood and agreed.

No OpenMC or mcslab setting is ever tuned to make the codes agree. The
benchmark settings, seeds and checks are those fixed in the plan. OpenMC
runs on one thread, so its results reproduce bit for bit; its run
directories stay outside the repository, and only the results JSON is
committed, with no absolute paths, user names, dates or timings.

## Regression references

Refactors must be bit-identical. Each kernel has a byte-exact regression
harness with a committed reference:

| Harness | Reference | Covers |
|---|---|---|
| `tests/test_regression.py` | `tests/reference/` | Phase 1 one-group kernel |
| `tests/test_regression_ce.py` | `tests/reference_ce/` | Phase 2a continuous-energy kernel |
| `tests/test_regression_kin.py` | `tests/reference_kin/` | Phase 2b kinematic kernel, problem D7 |
| `tests/test_regression_tally.py` | `tests/reference_tally/<problem>/` | Phase 3 response tallies (`kin_d7`, `kin_d7_reflect`) |

- `compare` must pass byte-exact. `compare --loose` (rtol 1e-6) is a
  diagnostic, never a pass criterion.
- A reference is regenerated only when results legitimately change (a
  physics change, a change in random-number consumption, a dependency
  upgrade), with `record --reason "..."`. The reason, date, commit and
  library versions are stored in the manifest with its history.
- Each reference is regenerated only for a change to its own kernel: a
  CE-only change never re-records the Phase 1 reference, a kinematic
  change never re-records Phase 1 or Phase 2a, and a tally-only change
  never re-records `reference_kin`. `kin_d7` also requires D7's transport
  arrays to hash as in `tests/reference_kin`.
- `test_regression_tally.py record --problem NAME` refuses to overwrite an
  existing reference unless `--overwrite` is given with a legitimate
  reason; adding a problem never re-records another.
- The CE, kinematic and tally manifests pin the sha256 of every nuclear
  data file used. Different data is a setup error, not a pass.
- `tests/test_reproducibility_kin.py` (batch splitting),
  `tests/test_tracks.py` (track recording) and
  `tests/test_tally_reproducibility.py` (tallies on/off, batch splitting,
  a fresh Numba cache) guard the same bit-identity.

## Reproducibility invariants

- Each batch writes only to its own tally row. The number of batches is a
  problem input, never derived from the thread count. There are no shared
  accumulators and no `prange` reductions on tallies.
- No `fastmath=True`: it permits reassociation, which changes bits.
- In RNG code every integer constant is an explicit `np.uint64`. Numba
  promotes `uint64 op int64` to `float64`, which would silently corrupt
  the generator.
- Secondary neutrons go on a per-history LIFO bank, local to the batch. A
  bank overflow raises.
- No neutron is dropped silently. Every particle ends absorbed, leaked,
  cut off by the energy cutoff (`cutoff_weight`, per region), ended by a
  zero-yield event (`zero_yield_weight`, per region) or lost (counted).
  `KinResults.balance()` has a residual of exactly 0.
- Track recording (`n_track > 0`) and the response tallies (`depth_bins`)
  only read particle state: they draw no random number, and every other
  output is byte-identical with them on or off. The tallies reuse
  macro_total's grid index and factor, and the depth mesh adds no
  surfaces and resamples no distances.
- The reflective boundary (`reflect_left`) draws no random number and
  flips u exactly. A reflection is not a leak; its weight goes to
  `reflected_weight`.

## Nuclear data and OpenMC

- HDF5 files are read with h5py only; the `openmc` package is imported
  only by `benchmark/run_openmc.py`, in OpenMC's own environment. The two
  sides exchange `benchmark/problems.json` and results JSON only.
- Only the secondary laws the data use are implemented
  (`docs/law_inventory.md`). The reader refuses unknown attributes,
  interpolation codes or extra regions, and the kinematic driver refuses
  any problem whose nuclides carry an unsupported law.
- Response data are never clamped or adjusted. A nuclide without a
  response MT scores exactly 0, as OpenMC does, and is flagged in
  `response_present`.
- A material's temperature selects the data temperature by OpenMC's
  nearest rule (10 K tolerance); there is no interpolation between data
  temperatures.
- OpenMC v0.16.0 is the physics reference. Code comments cite its source
  file and function, and every known deviation is recorded, with its
  reason and expected effect, in `docs/deviations_from_openmc.md`.

## Documentation

- Limitations are documented honestly in the README. No validation is
  claimed that has not been run, and each test's description says what it
  does and does not cover.
- Without photon transport, heating is reported as the MT 301 to MT 901
  bracket, never as a single number.
