# Phase 2a plan: nuclear data layer (approved 2026-09-29)

This file records the approved Phase 2a plan so the work survives a session
reset. The "Status" section at the bottom is updated as commits land.

## Scope (from the request)

- ONLY read and look up real continuous-energy cross sections. No
  energy-dependent scattering kinematics (that is Phase 2b).
- All Phase 1 tests and `tests/test_regression.py compare` must still pass
  byte-identically.
- Read the OpenMC HDF5 files directly with `h5py`. Do NOT install or import
  the `openmc` package.
- Data lives OUTSIDE the repo, located by a config path / the `MCSLAB_DATA`
  environment variable, and is never committed.
- Nuclides (294 K): H1 (test nuclide), Fe54, Fe56, Fe57, Fe58, W180, W182,
  W183, W184, W186, Li6, Li7, Be9, F19.

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Library | **ENDF/B-VIII.0** (OpenMC official HDF5, `endfb80.tar.xz`). The reader must be library-agnostic so **FENDL-3.2** can be added later as a cross-library comparison. |
| 2 | W-180 (0.12 % of natural W) | **Add it** (14 nuclides). |
| 3 | FLiBe density | **Janz correlation at 973 K** as the default. Document the 294 K cross-section / 973 K density mismatch in README "Limitations". |
| 4 | Data directory | `~/nuclear_data/endfb-viii.0-hdf5/` |
| 5 | Plot PNGs | Commit them to `docs/figures/`. |

## Plan

**Setup.**
- Branch `phase-2a` off `main`.
- The data is found through `MCSLAB_DATA` or an explicit `data_dir=`
  argument. `*.h5` is already gitignored.
- `scripts/fetch_data.sh` streams the archive
  (`curl | tar -xJf - <patterns>`), so only the needed nuclide files reach
  disk. It records a sha256 for each file. The whole 3.38 GB is still
  transferred over the network.

**Step 2: reader (`mcslab/nucdata.py`, h5py only).**
- For each nuclide at the requested temperature, it loads:
  - `energy/<T>K`
  - each `reactions/reaction_XXX/<T>K/xs` with `threshold_idx`, `mt`,
    `Q_value` and `redundant`
  - the AWR
  - product and distribution metadata
- The library is a directory plus a label (e.g. "ENDF/B-VIII.0",
  "FENDL-3.2"). Nothing in the reader is specific to ENDF/B-VIII.0.
- For Numba, everything is flattened into CSR-style plain arrays: grids and
  cross sections are concatenated, with offset, length and threshold-index
  tables.

**Step 3: inventory.**
- `scripts/make_inventory.py` writes `docs/data_inventory.md`. For each
  nuclide and reaction it lists:
  - MT and name
  - the threshold from the grid, next to the kinematic threshold from Q
  - Q and sigma(14.1 MeV)
  - the angle and energy distribution type of each outgoing neutron
- It reports MT 301 (heating), MT 444 (damage) and MT 203-207 (gas
  production) *as found*, not as expected.
- It gets the form of Li-7 tritium production right (ENDF's (n,n't)alpha
  via LR=33 inelastic levels versus explicit MT 205).

**Step 4: lookup and materials.**
- Interpolation is verified from the files and the format specs, not
  assumed. The lookup refuses any law it doesn't implement.
- Binary search finds the largest `i` with `E_i <= E`. On-grid energies give
  f = 0, so they return the stored value exactly. Below `threshold_idx` the
  result is exactly 0.
- Macroscopic Sigma(E) = sum_i N_i sigma_i(E), with each nuclide on its own
  grid. There is no unionized grid.
- `mcslab/ce_materials.py` defines the materials, citing a source for every
  number:
  - tungsten and iron with CRC densities and IUPAC/CIAAW natural abundances
  - FLiBe (2LiF-BeF2): Janz density at 973 K by default, and a Li-6
    enrichment parameter that defaults to natural
- Masses come from each file's AWR x m_n.

**Continuous-energy kernel for test (d) (`mcslab/transport_ce.py`, new).**
- Its only collision mode is "the first collision ends the history".
- Phase 1 files are untouched.
- The documented RNG use is 1 draw per flight, plus source-energy draws if
  the source isn't monoenergetic.

**Step 5: tests (`tests/test_nucdata.py`).** They skip cleanly when the data
is absent.
- (a) On-grid lookups return the stored value exactly.
- (b) Grids are strictly increasing, and every cross section is zero below
  its threshold.
- (c) MT 1 equals the sum of the non-redundant partials. Report the worst
  mismatch and justify the tolerance.
- (d) A 14.1 MeV beam through natural iron in first-collision-kills mode
  gives T = exp(-Sigma_t L) within 3 SE. Sigma_t is computed in the test
  directly from raw HDF5 with h5py and its own number-density formula, not
  through the code under test. The seed is fixed before the first run.
- (e) A new regression reference problem on real data: a W | FLiBe | Fe slab
  with a spread source spectrum.
  - It goes in a separate reference file, so the Phase 1 reference stays
    untouched.
  - Its manifest records the sha256 of each HDF5 file it uses.
  - It's recorded with `--reason`.

**Plots.** `scripts/plot_xs.py` draws log-log plots with a marker at
14.1 MeV:
- Fe-56 total
- Li-6 and Li-7 tritium production
- Be-9 (n,2n)

The PNGs go in `docs/figures/`.

**Hand-check list.** 3 to 5 values at 14.1 MeV, with step-by-step
instructions for checking them against an independent source (NNDC ENDF
retrieval and plotter).

**Commits, in order:**
1. Plan
2. Data config and fetch script
3. Reader
4. Inventory
5. Lookup and materials
6. CE kernel
7. Tests a-d
8. Regression reference
9. Plots
10. README "Limitations"

End with a Phase 1-style summary table and the hand-check list. **Stop
before Phase 2b.**

**Limitations to document:**
- no unresolved-resonance probability tables
- 294 K cross sections only (and the FLiBe density-temperature mismatch)
- no kinematics
- no unionized grid

## Status

- [x] Author/committer email on the 7 Phase 1 commits rewritten to the
      GitHub noreply address. Trees and dates are unchanged. `main` = `8e53f68`.
- [x] Branch `phase-2a` created off `main`.
- [x] Data fetched: 14 nuclides, ~276 MB, sha256 pinned (`c63c092`).
- [x] Reader (`0934e5d`).
- [x] Lookup and materials (`93e7cda`). Committed before the inventory,
      because the inventory uses the lookup for sigma(14.1 MeV).
- [x] Inventory (`54b6876`).
- [x] CE kernel (`9019c8e`).
- [x] Tests a-d (`52fb445`). MT 1 is not stored, so (c) checks for no double
      counting and the redundant sums, as explained in the README.
- [x] Regression reference (`b344ce6`).
- [x] Plots (`a551c21`).
- [x] Docs (README). This is the last Phase 2a commit.

Phase 2a is complete. Phase 2b (kinematics) has not been started.
