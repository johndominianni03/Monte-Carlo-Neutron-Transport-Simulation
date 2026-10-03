# Phase 3 plan: response tallies (dpa, helium, heating, tritium) (approved 2026-10-03)

This file records the approved Phase 3 plan so the work survives a session
reset. The "Status" section at the bottom is updated as commits land.

**Approval (2026-10-03):**
- D8: branch `phase-3` off `main` at `fa5b245`, not pushed.
- D9-D14, D16, D17, D21: accepted as recommended.
- D15: per-bin first-flight checks (50 bins), for bin-offset coverage.
- D18: hold the molar density; the natural-Li path stays bit-identical.
- D19: accepted, including the extra reflective D7 run. The figure caption
  and the CSV header state that all points share one seed, so the curves
  are correlated with each other.
- D20: full per-batch arrays, one reference file and manifest **per
  problem**. The D7 tally reference is recorded exactly once; the
  reflective problem is added later as its own file, with no re-record of
  D7.
- D22: Part B follows Part A without a pause.

**Amendments (2026-10-03):**
1. Failure protocol, fixed before any test exists: if a statistical check
   exceeds 3 SE, work stops and **one** diagnostic is run on seed
   20261039 with 400 x 2000 histories. Both results are reported. No other
   seeds, and no change to thresholds or bins. The original result stays
   in the record.
2. Check 4 compares the track-length absorption only with histories ended
   by a *sampled absorption* (`region_sums[ABSORPTION]`). Zero-yield kills
   and energy-cutoff kills are separate counters and are not included.
3. Documentation: W-186 MT 444 starts at 4 keV in the data (reported, not
   adjusted). MT 301 varies irregularly between the W isotopes at
   14.1 MeV (about 3.9e5 to 9.4e5 eV-b), so W heating is presented as the
   301-901 bracket, not as one number.
4. The work ends with a short status block (branch and head, test count,
   statistical checks and largest |n_SE|, regressions, D7 and Part B
   headline numbers, tally overhead, judgment calls).

Throughout, the project's 3 SE rule and seed policy apply: seeds and
thresholds below were fixed before any test was written, and a failing
check is reported, never fixed by changing them.

## Scope (from the request)

- Every tally equals an OpenMC v0.16.0 score, so Phase 4 compares like
  with like: flux; heating (MT 301); heating-local (MT 901); damage-energy
  (MT 444); H3-production (MT 205); He4-production (MT 207).
- **Part A:** track-length (primary) and collision (cross-check) estimators
  on a depth mesh in problem D7, per nuclide, consuming no random numbers;
  a per-layer flux spectrum; a pure-Python post-processing module (dpa,
  He appm, heating, tritium, scaled to 1 MW/m^2 and one full-power year).
- **Part B:** Li-6 enrichment as a material parameter, an optional
  reflective plasma-side boundary, and a FLiBe thickness x enrichment
  sweep of tritium per source neutron.

## Baseline (verified at `fa5b245`, 2026-10-03)

- 161 tests pass (93 s).
- Phase 1, Phase 2a and D7 regression compares are byte-exact. D7 is also
  byte-exact after a cold Numba compile (empty `NUMBA_CACHE_DIR`, +6 s).
- D7 per source neutron: 25.2 collisions, about 29 flight segments;
  2.3 s of transport per 1e5 histories (warm cache), plus about 1.8 s of
  data loading.
- OpenMC v0.16.0 was read from the GitHub tag tarball (sha256
  `cfdf10f8dc66e652668dbc94501d5ed99b3dfecabc11cf91989a29ba7984654f`).

## Findings

### Code

- **Phase 1 estimators** (`mcslab/transport.py`): track length `w d` per
  region on every segment; collision estimator `w / Sigma_t` at every
  collision, absorbing or not. Region level only. The kinematic kernel
  does the same and adds a per-region energy spectrum.
- **Per-batch rows:** each batch zeroes and writes only its own rows
  (`run_batches_kin`); `tallies.batch_stats` combines them two-pass in a
  fixed order.
- **Regression harnesses:** per-array sha256, manifest with reason and
  history, strict or `--loose` compare; the CE and kinematic harnesses pin
  the sha256 of every data file. D7 pins exactly 9 arrays, so new output
  arrays leave it untouched.
- **Grid index and interpolation factor:** `macro_total`
  (`mcslab/transport_kin.py`) stores `ci[jj]`, `cf[jj]`, `ct[jj]` per
  *material slot* `jj`, recomputed only when the material or the energy
  changes; the collision reuses them. Tallies reuse `ci`, `cf` through
  `interp_at` (no second search) and map `jj` to the global nuclide
  `k = mat_nuc[j]`.
- **Reader:** `read_nuclide` already loads redundant reactions; the five
  response MTs carry only `threshold_idx`, so no reader change is needed.
- `flibe(li6_fraction=...)` already exists and holds the *mass* density
  (Janz) fixed; see D18.

### Data (D7 nuclides at D7 temperatures: W, Fe 294 K; FLiBe 900 K)

- MT 207, 301, 444 and 901 exist for all 13 nuclides. **MT 205 is absent
  for all five W isotopes** (so are 204 and 206). Nothing else is missing.
- Units: 301, 901, 444 in eV-b (energy x cross section); 205, 207 in b
  (multiplicity-weighted production cross sections). All are
  `redundant = 1` and sit on the nuclide's own grid
  (`threshold_idx + len == grid size`).
- Thresholds: 301 and 901 at index 0 everywhere. 444 at index 0 except
  **W186, which starts at 3997.71 eV** (zero damage energy below 4 keV,
  while the other W isotopes have nonzero capture-recoil damage at thermal
  energies; reported, not adjusted). 205: Li6 1e-5 eV, Li7 3.1454 MeV,
  Be9 11.6085 MeV, F19 9.5 MeV, Fe54/56/57/58 5.5/6.0/5.7/5.5 MeV. 207:
  from 9.69 eV (W180) to 1.42 MeV (Fe58). Every thresholded reaction is
  exactly 0 at its first point.
- No negative values in 301, 901 or 444 at any temperature; 901 >= 301 at
  every grid point.
- Li-6: MT 205 == MT 105 exactly at 900 K. Li-7: MT 205 vs the sum of
  MT 52-82, max difference 4.83e-8 b (2.27e-7 relative).
- D7 at 14.1 MeV: Sigma_t = 0.339309 (W), 0.137609 (FLiBe), 0.218637 (Fe)
  1/cm; cumulative optical depth 0.1697, 2.9218, 5.1082; uncollided
  transmission 0.00605.

14.1 MeV values (raw h5py + `np.interp`, D7 temperatures):

| nuclide | 301 (eV-b) | 901 (eV-b) | 444 (eV-b) | 205 (b) | 207 (b) |
|---|---|---|---|---|---|
| W180 | 644991 | 9.13656e6 | 94559.3 | absent | 0.00231004 |
| W182 | 621866 | 1.01257e7 | 97064.4 | absent | 0.00137327 |
| W183 | 388748 | 1.13464e7 | 99850.5 | absent | 0.00135283 |
| W184 | 944698 | 9.62028e6 | 96310.7 | absent | 0.000658936 |
| W186 | 724359 | 9.42129e6 | 95915.4 | absent | 0.00044794 |
| Li6 | 4.86361e6 | 4.86942e6 | 12518.9 | 0.0258 | 0.577686 |
| Li7 | 3.33950e6 | 3.37124e6 | 13319.2 | 0.300646 | 0.320906 |
| Be9 | 2.91306e6 | 2.91671e6 | 19293.5 | 0.0208775 | 0.979366 |
| F19 | 3.56030e6 | 4.35250e6 | 97140.2 | 0.01303 | 0.410193 |
| Fe54 | 5.22357e6 | 1.15040e7 | 263040 | 4.07258e-9 | 0.0884604 |
| Fe56 | 1.75887e6 | 9.54478e6 | 257756 | 2.61359e-7 | 0.0436698 |
| Fe57 | 1.22002e6 | 5.65010e6 | 177274 | 1.04802e-4 | 0.0296651 |
| Fe58 | 904268 | 5.34999e6 | 258634 | 1.80315e-7 | 0.0217647 |

### OpenMC v0.16.0

- Score names (`src/reaction.cpp`, `REACTION_TYPE_MAP`): heating 301,
  heating-local 901, damage-energy 444, H3-production 205 (`N_XT`),
  He4-production 207 (`N_XA`).
- Default estimator: track length (`include/openmc/tallies/tally.h`).
  `Tally::set_scores` (`src/tallies/tally.cpp`) switches heating to the
  collision estimator only with photon transport on; nothing switches 205,
  207, 444, 901 or a mesh filter. In a neutron-only run every score here
  uses the track-length estimator.
- `src/tallies/tally_scoring.cpp`: `score_tracklength_tally`, flux
  `wgt * distance`; `score_collision_tally`, flux
  `wgt_last / macro_xs().total`, called after `collision()` with the
  pre-collision cross-section cache.
- Heating: `score_general_ce_nonanalog` -> `score_particle_heating` ->
  `score_neutron_heating` -> `get_nuclide_neutron_heating`: kerma =
  `rx.xs(micro)` of MT 301; keff reweighting only in eigenvalue mode.
- 444, 901, 205, 207: `default:` case -> `get_nuclide_xs`, N sigma flux;
  **0 when the nuclide lacks the reaction** (`reaction_index_ == C_NONE`),
  so OpenMC scores 0 tritium on W.
- Interpolation: `Reaction::xs` (`src/reaction.cpp`), 0 below threshold,
  else `(1-f) x[i] + f x[i+1]`, with `i`, `f` from `Nuclide::calculate_xs`
  (`src/nuclide.cpp`), shared with the total. mcslab computes
  `x[i] + f (x[i+1] - x[i])` with the same `i`, `f`: equal in exact
  arithmetic, about one ulp apart in floating point.
- Mesh: `MeshFilter::get_all_bins` (`src/tallies/filter_mesh.cpp`): track
  length via `bins_crossed` -> `raytrace_mesh` (`src/mesh.cpp`), weights =
  length fractions; collision estimator by the collision position.
- Uncollided: secondaries inherit `n_collision` (`src/particle.cpp`), and
  `collision()` increments it before the collision tally
  (`src/physics.cpp`). mcslab's uncollided track-length estimator =
  `CollisionFilter` bin 0; its first-collision estimator = bin 1 with the
  collision estimator.
- Data generation (`openmc/data/njoy.py`, `neutron.py`): 301 = ACE heating
  number x sigma_t from HEATR with local photon deposition off (photon
  energy excluded); 901 from a second HEATR run with local deposition on;
  444 from HEATR with `ed` defaulting to 0 (NJOY's built-in per-element
  value, not recorded in the HDF5 file); 203-207 from GASPR.
- Reflection: `ReflectiveBC::handle_particle`
  (`src/boundary_condition.cpp`) reflects, then renormalises the
  direction; no random number.

## The physics, in plain terms

- **Track-length estimator.** Flux is the total distance travelled by
  neutrons per unit volume. A flight of length l through a bin adds l;
  times Sigma_x(E) (the chance per cm of reaction x), it is the expected
  number of x reactions along that flight, whether or not one happened.
  It scores on every flight, so it stays quiet in thin bins.
- **Collision estimator.** At each real collision, add Sigma_x / Sigma_t,
  the fraction of collisions at that energy that are reaction x. Same
  expected value, different code path (collision sites instead of flight
  geometry), so agreement is a strong bug check.
- **Why expected-value scoring.** Li-7(n,n't)alpha is a scattering
  channel and nobody tracks the triton; heating and damage are energy
  deposits, not events. OpenMC scores all of these the same way.
- **Heating, MT 301 (KERMA).** NJOY's average kinetic energy given to
  charged particles (recoil nucleus, p, alpha, t) per reaction, times
  sigma. Gamma energy is excluded; with no photon transport it is simply
  missing, so 301 under-counts.
- **Heating-local, MT 901.** The same with gamma energy deposited at the
  collision site. Right in total if no gamma escapes, but too local (real
  gammas travel centimetres). The truth lies between 301 and 901; the gap
  is large in W and Fe (about 0.12 vs 1.8 MeV per collision in W at
  14 MeV).
- **Damage energy (MT 444) and NRT dpa.** The part of the recoil energy
  that displaces atoms rather than exciting electrons (NJOY,
  Lindhard/Robinson). NRT counts one displaced atom per 2 E_d / 0.8 of
  damage energy; E_d is the energy needed to knock an atom off its site.
- **Gas production (MT 205, 207).** Tritons or alphas per reaction, summed
  over all reactions; Li-6(n,t)alpha counts one of each.
- **Slab normalisation.** The source is one neutron per cm^2 of wall, so
  every "per source neutron" tally is per cm^2 of wall, and a bin of width
  dx holds N dx atoms per cm^2 (N in atoms/cm^3 = 1e24 x atoms/(b cm)).
- **Wall loading.** 1 MW/m^2 = 100 W/cm^2 of 14.1 MeV neutrons =
  4.4266e13 n/cm^2/s.
- **Normal incidence.** D7's beam enters straight in; plasma neutrons
  arrive at all angles, which raises near-surface rates at the same wall
  loading. A limitation of D7.
- **Uncollided.** Flights of source neutrons before their first collision;
  their energy is exactly 14.1 MeV, so the expected track length per bin
  is closed-form.
- **Reflective boundary.** At x = 0 the direction cosine along the normal
  flips (u -> -u), a crude stand-in for neutrons returning from the far
  side of the torus.
- **Li-6 enrichment.** Li-6 turns slow neutrons into tritium (about 940 b
  at thermal energy); Li-7 makes tritium only above 3.1 MeV.

## Design, Part A

**Kernel (`mcslab/transport_kin.py`).** Every addition only reads particle
state: no random number is drawn, and only tally rows are written. A flight
is a segment (x0, u, d, region r, E, w).

1. When `macro_total` reruns, refresh `ns[jj, s] = N_j interp_at(ci[jj],
   cf[jj], ...)` for the five MTs, plus absorption from the existing
   `absn`, and `ntot[s] = sum_jj ns[jj, s]` in the material's nuclide
   order.
2. Split the segment across the region's depth bins (`segment_pieces`):
   parametric edge distances `s_k = (e_k - x0) / u` (the formula of
   `distance_to_boundary`, so a boundary segment ends exactly on the layer
   edge); pieces are differences of `s_k` clipped to [0, d]; the start bin
   is chosen by direction (an edge belongs to the bin being entered); u = 0
   gives one piece. No surfaces are added and no distance is resampled.
3. Track length, per piece: `flux[bin, TL] += w l`;
   `tally[bin, k, s, TL] += (w l) ns[jj, s]`; the same for the total slot;
   the same into `TL_UNC` while the particle is an uncollided primary.
4. Collision estimator, at a collision, before the nuclide is sampled:
   bin from the collision position (as OpenMC's mesh); `(w / Sigma_t)
   ns[jj, s]`, plus `COLL_UNC` at the primary's first collision; then the
   particle is collided. Banked secondaries are always collided.
5. With tallies off, the kernel receives zero-size arrays and skips all of
   this.

**Storage.** `tally[B, bin, nuclide (13 + total), score (6), estimator
(4)]` and `mesh_flux[B, bin, estimator]`, float64. Each batch zeroes and
writes only its own row.

**Driver.** `KinRunConfig.depth_bins` (bins per region; None = off, the
default) and, in Part B, `reflect_left` (default False). `KinResults` gains
the tally arrays, mesh edges, nuclide labels, a `response_present` mask,
region atom densities, `reflected_weight[B]` and per-source helpers.

**Post-processing (`mcslab/postprocess.py`, pure Python).** T is a
per-source tally in a bin, N the bin's atom density (atoms/cm^3),
S = 4.4266e13 n/cm^2/s at 1 MW/m^2, Phi_FPY = S x 3.15576e7 s.

| quantity | formula |
|---|---|
| flux | T_flux / dx |
| dpa per FPY | 0.8 T_444 / (2 E_d) / (N dx) x Phi_FPY (W, Fe only) |
| He appm per FPY | T_207 / (N dx) x 1e6 x Phi_FPY (W, Fe only) |
| He/dpa | ratio of the two (ratio-estimator SE from per-batch values) |
| heating (W/cm^3) | T_301 / dx x S x 1.602176634e-19; same for 901 |
| heating fraction | sum T_301 / 14.1 MeV; sum T_901 / 14.1 MeV |
| tritium | sum T_205 per layer and per nuclide |

Reported: front bin (as the peak) and layer average (sum T / sum dx).
Every formula is linear per batch, so it is applied per batch and then
`batch_stats` is used. The output JSON records E_d, the wall loading, E_n,
the FPY length and the eV-to-J factor.

## Design, Part B

- **Li-6 enrichment** (D18): enriched FLiBe holds the formula-unit
  (molar) density of natural-Li FLiBe at the Janz density. The natural-Li
  code path is unchanged bit for bit.
- **Reflective plasma side** (D17): opt-in `reflect_left`. At x = 0:
  u -> -u, v and w unchanged, x = bounds[0] exactly, region 0 kept; no
  random number. Surface tallies are not touched (the balance stays exact
  with leak_left = 0); the weight goes to `reflected_weight[B]`; tracks get
  an `EV_REFLECT` event.
- **Sweep** (D19): a script, not a test.

## Decisions

| # | Question | Decision |
|---|---|---|
| D8 | Branch | `phase-3` off `main` (`fa5b245`); not pushed. First commit: this plan. |
| D9 | W has no MT 205 | Scored exactly 0, as OpenMC does; flagged "no data" (`response_present`, README footnote) so it is not read as a measured zero. |
| D10 | Tally contents | Scores 301, 901, 444, 205, 207 and absorption (for check 4; OpenMC's "absorption"). Nuclide axis: the 13 nuclides plus an in-kernel total (OpenMC's total-nuclide bin). Estimators TL, COLL, TL_UNC, COLL_UNC. Flux stored separately. |
| D11 | Depth mesh | W 10 x 0.05 cm, FLiBe 20 x 1 cm, Fe 20 x 0.5 cm (50 bins). Edges from `np.linspace` per layer with exact layer bounds. Collision bin from position; a point on an internal edge goes to the right-hand bin. Phase 4: an OpenMC RectilinearMesh with the same x edges. |
| D12 | Spectrum grid | 20 bins per decade, edges 10^(k/20) eV, k = -100..146: 1e-5 eV to 19.95 MeV, 246 bins; 14.1 MeV in [12.59, 14.13) MeV. Through the existing per-region spectrum (`energy_edges`); the D7 regression keeps its 40 bins. |
| D13 | Uncollided estimators | Always on (OpenMC `CollisionFilter` bins 0 and 1). |
| D14 | Batches | Validation runs use 100 batches: with 20, a true-zero difference exceeds 3 SE 0.74% of the time (Student t, 19 dof); with 100, 0.34%. README numbers come from D7 itself (20 x 5000, seed 20261023). |
| D15 | First-flight resolution | 3 SE check on each of the 50 bins (TL_UNC), per layer for COLL_UNC, plus deterministic ratio checks. |
| D16 | Conventions | Linear NRT on the integrated damage energy (NRT's low-energy steps cannot be applied after energy integration). E_d 40 eV (Fe), 90 eV (W) (ASTM E521), configurable. FPY = 365.25 d. He appm and He/dpa for W and Fe only; FLiBe gets heating and tritium. |
| D17 | Reflection | As above. OpenMC renormalises u after reflecting; mcslab's sign flip is exact (round-off only; listed as a deviation). |
| D18 | Enrichment density | Hold the molar density at the natural-Li Janz value (mass density -1.7% at 90% Li-6). |
| D19 | Part B scope | W 0.5 \| FLiBe L \| Fe 10, reflective plasma side, FLiBe at 900 K data and density; L in {10, 20, 30, 50, 75, 100} cm x Li-6 in {7.59 (natural), 20, 40, 60, 90}%; 30 runs of 20 x 5000, all on seed 20261035 (common random numbers). Figure and CSV, labelled an idealised 1D upper bound with no structure, ports or gaps, and stating the shared seed. Plus one reflective D7 run for the full Part A table. |
| D20 | References | Full per-batch arrays, `np.savez_compressed`, one reference file and manifest per problem under `tests/reference_tally/<problem>/`. D7 recorded once; the reflective problem added later as its own file. |
| D21 | Statistical budget | 84 new checks (below). |
| D22 | Part A -> Part B | No pause. |

## Validation (seeds, thresholds and bins fixed here)

**Seeds**

| seed | use |
|---|---|
| 20261023 | D7 with tallies: README numbers, figures, tally reference (transport byte-identical to `tests/reference_kin`) |
| 20261030 | validation run: D7, 100 x 2000; checks 2, 3, 4 and the diagnostics |
| 20261031 | NumPy generator for random test segments in check 1 (inputs only) |
| 20261032 | reproducibility runs: D7 geometry, 12 x 200 |
| 20261033 / 20261034 | Part B symmetry, problem A / problem B (100 x 1000 each) |
| 20261035 | Part B sweep (all points) and the reflective D7 problem |
| 20261036 | NumPy generator for random lookup energies in the packing test (inputs only) |
| 20261037 | reflective void test |
| 20261038 | reflective real-material balance test |
| 20261039 | failure-protocol diagnostic only (400 x 2000), used only if a statistical check fails |

**Deterministic checks**

1. Splitting: hand-built cases; 1e5 random segments against exact rational
   arithmetic (`fractions.Fraction`): each bin's length within 1e-12 d of
   exact, pieces summing to d within 1e-12 d, no negative piece, every
   piece inside its layer; segments ending exactly on a layer boundary.
   Kernel level: per layer and batch, the summed mesh track length
   (collision estimator) equals the region tally to 1e-11 relative.
2. Response packing: packed lookup == `xs.lookup` bit for bit for all 13
   nuclides and every response; against h5py + `np.interp` to 1e-12 x the
   reaction's maximum; W MT 205 exactly 0.
3. Uncollided ratios: uncollided response / uncollided flux ==
   N sigma(14.1 MeV) from raw files to 1e-11 relative, for every bin x
   nuclide x score, TL_UNC and COLL_UNC.
4. Bit-identity: D7 with tallies on gives the 9 transport arrays
   byte-identical to `tests/reference_kin`; tally arrays byte-identical for
   split, chunked and reversed batch order, for stale rows, and in a
   subprocess with an empty `NUMBA_CACHE_DIR`; Phase 1, 2a and D7 compares
   unchanged.
5. Exact identities: W tritium exactly 0 in every estimator; balance
   residual exactly 0 (with reflection too); reflective void: exact
   reflection count (replayed from the RNG), max draws exactly 1,
   u after a reflection == -u before, bit for bit.
6. Post-processing against hand-computed values to 1e-14 relative.

A bound that is exceeded is reported, never loosened.

**Statistical checks (3 SE, 100 batches)**

| test | checks |
|---|---|
| 2: first flight, TL_UNC per bin vs (e^-tau_a - e^-tau_b) / Sigma_t (independent raw-file code and number densities) | 50 |
| 2: COLL_UNC per layer, same formula | 3 |
| 3: TL vs COLL, paired per batch: flux + 6 scores x 3 layers, minus W tritium (exact check) | 20 |
| 3: FLiBe tritium by nuclide (Li6, Li7, Be9, F19), paired | 4 |
| 4: TL absorption vs sampled analog absorptions per layer, paired | 3 |
| Part B symmetry: reflective [0, 20] cm FLiBe vs vacuum [-20, 20] cm, isotropic 14.1 MeV source at 0: flux, absorption, right leakage, tritium (combined SE) | 4 |
| **total** | **84** |

Chance of at least one failure by luck: at most 20% at 0.27% per check
(Sidak bound; the checks are positively correlated), about 25% with the
Student-t rate at 100 batches. With the 74 earlier checks, 158 in all,
about 35%.

**Diagnostics (reported only):** Li-6 sigma_205 vs sigma_105; Li-7
sigma_205 vs the sum of sigma_52..82; per-nuclide sums vs the in-kernel
total; tally runtime overhead; the analog count of Li-7 MT 52-82 events vs
the track-length Li-7 tritium.

## Files

- New: `mcslab/responses.py`, `mcslab/depth_mesh.py`,
  `mcslab/postprocess.py`; `scripts/phase3_results.py`,
  `scripts/blanket_sweep.py`; `tests/test_tally_mesh.py`,
  `test_tally_physics.py`, `test_tally_reproducibility.py`,
  `test_regression_tally.py`, `tests/reference_tally/<problem>/`,
  `test_postprocess.py`, `test_blanket.py`; `docs/phase3_results.json`,
  `docs/blanket_sweep.csv`, `docs/figures/phase3_profiles.png`,
  `docs/figures/blanket_sweep.png`.
- Modified: `mcslab/transport_kin.py`, `config_kin.py`, `tallies.py`,
  `tracks.py`, `ce_materials.py`; README;
  `docs/deviations_from_openmc.md`.
- Untouched: the Phase 1 and 2a kernels and drivers, `xs.py`,
  `nucdata.py`, all existing references.

## Commits (on `phase-3`)

1. This plan.
2. Mesh, splitter and response packing; deterministic checks 1-2. No
   kernel change.
3. Kernel and driver tallies; check 4 (bit-identity). All three
   regressions byte-exact; a changed D7 bit is a bug to find, not a reason
   to re-record.
4. Validation: checks 2, 3 (statistical part), 5 and the diagnostics.
5. Tally regression harness and the D7 reference, recorded once with
   `--reason`; it also asserts that D7's transport hashes equal those of
   `tests/reference_kin`.
6. Post-processing module and its tests.
7. Results script and depth-profile figure.
8. Part A docs: README, deviations, this status.
9. Li-6 molar-density rule and its test.
10. Reflective boundary and its tests; the reflective problem's own
    reference file.
11. Sweep script, figure and CSV.
12. Part B docs and the final summary.

## Estimated runtimes

| item | estimate |
|---|---|
| D7 with tallies | 3.5-5 s (+50-100%; measured and reported) |
| validation run, 2e5 histories | 8-11 s |
| reproducibility tests incl. cold compile | about 25 s |
| tally regression compare | about 6 s |
| Part B tests | about 10 s |
| full pytest | 93 s -> about 2.5-3 min |
| sweep, 30 runs | 4-8 min |

## Limitations to document

- No photon transport: 301 and 901 bracket the deposited heating.
- 1D slab, beam at normal incidence.
- NRT dpa (no arc-dpa); the E_d choice; the E_d NJOY used inside MT 444 is
  not recorded in the files.
- No URR probability tables.
- D7 leaks 56% of the source back out of the plasma side, so its tritium
  number is not a TBR.
- Part B is an idealised upper bound.
- README standard errors come from 20 batches.
- Hand-check: MT 205 of Li-6 and Li-7 can be checked against ENDF MT 105
  and MT 52-82; MT 301, 444 and 901 are NJOY-derived and need
  independently processed data (e.g. JANIS with an ACE library).

## Status

- [x] Branch `phase-3` created from `fa5b245`.
- [x] Plan approved (2026-10-03).
- [x] 1. Plan: `0612734`.
- [x] 2. Mesh, splitter, response packing: `83e1893`. Check 1: worst
      per-bin error 2.9e-16 d and sum error 2.2e-16 d (bound 1e-12).
      Check 2: lookups bit-identical to the reader, 2.1e-16 from raw h5py.
- [x] 3. Kernel and driver tallies: `b0f0f8c`. D7 with tallies on is
      byte-identical to `tests/reference_kin`; Phase 1, 2a and D7
      compares byte-exact; 172 tests pass.
- [x] 4. Validation: `ad15aef`. All 80 statistical checks pass, largest
      |n_SE| 2.15 (TL_UNC, W bin 2). Uncollided ratios 7.3e-14, mesh vs
      region tallies 3.9e-14 (bounds 1e-11). The failure protocol was not
      needed. The first overhead diagnostic compared a cold first call
      with a warm one (+78%); it was corrected before the commit to two
      warm calls (+37% incl. packing); the statistical results were
      unchanged (same seed, same rows).
- [x] 5. Tally regression and D7 reference: `d0fe4a4`. Recorded once at
      `ad15aef` with a clean tree; 1033 KiB compressed.
- [x] 6. Post-processing: `b028cac`.
- [x] 7. Results script and figure: `811ca25`. D7 with tallies: 2.82 s vs
      2.18 s without (+29%, warm, incl. packing).
- [x] 8. Part A docs: README (results, 14.1 MeV response table, data notes,
      Part A limitations), deviations 13-14 and the mirrored scoring path.
- [ ] 9. Li-6 molar density.
- [ ] 10. Reflective boundary.
- [ ] 11. Sweep.
- [ ] 12. Part B docs.
