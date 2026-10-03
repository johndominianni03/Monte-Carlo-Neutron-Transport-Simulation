# Phase 2b plan: continuous-energy transport with kinematics (approved 2026-09-30)

This file records the approved Phase 2b plan so the work survives a
session reset. The "Status" section at the bottom is updated as commits
land.

**Approval (2026-09-30):**
- D1: `main` fast-forwarded to `phase-2a`; `phase-2b` stays based on
  `phase-2a`.
- D2, D3, D4, D6, D7: accepted as recommended.
- D5: accepted, **on the condition that neutrons are never dropped
  silently**:
  - The weight killed by the energy cutoff is tallied **per region**.
  - It is its own term in test (d)'s exact balance identity.
  - It is reported in every summary table, so any bias is visible.

## Scope (from the request)

- Neutrons scatter, lose energy and multiply, using the reactions and
  distribution laws listed in `docs/data_inventory.md`. Neutrons only; no
  photon transport.
- Mirror OpenMC's physics choices wherever possible (Phase 4 benchmarks
  against OpenMC). Record every known deviation in
  `docs/deviations_from_openmc.md`.
- OpenMC's C++ source and theory manual may be read and cited (MIT
  license). The `openmc` package is not installed or imported.
- **Part 1** (stop for review): reaction selection, absorption, elastic
  scattering (target at rest), discrete inelastic levels (MT 51-90), a
  secondary-neutron bank, and max RNG draws per history tracked against
  STRIDE. Tests a-e.
- **Part 2** (only after Part 1 is approved): the remaining distribution
  laws, multiplicities, free-gas thermal treatment, track recording and
  its animation. Tests f-h plus the track-recording invariance test.
- Small commits. End each part with a summary table, then stop.

## Starting point

**Branch base.** The request says "branch phase-2b off main (Phase 2a is
already merged)". In this repository it is not:

| ref | commit | contents |
|---|---|---|
| `main` | `8e53f68` | Phase 1 only |
| `phase-2a` | `525cff1` | Phase 1 + all of Phase 2a (strict descendant of `main`; merging is a fast-forward) |

No remote is configured, so a merge done elsewhere (e.g. on GitHub) is not
visible here. `phase-2b` has been created from `525cff1`, the tip of
`phase-2a`, which is exactly what `main` would be after the merge.

Resolved at approval: `main` was verified to be an ancestor of `phase-2a`
and fast-forwarded with `git branch -f main phase-2a`. `main`, `phase-2a`
and `phase-2b` now all start at `525cff1`.

Both regression compares pass on this base:
- Phase 1: 6 arrays, byte-exact.
- Phase 2a CE: 6 arrays, byte-exact, 13 data files with matching sha256.

**Step 0: Phase 1 manifest provenance.** `tests/reference/manifest.json`
names commit `91c1864`. That is the pre-rewrite hash of what is now
`cdcd09d` ("Add analytic physics validation suite"). The object still
exists locally but is unreachable from any branch.

1. Keep a copy of the current `reference.npz`.
2. Run `tests/test_regression.py record --reason "provenance update after
   author-email history rewrite; arrays unchanged"`.
3. Check every array in the new file against the old one: same dtype,
   shape and raw bytes, and the same sha256 in both manifests.
4. `np.savez` writes zip entry timestamps, so the new `.npz` container
   differs byte-wise even when every array is identical. After step 3
   passes, restore the committed `reference.npz`. The commit then changes
   only `manifest.json`. The new manifest's `history` keeps the old entry.
5. `compare` must pass.

## What OpenMC actually does (v0.16.0, verified from source)

I read OpenMC v0.16.0 (tag `v0.16.0`, released 2026-08-05) from GitHub.
Paths are relative to the OpenMC repository and will be cited in code
comments.

| topic | OpenMC v0.16.0 behaviour | where |
|---|---|---|
| Nuclide selection | `cutoff = prn * Sigma_t`; the first nuclide with cumulative `N_i sigma_t,i >= cutoff` is chosen (1 draw). | `src/physics.cpp` `sample_nuclide` |
| Absorption | Analog by default (`survival_biasing = false`). If `sigma_a > 0`, one draw: absorbed if `sigma_a > prn * sigma_t`. Absorption is a single lumped "disappearance"; the specific channel is not sampled. | `physics.cpp` `absorption`; `src/settings.cpp` |
| Absorption set | `sigma_a` = sum of non-redundant MTs with `is_disappearance(mt)`: MT 101-117, 600-849, 155, 182, 191-193, 197 (+ fission). | `src/endf.cpp`; `src/nuclide.cpp` `create_derived` |
| Scatter channel | One draw, `cutoff = prn * (sigma_t - sigma_a)`. Elastic if `sigma_el > cutoff`. Otherwise walk the non-redundant inelastic-scatter reactions (MT 5-99 except 27 and fission, non-disappearance 100-200, 875-891) in file order (= increasing MT), accumulating from `sigma_el`. | `physics.cpp` `scatter`; `endf.cpp` `is_inelastic_scatter`; `nuclide.cpp` constructor |
| Elastic | Velocity vectors in units `v = sqrt(E)`. `v_cm = (v_n + A v_t)/(A+1)`. `mu_cm` from the tabulated angle distribution at the **lab** energy (isotropic if absent). Rotate with `rotate_angle` (1 draw for phi), then transform back. | `physics.cpp` `elastic_scatter` |
| Target motion | Target at rest if `E >= free_gas_threshold * kT` **and** `awr > 1`. Otherwise free gas with the constant-cross-section (cxs) sampler. `free_gas_threshold = 400` by default. Resonance scattering (DBRC/RVS) is off by default. kT is the data temperature's kT. **H-1 (awr = 0.99917) therefore gets free gas at every energy.** | `physics.cpp:891` `sample_target_velocity`, `sample_cxs_target_velocity`; `settings.cpp` (`free_gas_threshold {400.0}`, `res_scat_on {false}`) |
| Non-elastic scattering | `products_[0].sample(E_in)` gives `(E, mu)`. If the reaction is in CM: `E_lab = E_cm + (E_in + 2 mu (A+1) sqrt(E_in E_cm)) / (A+1)^2`, `mu_lab = mu sqrt(E_cm/E_lab) + sqrt(E_in/E_lab)/(A+1)`. Then `rotate_angle`. | `physics.cpp` `inelastic_scatter` |
| Multiplicity | Integer yield `y`: bank `y - 1` secondaries that are **identical copies** of the post-collision neutron (same E, direction, position, time). Non-integer yield: `wgt *= y` (no roulette, because roulette runs only with survival biasing). | `physics.cpp:1192-1199`; `src/physics_common.cpp` `apply_russian_roulette` |
| Secondary bank | Per particle, **LIFO** (`back()` then `pop_back()`). A secondary continues on the parent's RNG stream; seeds are not reset. | `src/particle.cpp` `create_secondary`, `event_check_limit_and_revive` |
| Level inelastic | `E_cm = mass_ratio * (E - threshold)`, using the **stored** (rounded) `mass_ratio` and `threshold`. | `src/distribution_energy.cpp` `LevelInelastic::sample` |
| Tabular angle | `get_energy_index`, then one draw for stochastic interpolation between incident energies (`if r > prn: i += 1`, drawn even when r = 0). Then `Tabular` sampling from the stored CDF (histogram or lin-lin), normalised by `c[n-1]`. | `src/distribution_angle.cpp`, `src/distribution.cpp` `Tabular::sample_unbiased` |
| Continuous tabular (ACE law 4) | Stochastic choice of incident table, then outgoing-energy CDF sampling, then **scaled interpolation** between the two incident tables' `[E_1, E_K]`. | `distribution_energy.cpp` `ContinuousTabular::sample` |
| Correlated angle-energy (ACE law 61) | As law 4 for the energy (stored CDFs used as-is). Then `mu` from the angle table of the closer outgoing-energy point, `k` or `k+1` in CDF space (always `k` for histogram). | `src/secondary_correlated.cpp` `sample_dist` |
| Several distributions | One draw against the applicability functions (only when there is more than one distribution). | `src/reaction_product.cpp` `sample_dist` |
| Energy cutoff | Neutron cutoff 0 eV by default. Below the grid, cross sections are extrapolated linearly from the first two points. | `settings.cpp` `energy_cutoff`; `nuclide.cpp:719` |
| URR | Probability tables on by default (`urr_ptables_on = true`). | `settings.cpp` |
| Speed and time | `v = c sqrt(E(E + 2m)) / (E + m)`, `m = 939.56542052e6 eV` (CODATA 2018), `c = 2.99792458e10 cm/s`. `dt = distance / speed`. | `particle.cpp` `speed`; `include/openmc/constants.h` |

Checked against our data: on all 14 nuclides, mcslab's Phase 2a absorption
set ("non-redundant reactions with no outgoing neutron") is identical to
OpenMC's `is_disappearance` set. Every other non-redundant reaction is
elastic or OpenMC inelastic-scatter. No reaction falls between the two.

## What the data require (survey of the 14 HDF5 files)

Non-redundant reactions with an outgoing neutron, by law:

| law (HDF5 type) | frame | where | Part |
|---|---|---|---|
| uncorrelated: tabular angle, no energy law | CM | MT 2 elastic, all 14 nuclides (angle interpolation lin-lin) | 1 |
| uncorrelated: tabular angle + `level` | CM | MT 51-90: 231 reactions (220 with lin-lin angle tables, 11 with histogram) | 1 |
| `correlated` (ACE law 61), E_out lin-lin, mu lin-lin | CM | Fe: MT 5, 16, 91. W: MT 5, 16, 17, 28, 37, 41, 91 | 2 |
| `correlated`, E_out histogram | lab | Be9 MT 16; F19 MT 16, 22, 28, 91 | 2 |
| `correlated`, two distributions + applicability (0.5 / 0.5) | lab | F19 MT 16 | 2 |
| uncorrelated: tabular angle + `continuous` (ACE law 4) | lab | Li6 MT 24; Li7 MT 16, 24, 25 | 2 |

Yields: constant 1 (262 reactions), 2 (MT 16, 24, 41), 3 (MT 17, 25) and
4 (MT 37). **MT 5 in all Fe and W isotopes has an energy-dependent
`Tabulated1D` yield from 0 to 8.2.** For Fe56 it is 0 up to 6 MeV and 0.456
at 14 MeV. Its cross section is nonzero from 1 keV.

In the neutron data: no discrete lines (`n_discrete_lines = 0` everywhere),
every incident-energy interpolation is one lin-lin region, and every
correlated mu table is lin-lin. Kalbach-Mann, N-body, evaporation, Maxwell
and Watt laws do not occur.

**Level parameters are rounded.** The stored `mass_ratio` and `threshold`
differ from `(A/(A+1))^2` and `-Q (A+1)/A` (from the file's AWR and Q) by
up to 4.0e-8 and 4.7e-7 relative (worst cases: Fe56 MT 51 and Li7 MT 65).
This matters for test (c); see D4.

## Decisions (all accepted 2026-09-30; D5 with an added condition)

| # | Question | Decision |
|---|---|---|
| D1 | Branch base (see above) | `phase-2b` off `phase-2a`'s tip. `main` fast-forwarded to `phase-2a`. |
| D2 | Non-integer yields (MT 5 in Fe and W). OpenMC multiplies the weight by y(E). | **Analog integer sampling:** `n = floor(y) + [xi < y - floor(y)]` (one extra draw, only when y is not an integer). `n = 0` ends the neutron; `n >= 2` banks `n - 1` copies as OpenMC does for integer yields. Weights stay exactly 1, so the exact integer balance (d) also holds on real data in Part 2. The mean multiplicity is exactly y(E). Only the variance differs from OpenMC. Recorded as a deviation. |
| D3 | Secondaries from integer yields | **Mirror OpenMC:** `y - 1` identical copies of the post-collision neutron. Means are unaffected, but the two (n,2n) neutrons are perfectly correlated, as in OpenMC. The alternative (sample each outgoing neutron independently) would be a deviation. |
| D4 | Level inelastic energy | **Compute from AWR and Q:** `E_cm = (A/(A+1))^2 (E - (A+1)/A |Q|)`, i.e. exact two-body kinematics. Test (c) then holds to round-off. Using the stored rounded parameters, as OpenMC does, would give about 5e-7 relative energy non-conservation. Recorded as a deviation (at most 4.7e-7 relative in threshold, 4e-8 in E_cm scale). |
| D5 | Energies below the data grid (1e-5 eV) | **Energy cutoff**, a problem input defaulting to the largest grid minimum of the nuclides used (1e-5 eV). As in OpenMC, it is applied after the collision, and a secondary born below it is not created. mcslab's lookup refuses off-grid energies by design (Phase 2a), so this replaces OpenMC's linear extrapolation below the grid. Recorded as a deviation. **Condition (approval): never drop silently.** The killed weight is tallied per region (`cutoff_weight [B, R]`), it is a separate term in (d)'s exact balance, and it is reported in every summary table. |
| D6 | Tests beyond your list | Add three: (a2) chi-square of the tabular-angle sampler on real data in Part 1, because that sampler lands in Part 1 (this is the tabular-angle slice of (f), moved forward); a reproducibility test for the new kernel (split or reordered batches are bit-identical, including histories with secondaries); and in Part 2 a free-gas equilibrium test (below). |
| D7 | Test (h) geometry | W 0.5 cm \| FLiBe 20 cm \| Fe 10 cm, 14.1 MeV beam at x = 0, natural Li, 20 batches x 5000. Adjust if you have a design in mind. |

## Design

**New modules.** Existing kernels are not modified, so the Phase 1 and
Phase 2a regressions stay byte-identical by construction.

- `mcslab/nucdata.py` (extended): reads the secondary distributions
  themselves (angle tables, level parameters, yields, applicability; in
  Part 2 also correlated and continuous tables), not only their
  descriptions. Following the Phase 2a rule, it **refuses** anything it
  does not implement, with a clear error:
  - discrete lines
  - more than one interpolation region, or a law other than histogram or
    lin-lin
  - Kalbach-Mann, N-body, evaporation, Maxwell or Watt
  - an unknown attribute

  Cross-section reading is unchanged, and `docs/data_inventory.md` must
  regenerate byte-identically.
- `mcslab/distributions.py` (new): packs the laws CSR-style into a float64
  pool plus an int64 index pool, with one header per reaction product.
  Njit samplers transcribe the OpenMC functions above, with citations.
- `mcslab/collision.py` (new, njit):
  - `rotate_angle`, as in OpenMC
  - nuclide, absorption and scatter-channel selection
  - `elastic_scatter`
  - `inelastic_scatter` (CM->lab conversion plus yield handling)
  - in Part 2, free gas
- `mcslab/transport_kin.py` (new, njit): the history loop with the
  secondary bank, tallies, counters, energy cutoff and (Part 2) track
  recording.
- `mcslab/config_kin.py` (new): the Python driver (`KinRunConfig`,
  `run_kin`, `KinResults`). Python objects stop here.
- `mcslab/synthetic.py` (new): builds synthetic nuclides (constant cross
  sections, chosen AWR, isotropic-CM elastic, synthetic levels and
  multiplying reactions, kT = 0) as the same dataclasses the reader
  returns. A `SyntheticLibrary` duck-types `nucdata.Library`, so synthetic
  nuclides go through exactly the same packing and kernels as real ones.

**Particle state.** The kernel carries `x`, a full direction `(u, v, w)`,
`E`, `t` and the region. The slab axis is x. The source sets
`(mu, sqrt(1 - mu^2), 0)` with no extra draw. The slab is symmetric about
the x axis and every collision is rotationally covariant, so the initial
azimuth has no effect on any tally. This lets `rotate_angle`, elastic
scattering and free gas be transcribed as 3D vector code, exactly as in
OpenMC.

**RNG consumption contract** (documented in the kernel docstring; mirrors
OpenMC's draw order):
- source: energy 0 or 1, direction 0 or 1
- flight: 1, skipped in void
- collision:
  - nuclide: 1
  - absorption: 1, only if `sigma_a > 0`
  - scatter channel: 1
  - then the reaction:
    - elastic at rest: 2 for the angle (1 if isotropic), plus 1 for phi
    - level: 2 for the angle, plus 1 for phi
    - Part 2 laws: as in OpenMC, plus 1 for a non-integer yield (D2)
- Part 2 free gas: `3 + [1] + 2` per rejection-loop iteration, plus 1 for
  phi

**Secondary bank.** A per-batch fixed-capacity array (default 10000, the
same as OpenMC's `max_secondaries`) holding `x`, `(u, v, w)`, `E`, `t`, the
region and the track-particle index. It is LIFO and continues the parent's
RNG state. Overflow is a counted error, and the driver raises; nothing is
dropped silently. Max draws are counted per **history family** (the primary
plus all its secondaries), recorded per batch and checked against STRIDE,
as in Phase 1.

**Tallies (new kernel only).**
- `region_sums [B, 4, R]` and `surface_sums [B, 2, R+1]`, with the Phase 1
  layout. Absorption is analog here: the weight at absorption events, not
  the expected value used in 2a.
- `spectrum [B, 2, R, G]`: track-length and collision-estimator flux in
  energy bins given as a problem input.
- `cutoff_weight [B, R]`: the weight killed by the energy cutoff, per
  region (D5 condition). This includes secondaries that would have been
  born below the cutoff, which are scored in the region of their birth
  and not banked. The driver warns when it is nonzero, and `KinResults`
  reports it next to absorption and leakage.
- `counts [B, K]` int64:
  - sources, secondaries created, absorbed, zero-yield kills (Part 2),
    leaked left, leaked right, energy cutoff
  - elastic events, level events, other-inelastic events
  - bank overflow, lost, max draws

  Each batch writes only its own rows; the batch-independence invariants
  are unchanged.

**Guards.**
- If round-off makes a cumulative selection loop fall through, the last
  channel with nonzero cross section is chosen. OpenMC would pick the last
  reaction evaluated, which can be below threshold. This is recorded as a
  (round-off-level) deviation.
- Nuclide selection reuses the per-nuclide `N_i sigma_t,i` products
  computed for `Sigma_t`, in the same order. The final cumulative sum
  therefore equals `Sigma_t` bit for bit, and nuclide selection cannot fall
  through.
- In Part 1, the driver refuses a problem in which a reaction with an
  unimplemented law is reachable, i.e. it has a nonzero cross section below
  the maximum source energy. Energies never increase in Part 1 (target at
  rest), so this check is exact.

## Part 1

**Steps / commits:**
1. Approved plan (this file).
2. Step 0: Phase 1 manifest provenance.
3. Reader: angle tables, level law, constant yields, applicability; refusal
   of everything else. Check that the inventory regenerates
   byte-identically.
4. `distributions.py`: packing plus tabular-angle and level samplers, with
   test (a2).
5. `collision.py`: selection, elastic (target at rest), level inelastic,
   integer-yield copies.
6. `transport_kin.py`, `config_kin.py`, `synthetic.py`: kernel, bank,
   tallies, counters, cutoff, STRIDE tracking; reproducibility test.
7. Tests (a) and (b).
8. Tests (c) and (d).
9. Docs: `docs/deviations_from_openmc.md`, README (scope, tests,
   limitations),
   plan status. Summary table, then **stop**.

**Acceptance criteria.**
- Statistical checks use 3 SE.
- Chi-square and Hotelling tests pass if p >= 0.0027. That is the same
  false-alarm rate as a two-sided 3-sigma check.
- "Exact" means an integer identity on raw sums (all weights are 1.0), or
  round-off with a stated bound.
- Every test asserts `lost == 0` and `max_draws < STRIDE`.
- Every transport test reports the cutoff-killed weight per source neutron
  in the pytest summary table (D5 condition), even when it is 0.
- Seeds were fixed in this plan before any test exists, so they can be
  audited (project conventions: no seed-shopping).

| test | what it checks | seed |
|---|---|---|
| **a** | Synthetic nuclide, isotropic-CM elastic, target at rest, A in {1, 12, 184}, one incident energy. Per A: a chi-square of E_out on 50 equal bins on [alpha E, E] (uniform); the mean of ln(E/E') vs xi = 1 + alpha ln(alpha)/(1 - alpha) at 3 SE (xi = 1 for A = 1); all E_out in [alpha E, E] to round-off. Calls the same njit `elastic_scatter` the kernel uses. | 20261010 |
| **a2** | Tabular-angle sampler on real data (Fe56 MT 2 at a tabulated incident energy and at a midpoint; one histogram-interpolated level table). Chi-square of mu_cm against bin probabilities integrated in the test from the raw HDF5 tables (h5py, not the packer). | 20261011 |
| **b** | Infinite non-absorbing medium: one region of a synthetic constant-cross-section scatterer, a slab so thick that **zero leakage is asserted exactly**, so the realised histories are identical to infinite-medium ones. Mono source at E0 = 1 MeV, energy cutoff well below the window. For A = 1 the window is all of [cutoff, E0), where F(E) = 1/E is exact. For A = 12 the window starts at E0 alpha^8: the Placzek transient is below 1e-3 after 3 intervals (checked numerically for this plan) and the test documents the bound. Expected per bin and per source neutron: collision estimator `ln(E_hi/E_lo)/xi`, track length `ln(E_hi/E_lo)/(xi Sigma_s)`, i.e. phi proportional to 1/(xi Sigma_s E). One Hotelling T^2 test per (A, estimator) over about 10 log bins, using the batch covariance so correlations between bins are handled. Exact: no absorption, no leakage, every history ends at the cutoff. | 20261012 |
| **c** | Discrete inelastic conservation, per event. Real levels (Fe56 MT 51-60, Li7 MT 51-82, F19 levels, W184 levels) at several incident energies, plus a synthetic level with isotropic CM. Recoil momentum `p_R = p_in - p_out` (m_n = 1, M = AWR); check `E_in + Q - E_out - p_R^2/(2 AWR)` <= 1e-12 E_in and report the worst value. Also: elastic events conserve kinetic energy by the same check with Q = 0. | 20261013 |
| **d** | Exact neutron balance in a finite multiplying slab (two regions plus a void). Synthetic nuclide with elastic, capture, an (n,2n) (yield 2) and an (n,3n) (yield 3), each with a level-type energy law and a Q-value. The energy cutoff is set so that some weight is really killed; a zero cutoff term would test nothing. Exact integer identities on raw sums: `sources + created == absorbed + leaked_left + leaked_right + cutoff_weight`, with absorbed from the absorption tally, leakage from the surface currents and cutoff from the per-region `cutoff_weight` tally (each also cross-checked against its int64 counter); `created == 1 x N(n,2n) + 2 x N(n,3n)`. | 20261014 |
| **e** | `tests/test_regression.py compare` and `tests/test_regression_ce.py compare`, byte-exact. | n/a |
| repro | New kernel: repeated, split and reordered batch runs are bit-identical, including histories with secondaries. | 20261015 |

What Part 1 does **not** cover: real-data transport beyond single-collision
unit tests (the Fe and W MT 5 channel is reachable from 1 keV, so real Fe or
W problems need Part 2), free gas, and any law other than tabular angle and
level.

## Part 2 (only after Part 1 is approved)

**Steps / commits:**
1. Reader and packer: correlated (law 61), continuous tabular (law 4),
   applicability mixtures, `Tabulated1D` yields.
2. Samplers, transcribing `ContinuousTabular::sample` and
   `CorrelatedAngleEnergy::sample_dist`; CM->lab for every CM law;
   non-integer yields (D2). The Part 1 driver guard is lifted once every
   law in the inventory is implemented.
3. Free gas: the cxs sampler, applied when `E < 400 kT` or `awr <= 1`,
   with kT from the file (`kTs/294K`) and resonance scattering off, as in
   OpenMC's defaults. Synthetic nuclides with kT = 0 keep the target at
   rest (this guard avoids OpenMC's division by kT).
   **Hydrogen cutoff re-run (added 2026-09-30, at Part 1 review).**
   - Once free gas is in, re-run the Part 1 illustration with identical
     inputs: pure H-1 at 0.0708 g/cm^3, slab [0, 30] cm, 1 MeV beam at
     x = 0, 10 batches x 1000 histories, seed 3, default energy cutoff
     (1e-5 eV).
   - Report the cutoff fraction before and after, with the full balance.
   - Before (Part 1, target at rest, commit `de952b1`): source 10000,
     absorbed 1013, leaked left 1421, leaked right 625, cutoff 6941
     (0.6941), residual 0.
   - Expected after: the cutoff becomes the sub-1e-5 eV tail of a 294 K
     thermal population, i.e. near zero, with absorption and leakage
     taking up the difference.
   - This is a diagnostic comparison, not a statistical pass/fail test,
     so the seed is simply the one used before. The result goes into the
     Part 2 summary, the README and `docs/deviations_from_openmc.md`.
4. Tests (f), (g) and free gas.
5. Regression reference (h): a new harness `tests/test_regression_kin.py`
   with its own `tests/reference_kin/`, pinning data sha256s like the CE
   harness. The Phase 1 and 2a references are untouched.
6. Track recording and its invariance test.
7. `scripts/animate_tracks.py`, GIF and PNG.
8. Docs. Summary table, then **stop**.

**Tests:**

| test | what it checks | seed |
|---|---|---|
| **f** | Each law on real data, calling the njit samplers directly, with expected bin probabilities integrated in the test from raw HDF5 (h5py). Level: E_cm deterministic, exact comparison. Law 4 (Li7 MT 16): chi-square on E_out and mu, at a tabulated incident energy and at a midpoint (mixture plus scaled interpolation computed in the test). Law 61 CM, lin-lin (Fe56 MT 91, W184 MT 16): joint (E_out, mu) chi-square at a tabulated incident energy, using the exact conditional rule (0.5/0.5 between the angle tables at k and k+1), plus the E_out marginal at a midpoint. Law 61 lab, histogram (Be9 MT 16). Applicability (F19 MT 16): the fraction of events per distribution at 3 SE plus E_out chi-square. The CM->lab transform is checked separately by the conservation check of (c) applied to two-body-consistent samples. | 20261020 |
| **g** | Neutrons per event, from the new kernel's counters in single-collision runs. MT 16, 17, 24, 25, 37, 41 are exact integers (2, 3, 2, 3, 4, 2). MT 5 (Fe56, W184) at several energies: support in {floor(y), ceil(y)} is exact, and the mean equals y(E) at 3 SE, with y interpolated in the test from raw HDF5. | 20261021 |
| free gas | **Corrected 2026-10-01** (before any free-gas code or test existed; see "Part 2 detailed plan", C). Synthetic isotropic-CM scatterer with a constant free-atom cross section sigma_f, A in {1, 12}, kT = 0.0253 eV. Incident energies are drawn from the collision density pi(E) ~ sigma_eff(E) E exp(-E/kT), where sigma_eff is the free-gas (Doppler-broadened) cross section of a constant sigma_f. By detailed balance of the cxs kernel, E_out after one collision must follow pi: chi-square, plus the mean of pi at 3 SE. Plus exact checks that the target is at rest at `E >= 400 kT` for A > 1, and that A <= 1 always gets free gas. *The approved text said "drawn from the Maxwellian flux spectrum E exp(-E/kT) ... E_out must follow the same spectrum". That is wrong: the cxs kernel conditioned on a collision preserves pi, not E exp(-E/kT). A scratch relaxation run found mean collision energies of 1.750 kT (A = 1) and 1.961 kT (A = 12), not 2 kT. Only the flux pi/sigma_eff is Maxwellian.* | 20261022 |
| **h** | Regression reference: W / FLiBe / Fe slab, 14.1 MeV beam (D7). Region sums, surface sums, spectrum and counts, byte-exact. Recorded with `--reason`. | 20261023 |
| tracks | The same problem with recording on vs off: every tally and count array (including max draws) is byte-identical. Recorded tracks are sane: time non-decreasing along each particle, x inside the slab, E constant between collisions. | 20261024 |

**Track recording.**
- Opt-in, for history ids below `n_track` (default 200).
- Each recorded history owns a fixed slot range in preallocated arrays, so
  no two batches (or future threads) share memory.
- Per event it stores: history, family particle index, event type (source,
  collision with MT, surface crossing, absorption, leak, cutoff), `x`, `E`,
  `t` and the region.
- `t` advances by `d / v(E)` with OpenMC's relativistic speed and
  constants. A secondary inherits its parent's time.
- The recording code never draws a random number and never touches a
  tally. The tracks test proves this.
- Overflowing a history's slots sets a truncation flag; physics is
  unaffected.

**`scripts/animate_tracks.py`.**
- Runs the (h) geometry with recording on.
- x = depth, with region boundaries shaded and labelled (W / FLiBe / Fe);
  y = log10 E.
- Frames at log-spaced times: flights take about 1e-11 s at 14 MeV, and
  thermal neutrons live about 1e-4 s. Each neutron is a dot with a fading
  trail.
- Saves `docs/figures/tracks.gif` (`PillowWriter`) and a static
  `docs/figures/tracks.png` of the same tracks.

## Part 2 detailed plan (approved 2026-10-01)

This section refines the Part 2 outline above for the Part 2 request
(requirements A-H). Where the two differ, this section applies.

**Approval (2026-10-01), with conditions:**
- **P1: yes.** Temperatures change only in the D7 problem definition.
  `ce_materials.flibe()` keeps its default temperature and density, and
  the Phase 2a regression must stay byte-exact.
- **P2, P3, P4, P6: yes,** as recommended.
- **P5: yes, and required.** The specification under C ("Free-gas kernel
  shape") is the approved one:
  - fixed incident energies, so a kernel that leaves E unchanged fails
  - A = 1 against the analytic free-gas kernel for hydrogen, derived and
    cited in the docstring
  - A = 12 against a numerically integrated kernel built independently of
    the sampler
  - one A = 12 incident energy above 400 kT, with an exact check that the
    target-at-rest path is taken there
- **If any statistical check fails during implementation:** stop and
  report it. Seeds and thresholds are not changed.

**Baseline at `2c1490b`** (re-checked 2026-10-01):
- 119 tests pass.
- Phase 1 and Phase 2a compares are byte-exact.
- The Part 1 hydrogen run reproduces: cutoff 6941 / 10000, residual 0.

### Part 1 follow-ups

**1. Are the (a) / (a2) chi-square references independent of the sampler's CDF?**
- **(a), 3 tests: yes.** The expected distribution is uniform on
  [alpha E, E], which is analytic. No table or CDF is involved.
- **(a2), 4 tests: the code is separate, the construction is not.**
  - The expected probabilities come from `table_cdf` / `mixture_cdf` in
    the test: Python on raw h5py arrays, not the packer and not the njit
    sampler.
  - But they rebuild the sampler's own inverse-CDF map: the mass per
    interval from the stored CDF column c, the shape inside an interval
    from p, and OpenMC's table-choice rule.
  - So a misreading of c shared by the sampler and the test would not be
    caught.
- **Proposed fix (P4).** Deterministic, no new statistical check, existing
  expectation unchanged:
  - (a2) also builds its bin probabilities from p alone: exact integration
    of the piecewise-linear or piecewise-constant PDF, normalised by its
    own integral, never reading c.
  - It asserts agreement with the current probabilities to <= 1e-6 per bin
    and reports the worst value.
  - In the real tables, stored c and the integral of p agree to 9.2e-7
    (mu) and 1.1e-7 (E_out). The synthetic histogram tables agree exactly.
- Part 2's (f) uses the p-only construction as its primary reference.

**2. Test (d): geometry, and the ~1.31 secondaries per source.**
- **Geometry.**
  - [0, 4] cm: material a, X at 0.1 /b-cm.
  - [4, 5] cm: void.
  - [5, 12] cm: material b, X 0.05 + Y 0.04 /b-cm.
- **Nuclides.**
  - X: A = 9. Elastic 2 b, capture 0.1 b, a level (Q = -1 MeV, 0.5 b),
    (n,2n) (Q = -2 MeV, 0.4 b, yield 2) and (n,3n) (Q = -4 MeV, 0.3 b,
    yield 3).
  - Y: A = 56. Elastic 3 b, capture 0.05 b, a level (Q = -0.8 MeV, 1 b).
- **Run.** 14.1 MeV beam at x = 0, and an isotropic plane source at
  x = 2; cutoff 100 keV; 20 x 5000 histories; seed 20261014.
- **Measured (re-run 2026-10-01).**
  - Created per source: 1.309 (beam), 1.199 (isotropic).
  - Multiplying events per source (beam): 0.625 (n,2n) and 0.342 (n,3n).
- **Not physical.**
  - The synthetic multiplying law is a level-type two-body law, and each
    of the y identical copies gets the full two-body energy.
  - (n,2n): mean lab energy 9.76 MeV per neutron, so the family carries
    19.5 MeV out of the E + Q = 12.1 MeV available. For (n,3n): 23.9 vs
    10.1 MeV.
  - Energy is created. The secondaries stay above the 2.2 / 4.4 MeV
    thresholds, and the flat cross sections make 21% of every collision
    above 4.4 MeV a multiplying one.
  - Real (n,2n) neutrons are emitted with soft spectra, mostly below
    threshold (Be-9: E_out <= 0.88 E_in).
- **Consequence.** (d) is fine as an integer-balance test. The
  `synthetic.py` docstring and the README will say so explicitly, so that
  nobody reads 1.31 as physical. D7 gives the first physical
  multiplication number.

### A. Law inventory

**Method.** An h5py scan (independent of `nucdata.py`) of all 13 W /
FLiBe / Fe nuclides plus H1, at 294 K. It covers every non-redundant
reaction with an outgoing neutron.

| law as stored | frame | reactions | status |
|---|---|---|---|
| uncorrelated: tabular angle (lin-lin), no energy law | CM | MT 2, all 14 nuclides | Part 1 |
| uncorrelated: tabular angle + `level` | CM | MT 51-90: 231 reactions; 220 lin-lin angle tables, 11 histogram (Li7 MT 72-82) | Part 1 |
| `correlated` (ACE law 61): one lin-lin incident region, E_out lin-lin, mu lin-lin, no discrete lines | CM | Fe54-58 MT 5, 16, 91; W180-186 MT 5, 16, 17, 28, 37, 41, 91 (47 reactions) | **Part 2** |
| `correlated`, E_out histogram, mu lin-lin | lab | Be9 MT 16; F19 MT 16, 22, 28, 91 (5 reactions) | **Part 2** |
| two `correlated` distributions + applicability (0.5 / 0.5, histogram, 10.99-20 MeV) | lab | F19 MT 16 | **Part 2** |
| uncorrelated: tabular angle (lin-lin) + `continuous` (ACE law 4), one lin-lin incident region, E_out lin-lin, no discrete lines | lab | Li6 MT 24; Li7 MT 16, 24, 25 | **Part 2** |
| yield, constant integer | | 1; 2 (MT 16, 24, 41); 3 (MT 17, 25); 4 (MT 37) | Part 1 |
| yield, `Tabulated1D` (one lin-lin region), y from 0 to 8.23 | | MT 5 of all Fe and W (9 reactions) | **Part 2** |

In total, 56 reactions use a Part 2 law.

**What the scan found.**
- Absent everywhere:
  - discrete lines; Kalbach-Mann, N-body, evaporation, Maxwell and Watt
    laws
  - multi-region or histogram incident-energy interpolation
  - repeated incident energies in law 4 / law 61 tables
  - zero-width E_out tables
- No extrapolation is needed. In every law 4 / law 61 reaction, the cross
  section becomes positive at or above the first tabulated incident energy.
- Every E_out table starts at E = 0 with c = 0, and ends with c = 1 and
  p = 0.
- Every `correlated` distribution carries an `applicability` dataset, even
  when it is the only distribution (e.g. Fe54 MT 5, which is 0 below
  5.5 MeV). OpenMC reads applicability only when there is more than one
  distribution (`src/reaction_product.cpp`), and so does mcslab.
- **MT 5 of Fe and W.**
  - The cross section is at most 2e-7 b below 5 MeV.
  - y = 0 below 5-6.5 MeV, so such an event ends the neutron.
  - OpenMC does the same: `wgt *= 0`, and `alive()` is `wgt != 0`
    (`include/openmc/particle_data.h`).
- Stored c vs the integral of p: within 1.1e-7 (E_out) and 9.2e-7 (mu).
- Photon products are ignored (neutron-only transport, as documented).

**Called out for Phase 3 tritium breeding.**
- **Be-9 (n,2n)** is MT 16, the only neutron-multiplying Be-9 reaction in
  the file (there is no MT 875-891).
  - sigma(14.1 MeV) = 0.484 b, 32% of sigma_t.
  - Lab frame, law 61 with histogram E_out and lin-lin mu. Its 24 incident
    energies run from 1.749 to 20 MeV. Yield 2.
  - The second neutron is an identical copy (D3).
  - This is FLiBe's neutron multiplier. Be-9's own tritium comes from
    MT 105 via MT 700/701 (0.0209 b, an absorption).
- **Li-7 (n,n'alpha)t** is MT 52-82: the levels above the bound 0.478 MeV
  state, MT 51.
  - Level law in CM, with two-body kinematics from each level's Q. This is
    already implemented and conservation-tested in Part 1 (test c).
  - Their sum, 0.301 b at 14.1 MeV, equals MT 205.
  - The HDF5 drops ENDF's LR = 33 breakup flag. Only neutrons are
    transported. Phase 3 must score MT 205, or MT 52-82 as a set.
  - Part 2 matters to it indirectly: Li7 MT 16, 24 and 25 (law 4) are what
    currently blocks FLiBe at fusion energies.
- Li-6 (n,t)alpha is MT 105, an absorption, already handled in Part 1.

**Deliverables.**
- `scripts/law_inventory.py` (h5py only) generates `docs/law_inventory.md`:
  per nuclide x MT, the law, frame, interpolation, yield, threshold and
  sigma(14.1 MeV).
- A test checks that every non-redundant neutron reaction of the 14
  nuclides packs with zero unsupported channels, matching the inventory
  row by row.

**Unimplemented laws (policy change).**
- Once free gas is in, energies are no longer monotone, so Part 1's
  reachability exemption is removed.
- `run_kin` raises `NotImplementedError` (listing them) if any nuclide in
  the problem has a channel with an `UnsupportedLaw`, reachable or not.
  The kernel's `RuntimeError` stays as a backstop.
- The reader still refuses outright:
  - unknown attributes or keys
  - interpolation codes other than 1 or 2
  - more than one interpolation region
- It still keeps the following as `UnsupportedLaw` (named explicitly):
  - discrete lines (`n_discrete_lines > 0`)
  - Kalbach-Mann, N-body, evaporation, Maxwell and Watt laws
- None of these occur in the 14 files, so nothing is refused for W,
  FLiBe, Fe or H.

### B. New laws: samplers and tests

**Transcriptions** of OpenMC v0.16.0, re-read 2026-10-01. Each is cited in
the code.

- **`CorrelatedAngleEnergy::sample_dist`** (`src/secondary_correlated.cpp`):
  - `get_energy_index`, then 1 draw to choose the incident table
  - 1 draw for E_out from the stored CDF, then the scaled interpolation
  - the angle table at k or k+1, whichever outgoing point is closer in CDF
    space (always k for histogram E_out; `c_k1` starts at +inf)
  - 1 draw for mu (`Tabular` sampling)
- **`ContinuousTabular::sample`** (`src/distribution_energy.cpp`):
  - its own bracketing: below the first energy i = 0, r = 0; above the
    last, i = n - 2, r = 1
  - a table-choice draw unless the incident interpolation is histogram
  - 1 draw for E_out, then the scaling
  - the angle is sampled first (`UncorrelatedAngleEnergy::sample`)
- **`ReactionProduct::sample_dist`** (`src/reaction_product.cpp`): 1 draw;
  the first distribution with `c <= cumulative applicability`. If the
  cumulative sum falls short, it returns the last distribution. This is
  mirrored as is, so no mcslab guard is needed.
- **`Tabulated1D::operator()`** (`src/endf.cpp`): constant outside the
  table, lin-lin or histogram inside.
- **Yields (D2).** After sampling and rotation (OpenMC's order), y is
  evaluated at E_in:
  - integer y >= 1: y - 1 identical copies, no draw
  - y = 0: the neutron ends, no draw
  - otherwise: 1 draw, n = floor(y) + [xi < y - floor(y)]
- **Known OpenMC hazard.** At an E_in above a correlated law's last
  incident energy, OpenMC indexes past the end of its tables (undefined
  behaviour). In these data the last incident energy equals the grid
  maximum, and mcslab's lookups refuse energies above it, so the case is
  unreachable. mcslab raises if it ever happens.
- **Kernel draw contract**, per event:
  - correlated: 3, plus 1 for phi; plus 1 if there are two distributions
    (applicability); plus 1 for a non-integer yield
  - law 4: 2 for the angle, 2 for the energy, 1 for phi

**Test (f), seed 20261020 (approved).**
- **Replay (exact).**
  - Coverage: all 56 reactions at 3 incident energies (the first tabulated
    energy above threshold, a tabulated energy near 14.1 MeV, and the
    midpoint of the pair bracketing 14.1 MeV), 2000 events each.
  - Each event's draws are replayed in plain Python on raw h5py arrays.
  - These must match exactly: the incident table chosen, the distribution
    chosen, the outgoing bin k, and the angle table.
  - |F_l(E_out unscaled) - xi_E| and |F(mu) - xi_mu| must be <= 1e-9; the
    worst values are reported.
  - CM -> lab is recomputed from (E_cm, mu_cm) to <= 1e-12 relative.
- **Statistical, reference built from p only.**
  - The reference uses:
    - exact integration of the piecewise PDFs
    - the angle-table split at the CDF midpoint of each interval
    - the scaling map applied analytically
    - mixture weights 1 - r and r
  - Each check is a joint (E_out, mu) chi-square in the law's frame: 8
    E_out bins (equiprobable under the marginal) x 5 mu bins, 200 000
    events.
  - Cases, each at a tabulated energy and at a midpoint: Fe56 MT 91 (CM,
    lin-lin), W184 MT 16 (CM, lin-lin), Be9 MT 16 (lab, histogram),
    F19 MT 16 (applicability mixture), Li7 MT 16 (law 4 + angle).
  - **10 checks.**
- The approved outline's "fraction per distribution at 3 SE" for F19 is
  replaced by the exact replay of the distribution choice.

**Test (g), seed 20261021 (approved): yields.**
- **Integer yields, exact, from kernel counters on real data.** Per
  channel, created == (y - 1) x events and zero-yield == 0, for MT 16, 17,
  24, 25, 37 and 41.
  - W uses a 30 MeV mono source, so that MT 37 and 41 are open.
  - FLiBe at 14.1 MeV covers Li MT 24 and 25.
- **MT 5 (Fe56, W184): direct calls of the njit `inelastic_scatter`, at 3
  energies each.**
  - The energies are chosen so that y lies in (0,1), (1,2) and (2,3), and
    one of them falls between points of the yield table.
  - Exact: n is floor(y) or ceil(y), and no extra draw is made when y is
    an integer.
  - Mean n = y(E) at 3 SE, with y from `np.interp` on the raw h5py table.
  - **6 checks.**
- This replaces the outline's "single-collision kernel runs" (P6). The
  coverage is the same, with less machinery.

### C. Free gas and temperature

**The rule, as in OpenMC v0.16.0.**
- Sources: `src/physics.cpp`, `sample_target_velocity` (line 891) and
  `sample_cxs_target_velocity`; the defaults are in `src/settings.cpp`.
- The target is at rest iff `E >= free_gas_threshold * kT` and
  `awr > 1`. The threshold defaults to 400.
- Otherwise the cxs sampler is used: 5 or 6 draws per rejection
  iteration, then 1 for the target direction.
- Resonance scattering is off (OpenMC's default).
- mu_cm is sampled at the lab energy.
- kT is the kT of the selected data temperature (`scatter`:
  `nuc->kTs_[i_temp]` for non-multipole data).
- H-1 (awr 0.99917) gets free gas at all energies.
- Free gas applies to elastic scattering only.
- kT = 0 synthetic nuclides keep the target at rest (deviation 8).

**Temperature is a per-material input** (K).
- The default is 293.6 K (OpenMC's `temperature_default`).
- The data temperature is chosen by OpenMC's "nearest" rule
  (`src/nuclide.cpp`):
  - available temperatures are round(kT / k_B) from `kTs`: 250, 294,
    600, 900, 1200 and 2500 K in every file
  - the nearest is taken, and refused if |dT| >= 10 K
    (`temperature_tolerance`)
- Each (nuclide, data temperature) pair is packed as its own entry, with
  its own grid, cross sections and kT. The grids differ per temperature.
- Phase 2a's `config_ce` keeps its single temperature and refuses a
  material that asks for a different one. This is a guard only and
  changes no results.

**Code.**
- `collision.sample_cxs_target_velocity`.
- An elastic dispatcher that calls Part 1's `elastic_scatter`, unchanged,
  whenever the target is at rest. Results for kT = 0, or E >= 400 kT, are
  therefore bit-identical to Part 1.
- A counter `K_FREE_GAS`.
- `KinRunConfig.free_gas_threshold = 400.0`.
- A diagnostic switch `free_gas=True` (P3). False reproduces Part 1; it
  is mcslab-only and documented.

**Free-gas test, seed 20261022 (approved; corrected above).**
- **Setup.** A synthetic isotropic-CM scatterer, A in {1, 12},
  kT = 0.0253 eV.
- **Per event.**
  - E_in is drawn from pi by inverse CDF, using the first draw of the
    history's stream.
  - One collision follows, through the same njit dispatcher the kernel
    calls.
- **Checks.**
  - A chi-square of E_out against pi on 20 equiprobable bins.
  - The mean of E_out against kT (2 - 1 / (2 (A + 1))) at 3 SE.
  - 200 000 events per A. **4 checks.**
- **Derived in the docstring.**
  - **Effective cross section.** A constant free-atom sigma_f in a
    Maxwellian gas gives
    sigma_eff(E) = sigma_f [(1 + 1/(2a^2)) erf(a) + exp(-a^2) / (a sqrt(pi))],
    with a^2 = A E / kT.
  - **Detailed balance.** phi_M(E) sigma_eff(E) P(E->E') is symmetric
    in E and E', with phi_M ~ E exp(-E/kT). So the stationary collision
    density is pi ~ sigma_eff(E) E exp(-E/kT).
  - **Mean of pi.** From the split into centre-of-mass and relative
    velocity: kT (2 - 1/(2(A+1))), i.e. 1.75 kT for A = 1 and
    1.9615 kT for A = 12.
  - **Flux.** pi / sigma_eff is Maxwellian (mean 2 kT) only if the
    tabulated cross section is sigma_eff. A constant tabulated cross
    section gives (sigma_eff / sigma) x Maxwellian instead.
- **Numerical bound.** The inverse-CDF table and the bin edges both come
  from the closed-form pi, with a stated bound on the CDF table error
  (<= 1e-9).
- **Scratch evidence (numpy, 2026-09-30).** Collision chains starting at
  1 eV reach pi within about 20 (A = 1) and 75 (A = 12) collisions. Their
  mean collision energies are 1.7500 and 1.9615 kT, and their
  flux-weighted mean is 2.00 kT.

**Free-gas kernel shape, seed 20261025 (new, P5; required, as approved).**
- E_out from a **fixed** E_in in {kT, 20 kT}, for A in {1, 12}. A kernel
  that left E unchanged would fail.
- **A = 1:** a chi-square against the analytic free-gas kernel for
  hydrogen (Wigner-Wilkins). Its CDF is closed-form, and the docstring
  derives and cites it.
- **A = 12:** a chi-square against a kernel integrated numerically in the
  test, independently of the sampler:
  - Average over the Maxwellian target velocity, weighted by the relative
    speed.
  - Given the two velocities, E' is uniform on
    [(|V_cm| - r)^2, (|V_cm| + r)^2] (isotropic CM; Archimedes' hat-box
    theorem), with r the neutron's CM speed.
- Deterministic self-checks of the references:
  - the same integrator reproduces the closed-form A = 1 CDF
  - its normalisation gives sigma_eff(E)
  - quadrature refinement changes the probabilities by less than the
    stated bound
- **At rest above 400 kT, exact.** One extra A = 12 incident energy,
  1000 kT. The dispatcher must take the target-at-rest path: results
  bit-identical to Part 1's `elastic_scatter` on the same stream, and the
  same draw count.
- **Why it is needed.** Stationarity under pi cannot detect a wrong kernel
  that still satisfies detailed balance; this test can.
- **4 checks** (plus the exact at-rest check).

**Dispatch and temperature, exact, seed 20261026 (new).**
- **Free-gas threshold (A = 12).** At exactly E = 400 kT the target is at
  rest; at the next float below, free gas is used.
- **A <= 1.** A = 1 and H-1 get free gas at 14 MeV.
- **At rest.** The result is bit-identical to Part 1's `elastic_scatter`,
  and the draw counts are as documented.
- **Kernel counter.** `K_FREE_GAS` equals the number of elastic events
  for pure H-1. It is 0 for A = 12 with an energy cutoff >= 400 kT.
- **Temperature selection.**
  - 293.6 K selects 294K, 900 K selects 900K, and 950 K is refused.
  - The packed xs and kT equal the raw h5py values.
  - The same nuclide at two temperatures in two materials gives two
    entries.

**Not proposed.**
- A transport-level Maxwellian check. It would need the synthetic
  nuclide's tabulated elastic cross section to be sigma_eff(E), not a
  constant, plus a way to end histories in a non-absorbing medium (e.g.
  OpenMC's time cutoff).
- A multi-collision convergence test from a 1 eV start, like the scratch
  run.

### D. Hydrogen before / after

- `scripts/hydrogen_cutoff.py` runs the README problem with identical
  inputs, twice:
  - with `free_gas=False`. It must reproduce Part 1 exactly, and this is
    asserted: 10000 / 1013 / 1421 / 625 / 6941, residual 0.
  - with free gas on.
- It reports both balances, the cutoff fraction, and max draws vs STRIDE.
- **Expected after:** of order 1e-4, i.e. a few neutrons in 10^4.
  - Estimate: one thermal H collision sends the neutron below 1e-5 eV
    with probability about 4e-6, from the H kernel:
    (4 / (3 sqrt(pi))) eps_c^(3/2) with eps_c = 4e-4.
  - A thermalised neutron makes about 80 collisions before capture.
- **The residual** is the sub-1e-5 eV tail of the thermal population.
  OpenMC would keep transporting those neutrons on extrapolated cross
  sections (deviation 2).
- The result goes into the README, `docs/deviations_from_openmc.md` and
  this file.

### E. Regression (h), seed 20261023 (approved)

- **Harness.** `tests/test_regression_kin.py` with `tests/reference_kin/`,
  using the CE harness's CLI and rules. The manifest pins the sha256 of
  the 13 data files.
- **Problem D7.**
  - W [0, 0.5] | FLiBe [0.5, 20.5] | Fe [20.5, 30.5] cm.
  - 14.1 MeV beam at x = 0, natural Li, 20 x 5000 histories.
  - At 14.1 MeV it reaches every Part 2 law and yield type.
- **Arrays.**
  - region_sums and surface_sums
  - counts
  - spectrum: 2 estimators x 3 regions x 40 log bins, 1e-5 eV to 20 MeV
  - cutoff_weight and zero_yield_weight
  - chan_events and chan_created
- **Recording.** Done once, after all the physics is in (laws, free gas,
  temperature). The commit message and the manifest reason say why.
- **After recording.** Track recording lands next, and the compare must
  stay byte-exact.
- **Other references.**
  - The Phase 1 and 2a compares run after every commit.
  - `docs/data_inventory.md` must regenerate byte-identically after the
    reader change.

### F. Track recording, seed 20261024 (approved)

- **Opt-in**, for histories with id below `n_track`.
- **Memory.** Each recorded history owns a fixed range of slots. No two
  batches share memory.
- **Per event:**
  - history, particle index in the family, parent index
  - event code: source, collision + MT, surface, absorption, leak,
    cutoff, zero-yield, banked or popped
  - x, u, E, t and the region
- **Isolation.** The recorder never draws a random number and never
  touches a tally. Overflowing a history's slots sets its truncation flag.
- **Test** (D7, 4 x 500):
  - Recording on vs off: every tally and count array is byte-identical,
    max draws included.
  - Sanity checks:
    - t is non-decreasing along each particle
    - x stays within the slab
    - E is constant between collisions
    - each particle in a recorded family ends exactly once
    - the number of particles in each family is 1 + created

### G. GIF

- `scripts/record_tracks.py` runs D7 with recording on and writes
  `outputs/tracks_d7.npz` (gitignored).
- `scripts/animate_tracks.py` reads only that file.
- **Plot.**
  - x = depth, with W / FLiBe / Fe shaded and labelled; y = log10 E.
  - Frames at log-spaced times from 1e-11 s to about 1e-3 s.
  - Each neutron is a dot with a fading trail.
  - At most 60 families are drawn.
- **Output.** `PillowWriter`, about 100 frames, target <= 3 MB (the size
  is printed), plus a static PNG.
- Python 3.9 compatible.

### H. Reporting, balance, statistical budget

- **Summary table:** the same conftest columns as Part 1.
  - The max_draws column is filled for every new kind of history family:
    D7 real-data families, H-1 with free gas, and the unit draws of the
    law and free-gas tests.
  - Zero-yield kills appear as a row of the balance.
- **Balance**, exact:
  `sources + created == absorbed + leak_left + leak_right + cutoff + zero_yield (+ lost)`.
  - New: a per-region tally `zero_yield_weight [B, R]` and a counter
    `K_ZERO_YIELD` (P2).
  - It is asserted on real data in (g) and on D7.
- **Statistical checks:** 50 now, plus 24 (f 10, g 6, free gas 4, kernel
  shape 4), for 74 in total. The chance of at least one chance failure in
  a full run is about 18%.

### Decisions (approved 2026-10-01; conditions at the top of this section)

| # | Question | Recommendation (approved) |
|---|---|---|
| P1 | D7 temperatures | W and Fe at 293.6 K. FLiBe at 900 K data, with the Janz density at 900 K (1.973 g/cm^3). This removes the 2a density / data-temperature mismatch for this problem and puts per-material temperature in the reference. Alternative: everything at 293.6 K with the 2a FLiBe (973 K density). |
| P2 | Zero-yield kills (MT 5 with y = 0, or n = 0 sampled) | A separate per-region tally, counter and balance term, not absorption. OpenMC scores the event as a scatter whose weight drops to 0. |
| P3 | `free_gas` switch | Yes. A diagnostic only; it lets the hydrogen script re-derive the "before" number exactly. |
| P4 | (a2) p-only agreement check | Yes. Deterministic, and no expectation changes. |
| P5 | Free-gas kernel-shape test, new seed 20261025 | Yes. |
| P6 | (g) via direct calls plus kernel counters | Yes. |

New seeds fixed here, before any test exists: 20261025 (kernel shape) and
20261026 (dispatch and temperature).

### Commits (on `phase-2b`, no attribution trailers)

1. This plan section and the free-gas correction, marked approved.
2. `scripts/law_inventory.py` and `docs/law_inventory.md`.
3. Reader: correlated, continuous tabular, applicability, `Tabulated1D`
   yields, strict refusal. `docs/data_inventory.md` regenerates
   byte-identically.
4. Packing, samplers and D2 yields; test (f); the (a2) p-only check.
5. Kernel: new laws wired in, zero-yield tally, balance, unconditional
   refusal of unsupported laws; test (g).
6. Per-material temperature, free gas, the switch and `K_FREE_GAS`; the
   free-gas tests.
7. Hydrogen script and its numbers.
8. Regression harness (h) and its reference, recorded with `--reason`.
9. Track recording and its test. The (h) compare stays byte-exact.
10. GIF scripts and figures.
11. Docs: deviations, README (including the synthetic (d) caveat), the
    `synthetic.py` docstring, plan status. Summary table, then
    **stop**.

**Later, after the user's history rewrite (not now).** A provenance-only
update, like Step 0:
- The regression manifests (`tests/reference/`, `tests/reference_ce/`,
  `tests/reference_kin/`) and this file record commit hashes.
- Re-record each manifest with a provenance reason, and check that every
  array is byte-identical.
- Restore the `.npz` files, so that only `manifest.json` changes. Old
  entries stay in each manifest's history.
- Then update the hashes cited in the docs from the rewrite's commit map.

## Deviations from OpenMC (initial list for `docs/deviations_from_openmc.md`)

Each entry will give OpenMC's behaviour (with file and function), mcslab's
behaviour, why, and the expected effect on Phase 4 comparisons.

1. No URR probability tables (OpenMC default: on). Affects Fe58 and W
   between keV and MeV. This is the known 2a limitation.
2. Level inelastic uses AWR and Q, not the rounded stored parameters (D4):
   at most 4.7e-7 relative.
3. Non-integer yields sampled as integers, not weight x yield (D2): same
   mean, different variance.
4. Energy cutoff at the grid minimum instead of extrapolating below the
   grid (D5).
5. One RNG stream per history. OpenMC uses separate source, tracking and
   URR streams with its own seeding. Results agree statistically, never
   bit for bit.
6. Round-off fall-through in channel selection picks the last nonzero
   channel.
7. 1D slab: 3D direction with a fixed initial azimuth (no physical effect;
   noted for completeness).
8. kT = 0 synthetic nuclides keep the target at rest (OpenMC has no such
   case).
9. Temperature: 294 K data only, no multipole or interpolation, the same as
   OpenMC's default "nearest" at 293.6 K. The FLiBe density-temperature
   mismatch from 2a remains.

Anything else found during implementation is added when found.

## Limitations to document

- No S(alpha,beta): none of these materials has thermal scattering data in
  ENDF/B-VIII.0 anyway.
- No photon transport.
- No variance reduction.
- 294 K data only.
- No URR tables.
- Serial execution.
- (n,2n)-type copies are fully correlated, as in OpenMC.
- The energy cutoff at 1e-5 eV.

## Status

- [x] Branch `phase-2b` created from `phase-2a` tip `525cff1`. `main`
      (`8e53f68`, Phase 1 only) not touched.
- [x] Baseline: Phase 1 and Phase 2a regression compares pass byte-exact.
- [x] Plan approved (2026-09-30). `main` fast-forwarded to `525cff1`.
- [x] Part 1 (2026-09-30):
  - [x] 2. Step 0, `5a3efd7`. Arrays and `reference.npz` byte-identical
        to before; only `manifest.json` changed.
  - [x] 3. Reader, `96ae0ad`. Inventory regenerates byte-identically.
  - [x] 4. Samplers and (a2), `6686331`.
  - [x] 5. Collision physics, `8818916`.
  - [x] 6. Kernel, driver, bank, synthetic nuclides, reproducibility
        test, `21a70f3`.
  - [x] 7. Tests (a), (b), `3b9e8ea`.
  - [x] 8. Tests (c), (d), `de952b1`. Adds the K_BORN_BELOW_CUTOFF
        counter.
  - [x] 9. Docs: `docs/deviations_from_openmc.md`, README,
        this status, and the hydrogen re-run added to Part 2.
  - Note: a network drop interrupted the session during step 9. The
    repository was audited before continuing: no lock or in-progress git
    operation, steps 1-8 intact (119 tests pass, both regressions
    byte-exact), and the uncommitted step-9 files were inspected and their
    untested claims re-run.
- [x] Part 2 detailed plan approved 2026-10-01, with conditions ("Part 2
      detailed plan" above). The free-gas test description was corrected
      the same day, before any free-gas code or test existed.
- [x] Part 2 implementation (2026-10-01):
  - [x] 1. Plan, approved, with the free-gas correction: `5b1ef53`.
  - [x] 2. Law inventory script and `docs/law_inventory.md`: `504fbd2`.
  - [x] 3. Reader for law 61, law 4, applicability and `Tabulated1D`
        yields: `dfa5d78`.
        - `docs/data_inventory.md` regenerates byte-identically.
        - Found: 1275 outgoing-energy tables repeat their last point, with
          zero mass after c = 1. They are accepted, and recorded in the
          inventory.
  - [x] 4. Samplers, D2 yields, test (f) and the (a2) p-only check:
        `8144fee`.
        - OpenMC's last-bin angle-table quirk (`c_k1 == c_k`) is mirrored.
        - The law-frame mu may exceed 1 by about 1e-7, as in OpenMC (2.6e-8
          observed, bound 1e-6).
        - Chi-square cells expecting < 5 events are merged by a fixed rule.
          Only Be9 MT 16 needed it, one cell per E row. The rule was added
          after the expected-count assertion failed, before Be9's
          statistic was ever computed. The other 8 cases were unchanged.
  - [x] 5. Kernel wiring, zero-yield tally (P2), unconditional refusal, and
        test (g): `39f732b`. The (g) kernel runs use a 1 MeV energy cutoff
        to stay short.
  - [x] 6. Free gas, per-material temperature, `free_gas` switch (P3) and
        free-gas tests: `2adf679`.
        - A = 1 stationarity: p = 0.007, which passes. A diagnostic with
          2e6 events on three other seeds gave p = 0.40, 0.70 and 0.17.
        - A = 12 kernel shape: bin edges come from a 401-point CDF grid.
          A first run with 4001 points also passed (p = 0.98 / 0.92, now
          0.94 / 0.92).
  - [x] 7. Hydrogen before/after: `071ca87`. The cutoff fraction goes from
        0.6941 to 0.0001.
  - [x] 8. Regression (h), D7 reference recorded once: `da6c762`.
  - [x] 9. Track recording and its test: `6172d94`. D7 stays byte-exact.
  - [x] 10. Track scripts, GIF and PNG: `82ee86d`.
  - [x] 11. Docs: deviations, README, this status.
  - No statistical check failed. Seeds and thresholds are as planned.
- [ ] Later, after the user's history rewrite: provenance-only update of
      the regression manifests and of the hashes in this file.
