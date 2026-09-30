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
| `main` | `da7822a` | Phase 1 only |
| `phase-2a` | `397ff86` | Phase 1 + all of Phase 2a (strict descendant of `main`; merging is a fast-forward) |

No remote is configured, so a merge done elsewhere (e.g. on GitHub) is not
visible here. `phase-2b` has been created from `397ff86`, the tip of
`phase-2a`, which is exactly what `main` would be after the merge.

Resolved at approval: `main` was verified to be an ancestor of `phase-2a`
and fast-forwarded with `git branch -f main phase-2a`. `main`, `phase-2a`
and `phase-2b` now all start at `397ff86`.

Both regression compares pass on this base:
- Phase 1: 6 arrays, byte-exact.
- Phase 2a CE: 6 arrays, byte-exact, 13 data files with matching sha256.

**Step 0: Phase 1 manifest provenance.** `tests/reference/manifest.json`
names commit `91c1864`. That is the pre-rewrite hash of what is now
`c33fd6d` ("Add analytic physics validation suite"). The object still
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
| free gas | Synthetic constant-cross-section, isotropic-CM scatterer, A in {1, 12}, kT = 0.0253 eV. Incident energies are drawn from the Maxwellian flux spectrum E exp(-E/kT). By detailed balance of the cxs kernel, E_out after one collision must follow the same spectrum: chi-square. Plus exact checks that the target is at rest at `E >= 400 kT` for A > 1, and that A <= 1 always gets free gas. | 20261022 |
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

- [x] Branch `phase-2b` created from `phase-2a` tip `397ff86`. `main`
      (`da7822a`, Phase 1 only) not touched.
- [x] Baseline: Phase 1 and Phase 2a regression compares pass byte-exact.
- [x] Plan approved (2026-09-30). `main` fast-forwarded to `397ff86`.
- [ ] Part 1 (commits 2-9 above).
- [ ] Part 2.
