# Where mcslab deviates from OpenMC

Phase 4 benchmarks mcslab against OpenMC, so mcslab mirrors OpenMC's
physics wherever it can. This file lists every place where it knowingly
does not.

The reference is **OpenMC v0.16.0** (tag `v0.16.0`, released 2026-08-05),
read from source. File paths are relative to the OpenMC repository. The
`openmc` package itself is never installed or imported.

Status: Phase 2b Part 2. Every secondary law in the W / FLiBe / Fe data is
implemented, plus free-gas target motion and per-material temperature.

## What is mirrored (for orientation)

These follow OpenMC's algorithm and random-number order:

- **Collision sampling.** In order: nuclide (`sample_nuclide`), analog
  absorption (`absorption`), then the scatter channel, elastic first and
  then the inelastic reactions in file order (`scatter`). All three are in
  `src/physics.cpp`.
- **Absorption set.** It is identical to OpenMC's `is_disappearance`
  (`src/endf.cpp`) on all 14 nuclides. Checked when the Phase 2b plan was
  written.
- **Elastic scattering** (`elastic_scatter`): target velocity first (free
  gas, below), then the CM angle sampled at the lab energy, vector
  transforms in units of `v = sqrt(E)`, and `rotate_angle`
  (`src/math_functions.cpp`).
- **CM->lab transform** for other reactions, then the mu clamp and the
  rotation (`inelastic_scatter`).
- **Tabular angle sampling:** `AngleDistribution::sample` and
  `Tabular::sample_unbiased`, including `lower_bound_index` for repeated
  incident energies and normalisation by `c[n-1]`.
- **Correlated angle-energy (ACE law 61):**
  `CorrelatedAngleEnergy::sample_dist` (`src/secondary_correlated.cpp`).
  Mirrored as is:
  - The stored outgoing-energy CDF is used without normalising.
  - The angle table is chosen at the closer outgoing point, in CDF
    space.
  - A quirk is kept: when the search runs into the last bin, `c_k1`
    equals `c_k`, so that bin always takes angle table k+1.
  - The law-frame mu is not clamped. It can exceed +-1 by up to about
    1e-7 where a stored mu CDF exceeds its PDF's integral, and is clamped
    after the CM->lab transform, as in OpenMC.
- **Continuous tabular (ACE law 4):** `ContinuousTabular::sample`
  (`src/distribution_energy.cpp`). This includes its own incident-energy
  bracketing and its guard for zero-width bins.
- **Applicability mixtures:** `ReactionProduct::sample_dist`
  (`src/reaction_product.cpp`). One draw, the first law with
  `c <= cumulative applicability`, and the last law if the sum falls
  short. Applicability is read only when there is more than one law.
- **Yields:** `Tabulated1D::operator()` (`src/endf.cpp`), constant outside
  the table. Integer yields bank `y - 1` identical copies of the outgoing
  neutron (`inelastic_scatter`).
- **Secondary bank:** LIFO, and secondaries continue the parent's RNG
  stream (`src/particle.cpp` `event_check_limit_and_revive`).
- **Speed and time:** relativistic speed with CODATA 2018 constants
  (`Particle::speed`, `include/openmc/constants.h`).
- **Free-gas target motion (the rule).** mcslab uses OpenMC's own rule
  (`src/physics.cpp` `sample_target_velocity`, line 891, and
  `sample_cxs_target_velocity`; defaults in `src/settings.cpp`):
  - The target is **at rest iff `E >= free_gas_threshold * kT` and
    `awr > 1`**. `free_gas_threshold` is 400 by default and is an mcslab
    input with the same name and default.
  - Otherwise the target velocity comes from the constant-cross-section
    (cxs) sampler. Its rejection loop draws 3 + [1] + 2 numbers per
    iteration, then 1 for the target direction.
  - **H-1 (awr 0.99917) therefore gets free gas at every energy.**
  - kT is that of the nuclide's selected data temperature
    (`nuc->kTs_[i_temp]` in `scatter`, for non-multipole data).
  - Resonance scattering (DBRC / RVS) is off, as by default.
  - Inelastic reactions always see a target at rest.
- **Temperature selection:** each material has a temperature, defaulting
  to OpenMC's `temperature_default` of 293.6 K. The data temperature is
  OpenMC's `NEAREST` choice (`src/nuclide.cpp`):
  - the available temperatures are round(kT / k_B) of the `kTs` datasets,
    with k_B = 8.617333262e-5 eV/K
  - the nearest is taken, ties going to the lower temperature
  - it is refused unless it is within `temperature_tolerance` (10 K)

  The data hold 250, 294, 600, 900, 1200 and 2500 K.
- **Temperature for the 2a driver:** the Phase 2a driver (first-collision
  kernel) keeps one temperature, "294K", and refuses materials that would
  select other data.

## Deviations

### 1. Discrete-level kinematics use AWR and Q, not the stored parameters (approved D4)

- **OpenMC** (`src/distribution_energy.cpp`, `LevelInelastic::sample`):
  `E_cm = mass_ratio (E - threshold)`, using the `mass_ratio` and
  `threshold` attributes stored in the HDF5 file.
- **mcslab** (`mcslab/distributions.py`, `level_cm_energy`):
  `E_cm = (A/(A+1))^2 (E - thr)` with `thr = -Q (A+1)/A`, from the file's
  AWR and the reaction's Q.
- **Why:** the stored values are rounded (6-7 significant digits). Against
  the AWR/Q values they differ by up to 4.0e-8 relative in `mass_ratio`
  (Fe56 MT 51) and 4.7e-7 in `threshold` (Li7 MT 65). With them, energy is
  not conserved per event beyond about 5e-7. mcslab conserves energy and
  momentum to round-off (test c: at most 6.7e-16).
- **Effect on Phase 4:** outgoing energies differ by at most about 5e-7
  relative, far below any statistical resolution. Within about 5e-7 of a
  level's threshold, the two codes disagree on whether the reaction is
  open at all, but the cross section is zero there.

### 2. Energy cutoff at the bottom of the data grid (approved D5)

- **OpenMC:** the neutron energy cutoff defaults to 0 (`src/settings.cpp`
  `energy_cutoff`). Below the lowest grid energy, cross sections are
  extrapolated linearly from the first two points (`src/nuclide.cpp`,
  `calculate_xs`).
- **mcslab:** the lookup refuses energies off the grid (a Phase 2a design
  rule). So the kinematic kernel kills a neutron whose post-collision
  energy is below `energy_cutoff`. This defaults to the largest grid
  minimum of the nuclides used: 1e-5 eV for ENDF/B-VIII.0. As in OpenMC's
  `create_secondary`, a secondary born below the cutoff is not banked.
- **Never silent** (approval condition):
  - The killed weight is tallied per region (`cutoff_weight`).
  - It is its own term in the exact neutron balance (tests d and g).
  - It is shown in the pytest summary table (`cutoff/src`).
  - The driver issues `EnergyCutoffWarning` whenever it is nonzero.
  - Secondaries born below the cutoff are counted as created *and* as cut
    off. OpenMC does not count them at all. This is a bookkeeping
    difference only.
- **Effect, now that free gas is in:** what still reaches the cutoff is
  the sub-1e-5 eV tail of the thermal population. OpenMC would go on
  transporting those neutrons on extrapolated cross sections.
  - **Hydrogen illustration** (`scripts/hydrogen_cutoff.py`): pure H-1 at
    0.0708 g/cm^3, a 30 cm slab, 1 MeV beam, 10 x 1000 histories, seed 3.
    The cutoff fraction falls from **0.6941** (Part 1, target at rest) to
    **0.0001** (1 neutron in 10000) with free gas. The balance is exact
    in both runs:

    | run | absorbed | leaked left | leaked right | cutoff |
    |---|---|---|---|---|
    | target at rest | 1013 | 1421 | 625 | 6941 |
    | free gas | 5748 | 3006 | 1245 | 1 |

    Expected order: one thermal H collision lands below 1e-5 eV with
    probability about 4e-6, times about 80 collisions per thermalised
    neutron.
  - **Problem D7** (W / FLiBe / Fe, 14.1 MeV, 1e5 histories): 0 cut off.
- **Effect on Phase 4:** about 1e-4 of the source in a hydrogenous
  moderator, and none in the fusion problems so far.

### 3. Target motion: two additions to OpenMC's rule

The rule itself is OpenMC's (see "What is mirrored"). mcslab adds:

- **kT = 0 keeps the target at rest.** Synthetic test nuclides have
  kT = 0 by default. OpenMC has no such case: its sampler divides by kT.
- **`free_gas=False` switch.** A diagnostic that disables target motion
  everywhere. It reproduces Part 1 exactly, and `scripts/hydrogen_cutoff.py`
  uses it. The default is on.

**Effect:** none on real-data runs with the defaults.

### 4. Unimplemented laws refuse the whole problem

- **OpenMC** samples every law it can read.
- **mcslab** implements every law in the 14 files: tabular and isotropic
  angles, level, ACE laws 4 and 61, applicability, constant and
  `Tabulated1D` yields (see `docs/law_inventory.md`).
- **What else is refused:**
  - These are kept as `UnsupportedLaw`:
    - discrete lines
    - histogram incident-energy interpolation
    - Kalbach-Mann, N-body, evaporation, Maxwell and Watt laws
  - Polynomial yields of degree > 0 are refused when the problem is
    packed.
  - `run_kin` raises `NotImplementedError` if any nuclide of the problem
    has such a channel, whether or not the source can reach it. With
    target motion, energies are not monotone, so reachability cannot be
    bounded.
- **Effect:** none for these data. Other libraries (e.g. FENDL-3.2) may be
  refused until their laws are added.

### 5. Non-integer yields and zero-yield events (approved D2, P2)

- **OpenMC** (`inelastic_scatter`): a non-integer or zero yield `y`
  multiplies the neutron's weight by `y`. With `y = 0` the weight becomes
  0, and the particle is no longer `alive()`
  (`include/openmc/particle_data.h`). It is scored as a scatter, not an
  absorption.
- **mcslab:** analog. An integer `y` (including 0) gives exactly `y`
  neutrons with no draw. Otherwise one draw gives `floor(y)` or
  `floor(y) + 1`, with mean exactly `y`. Weights stay 1, and the integer
  balance stays exact.
  - This concerns MT 5 of Fe and W (y = 0 below 5-6.5 MeV, up to 8.23).
  - A multiplicity of 0 ends the neutron. Its weight is tallied per
    region in `zero_yield_weight` (also `K_ZERO_YIELD`, `chan_zero`).
  - It is a separate balance term: not absorption, not cutoff.
- **Effect:**
  - Expected values are the same; the variance differs.
  - Neither code counts zero-yield events as absorption.
  - In D7 they end 0.0046 neutrons per source.

### 6. Random-number streams

- **OpenMC** keeps separate streams per particle (source, tracking, URR,
  ...), each seeded by its own scheme.
- **mcslab** uses one stream per history (OpenMC's LCG and stride,
  `mcslab/rng.py`), and secondaries continue it.
- **Effect:** results agree statistically, never bit for bit.

### 7. Round-off guards in channel selection

- **Nuclide selection.** mcslab never picks a nuclide whose cross section
  is 0. OpenMC can, but only when the random number is exactly 0.
  mcslab's cumulative sum is also bit-identical to `Sigma_t`, so the
  selection cannot fall through.
- **Scatter channel.** If the running sum of the partials falls short of
  `prn * (sigma_t - sigma_a)` through round-off (partials and the
  pre-summed total are interpolated separately), OpenMC uses the last
  reaction it examined. That reaction can have zero cross section (below
  threshold). mcslab takes the last channel with a positive cross section.
- **Effect:** only on events of probability of order 1e-16.

### 8. Energies above the data grid are refused

- **OpenMC** clamps the grid index and extrapolates.
- **mcslab** refuses source energies outside the common data range of the
  problem's nuclides. A lookup off the grid raises.
- **Same cause, two law-61 cases:**
  - Above a law-61 table's last incident energy, OpenMC reads past its
    tables (undefined behaviour); mcslab raises.
  - A zero-width law-61 outgoing bin makes OpenMC divide by zero; mcslab
    raises.

  In these data both are unreachable. The last incident energy is the
  grid maximum. The 1275 zero-width pairs (all W law-61 tables, one Li6
  MT 24 table) sit after the CDF has reached 1, where no draw can land.
- **Effect:** none for fusion problems, since sources are at or below
  14.1 MeV and the grids reach 20 or 150 MeV.

### 9. Unresolved-resonance probability tables are not used

- **OpenMC:** `urr_ptables_on = true` by default (`src/settings.cpp`). In
  the URR, `elastic_scatter` then also skips target motion.
- **mcslab:** smooth (infinite-dilution) cross sections in the unresolved
  range (a Phase 2a limitation).
- **Effect:**
  - Self-shielding differences for Fe58 (0.35-3 MeV) and W182-186
    (5-100 keV).
  - Target motion is not affected, because the URR lies far above
    400 kT.
  - For Phase 4, run OpenMC with `urr_ptables_on = False` for a
    like-for-like comparison, or quantify the difference.

### 10. Slab geometry with a 3D direction

- **mcslab** carries a full direction `(u, v, w)`, with `u` along the slab
  normal, and starts sources at azimuth 0 (`(mu, sqrt(1-mu^2), 0)`), with
  no extra random number.
- **Effect:** none. The slab is symmetric about its normal and every
  collision is rotationally covariant. Noted only because the random
  number sequence differs from a 3D code that samples a source azimuth.

### 11. No event limit per particle

- **OpenMC** kills a particle after `max_particle_events` (1e6) events,
  with a warning.
- **mcslab** has no such limit. Every history ends by absorption, leakage,
  the energy cutoff or a zero-yield event.
- **Effect:** none in absorbing media. With free gas, a neutron in a large
  purely scattering medium could scatter very long without ending.
  OpenMC would stop it at 1e6 events. mcslab would keep going, and its
  draw count would exceed STRIDE, which the driver warns about. Every
  material used so far absorbs.

### 12. Temperature: no interpolation and no multipole

- **OpenMC** offers `temperature_method = "interpolation"` and windowed
  multipole data. Neither is the default.
- **mcslab** implements the default (`NEAREST`, 10 K tolerance) only.
- **FLiBe density:** `ce_materials.flibe()` still takes its density from
  a temperature (973 K by default). That parameter does not set the data
  temperature. Problem D7 uses 900 K for both. A Phase 2a run of the
  default FLiBe pairs the 973 K density with 294 K data, as before.
- **Effect:** none at the default settings. A material temperature more
  than 10 K from a data temperature is refused, as in OpenMC.
