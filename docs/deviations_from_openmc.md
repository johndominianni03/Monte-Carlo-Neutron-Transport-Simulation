# Where mcslab deviates from OpenMC

Phase 4 benchmarks mcslab against OpenMC, so mcslab mirrors OpenMC's
physics wherever it can. This file lists every place where it knowingly
does not.

The reference is **OpenMC v0.16.0** (tag `v0.16.0`, released 2026-08-05),
read from source. File paths are relative to the OpenMC repository. The
`openmc` package itself is never installed or imported.

Status: Phase 2b Part 1 (elastic and discrete-level kinematics, target at
rest). Entries marked *(Part 2)* describe decisions already approved for
code that does not exist yet.

## What is mirrored (for orientation)

These follow OpenMC's algorithm and random-number order:

- **Collision sampling.** In order: nuclide (`sample_nuclide`), analog
  absorption (`absorption`), then the scatter channel, elastic first and
  then the inelastic reactions in file order (`scatter`). All three are in
  `src/physics.cpp`.
- **Absorption set.** It is identical to OpenMC's `is_disappearance`
  (`src/endf.cpp`) on all 14 nuclides. Checked when the Phase 2b plan was
  written.
- **Elastic scattering** (`elastic_scatter`): CM angle sampled at the lab
  energy, vector transforms in units of `v = sqrt(E)`, and `rotate_angle`
  (`src/math_functions.cpp`).
- **CM->lab transform** for other reactions (`inelastic_scatter`).
- **Tabular angle sampling:** `AngleDistribution::sample` and
  `Tabular::sample_unbiased`, including `lower_bound_index` for repeated
  incident energies and normalisation by `c[n-1]`.
- **Multiplicity:** integer yields bank `y - 1` identical copies of the
  outgoing neutron (`inelastic_scatter`).
- **Secondary bank:** LIFO, and secondaries continue the parent's RNG
  stream (`src/particle.cpp` `event_check_limit_and_revive`).
- **Speed and time:** relativistic speed with CODATA 2018 constants
  (`Particle::speed`, `include/openmc/constants.h`).
- **Temperature:** 294 K data, the same as OpenMC's default "nearest"
  temperature for 293.6 K.

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
  - It is its own term in the exact neutron balance (test d).
  - It is shown in the pytest summary table (`cutoff/src`).
  - The driver issues `EnergyCutoffWarning` whenever it is nonzero.
  - Secondaries born below the cutoff are counted as created *and* as cut
    off. OpenMC does not count them at all. This is a bookkeeping
    difference only.
- **Effect on Phase 4:** none once thermal scattering exists (Part 2).
  There, sub-1e-5 eV neutrons are a negligible tail of a 294 K Maxwellian.
- **Part 1 caveat (large!):** Part 1 has no thermal treatment (item 3), so
  no collision can raise a neutron's energy. Neutrons slow down
  indefinitely instead of settling into a thermal population near kT. In
  a weakly absorbing medium most of them reach the cutoff.
  - **Example:** a 30 cm slab of pure H-1 at 0.0708 g/cm^3, 1 MeV beam,
    10 x 1000 histories, seed 3, loses 0.6941 of the source weight to the
    cutoff.
  - **Why hydrogen is so affected:** it removes on average a factor e per
    collision, so going from 1 eV to 1e-5 eV takes about 11.5
    collisions, each with at most a 1.4% capture probability.

  Part 1 results are therefore not physical below about 1 eV. The same
  problem is re-run after free gas is added (Part 2) and the cutoff
  fraction is reported before and after.

### 3. Target always at rest *(Part 1 only; Part 2 mirrors OpenMC)*

- **OpenMC** (`src/physics.cpp`, `sample_target_velocity`): free-gas target
  motion (the constant-cross-section sampler) when
  `E < free_gas_threshold * kT` (400 kT by default) or `awr <= 1`, so H-1
  always gets it.
- **mcslab Part 1:** the target is at rest at all energies.
- **Effect:** wrong physics at thermal and epithermal energies, and for H-1
  at all energies. It is removed in Part 2. Synthetic test nuclides have
  kT = 0 and will keep the target at rest even then, because OpenMC's
  sampler divides by kT.

### 4. Laws not implemented yet are refused *(Part 1 only)*

- **OpenMC** samples every law in these files.
- **mcslab Part 1** implements tabular angles, the level law and constant
  integer yields. `config_kin.run_kin` refuses a problem in which a
  reaction with another law can occur, i.e. whose cross section is
  positive below the maximum source energy. This check is exact because
  energies never increase in Part 1. The refused laws are correlated
  angle-energy, continuous tabular and energy-dependent yields. In
  practice every real Fe or W problem above 1 keV is refused (MT 5 is open
  from about 1 keV), as are Be, F and Li at fusion energies.
- **Effect:** none on results; the problems simply cannot run yet.

### 5. Non-integer yields *(Part 2; approved D2)*

- **OpenMC** (`inelastic_scatter`): for a non-integer yield `y`, the
  neutron's weight is multiplied by `y`.
- **mcslab (planned):** analog. `floor(y)` or `floor(y) + 1` neutrons, with
  mean exactly `y`, drawn with one extra random number. This keeps every
  weight at 1 and the integer balance exact. It concerns MT 5 of Fe and W
  (yield 0 to 8).
- **Effect:** same expected values; different variance.

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
- **Effect:** none for fusion problems, since sources are at or below
  14.1 MeV and the grids reach 20 or 150 MeV.

### 9. Unresolved-resonance probability tables are not used

- **OpenMC:** `urr_ptables_on = true` by default (`src/settings.cpp`).
- **mcslab:** smooth (infinite-dilution) cross sections in the unresolved
  range (a Phase 2a limitation).
- **Effect:** self-shielding differences for Fe58 (0.35-3 MeV) and W182-186
  (5-100 keV). For Phase 4, run OpenMC with `urr_ptables_on = False` for a
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
- **mcslab** has no such limit. Every history ends by absorption,
  leakage or the cutoff.
- **Effect:** none in practice. It would only matter for a history that
  never ends (not possible in Part 1, since energies fall monotonically
  to the cutoff).
