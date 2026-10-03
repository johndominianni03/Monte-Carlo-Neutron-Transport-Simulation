# Phase 4 plan: benchmark against OpenMC 0.16.0 (approved 2026-10-03)

This file records the approved Phase 4 plan so the work survives a session
reset. The "Status" section at the bottom is updated as commits land.

**Goal.** Show with pre-declared statistical tests whether mcslab and OpenMC
agree on the same problems with the same nuclear data, and explain any
difference that is real.

**Approval (2026-10-03):**
- D23 to D37, D39, D40: accepted as recommended (energy cutoff matched at
  1e-5 eV, OpenMC on one thread, P4 included, default-settings diagnostics
  run).
- D38, changed before any benchmark run: P3's Fe tritium moves from the
  primary checks to the diagnostics. About 70 contributing histories over
  100 batches is too few for the 3 SE rule, which assumes near-normal batch
  means. The rest of P3's Fe layer stays primary and is flagged as the
  weakest set of checks. The primary count is 82.
- D41: the running count of statistical checks is kept up to date (240).

**Amendments (2026-10-03):**
1. Every tracked file and commit message of this phase is written as a
   project document.
2. No absolute local paths or user names in any committed file (results
   JSON, embedded model XML, problem file). The 14-entry cross-section
   index lives in the run directory outside the repository and is passed
   to OpenMC through `OPENMC_CROSS_SECTIONS`, so its path never enters
   `model.xml` or the results.
3. Failure protocol, completed before any run (see "Failure protocol").
4. Deviation 6 states what mcslab's generator is and that OpenMC 0.16.0
   uses a different one. Which OpenMC versions used the older generator is
   not stated, since the repository history of OpenMC was not checked.
5. The README benchmark section reports the default-settings diagnostics
   (D37) as numbers: the change caused by probability tables and by the
   energy cutoff, in SE and in percent.
6. The work ends with a status block like Phase 3's, plus a list of every
   tracked file that records a git commit hash or plan hash.

Throughout, the project's 3 SE rule and seed policy apply: the seeds,
thresholds and checks below were fixed before any benchmark run, and a
failing check is reported, never fixed by changing them.

## Findings (read-only investigation, 2026-10-03)

### Data and the cross-section index

- The library is the ENDF/B-VIII.0 HDF5 set fetched by
  `scripts/fetch_data.sh`, located through `MCSLAB_DATA`.
- All 14 nuclide files and the library's `cross_sections.xml` match
  `scripts/checksums/endfb-viii.0.sha256`. The 13 non-H files also match the
  pins of all four data-pinning manifests (`tests/reference_ce`,
  `tests/reference_kin`, `tests/reference_tally/kin_d7`,
  `tests/reference_tally/kin_d7_reflect`). H-1 is pinned only by
  `scripts/checksums`.
- The library's `cross_sections.xml` indexes the whole library: 690 entries
  (556 neutron, 100 photon, 34 thermal), of which only the 14 nuclide files
  are present. OpenMC opens only the files its materials use, so the index
  works, but it does not make "these 14 files and nothing else"
  structural. The OpenMC runner therefore writes a 14-entry index
  (`openmc.data.DataLibrary.register_file`) into its run directory, after
  checking each file's sha256 against the pins (D24).

### OpenMC installation

- `openmc --version`: 0.16.0, commit
  `617d35a5063c57796b43428bc401e627d2011046`. This is the commit of the
  v0.16.0 tag tarball (`git get-tar-commit-id`; tarball sha256
  `cfdf10f8dc66e652668dbc94501d5ed99b3dfecabc11cf91989a29ba7984654f`, the one
  read in Phase 3), so the source read is the code that runs.
- Release build, Clang 21.1.8, no MPI, "Strict FP: no". The executable is an
  x86_64 (osx-64) build running under Rosetta on an Apple A18 Pro (6
  cores), in its own conda environment with Python 3.13.15, numpy 2.5.3 and
  h5py 3.16.0. The environment's `bin/` must be on `PATH` for
  `openmc.run`; the runner calls the executable itself.

### Smoke runs (scratch directory, D7 with every planned tally)

| run | result |
|---|---|
| source on the x = 0 plane, no cell on the x < 0 side | fatal: "Exceeded maximum number of source rejections per sample" |
| with a void buffer cell on x < 0 | runs, 0 lost |
| 10 x 10,000 histories, 1 thread | 9.6-10.2 s transport + 1.7-2.0 s data loading |
| same, 6 threads | 3.7-5.3 s (2.6 times faster) |
| P3 geometry, 10 x 2000, 1 thread | 3.1 s |
| P4 hydrogen slab, 10 x 10,000, 1 thread | 5.1 s |

- Two 6-thread runs differ in the last bits (up to 1e-13 relative): threads
  add into shared tally sums in a varying order. Two 1-thread runs are bit
  identical.
- The atom densities OpenMC used (read back from `summary.h5`) equal
  mcslab's D7 densities bit for bit for all 13 nuclides.
- The statepoint labels H3-production `(n,Xt)` and He4-production `(n,Xa)`.
  Every volume tally used the track-length estimator; the surface current
  is analog.
- At a reflective x = 0 the surface current is exactly 0.0: OpenMC scores
  each reflection twice, -w and then +w.
- **Disclosure.** The smoke runs (seed 99, 1e5 histories, probability tables
  off, energy cutoff 0) showed, among other numbers, FLiBe tritium per
  source of 0.1721 (Li-6) and 0.1179 (Li-7) and leakage of 0.559 (left)
  and 0.376 (right), informally in line with Phase 3's D7. They were seen
  before the rules below were fixed. No setting was chosen because of
  them, and seed 99 (and the mcslab timing seeds 777 and 888) are retired.

### Corrections to Phase 3 statements

1. **OpenMC's CollisionFilter bin 0 is not mcslab's uncollided estimator.**
   Phase 3 stated that mcslab's TL_UNC equals OpenMC's `CollisionFilter`
   bin 0 with a track-length estimator (and COLL_UNC bin 1 with a collision
   estimator), because secondaries inherit `n_collision`. In 0.16.0:
   - `Particle::create_secondary` (`src/particle.cpp`) never sets the
     bank site's `n_collision`, and `SourceSite::n_collision` defaults to
     0 (`include/openmc/particle_data.h`).
   - Only `Particle::split` copies it; that is the line Phase 3 cited.
   - Every (n,2n) or (n,3n) copy therefore starts its life "uncollided",
     and its first collision lands in bin 1.

   In the smoke run, OpenMC's bin 0 exceeded the first-flight formula by
   9-12% per bin in W, 13% in FLiBe and 80% in Fe; the W front bin even
   exceeded its own width (0.0539 cm against 0.05 cm). Restricting bin 0 to
   E = 14.1 MeV exactly (only source neutrons that never collided have that
   energy) restores agreement with the formula. Per layer: 0.4604 against
   0.4599 (W), 5.763 against 5.742 (FLiBe), 0.2156 against 0.2186 (Fe);
   largest per-bin |z| 2.5 of 50, with 10 batches. mcslab's estimators count
   primaries only, as designed and validated in Phase 3; only the claimed
   OpenMC equivalence was wrong. The statement is corrected in the README,
   `docs/deviations_from_openmc.md` and a comment in `mcslab/tallies.py`.
   `docs/phase3_plan.md` stays as the historical record; this section is
   its erratum.
2. **The random-number generators differ.** Deviation 6 said mcslab uses
   "OpenMC's LCG". OpenMC 0.16.0 uses PCG-RXS-M-XS (`src/random_lcg.cpp`,
   `prn`): a 64-bit LCG (multiplier 6364136223846793005, increment
   1442695040888963407) followed by an output permutation. mcslab uses the
   63-bit LCG s' = (2806196910506780709 s + 1) mod 2^63. The two codes can
   therefore never replay each other's random numbers, whatever the seeds.
3. **Mesh slivers (new, round-off).** OpenMC's mesh puts slivers of about
   4e-17 of a track into the neighbouring layer's boundary bin (for
   example 5.4e-19 tritons per source in W bin 9, from FLiBe tracks). This
   extends deviation 14. Exact-zero checks therefore use per-layer cell
   tallies, which see no slivers.

### Settings for a like-for-like run

Defaults from `src/settings.cpp` unless another file is named.

| setting | OpenMC 0.16.0 default | benchmark |
|---|---|---|
| run mode | must be given | fixed source |
| photon transport | off | off, explicit |
| URR probability tables (`ptables`) | **on** | **off** (mcslab uses smooth URR cross sections, deviation 9) |
| temperature method, tolerance, default | nearest, 10 K, 293.6 K | same, explicit |
| where the temperature comes from | the cell's if set, else the material's (`src/geometry_aux.cpp` `assign_temperatures`, `src/material.cpp` `Material::temperature`) | material temperatures only: W, Fe, H 293.6 K (294K data), FLiBe 900 K (900K data) |
| free-gas kT | the selected data temperature's `kTs` value (`src/physics.cpp`, `scatter`) | the same value mcslab uses |
| free-gas threshold | 400 kT, target at rest above it if A > 1 (`sample_target_velocity`) | 400, explicit |
| elastic target-velocity method | constant-cross-section sampler; DBRC/RVS only with resonance scattering | default |
| resonance scattering | off | off |
| S(alpha,beta) | none unless added to a material | none added |
| survival biasing | off | off, explicit |
| weight cutoff / survival weight | 0.25 / 1.0, used only with survival biasing (`src/physics_common.cpp`, `apply_russian_roulette` returns at once otherwise) | no roulette at all |
| weight windows | off | off |
| neutron energy cutoff | 0 eV | **1e-5 eV** (D28) |
| non-integer yield | weight times y; an integer y banks y - 1 copies (`src/physics.cpp`, `inelastic_scatter`) | inherent (deviation 5) |
| zero yield | weight times 0 ends the particle (`alive()` is wgt != 0) | inherent |
| max events / secondaries / lost particles | 1e6 / 10000 / 10 | defaults; lost must be 0 |
| shared secondary bank, event-based mode | off | off |
| threads | all cores | 1 (D30) |

### Remaining known differences

| difference | changes expected values? | in this benchmark |
|---|---|---|
| Dev. 1: level kinematics from AWR and Q | at most 5e-7 relative in outgoing energy | far below the resolution (about 1e-3) |
| Dev. 2: energy cutoff | yes if unmatched (cutoff fraction 0 in P1-P3, 1.6e-4 in P4, measured) | matched, so none; D37 measures the effect |
| Dev. 3, 4, 8, 11, 12 | no | not exercised |
| Dev. 5: non-integer yields (weights against integer sampling) | no, noise only | MT 5 of W and Fe |
| Dev. 6: random-number streams and generator | no, noise only | independent by construction |
| Dev. 7, 13, 14: round-off guards, interpolation formula, track splitting, mesh slivers | round-off | none |
| Dev. 9: URR probability tables | yes (self-shielding in Fe-58 and W) | off, so none; D37 measures the effect |
| Dev. 10: azimuth and 3D direction | no, noise only | none |
| Dev. 15: reflection renormalisation, surface current | round-off, bookkeeping | left current exactly 0 in both codes |
| secondaries' `n_collision` (new) | no; changes only what a diagnostic means | uncollided compared at E = 14.1 MeV |
| source-site check direction (new) | no; a model device | buffer cell (D26) |
| thread order of OpenMC tally sums (new) | last bits only | 1 thread (D30) |

### Normalisation

`Tally::accumulate` (`src/tallies/tally.cpp`) scales each batch by the
total source strength over the number of particles, 1/n for one source of
strength 1, and never divides by a volume; the flux score is weight times
track length. An OpenMC mesh value is therefore the sum of w l per source
particle, the same quantity as mcslab's per-source tally. Mirrors in y
and z change neither the x-motion nor any path length, so there is no area
factor. The smoke runs confirm it: the 14.1 MeV uncollided flux matches
e^-tau per layer.

## The physics, in plain terms

- **What agreement means.** Both codes read the same files and follow the
  same rules, so every tally has the same true (expected) value in both;
  only the random noise differs. A gap larger than the noise means a bug or
  an unlisted difference. Agreement verifies the implementation. It says
  nothing about whether the nuclear data match experiment.
- **z and the 3 SE rule.** z is the gap between the codes in units of its
  combined uncertainty. Chance alone gives |z| > 3 about once in 370
  checks; over 82 checks the chance of at least one false alarm is about
  20%, which is why a failure triggers a rerun rather than a verdict.
- **Probability tables off.** In the unresolved-resonance range the data
  give the average cross section of resonances too dense to resolve.
  OpenMC can sample the fluctuations around the average (self-shielding);
  mcslab cannot, so OpenMC is told to use the average too.
- **Temperature.** Hot atoms jiggle, so a slow neutron can gain energy when
  it hits one (the free-gas model). The scale is kT: 0.025 eV at 294 K,
  0.078 eV at 900 K. Both codes take kT from the same file entry. It
  matters below 400 kT, and for hydrogen at every energy.
- **Fractional yields.** When a reaction makes 1.6 neutrons on average,
  OpenMC sends one neutron with weight 1.6; mcslab sends 1 or 2 neutrons
  with probabilities 0.4 and 0.6. Same average, different noise.
- **Energy cutoff.** The data stop at 1e-5 eV and mcslab ends a neutron
  below it. OpenMC is set to do the same, so the bookkeeping is identical.
- **Source on a surface.** Before starting a neutron, OpenMC checks that
  the start point is inside the geometry, pretending the neutron points
  along +z (`Source::satisfies_spatial_constraints`). On the x = 0 plane
  that test says "outside", and every start is rejected. A thin empty slab
  to the left gives the check somewhere to land. The neutron itself is
  then located with its real direction, +x (`Particle::event_calculate_xs`
  and `Surface::sense`), so it starts in W at exactly x = 0, as in mcslab.
  Nothing about the start moves, so there is no bias. With a mirror at
  x = 0 the buffer sits behind the mirror and is never entered.
- **An infinite slab in 3D.** Mirrors at y, z = +-10 m make the slab
  infinite sideways in effect; a mirror changes neither the x-motion nor a
  path length.
- **Thread order.** Adding the same numbers in a different order changes
  the last digit. OpenMC's threads add into shared tallies in whatever
  order they finish, so only a one-thread run repeats exactly.

## Decisions

| # | Question | Decision |
|---|---|---|
| D23 | Branch | `main` fast-forwarded to `phase-3` (`4db8833`); branch `phase-4` from it; not pushed. Files are staged by explicit path. |
| D24 | Cross-section index | A 14-entry `cross_sections.xml` written into the run directory after the sha256 checks. |
| D25 | Run directory | Outside the repository (default `~/mcslab_openmc_runs/phase4/`, one subdirectory per run). It holds the XML, statepoints, `summary.h5` and OpenMC's log. Only the extracted JSON is committed. |
| D26 | Geometry | x-planes at mcslab's bounds; reflective y and z planes at +-1000 cm; a void buffer cell -1 < x < 0 with vacuum at x = -1; x = 0 an internal surface in P1 and P4, reflective in P2 and P3; vacuum on the right. |
| D27 | Source | `Point((0, 0, 0))`, `Monodirectional((1, 0, 0))`, `Discrete([E0], [1])`, strength 1. |
| D28 | Energy cutoff | OpenMC's neutron cutoff set to 1e-5 eV, mcslab's default (the grid minimum). Both kill after the collision with a strict `<` and drop secondaries born below it. |
| D29 | Other settings | As in the settings table, all set explicitly; the full `model.xml` is embedded in the results JSON. The runner also checks each nuclide's NEAREST-rule data temperature (from the file's `kTs`) against the problem file. |
| D30 | Threads | OpenMC on one thread: bit-reproducible on the same build, about 7.5 min for P1-P4. |
| D31 | Problems | P1: D7 (W 0.5 \| FLiBe 20, natural Li, 900 K \| Fe 10 cm), vacuum both sides. P2: D7 with a reflective plasma side. P3: W 0.5 \| FLiBe 100 (natural Li, 900 K) \| Fe 10, reflective plasma side. P4: H-1 at 0.0708 g/cm^3 and 293.6 K, [0, 30] cm, 1 MeV beam, vacuum both sides (free gas at every energy). The beam is 14.1 MeV in P1-P3. |
| D32 | Depth meshes | P1, P2: W 10, FLiBe 20, Fe 20 bins (Phase 3, D11). P3: 10, 50, 20 (2 cm FLiBe bins). P4: 30 bins of 1 cm. Edges from mcslab's `make_mesh`, exported at full precision; one y bin and one z bin over [-1000, 1000] cm. |
| D33 | OpenMC tallies | All track-length, set explicitly: mesh flux; mesh x 6 scores x (problem nuclides + total); cell flux; cell x 6 scores x nuclides; cell x 246-bin energy flux; surface current on both x boundaries; mesh x `CollisionFilter([0])` x energy bins {below, 14.1 MeV +- 1e-9 relative} (E0 for P4). Layer values come from the cell tallies: OpenMC stores per-bin sums and sums of squares only, so the SE of a sum of mesh bins cannot be recovered. mcslab's layer values are per-batch sums of its bins. mcslab's densities are exported from the kernel's own arrays (`PackedXS.mat_dens` via `pack_problem`). |
| D34 | Statistics | 100 batches x 10,000 histories in each code. z = (mcslab - OpenMC) / sqrt(SE_m^2 + SE_o^2); a check fails at \|z\| > 3. Relative SEs estimated before the plan: at most 0.3% everywhere except P3's Fe layer (2-12%), so gaps of about 0.4-1% are detectable. |
| D35 | Primary checks | 82, listed below. |
| D36 | Results files | `benchmark/results/{mcslab,openmc}_P{1..4}.json`: mean and SE per layer and per bin (nuclides present in each layer), spectra and uncollided values. OpenMC provenance: version, commit, the `--version` text, platform with the Rosetta flag, Python, numpy and h5py versions, the 14 sha256 values, the sha256 of the problem file, seeds, settings, `model.xml`, densities read back and the lost count. No dates, timings or paths, so a rerun reproduces a file byte for byte. mcslab provenance: commit, a content hash of the `mcslab/` sources and the runner, environment. |
| D37 | Default-settings diagnostics | OpenMC P1 with probability tables on, and OpenMC P4 with energy cutoff 0, once each on their own seeds, reported only: the change against the main OpenMC run in SE and percent, and the z distribution against mcslab. |
| D38 | P3's Fe layer | Fe tritium is a diagnostic (about 70 contributing histories). The other Fe quantities stay primary and are flagged as the weakest checks. |
| D39 | Regression references | No new byte-exact reference. The kernel paths are pinned by `kin_d7` and `kin_d7_reflect`; the mcslab results record the commit and source hash that produced them, and `run_mcslab.py check` reruns a problem and compares the result bytes (by hand, not in pytest). |
| D40 | Phase 3 statements | Corrected in the README, `docs/deviations_from_openmc.md` (uncollided paragraph and deviation 6) and a comment in `mcslab/tallies.py`; the Phase 3 plan is left as the record. |
| D41 | Check inventory | The running count of statistical checks becomes 240 (158 before Phase 4). |

## Validation (seeds, checks and protocol fixed here)

**Seeds**

| seed | use |
|---|---|
| 20261040-20261043 | mcslab P1-P4 |
| 20261044-20261047 | mcslab P1-P4, failure protocol (4x) |
| 20261048 | mcslab development runs (discarded) |
| 20261100, 20261110, 20261120, 20261130 | OpenMC P1-P4 |
| 20261140, 20261150, 20261160, 20261170 | OpenMC P1-P4, failure protocol (4x) |
| 20261180 | OpenMC P1 with probability tables on (D37) |
| 20261190 | OpenMC P4 with energy cutoff 0 (D37) |
| 20261199 | OpenMC development check (discarded) |

OpenMC seeds are spaced by 10 because OpenMC derives four stream seeds,
s to s + 3 (`init_particle_seeds`); no stream seed coincides with another
declared seed.

**Exact checks (pytest, deterministic)**

1. Both result files of a problem were built from the committed problem
   file (sha256).
2. The data sha256 values in both files equal the pins in
   `scripts/checksums`.
3. OpenMC's atom densities (from `summary.h5`) equal the problem file's bit
   for bit.
4. OpenMC's mesh x-edges and energy edges (from the statepoint) equal the
   problem file's bit for bit.
5. The recorded settings, OpenMC version and commit, seeds and sizes equal
   the declared ones.
6. Exactly 0 in both codes: W tritium per layer (total and the five W
   nuclides) in P1-P3; left leakage in P2 and P3; tritium and helium in P4.
7. mcslab's balance residual is exactly 0 and nothing is lost; OpenMC
   reports no lost particle.
8. No committed benchmark file contains an absolute path or the current
   user name.
9. `benchmark/compare.py` imports neither openmc, mcslab nor h5py (checked
   in a subprocess).

**Primary statistical checks (layer-integrated, |z| <= 3)**

| problem | flux | absorption, 301, 901, 444, 207 | H3 (205) | FLiBe H3 by nuclide | leakage | total |
|---|---|---|---|---|---|---|
| P1 | 3 | 15 | FLiBe, Fe | Li6, Li7, Be9, F19 | left, right | 26 |
| P2 | 3 | 15 | FLiBe, Fe | 4 | right | 25 |
| P3 | 3 | 15 | FLiBe | 4 | right | 24 |
| P4 | 1 | 4 (H-1 has no MT 207) | none | none | left, right | 7 |
| **total** | | | | | | **82** |

Absorption is counted once, as one of the six scores. W tritium, left
leakage with a reflective side and P4's tritium and helium are exact
checks. Chance of at least one false failure: at most 19.9% (normal
rate 0.27% per check, Sidak bound; the checks are positively correlated),
22.1% with the Student-t rate at about 198 degrees of freedom. The checks
flagged as weakest are P3's Fe flux, absorption, 301, 901, 444 and 207.

**Diagnostics (reported, no pass/fail)**

- Per-bin profiles of flux and of every score for each nuclide present and
  the total, as z per bin.
- The 246-bin spectrum of every layer.
- Uncollided flux: mcslab's TL_UNC against OpenMC's 14.1 MeV band, and
  each against the first-flight formula (Sigma_t at the source energy from
  raw h5py, in the problem file). OpenMC's bin-0 part from secondaries is
  reported separately.
- P3's Fe tritium (D38).
- z distribution per problem: fraction beyond 2 (4.55% for normal noise)
  and beyond 3 (0.27%), the maximum, and a histogram against N(0,1).
  Descriptive only: neighbouring bins are correlated. Excluded:
  structurally zero quantities (no data, such as W tritium) and bins where
  both codes are exactly 0; bins where exactly one code is 0 are listed.
- Bookkeeping: OpenMC's mesh sums against its cell tallies; mcslab's
  cutoff weight, zero-yield weight, reflected weight and maximum random
  numbers per history against STRIDE.
- D37: probability tables on (P1) and energy cutoff 0 (P4).

**Failure protocol**

If any primary |z| > 3, work stops (no figures, no documentation). That
problem is rerun once in both codes at 100 x 40,000 on its declared
second seed pair, and both results are reported; the original stays in the
record.

- If the same check exceeds 3 SE again, the difference is treated as real:
  it is diagnosed with extra read-only tallies (uncollided against
  collided, by nuclide, by energy), documented, and work stops for a
  decision before any change to mcslab's physics.
- If the rerun passes, both z values go into a named exceptions table in
  the pytest and in the README, with the reason; there is never a silent
  xfail. Work then continues with the figures and documentation, and the
  exception is put at the top of the final summary.

No other seeds, sizes, thresholds or settings are tried.

## Files

- New: `benchmark/__init__.py`; `benchmark/run_mcslab.py` (mcslab's venv;
  writes `benchmark/problems.json` and the mcslab results);
  `benchmark/run_openmc.py` (OpenMC's environment only; reads only the
  problem file and never imports mcslab); `benchmark/compare.py` (numpy
  only; matplotlib only with `--figures`); `benchmark/problems.json`;
  `benchmark/results/*.json`; `tests/test_benchmark_openmc.py`; this plan;
  `docs/figures/phase4_*.png`.
- Modified: README (Phase 4 section, tests, limitations, the uncollided
  statement); `docs/deviations_from_openmc.md`; `mcslab/tallies.py` (one
  comment).
- Untouched: every kernel and driver, all regression references,
  `docs/phase3_plan.md`.

## Commits (on `phase-4`)

1. This plan.
2. Corrections to Phase 3 statements (D40). All five regressions byte
   exact; full pytest.
3. mcslab side and the problem file (inputs only, no results).
4. OpenMC side (development check on seed 20261199, discarded).
5. Comparison script and pytest, written before the benchmark runs,
   committed with the results. Work stops here if a primary check fails
   (failure protocol).
6. Figures.
7. README, deviations, this status and the final summary.

## Estimated runtimes

| item | mcslab | OpenMC (1 thread, Rosetta) |
|---|---|---|
| P1 / P2 / P3 / P4, 1e6 histories | 30 / 35 / 45 / 25 s | 1.7 / 2 / 2.6 / 0.9 min |
| total | about 2.5 min | about 7.5 min |
| D37 diagnostics | none | about 2.5 min |
| failure protocol (4x), per problem | 1.5-3 min | 3.5-11 min |
| new pytest | under 1 s | none |

Basis, per 1e5 histories: mcslab 3.0 s (P1), 4.6 s (P3), 2.4 s (P4),
including packing; OpenMC 10 s (P1), 15.5 s (P3), 5.1 s (P4), plus about
2 s of data loading per run. P2 was not timed.

## Status

- [x] `main` fast-forwarded to `phase-3` (`4db8833`); branch `phase-4`
      created.
- [x] Plan approved (2026-10-03).
- [x] 1. Plan: `0b160e3`.
- [x] 2. Corrections to Phase 3 statements: `b1ee081`. New deviation 16
      (uncollided estimators), deviation 6 rewritten (generators), README
      and a comment in `mcslab/tallies.py`. All five regressions byte
      exact; 194 tests pass.
- [x] 3. mcslab side and problem file: `1e9c041`. The kernel's densities
      equal `CEMaterial.number_densities` and the D7 tally manifest bit
      for bit. A shared writer, `benchmark/jsonio.py` (standard library
      only), was added so both environments write identical JSON text.
- [x] 4. OpenMC side: `108e951`. Development check on seed 20261199
      (P1, 2 x 1000): densities, mesh and spectrum edges read back
      bit-identical, settings as declared in `model.xml`, no lost
      particle, no local path.
- [x] 5. Comparison, pytest and results: `5350c94`. `compare.py` and the
      pytest were written before the benchmark runs. All 82 primary
      checks pass; the failure protocol was not needed. The commit was
      amended once before any push (from `c53fc12`): the test for local
      paths matched its own pattern once the file was tracked, and the
      pattern is now assembled at run time.
- [x] 6. Figures: `08bcb89`.
- [x] 7. Documentation: README Phase 4 section (setup, results, what
      differs and why, the default-settings numbers, reproduction, tests,
      limitations); deviations 2, 9 and 14 measured, new deviation 17;
      this status and the final summary.

**As run** (where the runs differ from the text of this plan):
- The matched energy cutoff (D28, "1e-5 eV") is 9.999999999999999e-06 eV,
  one ulp below 1e-5. It is mcslab's default, the largest data-grid
  minimum of the nuclides as stored in the files
  (`config_kin._energy_cutoff`), and it reaches OpenMC unchanged through
  `benchmark/problems.json`.
- `benchmark/jsonio.py` (standard library only) was added, a file the plan
  did not list, so that both environments write identical JSON text.
- The palette validator did not run (no Node.js). The figures use slots 1
  and 2 of the reference palette, which is documented as validated for its
  first three slots; no local validation was made.

## Final summary (2026-10-03)

- **Protocol exceptions: none.** No primary check exceeded 3 SE, so the
  4x reruns were not run and `PROTOCOL_EXCEPTIONS` is empty.
- **Result:** mcslab and OpenMC 0.16.0 agree on all four problems.
  - All 82 primary checks pass: largest |z| 2.63 (P1 Fe He4-production).
    Per problem, the largest |z| is 2.63, 2.05, 1.52 and 1.21, and the
    mean z +0.89, -0.77, +0.68 and +0.53.
  - All 66 exact checks pass (16, 17, 17 and 16 per problem).
  - 74 of the 82 checks resolve differences of about 0.5% (median relative
    SE of the difference 0.16%). The largest gap among them is +0.86%
    (P1 Fe tritium, 2.3 SE).
- **Headline values** (mcslab / OpenMC, per source neutron):
  - FLiBe tritium: P1 0.29905 / 0.29852, P2 0.67976 / 0.68097, P3
    1.2289 / 1.2266
  - H-1 absorption (P4): 0.57202 / 0.57120
- **Tests:** 208 pass (194 before Phase 4, 14 new). Statistical checks:
  240 in all (82 new). Regressions: Phase 1, Phase 2a, D7 and both tally
  problems byte-exact; no reference was recorded or changed.
- **Diagnostics** (no pass/fail):
  - **Per-bin z beyond 2:** 13.4% (P1), 3.3% (P2), 1.3% (P3) and 6.7% (P4).
    The P1 excess is one run-wide correlated fluctuation (Fe fast
    scores, plasma-side region; largest per-bin |z| 4.33). P2's Fe has
    the opposite sign.
  - **Uncollided flux:** agrees with the first-flight formula in both
    codes once OpenMC's bin 0 is restricted to the source energy.
  - **D37:** probability tables on change D7 by up to 0.92% (W
    absorption, 1.2 SE) and the left leakage by +0.59% (3.6 SE); an
    energy cutoff of 0 changes P4 by at most 0.40% (2.5 SE, mostly
    noise). In both runs every check still passes against mcslab
    (largest |z| 2.47 and 1.87).
- **Findings:**
  - OpenMC 0.16.0's `CollisionFilter` counts (n,2n)-type secondaries as
    uncollided (deviation 16; a Phase 3 statement corrected).
  - The two codes use different random-number generators (deviation 6
    corrected).
  - An energy exactly on a bin edge is binned in opposite directions
    (deviation 17; P4's 1 MeV source).
  - OpenMC's mesh leaks round-off slivers across layer boundaries
    (deviation 14).
  - OpenMC's source-site check uses a fixed direction, which needs the
    buffer cell.
  - OpenMC tallies are bit-reproducible only on one thread.
- **Runtimes** (CPU time, not a speed comparison):
  - mcslab: 22-44 s per problem, 2.5 min for P1-P4 (run alongside OpenMC).
  - OpenMC, one thread under Rosetta: 158 s (P1), 266 s (P2), 245 s (P3),
    75 s (P4); 151 s and 67 s for the two D37 runs; about 16 min in all.
    This is more than estimated. P2's wall time was 886 s, with the
    machine busy or idle.
  - New pytest: under 1 s.
- **Judgment calls:**
  - The matched energy cutoff is mcslab's actual default, the grid
    minimum 9.999999999999999e-06 eV, rather than the rounded 1e-5 of
    the plan text.
  - `benchmark/jsonio.py` was added (not in the plan's file list).
  - Two diagnostic-only rules were added after seeing the runs, and are
    stated as such in `compare.py` and the README:
    - the two spectrum bins at a source energy on an edge are reported
      separately
    - bins without an uncollided event are counted instead of compared
      with the formula
    The first comparison attempt also stopped on an infinite z in that
    second diagnostic before it was handled. No primary check, seed,
    threshold or setting changed.
  - The P1 per-bin excursions were reported, not investigated with extra
    runs: no primary check failed, and the protocol allows extra runs
    only after a failure.
  - mcslab keeps its bin-edge convention (deviation 17), since changing it
    would move byte-exact references for a pure convention.
  - mcslab's results record a content hash and a dirty flag over
    `mcslab/` and the runner only, so unrelated local edits do not mark
    them dirty.
  - The figures put the depth-bin index on the x-axis, so the thin W bins
    are visible. Points beyond a panel's range are drawn as triangles on
    its edge. The spectrum ratio shows bins with SE below 20%.
  - The palette validator script could not run (no Node.js). The two
    series colours are slots 1 and 2 of the reference palette, which is
    documented as validated for its first three slots, and the codes also
    differ in line style.
