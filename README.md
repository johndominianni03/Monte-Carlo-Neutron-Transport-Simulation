# mcslab: 1D Monte Carlo neutron transport

A 1D slab Monte Carlo neutron transport code in Python + Numba. The eventual
goal is modelling 14.1 MeV D-T fusion neutrons in a first wall / blanket
(heating, dpa, helium production, tritium breeding ratio) with ENDF/B-VIII.0
data.

## Current scope

### Phase 1: transport engine (one-group)

- 1D slab of contiguous regions, vacuum boundaries on both sides (units: cm).
- One-group, made-up materials (Sigma_t = Sigma_s + Sigma_a), isotropic
  lab-frame scattering. Void regions (Sigma_t = 0) are allowed.
- Sources: monodirectional beam at x = 0 (mu = 1); isotropic plane source.
- Analog transport. Tallies per region are collisions, absorptions,
  track-length flux and collision-estimator flux, plus currents in both
  directions at every region interface. Means and standard errors come from
  batch statistics.
- Reproducible per-history RNG (OpenMC-style LCG with skip-ahead). Results
  are bit-identical however the batches are split across calls or, later,
  threads.

### Phase 2a: nuclear data layer (continuous energy, lookup only)

- `mcslab/nucdata.py` reads official OpenMC HDF5 incident-neutron files
  (format 3.x) directly with h5py. The `openmc` package is not used. The
  library is a directory plus a label, so ENDF/B-VIII.0 (used now) and
  FENDL-3.2 (planned comparison) are read the same way.
- `mcslab/xs.py` does the lookup: binary search plus lin-lin interpolation,
  exactly 0 below threshold, and it refuses energies off the grid. It also
  packs the data CSR-style for the Numba kernels.
- `mcslab/ce_materials.py` defines natural W, natural Fe and FLiBe
  (Li2BeF4, with a Li-6 enrichment parameter). Densities and abundances are
  cited in the code.
- `mcslab/transport_ce.py` and `config_ce.py` form a continuous-energy
  kernel in which **the first collision ends the history**. There are no
  scattering kinematics yet; that's Phase 2b.
- `docs/data_inventory.md` is generated from the data. It lists, per
  nuclide, every reaction with its MT, thresholds, Q, sigma(14.1 MeV) and
  secondary-distribution types, plus what heating, damage and
  gas-production data exist.
- Plots of the key cross sections are in `docs/figures/`.

### Phase 2b, Part 1: collision kinematics (elastic and discrete levels)

A new continuous-energy kernel (`mcslab/transport_kin.py`, driver
`mcslab/config_kin.py`) in which neutrons scatter, lose energy and
multiply. It mirrors OpenMC v0.16.0's physics, cited in the code. The
Phase 1 and Phase 2a kernels are unchanged.

- **Collisions.** The nuclide is chosen in proportion to N sigma_t. Then
  analog absorption (capture, (n,p), (n,a), Li-6 (n,t), ...) ends the
  neutron. Otherwise an elastic or inelastic channel is chosen in
  proportion to its cross section, in OpenMC's order and draw count.
- **Elastic scattering.** Two-body kinematics with the tabulated CM angular
  distribution, target at rest.
- **Discrete inelastic levels (MT 51-90).** Two-body kinematics with the
  level's Q. The CM energy comes from AWR and Q, so energy and momentum
  are conserved to round-off. This is a documented deviation from
  OpenMC, which uses the rounded stored parameters.
- **Secondary neutrons.** Reactions with yield y > 1 bank y - 1 identical
  copies, as OpenMC does. The bank is per history and LIFO, and
  continues the history's random-number stream, so results stay
  bit-identical however batches are split.
- **Energy cutoff.** Defaults to the bottom of the data grid (1e-5 eV).
  Killed weight is never dropped silently: it is tallied per region, is a
  term in the exact neutron balance, triggers `EnergyCutoffWarning`, and is
  shown in the test summary.
- **Tallies.** The Phase 1 region and surface tallies, plus:
  - a spectrum tally (track-length and collision estimators in energy bins)
  - per-channel event and secondary counts
  - counters for the neutron balance
- **Synthetic nuclides** (`mcslab/synthetic.py`): constant cross sections,
  chosen mass, isotropic CM. Used for the analytic tests.
### Phase 2b, Part 2: every law in the data, free gas, tracks

- **Every secondary law in the W / FLiBe / Fe data** (inventory:
  `docs/law_inventory.md`, generated from the HDF5 files by
  `scripts/law_inventory.py` with h5py only):
  - correlated angle-energy (ACE law 61), in the CM frame (Fe, W) and in
    the lab frame (Be-9 (n,2n), F-19)
  - F-19 (n,2n)'s two-law applicability mixture
  - continuous tabular (ACE law 4) with tabular angles (Li-6, Li-7
    (n,2n)-type reactions)
  - energy-dependent yields (MT 5 of Fe and W)

  Each transcribes OpenMC v0.16.0's sampler, cited in the code. The driver
  refuses any problem whose nuclides have a law that is not implemented.
  None of the 14 files has one.
- **Multiplicity** (approved deviation D2). A non-integer yield y gives
  floor(y) or floor(y) + 1 neutrons, with mean y, and weights stay 1. A
  multiplicity of 0 (e.g. Fe and W MT 5 below ~6 MeV) ends the neutron.
  Its weight is tallied per region (`zero_yield_weight`) as its own term
  of the exact balance.
- **Free-gas target motion**, OpenMC's constant-cross-section model and
  rule: the target is at rest only if E >= 400 kT and A > 1, so H-1 gets
  free gas at every energy. kT is that of the data temperature.
- **Temperature is a per-material input** (`CEMaterial.temperature`,
  default 293.6 K). It selects the data temperature (250, 294, 600, 900,
  1200 or 2500 K) by OpenMC's nearest rule, within 10 K.
- **Track recording** (`KinRunConfig.n_track`) stores every event of
  chosen histories (position, direction, energy, time, family particle
  id, event type, MT). It draws no random number and touches no tally:
  results are byte-identical with it on or off.
- **Animation:** `scripts/record_tracks.py` saves tracks, and
  `scripts/animate_tracks.py` turns them into `docs/figures/tracks.gif`
  (depth vs log energy over time) and `docs/figures/tracks.png`.
- **Regression reference** for the kinematic kernel, problem D7 (below).

![Neutron tracks in W | FLiBe | Fe](docs/figures/tracks.gif)

`docs/deviations_from_openmc.md` lists every known difference from OpenMC.

## Nuclear data setup

The data live outside the repository and are never committed.

```bash
scripts/fetch_data.sh                 # streams endfb80.tar.xz (3.38 GB), keeps 14 nuclides (~276 MB)
export MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5
```

The extracted files are checked against the sha256 sums pinned in
`scripts/checksums/endfb-viii.0.sha256`. Without `MCSLAB_DATA`, the Phase 2a
tests skip and the Phase 1 tests run as before.

## Usage

```bash
./venv/bin/python -m mcslab                                # Phase 1 demo run
./venv/bin/python -m pytest                                # all tests
./venv/bin/python tests/test_regression.py compare         # Phase 1 bit-identical regression
./venv/bin/python tests/test_regression_ce.py compare      # Phase 2a (CE) bit-identical regression
./venv/bin/python tests/test_regression_kin.py compare     # Phase 2b kinematic bit-identical regression (D7)
./venv/bin/python scripts/make_inventory.py                # regenerate docs/data_inventory.md
./venv/bin/python scripts/law_inventory.py                 # regenerate docs/law_inventory.md
./venv/bin/python scripts/plot_xs.py                       # regenerate docs/figures/*.png
./venv/bin/python scripts/hydrogen_cutoff.py               # H-1 cutoff fraction, target at rest vs free gas
./venv/bin/python scripts/record_tracks.py                 # D7 tracks -> outputs/tracks_d7.npz
./venv/bin/python scripts/animate_tracks.py                # -> docs/figures/tracks.gif, tracks.png
```

A kinematic run (Phase 2b), e.g. a synthetic scatterer:

```python
from mcslab import synthetic as S
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.sources import BeamSource

x = S.nuclide("X", awr=12.0, elastic_b=4.0, capture_b=0.01)
cfg = KinRunConfig(SlabGeometry([0.0, 10.0], [S.SyntheticMaterial("x", (("X", 0.1),))]),
                   BeamSource(), MonoEnergetic(2.0e6), S.SyntheticLibrary({"X": x}),
                   n_batches=20, histories_per_batch=1000, energy_cutoff=1.0e3)
res = run_kin(cfg)
print(res.balance())       # exact: source + created == absorbed + leaked + cutoff + zero_yield
```

A real-data run with a material temperature and recorded tracks:

```python
import dataclasses
from mcslab import ce_materials as cm
from mcslab.nucdata import Library

flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)  # 900 K data
cfg = KinRunConfig(SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()]),
                   BeamSource(), MonoEnergetic(14.1e6), Library.open(),
                   n_batches=20, histories_per_batch=5000, n_track=100)
res = run_kin(cfg)
res.tracks.save("tracks.npz")     # then: scripts/animate_tracks.py tracks.npz
```

## Tests: what they do and do not cover

- `tests/test_physics.py` checks the Phase 1 engine against analytic results
  (see its docstring).
- `tests/test_nucdata.py` covers Phase 2a and needs the data:
  - **(a)** On-grid lookups return the stored values bit for bit, for every
    reaction of all 14 nuclides. This tests the search and interpolation
    code, not the accuracy of the data.
  - **(b)** Grids are strictly increasing. Every reaction is exactly 0 below
    its threshold index, and at every grid energy below the kinematic
    threshold -Q(A+1)/A.
  - **(c)** **MT 1 (the total) is not stored in the OpenMC HDF5 files**, so
    "total = sum of partials" cannot be checked against an evaluated total
    here. What *is* checked:
    - No reaction is counted twice.
    - All 24 stored redundant sums (MT 4, 103-107, and Li-7 MT 205) match
      their components. The tolerance is the ENDF-format rounding bound (7
      significant digits, or 6 below 1e-9 b) plus 1e-9 b. The worst excess
      over the rounding bound is 8.5e-11 b, in sub-threshold tails of MT
      103/107, where the stored sum is nonzero below all its components'
      thresholds.
    - The kernel's pre-summed total matches the per-reaction lookups to
      3e-15 relative.

    The evaluated total itself is only checked by hand (below).
  - **(d)** A 14.1 MeV beam through 5 cm of natural iron, with the first
    collision ending the history. The test compares T with
    exp(-Sigma_t L) at 3 SE, with Sigma_t computed in the test from the raw
    HDF5 files (h5py + np.interp). This validates the lookup and sampling
    chain end to end at one energy. It does not validate any energy
    dependence of transport.
  - Also covered: the data checksums, and mean atomic masses from IUPAC
    abundances x AWR against standard atomic weights.
- `tests/test_regression_ce.py` **(e)** is a byte-exact regression of the CE
  kernel on real data: a W | void | FLiBe | Fe slab with a log-uniform
  1 keV-14.1 MeV source. Its reference (`tests/reference_ce/`) pins the
  sha256 of every data file it uses.

- `tests/test_kinematics.py` covers Phase 2b Part 1. Seeds were fixed in
  `docs/phase2b_plan.md` before the tests were written.
  - **(a)** Elastic scattering off a target at rest, synthetic isotropic-CM
    nuclide with A = 1, 12, 184. E' is uniform on [alpha E, E]
    (chi-square) and the mean ln(E/E') is xi (3 SE). Per event,
    E' = E((1+alpha) + (1-alpha) mu_cm)/2 to round-off, with mu_cm
    replayed from the event's random number.
  - **(a2)** The tabular-angle sampler on real tables (Fe56 and W184
    elastic, at on-grid and between-grid energies) and on synthetic
    histogram tables. Chi-square against probabilities computed from the
    raw HDF5 files, plus an exact inverse-CDF replay (1.7e-15).
  - **(b)** Slowing down in an infinite non-absorbing medium, through the
    full kernel. Both flux estimators follow 1/(xi Sigma_s E): a
    Hotelling T^2 test over 10 energy bins, for A = 1 (exact) and A = 12
    (8 collision intervals below the source, where the Placzek transient
    is bounded at 1.3e-8 by a deterministic solve in the test). Leakage
    is asserted to be exactly 0.
  - **(c)** Per-event energy and momentum conservation for all 98 discrete
    levels of Fe56, Li7, F19 and W184 (real angle data), synthetic levels,
    and elastic scattering: at most 7.4e-16 relative.
  - **(d)** Exact integer neutron balance in a multiplying slab with
    (n,2n) and (n,3n):
    `sources + created == absorbed + leaked + cutoff`, per batch, with
    every term cross-checked against an independent counter, and
    `created == (y-1) x events` per channel.
    - **Its multiplication is not physical.** The synthetic (n,2n) and
      (n,3n) laws give each identical copy the full two-body energy. A
      14.1 MeV (n,2n) family on A = 9 therefore carries 2 x 9.76 =
      19.5 MeV out of the 12.1 MeV available (E + Q). The secondaries
      stay above threshold and multiply again, which gives 1.31
      secondaries per source (beam) or 1.20 (isotropic source).
    - That is fine for a bookkeeping test, but do not read it as a
      physical number. The real D7 slab gives 0.26.
  - **(a2)** also rebuilds its expected probabilities from the PDF alone,
    never reading the stored CDF that the sampler uses. They must agree
    to 1e-6 per bin (worst 2.8e-7).

- `tests/test_laws.py` covers the Part 2 laws on real data. Seeds were
  fixed in the plan before the tests existed.
  - **Reader and packing.** All 301 non-redundant neutron reactions of
    the 14 nuclides read and pack exactly as the independent h5py
    inventory lists them. Edited copies of real files with unimplemented
    or malformed laws are refused, and so is a whole problem containing
    one, even below that reaction's threshold.
  - **(f), replay.** All 56 reactions with a Part 2 law, at 3 incident
    energies, 336000 events in all. Each event is replayed from its random
    numbers on raw h5py arrays:
    - E_out matches the replay to 2.6e-16 E_in.
    - The forward CDF matches the draw: 2.5e-13 for E_out, 3.0e-10 for
      mu.
    - The CM->lab transform matches to 4.4e-16.
    - The multiplicity is exact.
  - **(f), statistical.** Joint (E_out, mu) chi-squares for Fe56 MT 91,
    W184 MT 16, Be9 MT 16, F19 MT 16 and Li7 MT 16, each at an on-grid
    and a between-grid incident energy (10 checks). The reference is
    built from the PDFs alone; the stored CDF is never read.
  - **(g), MT 5 multiplicity.** Fe56 and W184 at 6 energies: n is
    floor(y) or ceil(y), and its mean equals y(E) at 3 SE. Integer yields
    draw nothing extra.
  - **(g), kernel.** On real W | Fe (30 MeV) and FLiBe (14.1 MeV) slabs,
    created == (y - 1) x events per channel, and the balance with the
    zero-yield term is exact per batch.
- `tests/test_freegas.py` covers free gas and temperatures. The docstring
  derives every expected value.
  - **Stationarity.** E_in is drawn from the collision density
    pi ~ sigma_eff(E) E exp(-E/kT). After one collision, E_out follows pi
    again (chi-square), and its mean is kT (2 - 1/(2(A+1))), for A = 1
    and 12.
    - pi is not E exp(-E/kT). The flux pi / sigma_eff is Maxwellian only
      if the tabulated cross section is sigma_eff.
  - **Kernel shape** at fixed E_in = kT and 20 kT:
    - A = 1 against the closed-form hydrogen (Wigner-Wilkins) kernel.
    - A = 12 against a kernel integrated numerically over the Maxwellian
      target, independently of the sampler.
    - Above 400 kT (A = 12), the target-at-rest path is taken, bit for
      bit.
  - **Exact checks:**
    - the dispatch rule
    - the kernel's free-gas counter
    - OpenMC's nearest-temperature selection
    - per-material packing against h5py
    - the Phase 2a driver refusing other temperatures
- `tests/test_tracks.py`:
  - Track recording on, off or truncated leaves every tally and count
    byte-identical.
  - Recorded tracks are consistent, to 1.1e-14 cm, and do not depend on
    the batch split.
  - The recorded events match the kernel's counters exactly.
- `tests/test_regression_kin.py` **(h)** is a byte-exact regression of the
  kinematic kernel on problem D7, with its own `tests/reference_kin/` and
  pinned data sha256s.
  - **Problem:** W 0.5 cm, then FLiBe 20 cm (900 K data and density),
    then Fe 10 cm, with a 14.1 MeV beam, 20 x 5000 histories.
  - **Per source:** 0.263 secondaries created, 0.322 absorbed, 0.562
    leaked left, 0.375 leaked right, 0.0046 ended by zero yield, 0 cut
    off.
  - **What it is:** a bit-identity contract, not a validation of those
    numbers.

  Not covered yet: any comparison of real-data transport results with
  OpenMC or with measurements (Phase 4), and any transport-level thermal
  spectrum test.
- `tests/test_reproducibility_kin.py`: the kinematic kernel is
  bit-identical under repeated, split and reordered batch runs with
  secondaries present. Bank overflow raises instead of dropping neutrons.

## Hand-checking cross sections against an independent source

These are values at 14.1 MeV from mcslab (ENDF/B-VIII.0, 294 K, lin-lin on
the NJOY grid):

| quantity | ENDF MT | mcslab value |
|---|---|---|
| Fe-56 total | 1 | 2.57908 b |
| Fe-56 (n,2n) | 16 | 0.4316 b |
| W-184 total | 1 | 5.38633 b |
| Li-6 (n,t) | 105 | 0.0258 b (25.8 mb) |
| Be-9 (n,2n) | 16 | 0.484483 b |
| Li-7 tritium production (optional, harder) | 205, or the sum of MT 52-82 | 0.300646 b |

Steps:
1. Open the NNDC ENDF retrieval and plotting tool, Sigma
   (https://www.nndc.bnl.gov/sigma/), or the IAEA ENDF database
   (https://www-nds.iaea.org/exfor/endf.htm).
2. Select the library **ENDF/B-VIII.0**, then the target (e.g. Fe-56), then
   the reaction by MT number (MF=3, cross sections).
3. Read the evaluated cross section at 1.41e7 eV. Use the tool's
   interpolated value or data table, or interpolate between the two
   tabulated points around 14.1 MeV yourself.
4. Compare. Expect agreement to about 0.1%. NJOY reconstructs the pointwise
   data to a 0.1% tolerance, and Doppler broadening at 294 K is negligible at
   14 MeV. A difference of a few percent or more means something is wrong:
   the wrong nuclide, MT, library version or units.
5. For Li-7, ENDF usually has no MT 205. Tritium comes from the breakup
   levels MT 52-82 (LR = 33), so sum those at 14.1 MeV, or use a tool that
   reports tritium production.

The exact menu names in these web tools may differ from the description
above.

## Limitations

### Phase 1
- One energy group, made-up cross sections, isotropic lab-frame scattering only.
- Analog only: no variance reduction. The particle weight is carried but is always 1.
- Serial execution. The batch loop is structured for `prange`, but it has not
  been parallelised or tested multithreaded.
- Bit-identity holds on the same platform with the same numpy/numba/llvmlite.
  `log` may differ in the last bit across platforms or library versions.
- If a history uses more than `STRIDE` (152917) random numbers, its stream
  overlaps the next history's. That doesn't break reproducibility, but the
  streams are no longer independent. The kernel records the maximum draws per
  history so this can be checked.
- Standard errors come from batch means and assume batches are i.i.d. and
  approximately normal (at least 20 batches recommended).

### Phase 2a
- **No scattering kinematics in the Phase 2a kernel.** In
  `transport_ce.py` every collision ends the history, and a neutron keeps
  its source energy until it collides or leaks. Its results are
  uncollided-flux quantities only. The Phase 2b kernel adds kinematics.
- **294 K cross sections only.** Other temperatures in the files are
  readable but unused, and there is no on-the-fly Doppler broadening.
- **FLiBe temperature mismatch.** The default FLiBe density is the Janz
  correlation at 973 K (1.938 g/cm^3), but its Li, Be and F cross sections
  are at 294 K. At 14 MeV the Doppler effect is negligible. At low
  energies (resonances, thermal) this mismatch is real and is not corrected.
- **No unresolved-resonance probability tables.** URR tables exist in the
  files for Fe58 and W182-186 but are not used, so there is no self-shielding
  in the unresolved range.
- **No thermal scattering (S(alpha,beta))**, no unionized energy grid, no
  fission (none of the nuclides fission), and no photon transport.
- **Materials.** W and Fe densities are room-temperature handbook values.
  Natural lithium uses the IUPAC representative composition. Real lithium
  varies, and IUPAC's standard atomic weight for Li is an interval.
- **Validated at 14.1 MeV only.** Test (d) is a single-energy, uncollided
  check. The evaluated total cross section is compared with an independent
  source only through the manual hand-check above, which has not yet been
  done.
- **Tritium, heating and damage** data are inventoried but not yet tallied.
  For Li-7, tritium production is only available as the redundant MT 205
  (see `docs/data_inventory.md`).
- **FENDL-3.2** is supported by the reader in principle, but it has not been
  fetched or tested.

### Phase 2b, Part 1
- **(Fixed in Part 2.) No thermal treatment: the target was always at
  rest.** Neutrons did not thermalise; they kept slowing down until the
  energy cutoff (1e-5 eV) killed them. In weakly absorbing media this was
  most of the neutrons.
  - **Example:** a 30 cm slab of pure H-1 at 0.0708 g/cm^3 (294 K cross
    sections), 1 MeV beam, 10 batches x 1000 histories, seed 3, lost
    69.41% of the source weight to the cutoff.
  - **With free gas** (`scripts/hydrogen_cutoff.py`, identical inputs),
    the balance is exact in both runs:

    | run | absorbed | leaked left | leaked right | cutoff (fraction) |
    |---|---|---|---|---|
    | target at rest (Part 1) | 1013 | 1421 | 625 | 6941 (0.6941) |
    | free gas (Part 2) | 5748 | 3006 | 1245 | 1 (0.0001) |

    The one remaining neutron is the sub-1e-5 eV tail of the thermal
    population, which OpenMC would keep transporting.
  - **Cause.** With the target at rest, no collision can raise a
    neutron's energy. Hydrogen removes on average a factor e per
    collision, so a neutron at 1 eV reaches 1e-5 eV in about 11.5
    collisions, each with at most a 1.4% chance of capture.
  - **With target motion** (Part 2's free gas, as in OpenMC), collisions
    below a few kT raise the energy as often as they lower it. Neutrons
    then settle into a thermal population near kT = 0.025 eV and end by
    absorption or leakage instead.
  - **Not a physical material.** 0.0708 g/cm^3 is the density of liquid
    H2, and real hydrogen is cold and molecular, so its thermal
    scattering would need S(alpha,beta) data. The 294 K data's elastic
    cross section is already Doppler-broadened (1160 b at 1e-5 eV), but
    broadening the cross section moves no energy. Only target motion in
    the kinematics does.

  Part 1 results below about 1 eV were not physical. The lost weight was
  always reported, never hidden.
- **Energy cutoff instead of extrapolating below the data grid.** This
  deviates from OpenMC, which has a zero cutoff and extrapolates.
- **(n,2n)-type secondaries are identical copies** of the outgoing neutron
  (same energy and direction), as in OpenMC. Mean values are unaffected;
  the copies are fully correlated.

### Phase 2b, Part 2
- **Free gas only; no S(alpha,beta).** Target motion is OpenMC's
  constant-cross-section free gas, with resonance scattering (DBRC/RVS)
  off, as in OpenMC's defaults. None of these materials has thermal
  scattering data in ENDF/B-VIII.0. Hydrogen is treated as a free gas of
  protons, not as H2.
- **Temperature.** Only OpenMC's default "nearest" rule: no interpolation
  between data temperatures, no multipole, no on-the-fly broadening. A
  material more than 10 K from 250, 294, 600, 900, 1200 or 2500 K is
  refused. `flibe()`'s temperature argument still sets only its density.
- **Validated by unit-level and synthetic checks only.**
  - What is checked: every new law is checked per event and statistically
    against the raw tables, free gas against analytic kernels, and the
    real-data runs for exact balance.
  - What is not: no real-data transport result has been compared with
    OpenMC or experiment yet (Phase 4). The D7 reference pins the code's
    behaviour; it does not show the numbers are right.
- **Multiplicity variance differs from OpenMC** (D2): analog integer
  sampling, not weight x yield. Means agree.
- **No event limit per particle** (OpenMC stops at 1e6 events). A neutron
  in a large non-absorbing medium could scatter for a very long time.
- **Photons are ignored**, including photon products of every reaction.
  So are the charged particles and recoils of (n,n'alpha)t and the like:
  tritium and gas production are not tallied yet (Phase 3).
