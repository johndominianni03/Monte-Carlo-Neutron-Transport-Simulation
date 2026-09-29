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
./venv/bin/python scripts/make_inventory.py                # regenerate docs/data_inventory.md
./venv/bin/python scripts/plot_xs.py                       # regenerate docs/figures/*.png
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
- **No scattering kinematics.** In the CE kernel every collision ends the
  history, and a neutron keeps its source energy until it collides or
  leaks. CE results are therefore uncollided-flux quantities only (Phase 2b).
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
