# mcslab: 1D Monte Carlo neutron transport

A 1D slab Monte Carlo neutron transport code in Python + Numba. The eventual
goal is modelling 14.1 MeV D-T fusion neutrons in a first wall / blanket
(heating, dpa, helium production, tritium breeding ratio) with ENDF/B-VIII.0
data.

## Current scope: Phase 1 (transport engine only)

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

No real nuclear data, energy dependence or plotting yet.

## Usage

```bash
./venv/bin/python -m mcslab                                # small demo run
./venv/bin/python -m pytest tests/test_physics.py -v       # analytic validation
./venv/bin/python tests/test_regression.py compare         # bit-identical regression
```

## Limitations (Phase 1)

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
