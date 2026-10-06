# Monte Carlo Neutron Transport Simulation

A 1D Monte Carlo neutron transport code in Python and Numba (the Python
package is named `mcslab`). It follows
14.1 MeV D-T fusion neutrons through a layered wall consisting of tungsten, FLiBe molten
salt and iron; it uses ENDF/B-VIII.0 nuclear data, and scores tritium production,
displacement damage, helium production and heating. Its physics mirrors
OpenMC 0.16.0, and it is benchmarked against OpenMC on the same data.

![Neutron tracks in W | FLiBe | Fe](docs/figures/tracks.gif)

*40 source neutrons and their secondaries: depth against log energy over time.*

## What it does

- **Transport.** Continuous-energy neutron transport in a slab of layers,
  with elastic and inelastic scattering, (n,2n)-type multiplication and
  free-gas thermal motion of the target nuclei.
- **Data.** Reads the official OpenMC HDF5 files for ENDF/B-VIII.0 directly
  with h5py. The `openmc` package is not a dependency.
- **Tallies.** Flux, heating, damage energy, helium and tritium production
  per depth bin and per nuclide, converted to dpa, He appm and W/cm^3 at a
  1 MW/m^2 neutron wall loading.
- **Reproducibility.** Every history has its own random-number stream, so
  results are bit-identical however the batches are split. Each kernel has a
  byte-exact regression reference.

## Results at a glance

Reference wall: W 0.5 cm | FLiBe 20 cm | Fe 10 cm, 14.1 MeV beam at normal
incidence.

| Quantity | Value |
|---|---|
| Agreement with OpenMC 0.16.0 | 82 pre-declared checks over 4 problems, none beyond 3 standard errors (largest \|z\| 2.63) |
| Tritium per source neutron, vacuum on the plasma side | 0.298 +- 0.002 |
| Tritium per source neutron, reflective plasma side | 0.682 +- 0.003 |
| Tritium per source neutron, 75-100 cm of FLiBe, reflective | about 1.23 (idealised upper bound) |
| Damage at 1 MW/m^2, front of each layer | W 0.82 dpa, Fe 2.16 dpa per full-power year |

![Tritium per source neutron vs FLiBe thickness and Li-6 enrichment](docs/figures/blanket_sweep.png)

The sweep is an idealised 1D upper bound, not the tritium breeding ratio of a
real design: a reflective plane returns every neutron that leaves the plasma
side, and there is no structure, coolant, ports or gaps.

## Quick start

Developed on Python 3.9.6 (macOS, Apple silicon). Dependencies are pinned
exactly, because the bit-identical references depend on them.

```bash
git clone https://github.com/johndominianni03/Monte-Carlo-Neutron-Transport-Simulation.git
cd Monte-Carlo-Neutron-Transport-Simulation
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

./venv/bin/python -m mcslab     # demo: needs no nuclear data
./venv/bin/python -m pytest     # tests that need nuclear data are skipped without it
```

The demo runs a one-group beam through a three-layer slab and prints a table
of flux, absorptions and leakage. It prints text only and writes no files.
Run it as a module from the repository root, as shown: `python
mcslab/__main__.py` fails, because the package uses relative imports.

## Nuclear data

The data are not in the repository. Fetch the 14 nuclide files once:

```bash
scripts/fetch_data.sh           # streams a 3.38 GB archive, keeps only the 14 files
export MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5
```

The files are verified against the checksums in
`scripts/checksums/endfb-viii.0.sha256`.

## Running a real problem

```python
import dataclasses
from mcslab import ce_materials as cm
from mcslab.config_ce import MonoEnergetic
from mcslab.config_kin import KinRunConfig, run_kin
from mcslab.geometry import SlabGeometry
from mcslab.nucdata import Library
from mcslab.sources import BeamSource

flibe = dataclasses.replace(cm.flibe(temperature_K=900.0), temperature=900.0)
cfg = KinRunConfig(
    SlabGeometry([0.0, 0.5, 20.5, 30.5], [cm.tungsten(), flibe, cm.iron()]),
    BeamSource(), MonoEnergetic(14.1e6), Library.open(),
    n_batches=20, histories_per_batch=5000)
res = run_kin(cfg)
print(res.balance())    # every neutron is accounted for: the residual is exactly 0
```

## Reproducing the figures and results

All commands run from the repository root with `./venv/bin/python`.

| Command | Writes | Needs data |
|---|---|---|
| `scripts/plot_xs.py` | four cross-section figures | yes |
| `scripts/record_tracks.py`, then `scripts/animate_tracks.py` | `docs/figures/tracks.gif`, `tracks.png` | yes |
| `scripts/phase3_results.py` | `docs/phase3_results.json`, `docs/figures/phase3_profiles.png` | yes |
| `scripts/blanket_sweep.py` | `docs/blanket_sweep.csv`, `docs/figures/blanket_sweep.png` | yes |
| `benchmark/compare.py --figures` | `benchmark/results/comparison.json`, `docs/figures/phase4_*.png` | no |

`benchmark/compare.py` works from the committed result files, so the OpenMC
comparison can be re-checked without OpenMC or the nuclear data installed.
Re-running the benchmark itself is described at the top of
`benchmark/run_mcslab.py` and `benchmark/run_openmc.py`.

## Repository layout

```
mcslab/      the package: data reader, cross-section lookup, collision physics,
             transport kernels, tallies, post-processing
scripts/     data download, inventories, figures, results
benchmark/   the OpenMC comparison: problem file, both runners, results, checks
tests/       analytic tests, bit-identical regression references, benchmark checks
docs/        development rules, deviations from OpenMC, phase plans, figures
```

## How it was checked

- **Analytic tests.** Statistical checks against exact results, each with its
  seed and its 3 standard-error threshold fixed before the test was written.
- **Bit-identical regression.** Refactors must reproduce the committed
  reference arrays byte for byte.
- **Code-to-code benchmark.** Four problems run in both mcslab and OpenMC
  0.16.0 with the same data, densities and settings.

The rules are in `docs/development.md`. Every known difference from OpenMC is
listed in `docs/deviations_from_openmc.md`.

## Limitations

- 1D slab with a beam at normal incidence.
- No photon transport, so heating is reported as a range (MT 301 to MT 901),
  not a single number.
- No unresolved-resonance probability tables and no thermal scattering data.
- Analog and serial: no variance reduction, not parallelised.
- Verified against OpenMC on the same data, not validated against experiment.

The full write-up, with every result table, the benchmark discussion and the
complete list of limitations, is in `docs/report.md`.

## Acknowledgements

The physics follows [OpenMC](https://github.com/openmc-dev/openmc) v0.16.0
(MIT license); the samplers in `mcslab/collision.py` and
`mcslab/distributions.py` are transcribed from its source and cite the
original functions. OpenMC's license is reproduced in `LICENSE-OpenMC`.
Nuclear data: ENDF/B-VIII.0, in the HDF5 library
distributed by the OpenMC project.