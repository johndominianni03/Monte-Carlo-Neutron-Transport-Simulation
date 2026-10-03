"""Post-processing of the Phase 3 response tallies (pure Python, no Numba).

Normalisation in a 1D slab. The slab is infinite sideways and the source
emits one neutron per cm^2 of wall, so every tally "per source neutron" is
per source neutron per cm^2 of wall, i.e. per unit source fluence. A depth
bin of width dx (cm) of a material with N atoms/cm^3 (= 1e24 x the
atoms/(b cm) used by the kernel) holds N dx atoms per cm^2 of wall.

T denotes a track-length tally of one bin, per source neutron
(KinResults.tally_batches): T_flux in cm, T_301 / T_901 / T_444 in eV,
T_205 in tritons, T_207 in alphas. Formulas (each applied to every batch,
then mean and SE over batches with tallies.batch_stats):

    flux (bin average)      phi = T_flux / dx                      (per unit fluence)
    damage energy per atom  e   = T_444 / (N dx)                   (eV cm^2)
    dpa (NRT)               d   = 0.8 e / (2 E_d)                  per unit fluence
    He appm                 h   = T_207 / (N dx) x 1e6             per unit fluence
    He / dpa                h / d                                  appm per dpa
    heating density         q   = T_301 / dx x S x e_J             W/cm^3 (same for 901)
    heating fraction        sum over all bins of T_301 / E_source  (same for 901)
    tritium                 sum of T_205 over bins (and per nuclide)

Scaling: S = P / (E_source e_J) source neutrons per cm^2 per s for a
neutron wall loading P (1 MW/m^2 = 100 W/cm^2; with E_source = 14.1 MeV,
S = 4.4266e13 /cm^2/s), and Phi_FPY = S x 3.15576e7 s for one full-power
year (365.25 days). dpa and He appm per FPY are d Phi_FPY and h Phi_FPY.

"Front" is the bin on the plasma side of a layer (smallest x); "avg" is
the layer average, sum T / sum dx (i.e. T summed over the layer's bins,
divided by the layer width; for dpa and appm also by N).

NRT model (M. J. Norgett, M. T. Robinson, I. M. Torrens, Nucl. Eng. Des.
33, 50 (1975); ASTM E693): displacements = 0.8 E_dam / (2 E_d). It is
applied to the *integrated* damage energy (the usual sigma_dpa =
0.8 sigma_444 / (2 E_d)), so NRT's low-energy steps (0 below E_d, 1 between
E_d and 2.5 E_d) cannot be applied after energy integration. E_d defaults
to ASTM E521's recommended values, 90 eV (W) and 40 eV (Fe), and is
recorded in the output. dpa and He appm are given only for the materials in
E_d (not for the liquid FLiBe). The damage energy itself (MT 444) comes
from NJOY HEATR with its own E_d threshold, which the data files do not
record.

The SE of a ratio of two batch means (He/dpa) uses the ratio estimator:
R = mean(a) / mean(b), SE = sqrt(sum_i (a_i - R b_i)^2 / (B (B - 1))) / mean(b).
"""
from __future__ import annotations

import json
import math
from typing import Dict, Optional

import numpy as np

from . import tallies as T

EV_TO_J = 1.602176634e-19          # J/eV, exact (SI 2019)
E_SOURCE_EV = 14.1e6               # D-T neutron energy used for the wall loading
SECONDS_PER_FPY = 365.25 * 86400.0  # one full-power year
NRT_EFFICIENCY = 0.8
E_D_ASTM_E521 = {"W": 90.0, "Fe": 40.0}   # eV, ASTM E521 recommended values
BARN_CM2 = 1.0e-24
W_CM2_PER_MW_M2 = 100.0


def source_rate(wall_loading_MW_m2: float = 1.0, e_source_eV: float = E_SOURCE_EV) -> float:
    """Source neutrons per cm^2 per second for a neutron wall loading in
    MW/m^2 of neutrons of energy e_source_eV."""
    return wall_loading_MW_m2 * W_CM2_PER_MW_M2 / (e_source_eV * EV_TO_J)


def _stat(x) -> Dict[str, float]:
    m, s = T.batch_stats(np.asarray(x, dtype=np.float64))
    return {"mean": float(m), "se": float(s)}


def _ratio(a, b) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    ma, mb = a.sum() / n, b.sum() / n
    r = ma / mb
    dev = a - r * b
    return {"mean": float(r), "se": float(math.sqrt((dev * dev).sum() / (n * (n - 1))) / mb)}


def _tl(res, resp, nuclide=None):
    """(B, n_bins) track-length tally per source neutron."""
    k = len(res.tally_nuclides) - 1 if nuclide is None else res.tally_nuclides.index(nuclide)
    return res.tally[:, :, k, resp, T.EST_TL] / float(res.config.histories_per_batch)


def summarize(res, E_d: Optional[Dict[str, float]] = None, wall_loading_MW_m2: float = 1.0,
              e_source_eV: float = E_SOURCE_EV, fpy_s: float = SECONDS_PER_FPY) -> dict:
    """Per-layer results of a run with response tallies, as a JSON-ready
    dict of {"mean", "se"} entries (see the module docstring)."""
    E_d = dict(E_D_ASTM_E521 if E_d is None else E_d)
    S = source_rate(wall_loading_MW_m2, e_source_eV)
    phi_fpy = S * fpy_s
    mesh = res.mesh
    width = mesh.bin_width
    region = mesh.bin_region
    n_per = float(res.config.histories_per_batch)
    flux = res.mesh_flux[:, :, T.EST_TL] / n_per
    tl = {s: _tl(res, s) for s in range(T.N_RESP)}
    out = {
        "inputs": {
            "wall_loading_MW_m2": wall_loading_MW_m2,
            "source_energy_eV": e_source_eV,
            "source_rate_n_cm2_s": S,
            "full_power_year_s": fpy_s,
            "fluence_per_fpy_n_cm2": phi_fpy,
            "eV_to_J": EV_TO_J,
            "nrt_efficiency": NRT_EFFICIENCY,
            "E_d_eV": E_d,
            "n_batches": int(res.tally.shape[0]),
            "histories_per_batch": int(n_per),
            "seed": int(res.config.seed),
            "depth_bins": list(mesh.counts),
            "estimator": "tracklength",
        },
        "layers": {},
    }
    names = [m.name for m in res.config.geometry.region_materials]
    for r, name in enumerate(names):
        bins = np.flatnonzero(region == r)
        dx = width[bins]
        L = float(dx.sum())
        N = float(res.region_atom_density[r]) / BARN_CM2          # atoms / cm^3
        f = bins[0]
        lay = {"x_cm": [float(mesh.bin_edges[bins[0]]), float(mesh.bin_edges[bins[-1] + 1])],
               "front_bin_cm": [float(mesh.bin_edges[f]), float(mesh.bin_edges[f + 1])],
               "atoms_per_cm3": N}
        lay["flux_front"] = _stat(flux[:, f] / width[f])
        lay["flux_avg"] = _stat(flux[:, bins].sum(axis=1) / L)
        for s, key in ((T.R_HEATING, "heating_301"), (T.R_HEATING_LOCAL, "heating_901")):
            lay[f"{key}_eV_per_source"] = _stat(tl[s][:, bins].sum(axis=1))
            lay[f"{key}_W_cm3_front"] = _stat(tl[s][:, f] / width[f] * S * EV_TO_J)
            lay[f"{key}_W_cm3_avg"] = _stat(tl[s][:, bins].sum(axis=1) / L * S * EV_TO_J)
        lay["tritium_per_source"] = _stat(tl[T.R_H3][:, bins].sum(axis=1))
        nucs = [n for n in res.config.geometry.region_materials[r].nuclide_names]
        lay["tritium_per_source_by_nuclide"] = {
            n: _stat(_tl(res, T.R_H3, n)[:, bins].sum(axis=1)) for n in nucs}
        lay["helium_atoms_per_source"] = _stat(tl[T.R_HE4][:, bins].sum(axis=1))
        lay["absorption_per_source"] = _stat(tl[T.R_ABSORPTION][:, bins].sum(axis=1))
        lay["damage_energy_eV_per_source"] = _stat(tl[T.R_DAMAGE][:, bins].sum(axis=1))
        if name in E_d:
            k_dpa = NRT_EFFICIENCY / (2.0 * E_d[name])
            d_front = k_dpa * tl[T.R_DAMAGE][:, f] / (N * width[f])
            d_avg = k_dpa * tl[T.R_DAMAGE][:, bins].sum(axis=1) / (N * L)
            h_front = tl[T.R_HE4][:, f] / (N * width[f]) * 1e6
            h_avg = tl[T.R_HE4][:, bins].sum(axis=1) / (N * L) * 1e6
            lay["dpa_per_fpy_front"] = _stat(d_front * phi_fpy)
            lay["dpa_per_fpy_avg"] = _stat(d_avg * phi_fpy)
            lay["he_appm_per_fpy_front"] = _stat(h_front * phi_fpy)
            lay["he_appm_per_fpy_avg"] = _stat(h_avg * phi_fpy)
            lay["he_appm_per_dpa_front"] = _ratio(h_front, d_front)
            lay["he_appm_per_dpa_avg"] = _ratio(h_avg, d_avg)
        out["layers"][name] = lay
    out["totals"] = {
        "heating_301_fraction_of_source_energy": _stat(tl[T.R_HEATING].sum(axis=1)
                                                       / e_source_eV),
        "heating_901_fraction_of_source_energy": _stat(tl[T.R_HEATING_LOCAL].sum(axis=1)
                                                       / e_source_eV),
        "tritium_per_source": _stat(tl[T.R_H3].sum(axis=1)),
    }
    return out


def profiles(res, E_d: Optional[Dict[str, float]] = None, wall_loading_MW_m2: float = 1.0,
             e_source_eV: float = E_SOURCE_EV, fpy_s: float = SECONDS_PER_FPY) -> dict:
    """Per-bin (mean, SE) arrays for plotting: dpa and He appm per FPY (NaN
    outside the materials in E_d), heating density (W/cm^3, 301 and 901),
    tritium production density (tritons per source neutron per cm)."""
    E_d = dict(E_D_ASTM_E521 if E_d is None else E_d)
    S = source_rate(wall_loading_MW_m2, e_source_eV)
    phi_fpy = S * fpy_s
    mesh = res.mesh
    width = mesh.bin_width
    names = [m.name for m in res.config.geometry.region_materials]
    N = np.array([res.region_atom_density[r] / BARN_CM2 for r in mesh.bin_region])
    k_dpa = np.array([NRT_EFFICIENCY / (2.0 * E_d[names[r]]) if names[r] in E_d else np.nan
                      for r in mesh.bin_region])
    t444, t207 = _tl(res, T.R_DAMAGE), _tl(res, T.R_HE4)
    out = {"edges_cm": mesh.bin_edges, "region": mesh.bin_region,
           "region_names": tuple(names)}
    for key, batches in (
            ("dpa_per_fpy", k_dpa * t444 / (N * width) * phi_fpy),
            ("he_appm_per_fpy", np.where(np.isnan(k_dpa), np.nan, 1.0)
             * t207 / (N * width) * 1e6 * phi_fpy),
            ("heating_301_W_cm3", _tl(res, T.R_HEATING) / width * S * EV_TO_J),
            ("heating_901_W_cm3", _tl(res, T.R_HEATING_LOCAL) / width * S * EV_TO_J),
            ("tritium_per_source_per_cm", _tl(res, T.R_H3) / width)):
        with np.errstate(invalid="ignore"):
            out[key] = T.batch_stats(batches)
    return out


def write_json(summary: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
