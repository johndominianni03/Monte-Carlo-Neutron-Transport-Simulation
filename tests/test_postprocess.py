"""Deterministic checks of mcslab/postprocess.py (plan check 6): every
formula against a hand computation on a small synthetic result, to 1e-14
relative. No transport."""
import math
from types import SimpleNamespace

import numpy as np

from mcslab import postprocess as P
from mcslab import tallies as T
from mcslab.depth_mesh import make_mesh

REL = 1e-14
B, N_PER = 3, 10


def _fake():
    """Two layers: "W" (nuclide A, 2 bins of 0.5 cm) and "FLiBe" (nuclide B,
    1 bin of 2 cm); 3 batches of 10 histories."""
    mesh = make_mesh([0.0, 1.0, 3.0], (2, 1))
    tally = np.zeros((B, 3, 3, T.N_RESP, T.N_EST))
    flux = np.zeros((B, 3, T.N_EST))
    for b in range(B):
        for i in range(3):
            k = 0 if i < 2 else 1
            for s in range(T.N_RESP):
                v = (b + 1) * (i + 2) * (s + 3) * 1.5
                tally[b, i, k, s, T.EST_TL] = v
                tally[b, i, 2, s, T.EST_TL] = v
            flux[b, i, T.EST_TL] = (b + 2) * (i + 1)
    mats = (SimpleNamespace(name="W", nuclide_names=("A",)),
            SimpleNamespace(name="FLiBe", nuclide_names=("B",)))
    cfg = SimpleNamespace(histories_per_batch=N_PER, seed=7,
                          geometry=SimpleNamespace(region_materials=mats))
    return SimpleNamespace(mesh=mesh, tally=tally, mesh_flux=flux, config=cfg,
                           tally_nuclides=("A", "B", "total"),
                           region_atom_density=np.array([0.06, 0.08]))


def _close(a, b):
    assert math.isclose(a, b, rel_tol=REL), (a, b)


def _mean_se(x):
    x = [float(v) for v in x]
    m = sum(x) / len(x)
    se = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) * (len(x) - 1)))
    return m, se


def test_source_rate_and_fpy():
    S = P.source_rate()
    _close(S, 100.0 / (14.1e6 * 1.602176634e-19))
    assert abs(S / 4.4266e13 - 1.0) < 1e-4
    _close(P.SECONDS_PER_FPY, 365.25 * 24 * 3600)
    _close(P.source_rate(2.0, 2.45e6), 200.0 / (2.45e6 * 1.602176634e-19))


def test_summary_formulas_by_hand():
    res = _fake()
    out = P.summarize(res)
    S = 100.0 / (14.1e6 * 1.602176634e-19)
    phi = S * 365.25 * 86400.0
    t = res.tally[:, :, 2, :, T.EST_TL] / N_PER          # (B, bins, resp), total slot
    N_w = 0.06e24
    w = out["layers"]["W"]
    # dpa: 0.8 T_444 / (2 * 90 eV) / (N dx) * Phi_FPY, front bin 0.5 cm, avg over 1 cm
    m, se = _mean_se([0.8 * t[b, 0, T.R_DAMAGE] / (2 * 90.0) / (N_w * 0.5) * phi
                      for b in range(B)])
    _close(w["dpa_per_fpy_front"]["mean"], m)
    _close(w["dpa_per_fpy_front"]["se"], se)
    m, _ = _mean_se([0.8 * (t[b, 0, T.R_DAMAGE] + t[b, 1, T.R_DAMAGE]) / (2 * 90.0)
                     / (N_w * 1.0) * phi for b in range(B)])
    _close(w["dpa_per_fpy_avg"]["mean"], m)
    # He appm
    he = [t[b, 0, T.R_HE4] / (N_w * 0.5) * 1e6 for b in range(B)]
    dpa = [0.8 * t[b, 0, T.R_DAMAGE] / (2 * 90.0) / (N_w * 0.5) for b in range(B)]
    _close(w["he_appm_per_fpy_front"]["mean"], _mean_se([h * phi for h in he])[0])
    # He/dpa: ratio estimator
    R = (sum(he) / B) / (sum(dpa) / B)
    se_r = math.sqrt(sum((h - R * d) ** 2 for h, d in zip(he, dpa)) / (B * (B - 1))) \
        / (sum(dpa) / B)
    _close(w["he_appm_per_dpa_front"]["mean"], R)
    _close(w["he_appm_per_dpa_front"]["se"], se_r)
    # heating density, both kerma variants
    for s, key in ((T.R_HEATING, "heating_301"), (T.R_HEATING_LOCAL, "heating_901")):
        _close(w[f"{key}_W_cm3_front"]["mean"],
               _mean_se([t[b, 0, s] / 0.5 * S * 1.602176634e-19 for b in range(B)])[0])
        _close(w[f"{key}_eV_per_source"]["mean"],
               _mean_se([t[b, 0, s] + t[b, 1, s] for b in range(B)])[0])
    # flux
    _close(w["flux_front"]["mean"],
           _mean_se([res.mesh_flux[b, 0, T.EST_TL] / N_PER / 0.5 for b in range(B)])[0])
    # FLiBe: no dpa / appm (liquid, not in E_d); tritium by nuclide
    f = out["layers"]["FLiBe"]
    assert not any(k.startswith(("dpa", "he_appm")) for k in f)
    _close(f["tritium_per_source"]["mean"], _mean_se([t[b, 2, T.R_H3] for b in range(B)])[0])
    _close(f["tritium_per_source_by_nuclide"]["B"]["mean"],
           _mean_se([res.tally[b, 2, 1, T.R_H3, T.EST_TL] / N_PER for b in range(B)])[0])
    # totals
    _close(out["totals"]["heating_301_fraction_of_source_energy"]["mean"],
           _mean_se([t[b, :, T.R_HEATING].sum() / 14.1e6 for b in range(B)])[0])
    _close(out["totals"]["tritium_per_source"]["mean"],
           _mean_se([t[b, :, T.R_H3].sum() for b in range(B)])[0])
    assert out["inputs"]["E_d_eV"] == {"W": 90.0, "Fe": 40.0}
    _close(out["inputs"]["fluence_per_fpy_n_cm2"], phi)


def test_custom_displacement_energy_and_profiles():
    res = _fake()
    a = P.summarize(res)["layers"]["W"]["dpa_per_fpy_front"]["mean"]
    b = P.summarize(res, E_d={"W": 45.0})["layers"]["W"]["dpa_per_fpy_front"]["mean"]
    _close(b, 2.0 * a)
    prof = P.profiles(res)
    mean, _ = prof["dpa_per_fpy"]
    _close(mean[0], a)
    assert np.isnan(mean[2]) and np.isnan(prof["he_appm_per_fpy"][0][2])
    _close(prof["heating_901_W_cm3"][0][1],
           P.summarize(res)["layers"]["W"]["heating_901_W_cm3_avg"]["mean"] * 2.0
           - prof["heating_901_W_cm3"][0][0])
