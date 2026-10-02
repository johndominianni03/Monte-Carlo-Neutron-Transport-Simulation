#!/usr/bin/env python3
"""Generate docs/law_inventory.md: every outgoing-neutron law in the data.

    MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5 ./venv/bin/python scripts/law_inventory.py

The scan reads the HDF5 files with h5py only. It does not use
mcslab.nucdata, so it is an independent account of what the reader and the
samplers must handle. tests/test_laws.py compares the packed physics
against scan() row by row.

Scope: every non-redundant reaction with an outgoing neutron, in the 14
nuclides mcslab uses (W, Fe, FLiBe = Li/Be/F, and H1), at 294 K. Photon
products are listed only as a count (mcslab transports neutrons only).
"""
import hashlib
import os
import sys
from collections import OrderedDict

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "law_inventory.md")
NUCLIDES = ["H1", "Fe54", "Fe56", "Fe57", "Fe58", "W180", "W182", "W183", "W184",
            "W186", "Li6", "Li7", "Be9", "F19"]
TEMPERATURE = "294K"
E14 = 14.1e6
INTERP = {1: "histogram", 2: "lin-lin"}


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _codes(values):
    return tuple(sorted({int(round(float(v))) for v in np.asarray(values).ravel()}))


def _ints(values):
    return "/".join(str(v) for v in values)


def _interp_text(codes):
    return "/".join(INTERP.get(c, f"code {c}") for c in codes)


def _regions(attr):
    """(breakpoints, codes) of an `interpolation` attribute stored as a
    (2, n_regions) array."""
    a = np.asarray(attr)
    return tuple(int(v) for v in a[0]), tuple(int(v) for v in a[1])


def _tables(ds, n):
    data = ds[()]
    offs = [int(o) for o in ds.attrs["offsets"]] + [data.shape[1]]
    return data, [(offs[i], offs[i + 1]) for i in range(n)]


def _cdf_mismatch(x, p, c, interp):
    """max |c / c[-1] - C / C[-1]|, C the integral of the PDF p alone."""
    if interp == 1:
        C = np.concatenate([[0.0], np.cumsum(p[:-1] * np.diff(x))])
    else:
        C = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return float(np.max(np.abs(c / c[-1] - C / C[-1])))


def _angle(group):
    mu = group["mu"]
    data, spans = _tables(mu, group["energy"].shape[0])
    ints = [int(v) for v in mu.attrs["interpolation"]]
    worst = max(_cdf_mismatch(data[0, a:b], data[1, a:b], data[2, a:b], it)
                for (a, b), it in zip(spans, ints))
    return {"n_energy": int(group["energy"].shape[0]), "interp": _codes(ints),
            "cdf_mismatch": worst}


def _continuous(group):
    ein = group["energy"]
    bp, inc = _regions(ein.attrs["interpolation"])
    dist = group["distribution"]
    data, spans = _tables(dist, ein.shape[0])
    ints = [int(v) for v in dist.attrs["interpolation"]]
    tabs = [(data[0, a:b], data[1, a:b], data[2, a:b]) for a, b in spans]
    return {"n_energy": int(ein.shape[0]), "e_range": (float(ein[0]), float(ein[-1])),
            "incident": inc, "n_regions": len(bp), "eout_interp": _codes(ints),
            "n_discrete": _codes(dist.attrs["n_discrete_lines"]),
            "cdf_mismatch": max(_cdf_mismatch(x, p, c, it) for (x, p, c), it in zip(tabs, ints)),
            "min_points": min(x.size for x, _, _ in tabs),
            "first_e_out": sorted({float(x[0]) for x, _, _ in tabs}),
            "c_ends": sorted({(float(c[0]), float(c[-1])) for _, _, c in tabs}),
            "last_p": max(float(p[-1]) for _, p, _ in tabs),
            "incident_energies": ein[()]}


def _correlated(group):
    ein = group["energy"]
    bp, inc = _regions(ein.attrs["interpolation"])
    eo = group["energy_out"]
    data, spans = _tables(eo, ein.shape[0])
    ints = [int(v) for v in eo.attrs["interpolation"]]
    mu = group["mu"][()]
    mu_off = np.round(data[4]).astype(int)
    mu_end = np.append(mu_off[1:], mu.shape[1])
    mu_int = np.round(data[3]).astype(int)
    worst_mu = max(_cdf_mismatch(mu[0, a:b], mu[1, a:b], mu[2, a:b], int(it))
                   for a, b, it in zip(mu_off, mu_end, mu_int))
    tabs = [(data[0, a:b], data[1, a:b], data[2, a:b]) for a, b in spans]
    return {"n_energy": int(ein.shape[0]), "e_range": (float(ein[0]), float(ein[-1])),
            "incident": inc, "n_regions": len(bp), "eout_interp": _codes(ints),
            "mu_interp": _codes(mu_int), "n_discrete": _codes(eo.attrs["n_discrete_lines"]),
            "cdf_mismatch": max(_cdf_mismatch(x, p, c, it) for (x, p, c), it in zip(tabs, ints)),
            "cdf_mismatch_mu": worst_mu,
            "min_points": min(x.size for x, _, _ in tabs),
            "first_e_out": sorted({float(x[0]) for x, _, _ in tabs}),
            "c_ends": sorted({(float(c[0]), float(c[-1])) for _, _, c in tabs}),
            "last_p": max(float(p[-1]) for _, p, _ in tabs),
            "incident_energies": ein[()]}


def describe_distribution(d):
    """-> dict with 'kind' in {"angle-only", "level", "continuous",
    "correlated", <other type>} and the details of the law."""
    typ = _s(d.attrs["type"])
    out = {"type": typ, "has_applicability": "applicability" in d}
    if typ == "correlated":
        out.update(kind="correlated", **_correlated(d))
    elif typ == "uncorrelated":
        out["angle"] = _angle(d["angle"]) if "angle" in d else None
        if "energy" not in d:
            out["kind"] = "angle-only"
        else:
            etype = _s(d["energy"].attrs["type"])
            if etype == "level":
                out["kind"] = "level"
            elif etype == "continuous":
                out.update(kind="continuous", **_continuous(d["energy"]))
            else:
                out["kind"] = f"uncorrelated/{etype}"
    else:
        out["kind"] = typ
    if out["has_applicability"]:
        a = d["applicability"]
        out["applicability"] = (tuple(int(v) for v in np.asarray(a.attrs["breakpoints"]).ravel()),
                                _codes(a.attrs["interpolation"]), a[()])
    return out


def describe_yield(y):
    t = _s(y.attrs["type"])
    if t == "Polynomial":
        coef = tuple(float(v) for v in np.asarray(y[()]).ravel())
        return {"type": t, "coefficients": coef}
    data = y[()]
    return {"type": t, "breakpoints": tuple(int(v) for v in np.asarray(y.attrs["breakpoints"]).ravel()),
            "interp": _codes(y.attrs["interpolation"]), "x": data[0], "y": data[1]}


def scan(root):
    """-> list of row dicts, one per non-redundant reaction with an outgoing
    neutron, in NUCLIDES order and increasing MT."""
    rows = []
    for name in NUCLIDES:
        path = os.path.join(root, "neutron", f"{name}.h5")
        with h5py.File(path, "r") as f:
            g = f[name]
            E = g["energy"][TEMPERATURE][()]
            for key in sorted(g["reactions"], key=lambda k: int(g["reactions"][k].attrs["mt"])):
                r = g["reactions"][key]
                if bool(r.attrs["redundant"]):
                    continue
                prods = [r[k] for k in sorted(r) if k.startswith("product_")]
                neut = [p for p in prods if _s(p.attrs["particle"]) == "neutron"]
                if not neut:
                    continue
                xs = r[TEMPERATURE]["xs"][()]
                thr = int(r[TEMPERATURE]["xs"].attrs["threshold_idx"])
                pos = np.flatnonzero(xs > 0.0)
                e_pos = float(E[max(thr + int(pos[0]) - 1, 0)]) if pos.size else float("inf")
                s14 = float(np.interp(E14, E[thr:], xs, left=0.0))
                p0 = prods[0]
                n_dist = int(p0.attrs["n_distribution"])
                rows.append({
                    "nuclide": name, "mt": int(r.attrs["mt"]), "label": _s(r.attrs["label"]),
                    "cm": bool(r.attrs["center_of_mass"]), "q": float(r.attrs["Q_value"]),
                    "xs_positive_above": e_pos, "sigma14": s14,
                    "product0_is_neutron": _s(p0.attrs["particle"]) == "neutron",
                    "n_neutron_products": len(neut),
                    "n_other_products": len(prods) - len(neut),
                    "yield": describe_yield(p0["yield"]),
                    "distributions": [describe_distribution(p0[f"distribution_{j}"])
                                      for j in range(n_dist)],
                })
    return rows


def law_label(d):
    k = d["kind"]
    ang = d.get("angle")
    angle = ("isotropic" if ang is None else f"tabular angle ({_interp_text(ang['interp'])})")
    if k == "angle-only":
        return f"uncorrelated: {angle}, no energy law"
    if k == "level":
        return f"uncorrelated: {angle} + level"
    if k == "continuous":
        return (f"uncorrelated: {angle} + continuous tabular (ACE law 4); incident "
                f"{_interp_text(d['incident'])} ({d['n_regions']} region), E_out "
                f"{_interp_text(d['eout_interp'])}, discrete lines {_ints(d['n_discrete'])}")
    if k == "correlated":
        return (f"correlated (ACE law 61); incident {_interp_text(d['incident'])} "
                f"({d['n_regions']} region), E_out {_interp_text(d['eout_interp'])}, mu "
                f"{_interp_text(d['mu_interp'])}, discrete lines {_ints(d['n_discrete'])}")
    return k


def yield_text(y):
    if y["type"] == "Polynomial":
        c = y["coefficients"]
        return f"{c[0]:g}" if len(c) == 1 else f"polynomial {c}"
    return (f"Tabulated1D ({_interp_text(y['interp'])}, {len(y['breakpoints'])} region), "
            f"y in [{y['y'].min():.4g}, {y['y'].max():.4g}]")


def _fmt(x, spec=".4g"):
    return "0" if x == 0 else format(x, spec)


def _mts(mts):
    """Compact MT list: runs of consecutive MTs as a-b."""
    mts = sorted(mts)
    out, start, prev = [], mts[0], mts[0]
    for m in mts[1:] + [None]:
        if m is not None and m == prev + 1:
            prev = m
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if m is not None:
            start = prev = m
    return ", ".join(out)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = os.environ.get("MCSLAB_DATA")
    if not root:
        sys.exit("set MCSLAB_DATA (see scripts/fetch_data.sh)")
    root = os.path.expanduser(root)
    rows = scan(root)
    L = []
    w = L.append

    w("# Secondary-neutron law inventory (Phase 2b)")
    w("")
    w("Generated by `scripts/law_inventory.py` from the ENDF/B-VIII.0 OpenMC HDF5 files at "
      f"{TEMPERATURE}, read with h5py only (not through `mcslab.nucdata`). It lists every "
      "non-redundant reaction with an outgoing neutron and the law that samples it. "
      "`tests/test_laws.py` checks the packed physics against this scan row by row.")
    w("")
    w("| file | sha256 |")
    w("|---|---|")
    for n in NUCLIDES:
        w(f"| `neutron/{n}.h5` | `{_sha256(os.path.join(root, 'neutron', n + '.h5'))}` |")
    w("")

    # ------------------------------------------------------------------ by law
    groups = OrderedDict()
    for r in rows:
        for j, d in enumerate(r["distributions"]):
            key = (law_label(d), "CM" if r["cm"] else "lab")
            groups.setdefault(key, OrderedDict()).setdefault(r["nuclide"], []).append(r["mt"])
    w("## Laws, by type")
    w("")
    w("A reaction with two distributions (F19 MT 16) is listed once per distribution.")
    w("")
    w("| law | frame | reactions | where |")
    w("|---|---|---|---|")
    for (label, frame), where in sorted(groups.items(), key=lambda kv: (-sum(len(v) for v in kv[1].values()), kv[0])):
        n = sum(len(v) for v in where.values())
        text = "; ".join(f"{nuc} MT {_mts(m)}" for nuc, m in where.items())
        w(f"| {label} | {frame} | {n} | {text} |")
    w("")

    ylds = OrderedDict()
    for r in rows:
        ylds.setdefault(yield_text(r["yield"]) if r["yield"]["type"] == "Polynomial"
                        else "Tabulated1D", []).append(f"{r['nuclide']}:{r['mt']}")
    w("**Yields of the neutron product:**")
    w("")
    w("| yield | reactions | MTs |")
    w("|---|---|---|")
    for y, where in ylds.items():
        mts = sorted({int(s.split(':')[1]) for s in where})
        w(f"| {y} | {len(where)} | {_mts(mts)} |")
    w("")

    # ------------------------------------------------------------------ findings
    kinds = sorted({d["kind"] for r in rows for d in r["distributions"]})
    extrap = [(r["nuclide"], r["mt"]) for r in rows for d in r["distributions"]
              if d["kind"] in ("correlated", "continuous")
              and r["xs_positive_above"] < d["e_range"][0] * (1.0 - 1e-12)]
    law_rows = [d for r in rows for d in r["distributions"] if d["kind"] in ("correlated", "continuous")]
    single_app = sorted({(r["nuclide"], r["mt"]) for r in rows
                         if len(r["distributions"]) == 1 and r["distributions"][0]["has_applicability"]})
    angle_mm = max(d["angle"]["cdf_mismatch"] for r in rows for d in r["distributions"]
                   if d.get("angle") is not None)
    w("## Findings")
    w("")
    w(f"- Distribution kinds present: {', '.join(kinds)}. No Kalbach-Mann, N-body, "
      "evaporation, Maxwell or Watt law occurs.")
    w(f"- Product 0 is the neutron in every reaction: "
      f"{all(r['product0_is_neutron'] for r in rows)}. Reactions with more than one neutron "
      f"product: {sum(r['n_neutron_products'] > 1 for r in rows)}.")
    w(f"- Law 4 / law 61 tables: incident interpolation "
      f"{sorted({(d['n_regions'],) + d['incident'] for d in law_rows})} (regions, code); "
      f"discrete lines {sorted({v for d in law_rows for v in d['n_discrete']})}; "
      f"smallest table {min(d['min_points'] for d in law_rows)} points; first E_out "
      f"{sorted({v for d in law_rows for v in d['first_e_out']})} eV; (c first, c last) "
      f"{sorted({v for d in law_rows for v in d['c_ends']})}; largest last p "
      f"{max(d['last_p'] for d in law_rows):g}.")
    w(f"- Incident energies strictly increasing in every law 4 / law 61 table: "
      f"{all(bool(np.all(np.diff(d['incident_energies']) > 0)) for d in law_rows)}.")
    w(f"- Reactions whose cross section is positive below their law's first incident energy "
      f"(would need extrapolation): {len(extrap)}{' ' + str(extrap) if extrap else ''}.")
    w(f"- Stored CDF c vs the integral of the PDF p alone, max |c/c_last - C/C_last|: "
      f"E_out tables {max(d['cdf_mismatch'] for d in law_rows):.2e}; correlated mu tables "
      f"{max(d['cdf_mismatch_mu'] for d in law_rows if d['kind'] == 'correlated'):.2e}; "
      f"tabular angle distributions {angle_mm:.2e}.")
    w(f"- Single distributions that still carry an `applicability` dataset: {len(single_app)}. "
      "OpenMC reads applicability only when a product has more than one distribution "
      "(`src/reaction_product.cpp`), and so does mcslab.")
    w("")

    # ------------------------------------------------------------------ callouts
    by = {(r["nuclide"], r["mt"]): r for r in rows}
    w("## Reactions that matter for tritium breeding (Phase 3)")
    w("")
    be = by[("Be9", 16)]
    bd = be["distributions"][0]
    be_tot = sum(r["sigma14"] for r in rows if r["nuclide"] == "Be9")
    w(f"- **Be-9 (n,2n), MT 16**: the only neutron-multiplying Be-9 reaction in the file "
      f"(no MT 875-891). sigma(14.1 MeV) = {be['sigma14']:.4g} b. Frame "
      f"{'CM' if be['cm'] else 'lab'}; {law_label(bd)}; {bd['n_energy']} incident energies "
      f"from {bd['e_range'][0]:.6g} to {bd['e_range'][1]:.6g} eV; yield "
      f"{yield_text(be['yield'])}. Its share of the scattering cross section at 14.1 MeV is "
      f"{be['sigma14'] / be_tot:.3f} (Be-9 absorption is small there).")
    li = [by[("Li7", m)] for m in range(52, 83) if ("Li7", m) in by]
    li_sum = sum(r["sigma14"] for r in li)
    w(f"- **Li-7 (n,n'alpha)t, MT 52-82**: {len(li)} discrete levels above the bound "
      f"0.478 MeV state (MT 51), each a level law in CM ("
      f"{sum(1 for r in li if r['distributions'][0]['angle']['interp'] == (2,))} with lin-lin "
      f"and {sum(1 for r in li if r['distributions'][0]['angle']['interp'] == (1,))} with "
      f"histogram angle tables). Their sum at 14.1 MeV is {li_sum:.4g} b; "
      "`docs/data_inventory.md` shows it equals the redundant MT 205. The HDF5 files do not "
      "keep ENDF's LR = 33 breakup flag, and mcslab transports neutrons only, so Phase 3 "
      "must score MT 205 (or MT 52-82 as a set).")
    w("- **Li-6 (n,t)alpha, MT 105**: no outgoing neutron, so an absorption (not listed above).")
    w("")

    # ------------------------------------------------------------------ per nuclide
    w("## Per nuclide")
    w("")
    w("Discrete levels (MT 51-90) are summarised in one row per nuclide. **xs > 0 above** is "
      "the grid point below the first positive cross section.")
    w("")
    for n in NUCLIDES:
        nr = [r for r in rows if r["nuclide"] == n]
        w(f"### {n}")
        w("")
        w("| MT | label | frame | Q (MeV) | xs > 0 above (MeV) | sigma(14.1 MeV) b | yield | law |")
        w("|---|---|---|---|---|---|---|---|")
        levels = [r for r in nr if 51 <= r["mt"] <= 90]
        for r in nr:
            if 51 <= r["mt"] <= 90:
                if r is not levels[0]:
                    continue
                kinds_l = OrderedDict()
                for lv in levels:
                    kinds_l.setdefault(law_label(lv["distributions"][0]), []).append(lv["mt"])
                law = "; ".join(f"MT {_mts(m)}: {k}" for k, m in kinds_l.items())
                s14 = sum(lv["sigma14"] for lv in levels)
                w(f"| {_mts([lv['mt'] for lv in levels])} | {len(levels)} levels | "
                  f"{'CM' if r['cm'] else 'lab'} | | | {_fmt(s14)} (sum) | "
                  f"{yield_text(r['yield'])} | {law} |")
                continue
            laws = "<br>".join(
                law_label(d) + (f"; applicability {_interp_text(d['applicability'][1])} "
                                f"{np.unique(d['applicability'][2][1]).tolist()}"
                                if len(r["distributions"]) > 1 else "")
                for d in r["distributions"])
            w(f"| {r['mt']} | {r['label']} | {'CM' if r['cm'] else 'lab'} | "
              f"{_fmt(r['q'] / 1e6, '.6g')} | {_fmt(r['xs_positive_above'] / 1e6, '.6g')} | "
              f"{_fmt(r['sigma14'])} | {yield_text(r['yield'])} | {laws} |")
        w("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(L).rstrip() + "\n")
    print(f"wrote {os.path.relpath(OUT, REPO)} ({len(rows)} reactions)")


if __name__ == "__main__":
    main()
