"""Small demo run:  ./venv/bin/python -m mcslab"""
import time

from .config import RunConfig, run
from .geometry import SlabGeometry
from .materials import VOID, Material
from .rng import STRIDE
from .sources import BeamSource


def main():
    shield = Material("absorber", sigma_s=0.5, sigma_a=1.0)
    moderator = Material("scatterer", sigma_s=0.9, sigma_a=0.1)
    geom = SlabGeometry([0.0, 1.0, 1.5, 4.0], [shield, VOID, moderator])
    cfg = RunConfig(geom, BeamSource(), n_batches=50, histories_per_batch=20_000, seed=1)

    t0 = time.perf_counter()
    res = run(cfg)
    dt = time.perf_counter() - t0

    print(f"Beam on a 3-region slab: {cfg.n_histories:,} histories in "
          f"{cfg.n_batches} batches, {dt:.2f} s (includes JIT load)")
    print(f"{'region':<10}{'x range':<14}{'flux (TL)':>22}{'flux (coll)':>22}"
          f"{'absorptions':>22}")
    ftl, fco, ab = (res.stats(a) for a in (res.flux_tl, res.flux_coll, res.absorptions))
    for r, m in enumerate(geom.region_materials):
        rng_s = f"[{geom.bounds[r]:g}, {geom.bounds[r + 1]:g}]"
        print(f"{m.name:<10}{rng_s:<14}{fmt(ftl, r):>22}{fmt(fco, r):>22}{fmt(ab, r):>22}")
    for name, a in (("left leakage", res.leakage_left), ("right leakage", res.leakage_right)):
        m, s = res.stats(a)
        print(f"{name:<24}{m:.6f} +/- {s:.6f}")
    print(f"max random numbers per history: {res.max_draws} (STRIDE = {STRIDE}); "
          f"lost histories: {res.lost}")


def fmt(ms, r):
    m, s = ms
    return f"{m[r]:.5f} +/- {s[r]:.5f}"


if __name__ == "__main__":
    main()
