"""One-group materials: macroscopic cross sections in 1/cm."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Material:
    """One-group material. Sigma_t defaults to Sigma_s + Sigma_a; if given
    explicitly it must agree with that sum (relative tolerance 1e-12)."""
    name: str
    sigma_s: float
    sigma_a: float
    sigma_t: Optional[float] = None

    def __post_init__(self):
        if self.sigma_s < 0.0 or self.sigma_a < 0.0:
            raise ValueError(f"{self.name}: cross sections must be >= 0")
        total = self.sigma_s + self.sigma_a
        if self.sigma_t is None:
            object.__setattr__(self, "sigma_t", total)
        elif not math.isclose(self.sigma_t, total, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                f"{self.name}: Sigma_t={self.sigma_t} != Sigma_s + Sigma_a = {total}")

    @property
    def is_void(self) -> bool:
        return self.sigma_t == 0.0


VOID = Material("void", 0.0, 0.0)


def pack_materials(materials: Sequence[Material]):
    """Materials -> (sig_t, sig_s, sig_a) float64 arrays indexed by material id."""
    sig_t = np.array([m.sigma_t for m in materials], dtype=np.float64)
    sig_s = np.array([m.sigma_s for m in materials], dtype=np.float64)
    sig_a = np.array([m.sigma_a for m in materials], dtype=np.float64)
    return sig_t, sig_s, sig_a
