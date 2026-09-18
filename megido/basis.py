"""Smooth positive correction to the analytic acceptance.

B(d) = A_geom(d) * exp(S(d)). A_geom carries the geometry exactly and has no
free parameters; S absorbs per-channel efficiency structure — the per-ASIC gain
offsets Phase 1 measured, chiefly ASIC 3 sitting about 15% high.

S is deliberately low-dimensional: 64 coefficients against roughly 1800 occupied
analysis bins. A basis flexible enough to follow the data bin by bin would
absorb the rock's absorption into the detector response and quietly return a
flat sky.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothBasis:
    centers: np.ndarray
    sigma: float

    @property
    def n_coeff(self) -> int:
        return int(self.centers.shape[0])

    def design(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """[n_points, n_coeff] Gaussian radial basis design matrix."""
        tx = np.asarray(tx, dtype=np.float64).reshape(-1)
        ty = np.asarray(ty, dtype=np.float64).reshape(-1)
        dx = tx[:, None] - self.centers[None, :, 0]
        dy = ty[:, None] - self.centers[None, :, 1]
        return np.exp(-0.5 * (dx**2 + dy**2) / self.sigma**2)

    def evaluate(self, coeffs: np.ndarray, tx: np.ndarray,
                 ty: np.ndarray) -> np.ndarray:
        tx_arr = np.asarray(tx, dtype=np.float64)
        flat = self.design(tx_arr, ty) @ np.asarray(coeffs, dtype=np.float64)
        return flat.reshape(tx_arr.shape)


def make_smooth_basis(t_max: float = 1.25, n_per_axis: int = 8) -> SmoothBasis:
    """Centres on a square grid, with sigma set to the centre spacing.

    Sigma equal to the spacing keeps neighbouring basis functions overlapping
    enough to represent a smooth field without leaving gaps, and coarse enough
    that the basis cannot chase individual bins.
    """
    axis = np.linspace(-t_max, t_max, n_per_axis)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    centers = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    spacing = (2.0 * t_max) / (n_per_axis - 1)
    return SmoothBasis(centers=centers, sigma=float(spacing))
