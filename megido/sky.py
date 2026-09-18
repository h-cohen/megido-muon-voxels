"""Detector frame to sky frame, and the cosmic flux shape.

The detector response B(d) is fixed in the DETECTOR frame; the rock's opacity is
fixed in the SKY frame. Tilting the detector 20 degrees moves every detector bin
to a different patch of sky while B does not move, and that displacement is the
entire reason the two can be separated without an open-sky calibration run.

Directions are carried as tangents (tx, ty) meaning the unit vector along
(tx, ty, 1). Rotation is applied to the unit vector, not to the tangents, which
is why the map is non-linear in tangent space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import Pose

_HORIZON_EPS = 1e-6


def detector_to_sky(tx: np.ndarray, ty: np.ndarray,
                    pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map detector-frame tangents to sky-frame tangents.

    Returns (sx, sy, valid). `valid` is False where the rotated ray points at or
    below the horizon, where a tangent representation is meaningless.
    """
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    vec = np.stack([tx, ty, np.ones_like(tx)], axis=-1)
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    rotated = vec @ pose.rotation().T
    z = rotated[..., 2]
    valid = z > _HORIZON_EPS

    safe_z = np.where(valid, z, 1.0)
    sx = np.where(valid, rotated[..., 0] / safe_z, np.nan)
    sy = np.where(valid, rotated[..., 1] / safe_z, np.nan)
    return sx, sy, valid


def flux_shape(sx: np.ndarray, sy: np.ndarray, index: float) -> np.ndarray:
    """Cosmic-ray angular shape, cos(theta_sky) ** index.

    Only the SHAPE matters: the absolute flux normalization is degenerate with
    the per-exposure normalization and is absorbed there, which is why a single
    free exponent is enough and a full Gaisser parametrisation would add
    parameters without adding information.
    """
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    cos_theta = 1.0 / np.sqrt(1.0 + sx**2 + sy**2)
    return cos_theta**index


@dataclass(frozen=True)
class SkyGrid:
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def flat_size(self) -> int:
        return self.n_bins * self.n_bins

    def bin_index(self, sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat bin index and an in-range mask. Out-of-range entries get index 0
        and mask False — callers must apply the mask, never trust the index."""
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        finite = np.isfinite(sx) & np.isfinite(sy)

        i = np.digitize(np.where(finite, sx, 0.0), self.edges) - 1
        j = np.digitize(np.where(finite, sy, 0.0), self.edges) - 1
        ok = finite & (i >= 0) & (i < self.n_bins) & (j >= 0) & (j < self.n_bins)

        flat = np.where(ok, i * self.n_bins + j, 0)
        return flat.astype(np.int64), ok


def make_sky_grid(t_max: float = 2.5, n_bins: int = 100) -> SkyGrid:
    """Sky-frame tangent grid, at the same 0.05 resolution as the analysis grid.

    Sizing this by `detector_max_tan + tan(tilt)` is wrong: that assumes the tilt
    displaces directions along an axis. At an oblique azimuth the tangent-plane
    blowup near grazing incidence is far larger — for the campaign's 20 degree
    tilt at azimuth 241, the acceptance corner (1.208, -1.208) maps to sky
    (-3.37, -1.34).

    Containing every geometric corner would need t_max above 3.4 and leave most
    of the grid empty, so the span is set by where the COUNTS are instead.
    Measured on the real campaign data, the fraction of counts falling outside
    the grid for the tilted exposures is 0.66-0.83% at t_max=1.75, and
    0.007-0.010% at 2.5. Untilted exposures lose nothing at any setting.
    """
    return SkyGrid(edges=np.linspace(-t_max, t_max, n_bins + 1))
