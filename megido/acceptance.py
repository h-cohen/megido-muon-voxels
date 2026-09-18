"""Exact four-layer coincidence acceptance — no free parameters.

A track of slope tx entering the lower X layer at x0 arrives at x0 + tx*dz in
the upper one. Both must fall inside the active width W, so the accepted entry
positions span max(W - |tx|*dz, 0). The y coordinate is independent, and the
conversion from per-(dtx dty) to per-solid-angle adds (1 + tx^2 + ty^2)^(-3/2).

The zero crossing is W/dz = 38.4/31.5 = 1.219, the same limit Phase 1's
acceptance_cutoff gate measured at 1.248 on real tracks. Fitting this shape
instead of deriving it would have thrown that cross-check away.
"""
from __future__ import annotations

import numpy as np

from megido.detector import DetectorGeometry


def geometric_acceptance(tx: np.ndarray, ty: np.ndarray,
                         geom: DetectorGeometry) -> np.ndarray:
    """Relative acceptance per angular bin. Absolute scale is arbitrary —
    it is degenerate with the per-exposure normalization and absorbed there."""
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    width = geom.bar.active_width_cm
    span_x = np.clip(width - np.abs(tx) * geom.dz_cm("x"), 0.0, None)
    span_y = np.clip(width - np.abs(ty) * geom.dz_cm("y"), 0.0, None)

    solid_angle = (1.0 + tx**2 + ty**2) ** -1.5
    return span_x * span_y * solid_angle
