"""Track angles from two-point slopes.

Each coordinate is measured by exactly TWO layers, so the straight-line fit is
exactly determined and chi2 is identically zero. There is no fit-quality cut to
be had here; selection comes from cluster topology in megido.hits instead.
(The same degeneracy is documented in layers_fit_calibration/method.md.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.detector import DetectorGeometry
from megido.hits import REJECT_OK, LayerHits


@dataclass(frozen=True)
class Tracks:
    tan_x: np.ndarray
    tan_y: np.ndarray
    x_bot_cm: np.ndarray
    y_bot_cm: np.ndarray
    valid: np.ndarray

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    def theta_deg(self) -> np.ndarray:
        return np.degrees(np.arctan(np.hypot(self.tan_x, self.tan_y)))

    def phi_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.tan_y, self.tan_x))


def fit_tracks(hits: LayerHits, geom: DetectorGeometry) -> Tracks:
    valid = (hits.reject == REJECT_OK).all(axis=1)

    x_lo_asic, x_hi_asic = geom.asics_for_coord("x")
    y_lo_asic, y_hi_asic = geom.asics_for_coord("y")
    dz_x = geom.dz_cm("x")
    dz_y = geom.dz_cm("y")

    x_bot = hits.position_cm[:, x_lo_asic]
    y_bot = hits.position_cm[:, y_lo_asic]

    tan_x = (hits.position_cm[:, x_hi_asic] - x_bot) / dz_x
    tan_y = (hits.position_cm[:, y_hi_asic] - y_bot) / dz_y

    tan_x = np.where(valid, tan_x, np.nan)
    tan_y = np.where(valid, tan_y, np.nan)

    return Tracks(tan_x=tan_x, tan_y=tan_y, x_bot_cm=np.where(valid, x_bot, np.nan),
                  y_bot_cm=np.where(valid, y_bot, np.nan), valid=valid)
