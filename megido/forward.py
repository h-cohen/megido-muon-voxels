"""The one forward model shared by inversion, phantom generation and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.fitdata import RowIndex, position_origins
from megido.raycast import build_system_matrix
from megido.voxels import VoxelGrid, auto_grid


@dataclass(frozen=True)
class ForwardModel:
    A: sparse.csr_matrix
    grid: VoxelGrid
    rows: RowIndex

    @property
    def n_rows(self) -> int:
        return self.rows.n_rows

    def predict(self, x: np.ndarray, offsets: dict[str, float] | None = None
                ) -> np.ndarray:
        """Volume (flat or 3D) -> predicted optical depth per row."""
        y = self.A @ np.asarray(x, dtype=np.float64).ravel()
        if offsets:
            add = np.array([offsets.get(pid, 0.0) for pid in self.rows.position_ids])
            y = y + add[self.rows.pos_of_row]
        return y

    def to_sky_image(self, values: np.ndarray, pid: str, n_sky: int) -> np.ndarray:
        """Scatter a per-row quantity back onto one position's sky grid.

        Bins with no row are NaN, never zero: "not measured" and "measured as
        zero opacity" are different statements and the viewer must not conflate
        them.
        """
        img = np.full(n_sky * n_sky, np.nan)
        sel = self.rows.mask_for(pid)
        img[self.rows.sky_flat[sel]] = np.asarray(values)[sel]
        return img.reshape(n_sky, n_sky)


def build_forward_model(rows: RowIndex, cfg: SiteConfig, *,
                        geom: DetectorGeometry | None = None,
                        grid: VoxelGrid | None = None,
                        cache_dir: str | Path | None = "runs/.cache"
                        ) -> ForwardModel:
    geom = geom or DetectorGeometry.for_site(cfg)
    origins = position_origins(cfg)
    if grid is None:
        grid = auto_grid(cfg.volume, origins, t_reach=rows.t_reach(),
                         aperture_m=geom.aperture_m)
    A = build_system_matrix(rows, origins, grid,
                            aperture_m=geom.aperture_m,
                            n_sub=cfg.volume.n_aperture_sub,
                            cache_dir=cache_dir)
    return ForwardModel(A=A, grid=grid, rows=rows)
