"""Orchestration: baseline solution in, voxel volumes out.

Alongside the full fit, one volume per position is produced from that position
ALONE. With two positions 2.2 m apart, agreement between those single-view fits
is the only direct evidence that the depth structure is measured rather than
assumed, so they are produced by default and not as an opt-in diagnostic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from megido.baseline import BaselineSolution
from megido.config import SiteConfig
from megido.fitdata import build_fit_data
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.raycast import INVERSION_VERSION
from megido.voxels import VoxelGrid


@dataclass(frozen=True)
class VoxelSolution:
    rho: np.ndarray                     # [n_voxels] opacity density, 1/m, >= 0
    grid: VoxelGrid
    offsets: dict[str, float]
    position_ids: tuple[str, ...]
    info: dict = field(default_factory=dict, repr=False)
    version: int = INVERSION_VERSION

    def rho3(self) -> np.ndarray:
        return self.rho.reshape(self.grid.shape)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rho=self.rho3().astype(np.float32),
            origin=np.asarray(self.grid.origin, dtype=np.float64),
            spacing=np.asarray(self.grid.spacing, dtype=np.float64),
            shape=np.asarray(self.grid.shape, dtype=np.int64),
            version=np.asarray(self.version, dtype=np.int64),
            meta=np.array(json.dumps({
                "offsets": self.offsets,
                "position_ids": list(self.position_ids),
                "info": self.info,
            })),
        )

    @staticmethod
    def load(path: str | Path) -> "VoxelSolution":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        grid = VoxelGrid(origin=tuple(float(v) for v in d["origin"]),
                         spacing=float(d["spacing"]),
                         shape=tuple(int(v) for v in d["shape"]))
        return VoxelSolution(
            rho=d["rho"].astype(np.float64).ravel(),
            grid=grid,
            offsets={k: float(v) for k, v in meta["offsets"].items()},
            position_ids=tuple(meta["position_ids"]),
            info=meta["info"],
            version=int(d["version"]),
        )


def solve_voxels(sol: BaselineSolution, cfg: SiteConfig, *,
                 sigma: dict[str, np.ndarray] | None = None,
                 max_tan: float = 1.25,
                 transparent_quantile: float = 0.05,
                 cache_dir: str | Path | None = "runs/.cache",
                 holdouts: bool = True,
                 grid: VoxelGrid | None = None) -> dict[str, VoxelSolution]:
    """Full fit plus one single-position fit per position.

    Every fit shares one system matrix; a holdout is a row-weight mask, not a
    rebuild. That is why the matrix cache pays for itself even on a single run.

    `grid` defaults to None, which keeps today's behaviour: the grid is
    auto-derived from this call's own `rows.t_reach()`. A caller that must
    keep several calls on one identical lattice (e.g. a bootstrap stacking
    replicas into one array) passes the grid explicitly instead.
    """
    data = build_fit_data(sol, cfg, sigma=sigma, max_tan=max_tan,
                          transparent_quantile=transparent_quantile)
    fwd = build_forward_model(data.rows, cfg, grid=grid, cache_dir=cache_dir)
    rc = cfg.reconstruction

    def run(keep: np.ndarray) -> VoxelSolution:
        restricted = data.restricted(keep)
        x, info = solve(fwd, restricted, rc)
        info["n_rows_used"] = int(np.count_nonzero(restricted.w))
        info["algorithm"] = rc.algorithm
        return VoxelSolution(rho=x, grid=fwd.grid,
                             offsets=info.pop("offsets"),
                             position_ids=fwd.rows.position_ids,
                             info=info)

    out = {"full": run(np.ones(data.rows.n_rows, dtype=bool))}
    if holdouts:
        for pid in fwd.rows.position_ids:
            out[f"holdout_{pid}"] = run(fwd.rows.mask_for(pid))
    return out
