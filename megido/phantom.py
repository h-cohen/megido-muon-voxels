"""Synthetic scenes, forward projection, and round-trip scoring.

Phantoms answer two different questions and it matters which is which:

  * Task 7 uses 01_data/raw/topography.csv under a two-detector geometry with
    generous parallax. It asks whether the CODE is right.
  * Task 8 uses the real campaign poses from configs/megido.yaml. It asks what
    the CAMPAIGN can resolve, which is a much harsher question.

Passing the first says nothing about the second. 01_data/raw has an open-sky
calibration file beside the topography; it is deliberately never read, because
this campaign has no open-sky run and a phantom that used one would validate a
pipeline that does not exist.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from megido.fitdata import FitData, RowIndex
from megido.forward import ForwardModel
from megido.voxels import VoxelGrid


def load_topography(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an x,y,z surface CSV into (xs, ys, z[nx, ny]) on its own lattice."""
    pts: list[tuple[float, float, float]] = []
    with Path(path).open() as fh:
        for row in csv.DictReader(fh):
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    arr = np.array(pts)
    xs = np.unique(arr[:, 0])
    ys = np.unique(arr[:, 1])
    if xs.size * ys.size != arr.shape[0]:
        raise ValueError(f"{path} is not a complete regular grid")
    ix = np.searchsorted(xs, arr[:, 0])
    iy = np.searchsorted(ys, arr[:, 1])
    z = np.full((xs.size, ys.size), np.nan)
    z[ix, iy] = arr[:, 2]
    return xs, ys, z


def surface_volume(grid: VoxelGrid, xs: np.ndarray, ys: np.ndarray,
                   z_surface: np.ndarray, density: float = 1.0) -> np.ndarray:
    """Flat truth volume: `density` where a voxel centre lies below the surface.

    The surface is sampled at each voxel column by nearest neighbour on its own
    lattice, which is exact when the surface is coarser than the voxels — the
    case for every phantom here.
    """
    gx = grid.axis_centers(0)
    gy = grid.axis_centers(1)
    gz = grid.axis_centers(2)
    ix = np.abs(xs[None, :] - gx[:, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - gy[:, None]).argmin(axis=1)
    h = z_surface[np.ix_(ix, iy)]                     # [nx, ny]
    rho = np.where(gz[None, None, :] < h[:, :, None], float(density), 0.0)
    return rho.ravel()


def sky_rows(position_ids: tuple[str, ...], t_max: float, n_bins: int) -> RowIndex:
    """A full regular (position, direction) row set, for synthetic campaigns.

    Real rows come from megido.fitdata.build_fit_data, which keeps only the
    directions Phase 2 actually constrained. A phantom has no such holes.
    """
    edges = np.linspace(-t_max, t_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ii, jj = np.meshgrid(np.arange(n_bins), np.arange(n_bins), indexing="ij")
    sx = centers[ii].ravel()
    sy = centers[jj].ravel()
    flat = (ii * n_bins + jj).ravel()

    n = len(position_ids)
    return RowIndex(
        position_ids=tuple(position_ids),
        pos_of_row=np.repeat(np.arange(n, dtype=np.int64), sx.size),
        sx=np.tile(sx, n), sy=np.tile(sy, n),
        sky_flat=np.tile(flat.astype(np.int64), n),
    )


def project(fwd: ForwardModel, truth: np.ndarray, *,
            sigma: float = 0.0, seed: int = 0) -> FitData:
    """Forward-project a truth volume into a measurement, optionally noisy.

    Noise is Gaussian on lambda rather than Poisson on counts. lambda is already
    a log-ratio of two large counts, so its error is Gaussian to well within the
    precision any phantom gate asks for, and this keeps the phantom independent
    of the Phase 2 count model.
    """
    lam = fwd.predict(truth)
    if sigma > 0:
        lam = lam + np.random.default_rng(seed).normal(scale=sigma, size=lam.size)
    w = np.full(lam.size, 1.0 / sigma**2 if sigma > 0 else 1.0)
    return FitData(lam=lam, w=w, rows=fwd.rows)


def interface_height(rho3: np.ndarray, grid: VoxelGrid,
                     frac: float = 0.5) -> np.ndarray:
    """Top of the material in each column: the highest z whose density exceeds
    `frac` of that column's own maximum. NaN where a column is empty."""
    zc = grid.axis_centers(2)
    peak = rho3.max(axis=2)
    above = rho3 >= frac * peak[:, :, None]
    above &= peak[:, :, None] > 0
    any_above = above.any(axis=2)
    # argmax on the reversed axis gives the LAST True
    top = rho3.shape[2] - 1 - above[:, :, ::-1].argmax(axis=2)
    return np.where(any_above, zc[top], np.nan)


def score(recovered: np.ndarray, truth: np.ndarray, grid: VoxelGrid) -> dict:
    """Voxel correlation plus interface-height error over commonly filled columns."""
    r3 = np.asarray(recovered).reshape(grid.shape)
    t3 = np.asarray(truth).reshape(grid.shape)

    rf, tf = r3.ravel(), t3.ravel()
    corr = (float(np.corrcoef(rf, tf)[0, 1])
            if rf.std() > 0 and tf.std() > 0 else float("nan"))

    hr = interface_height(r3, grid)
    ht = interface_height(t3, grid)
    both = np.isfinite(hr) & np.isfinite(ht)
    if both.any():
        d = hr[both] - ht[both]
        rmse = float(np.sqrt(np.mean(d**2)))
        bias = float(np.mean(d))
    else:
        rmse = bias = float("nan")

    return {"corr": corr, "interface_rmse_m": rmse,
            "interface_bias_m": bias, "n_columns": int(both.sum())}


def anomaly_from_topography(grid: VoxelGrid, xs: np.ndarray, ys: np.ndarray,
                            z_surface: np.ndarray, z_anomaly_m: float, *,
                            quantile: float = 0.5, density: float = 1.0) -> np.ndarray:
    """A thin one-layer density anomaly at height z_anomaly_m.

    The anomaly is present in every voxel column whose sampled surface height
    exceeds the `quantile` of the surface over the grid — so its lateral
    footprint is the topography's high ground, but its DEPTH is a single known
    layer. This is the phantom a one-sided geometry can actually be tested
    against: the lateral pattern is recoverable, and placing it at one depth
    makes the depth null space measurable (a correct reconstruction smears it).
    """
    gx = grid.axis_centers(0)
    gy = grid.axis_centers(1)
    ix = np.abs(xs[None, :] - gx[:, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - gy[:, None]).argmin(axis=1)
    h = z_surface[np.ix_(ix, iy)]                       # [nx, ny]
    footprint = h > np.quantile(h, quantile)

    iz = int((z_anomaly_m - grid.origin[2]) // grid.spacing)
    rho = np.zeros(grid.shape)
    rho[footprint, iz] = float(density)
    return rho.ravel()


def depth_localization(rho3: np.ndarray, grid: VoxelGrid) -> float:
    """Fraction of total mass in the single densest z-layer.

    1.0 means the reconstruction put everything at one depth; ~1/nz means it
    smeared uniformly. For a one-sided muon geometry a correct reconstruction of
    a single-layer truth scores LOW here — that is the honest signature of the
    depth null space, not a failure.
    """
    zsum = np.asarray(rho3).reshape(grid.shape).sum(axis=(0, 1))
    total = zsum.sum()
    return float(zsum.max() / total) if total > 0 else 0.0
