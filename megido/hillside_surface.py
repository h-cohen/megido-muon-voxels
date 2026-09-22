"""Linear regularized hillside-surface inversion core (Phase 5b, Task 1).

The hillside silhouette is fit as a smooth height field H(x, y) on a
rectangular grid. Each muon ray with a measured path length places a surface
point at its exit point ``o + L * d_hat``. Bilinear interpolation of the grid
at the exit point's (x, y) equals the exit point's z -- this is LINEAR in the
grid heights, unlike an iterative ray-march forward model (which is a
non-differentiable step function and stalls gradient-based optimizers on a
flat surface). This module builds the sparse bilinear sampling matrix, a
Laplacian smoothness operator, and the regularized normal-equations solve.

Grid flattening convention: for grids ``gx`` (length nx) and ``gy`` (length
ny), grid index ``(i, j)`` (i indexing gx, j indexing gy) flattens to
``flat = i * ny + j`` (i.e. ``np.meshgrid(gx, gy, indexing="ij").ravel()``).
Downstream consumers (Task 2's driver) must use this same convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from megido.baseline import position_ids


def bilinear_matrix(xy: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """Build the sparse bilinear sampling matrix for points ``xy`` on grid (gx, gy).

    Parameters
    ----------
    xy : (N, 2) array of (x, y) sample points.
    gx : (nx,) strictly increasing grid coordinates along x.
    gy : (ny,) strictly increasing grid coordinates along y.

    Returns
    -------
    B : scipy.sparse.csr_matrix, shape (N, nx*ny)
        Row n has up to 4 nonzero bilinear weights (summing to 1) for the
        grid cell containing point n, flattened as ``i * ny + j``. Points
        outside ``[gx[0], gx[-1]] x [gy[0], gy[-1]]`` get an all-zero row.
    support : (nx*ny,) ndarray
        Column sums of B -- total data support at each grid node.
    """
    xy = np.asarray(xy, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    nx, ny = len(gx), len(gy)
    n = xy.shape[0]

    x = xy[:, 0]
    y = xy[:, 1]

    inside = (
        (x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])
    )

    rows = []
    cols = []
    vals = []

    # searchsorted gives the insertion index; clip so i0 in [0, nx-2].
    i0 = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, nx - 2)
    j0 = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, ny - 2)

    dx = gx[i0 + 1] - gx[i0]
    dy = gy[j0 + 1] - gy[j0]
    tx = np.where(dx > 0, (x - gx[i0]) / np.where(dx > 0, dx, 1.0), 0.0)
    ty = np.where(dy > 0, (y - gy[j0]) / np.where(dy > 0, dy, 1.0), 0.0)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    corners = (
        (i0, j0, w00),
        (i0 + 1, j0, w10),
        (i0, j0 + 1, w01),
        (i0 + 1, j0 + 1, w11),
    )
    for ci, cj, cw in corners:
        flat = ci * ny + cj
        for row_idx in np.nonzero(inside)[0]:
            rows.append(row_idx)
            cols.append(flat[row_idx])
            vals.append(cw[row_idx])

    G = nx * ny
    B = sp.coo_matrix((vals, (rows, cols)), shape=(n, G)).tocsr()
    support = np.asarray(B.sum(axis=0)).ravel()
    return B, support


def laplacian_matrix(nx: int, ny: int):
    """Interior 2nd-difference (5-point Laplacian) operator over an nx x ny grid.

    Flat index convention matches ``bilinear_matrix``: ``flat = i * ny + j``.
    Only interior nodes (1 <= i <= nx-2, 1 <= j <= ny-2) produce a row; a
    plane sampled on the grid has zero 2nd difference at every interior node.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ((nx-2)*(ny-2), nx*ny) for nx, ny >= 3;
    an empty (0, nx*ny) matrix if there are no interior nodes.
    """
    G = nx * ny
    if nx < 3 or ny < 3:
        return sp.csr_matrix((0, G))

    rows = []
    cols = []
    vals = []
    row_idx = 0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            center = i * ny + j
            xp = (i + 1) * ny + j
            xm = (i - 1) * ny + j
            yp = i * ny + (j + 1)
            ym = i * ny + (j - 1)
            for col, val in ((center, 4.0), (xp, -1.0), (xm, -1.0), (yp, -1.0), (ym, -1.0)):
                rows.append(row_idx)
                cols.append(col)
                vals.append(val)
            row_idx += 1

    return sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, G)).tocsr()


def solve_surface(B, z: np.ndarray, w: np.ndarray, lap, mu: float) -> np.ndarray:
    """Solve ``min ||W(Bh - z)||^2 + mu^2 ||lap h||^2`` for the flat grid heights h.

    Solved via the normal equations ``(B^T W B + mu^2 L^T L) h = B^T W z``.
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    B = sp.csr_matrix(B)
    lap = sp.csr_matrix(lap)

    W = sp.diags(w)
    BtW = B.T @ W
    A = (BtW @ B + (mu ** 2) * (lap.T @ lap)).tocsc()
    b = BtW @ z

    h = spla.spsolve(A, b)
    return np.asarray(h).ravel()


def variance_explained(B, z: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    """Weighted variance-explained: ``1 - sum(w*(Bh-z)^2) / sum(w*(z-zbar)^2)``."""
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    B = sp.csr_matrix(B)

    pred = B @ h
    resid = pred - z
    num = np.sum(w * resid ** 2)

    zbar = np.sum(w * z) / np.sum(w)
    denom = np.sum(w * (z - zbar) ** 2)

    return 1.0 - num / denom


def _positions(cfg):
    """Position id (config order, "pos0"/"pos1"/...) -> the pose of that
    position's first exposure, in config order. Position id assignment and
    grouping is owned by `megido.baseline.position_ids`; this just picks a
    representative pose per group."""
    pids = position_ids(cfg)
    out: dict[str, object] = {}
    for exp in cfg.exposures:
        pid = pids[exp.id]
        if pid not in out:
            out[pid] = exp.pose
    return out


def exit_points(sol, cfg, a: float = 1.0, transparent_quantile: float = 0.05):
    """Exit points of every finite, positive opacity sky pixel, at all positions.

    exit = o + a * lam * d_hat, where o is the detector's world position for
    that position (metres), lam is the position's normalized opacity (Phase 2's
    gauge-pinned convention, never the raw `.opacity`), and d_hat = (sx, sy, 1)
    / ||.|| is the sky-frame ray direction for that pixel. `a` is an ASSUMED
    inverse-density scale (opacity-units per metre) -- it is not measured by
    this campaign's 2.2 m baseline, only assumed.

    Returns
    -------
    pts : (N, 3) ndarray of (x, y, z) exit points, metres.
    pos_index : (N,) int ndarray, index into `sorted(positions)` for each point.
    """
    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)

    centers = sol.sky.centers
    sx_grid, sy_grid = np.meshgrid(centers, centers, indexing="ij")
    sx_flat = sx_grid.ravel()
    sy_flat = sy_grid.ravel()

    pts_list = []
    idx_list = []
    for k, pid in enumerate(pids_sorted):
        pose = pos_pose[pid]
        lam = sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        mask = np.isfinite(lam) & (lam > 0)
        if not mask.any():
            continue
        sx = sx_flat[mask]
        sy = sy_flat[mask]
        l = lam[mask]
        norm = np.sqrt(sx ** 2 + sy ** 2 + 1.0)
        dx, dy, dz = sx / norm, sy / norm, 1.0 / norm
        x = pose.x + a * l * dx
        y = pose.y + a * l * dy
        z = pose.z + a * l * dz
        pts_list.append(np.stack([x, y, z], axis=1))
        idx_list.append(np.full(mask.sum(), k, dtype=int))

    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)

    pts = np.concatenate(pts_list, axis=0)
    pos_index = np.concatenate(idx_list, axis=0)
    return pts, pos_index


def _solve_with_ridge(B, z: np.ndarray, w: np.ndarray, lap, mu: float,
                       ridge: float) -> np.ndarray:
    """Same normal-equations solve as `solve_surface`, plus a tiny ridge term.

    A grid node with zero data support AND no Laplacian coupling (an isolated
    corner, say) makes B^T W B + mu^2 L^T L singular. `ridge * I` guarantees a
    non-singular solve everywhere; such nodes are masked to NaN downstream by
    the coverage mask, so the ridge never fabricates a value that survives.
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    B = sp.csr_matrix(B)
    lap = sp.csr_matrix(lap)

    W = sp.diags(w)
    BtW = B.T @ W
    G = B.shape[1]
    A = (BtW @ B + (mu ** 2) * (lap.T @ lap) + ridge * sp.eye(G)).tocsc()
    b = BtW @ z

    h = spla.spsolve(A, b)
    return np.asarray(h).ravel()


def _footprint_grid(xy: np.ndarray, cell_m: float):
    """Grid nodes spanning the exit-point footprint, robust to outlier rays.

    Bounds are the 2nd/98th percentile of exit x and y (not min/max) so a
    handful of grazing-incidence rays near the horizon cannot blow the grid
    out to a near-empty extent.
    """
    lo_x, hi_x = np.percentile(xy[:, 0], [2, 98])
    lo_y, hi_y = np.percentile(xy[:, 1], [2, 98])
    if hi_x <= lo_x:
        hi_x = lo_x + cell_m
    if hi_y <= lo_y:
        hi_y = lo_y + cell_m

    gx = np.arange(lo_x, hi_x + cell_m, cell_m)
    gy = np.arange(lo_y, hi_y + cell_m, cell_m)
    if len(gx) < 3:
        gx = np.linspace(lo_x, hi_x, 3)
    if len(gy) < 3:
        gy = np.linspace(lo_y, hi_y, 3)
    return gx, gy


@dataclass(frozen=True)
class SurfaceResult:
    """The fitted hillside height field, with its honesty bookkeeping.

    `H` and `sigma` are NaN off the data-supported region -- an unconstrained
    grid node is NaN, never 0 or an extrapolated guess (repo convention: NaN
    means "not constrained", not "measured as zero").
    """
    H: np.ndarray
    sigma: np.ndarray
    support: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    a: float
    variance_explained: float
    coverage_frac: float
    coverage_radius_m: float
    n_rays: int
    detectors: list = field(default_factory=list)
    scale_assumed: bool = True
    note: str = ""


def fit_surface(sol, cfg, *, a: float = 1.0, cell_m: float = 1.0, mu: float = 0.3,
                 min_support: float = 0.5, ridge: float = 1e-6, n_boot: int = 8,
                 quantiles: list[float] | None = None) -> SurfaceResult:
    """Fit the hillside height field from a baseline solution's opacity maps.

    Linear inversion only (bilinear sampling + Laplacian smoothness, solved by
    normal equations) -- no ray-marching, which is a non-differentiable step
    function that would stall a gradient-based fit on a flat surface.

    Absolute height is on an ASSUMED inverse-density scale `a`
    (opacity-units per metre): this campaign's 2.2 m position baseline does
    not determine it. Only the fitted SHAPE, at that assumed scale, is a
    genuine data-driven result. See `SurfaceResult.note`.
    """
    pts, _ = exit_points(sol, cfg, a=a)
    if pts.shape[0] == 0:
        raise ValueError("no finite, positive-opacity sky pixels to fit a surface from")

    xy_all = pts[:, :2]
    z_all = pts[:, 2]

    gx, gy = _footprint_grid(xy_all, cell_m)
    nx, ny = len(gx), len(gy)

    # Rays landing outside the grid footprint get an all-zero bilinear row
    # (see bilinear_matrix): drop them before fitting/scoring so they cannot
    # inflate variance_explained's residual against a prediction of zero.
    inside = (
        (xy_all[:, 0] >= gx[0]) & (xy_all[:, 0] <= gx[-1])
        & (xy_all[:, 1] >= gy[0]) & (xy_all[:, 1] <= gy[-1])
    )
    xy = xy_all[inside]
    z = z_all[inside]

    B, support = bilinear_matrix(xy, gx, gy)
    lap = laplacian_matrix(nx, ny)
    w = np.ones(len(z))

    h = _solve_with_ridge(B, z, w, lap, mu, ridge)
    ve = float(variance_explained(B, z, w, h))

    covered = support >= min_support
    h_masked = np.where(covered, h, np.nan)
    H = h_masked.reshape(nx, ny)
    support_grid = support.reshape(nx, ny)

    coverage_frac = float(np.mean(covered))

    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)
    detectors = [
        {"id": pid, "x": pos_pose[pid].x, "y": pos_pose[pid].y, "z": pos_pose[pid].z}
        for pid in pids_sorted
    ]
    cx = float(np.mean([d["x"] for d in detectors]))
    cy = float(np.mean([d["y"] for d in detectors]))
    radii = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    coverage_radius_m = float(np.percentile(radii, 95))

    # Bootstrap: re-derive exit points at varied transparent_quantile (Phase
    # 2's gauge systematic) and refit on the SAME grid, per-cell std across
    # replicas. This is an honesty check on the opacity-zero-point convention,
    # not a statistical error bar in the usual sense.
    if quantiles is None:
        quantiles = [0.02, 0.05, 0.1, 0.2]
    boots = np.empty((n_boot, nx * ny))
    for b in range(n_boot):
        q = quantiles[b % len(quantiles)]
        pts_b, _ = exit_points(sol, cfg, a=a, transparent_quantile=q)
        Bb, _ = bilinear_matrix(pts_b[:, :2], gx, gy)
        wb = np.ones(pts_b.shape[0])
        boots[b] = _solve_with_ridge(Bb, pts_b[:, 2], wb, lap, mu, ridge)

    sigma_flat = np.where(covered, np.std(boots, axis=0), np.nan)
    sigma = sigma_flat.reshape(nx, ny)

    note = (
        f"Height is on an ASSUMED inverse-density scale a={a:g} "
        "(opacity-units per metre) -- this campaign's 2.2 m position baseline "
        "does NOT determine that scale, only the fitted SHAPE at it is a "
        f"genuine result. {coverage_frac:.0%} of the grid within "
        f"~{coverage_radius_m:.1f} m of the detector centroid is data-supported "
        "(support >= min_support); the rest is NaN (unconstrained, never "
        f"fabricated). variance_explained={ve:.3f} over {len(z)} rays."
    )

    return SurfaceResult(
        H=H,
        sigma=sigma,
        support=support_grid,
        gx=gx,
        gy=gy,
        a=a,
        variance_explained=ve,
        coverage_frac=coverage_frac,
        coverage_radius_m=coverage_radius_m,
        n_rays=int(len(z)),
        detectors=detectors,
        scale_assumed=True,
        note=note,
    )
