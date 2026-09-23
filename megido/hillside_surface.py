"""Gaussian-process hillside-surface inversion driver (Phase 5b, Task 2).

The hillside silhouette is fit as a smooth height field H(x, y). Each muon
ray with a measured (gauge-pinned, ASSUMED-scale) opacity places a surface
point at its exit point ``o + a * lam * d_hat``. Those exit points are the
training data for a Matern-5/2 Gaussian process (`megido.hillside_gp`),
fit by ML-II with a heteroscedastic noise model built from Poisson counts
when available (or geometric leverage otherwise). The GP replaces the
earlier bilinear + Laplacian linear solver: it gives a real posterior
per-cell error bar instead of a bootstrap-over-gauge proxy, and needs no
smoothness-penalty tuning knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from megido.baseline import position_ids
from megido.hillside_gp import fit_hyperparams, predict


def variance_explained(z, pred, w=None) -> float:
    """Weighted variance-explained: ``1 - sum(w*(pred-z)^2) / sum(w*(z-zbar)^2)``."""
    z = np.asarray(z, float)
    pred = np.asarray(pred, float)
    w = np.ones_like(z) if w is None else np.asarray(w, float)
    resid = pred - z
    num = np.sum(w * resid ** 2)
    zbar = np.sum(w * z) / np.sum(w)
    denom = np.sum(w * (z - zbar) ** 2)
    return float(1.0 - num / denom)


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
    dz : (N,) float ndarray, the sky-frame vertical direction cosine, used for
        the cosmic-ray flux weight in `fit_surface`.
    sky_flat : (N,) int ndarray, the ray's flat sky-bin index into `sol.sky`.
    """
    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)

    centers = sol.sky.centers
    sx_grid, sy_grid = np.meshgrid(centers, centers, indexing="ij")
    sx_flat = sx_grid.ravel()
    sy_flat = sy_grid.ravel()

    pts_list, idx_list, dz_list, flat_list = [], [], [], []
    for k, pid in enumerate(pids_sorted):
        pose = pos_pose[pid]
        lam = sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        mask = np.isfinite(lam) & (lam > 0)
        if not mask.any():
            continue
        sx, sy, l = sx_flat[mask], sy_flat[mask], lam[mask]
        norm = np.sqrt(sx ** 2 + sy ** 2 + 1.0)
        dx, dy, dz = sx / norm, sy / norm, 1.0 / norm
        x = pose.x + a * l * dx
        y = pose.y + a * l * dy
        z = pose.z + a * l * dz
        pts_list.append(np.stack([x, y, z], axis=1))
        idx_list.append(np.full(int(mask.sum()), k, dtype=int))
        dz_list.append(dz)
        flat_list.append(np.nonzero(mask)[0])

    if not pts_list:
        z0 = np.zeros((0,), dtype=int)
        return np.zeros((0, 3)), z0, z0.astype(float), z0
    return (np.concatenate(pts_list), np.concatenate(idx_list),
            np.concatenate(dz_list), np.concatenate(flat_list))


def _weighted_quantile(v, w, q: float) -> float:
    """Weighted quantile of values ``v`` with non-negative weights ``w`` at level
    ``q`` in [0,1], midpoint (Hazen-like) convention. Falls back to the unweighted
    quantile if all weights are zero."""
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if total <= 0:
        return float(np.quantile(v, q))
    cq = (np.cumsum(w) - 0.5 * w) / total
    return float(np.interp(q, cq, v))


def _upper_envelope_cells(X, z, dz, gx, gy, *, index: float = 2.0,
                          q_hi: float = 0.85, min_count: int = 8,
                          ray_counts=None):
    """Reduce exit points to one flux-weighted upper-quantile target per grid cell.

    The true surface is the MAXIMUM overburden per column; the low exit points are
    artifacts of radial opacity sorting (see the spec). Per cell we take the
    ``q_hi`` weighted quantile of exit-z, weighting each ray by the cosmic-ray flux
    ``dz**index`` (the known cos^index angular distribution) and, when available,
    its Poisson counts -- so the sparse high-flux near-vertical rays that actually
    sample the crown drive the target. Cells with fewer than ``min_count`` points
    are dropped (unconstrained -> NaN downstream, never 0).

    Returns (X_cell (M,2) centroids, y_cell (M,) targets, noise_cell (M,) variances).
    """
    X = np.asarray(X, float); z = np.asarray(z, float); dz = np.asarray(dz, float)
    nx, ny = len(gx), len(gy)
    ix = np.clip(np.searchsorted(gx, X[:, 0]) - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(gy, X[:, 1]) - 1, 0, ny - 1)
    cell = ix * ny + iy
    fw = np.clip(dz ** index, 1e-3, None)
    if ray_counts is not None:
        fw = fw * np.clip(np.asarray(ray_counts, float), 1.0, None)

    xs, ys, targets, noises = [], [], [], []
    for c in np.unique(cell):
        m = cell == c
        if int(m.sum()) < min_count:
            continue
        zc, wc, xc, yc = z[m], fw[m], X[m, 0], X[m, 1]
        target = _weighted_quantile(zc, wc, q_hi)
        wsum = wc.sum()
        n_eff = (wsum ** 2) / np.sum(wc ** 2)          # Kish effective count
        wmean = np.sum(wc * zc) / wsum
        wvar = np.sum(wc * (zc - wmean) ** 2) / wsum
        sigma = np.sqrt(wvar) / np.sqrt(max(n_eff, 1.0)) + 0.02 * abs(target) + 1e-3
        xs.append(np.average(xc, weights=wc))
        ys.append(np.average(yc, weights=wc))
        targets.append(target)
        noises.append(sigma ** 2)

    if not xs:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))
    return (np.column_stack([xs, ys]), np.asarray(targets), np.asarray(noises))


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
    means "not constrained", not "measured as zero"). `support` carries
    `1 - sigma/signal_std`, a data-influence proxy (0 at the prior, 1 where
    the posterior has collapsed onto the data).
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
    length_scale_m: float = float("nan")
    signal_std: float = float("nan")
    noise_floor: float = float("nan")


def fit_surface(sol, cfg, *, a: float = 8.0, cell_m: float = 1.0,
                 sky_counts: dict | None = None, cov_tau: float = 0.9,
                 n_restarts: int = 3, max_points: int = 1000,
                 q_hi: float = 0.85, min_count: int = 8) -> SurfaceResult:
    """Fit the hillside height field from a baseline solution's opacity maps.

    A Matern-5/2 Gaussian process (`megido.hillside_gp`) fit by ML-II,
    replacing the earlier bilinear + Laplacian linear solve. `sigma` is the
    GP's real posterior standard deviation per grid cell -- a genuine error
    bar, not a bootstrap-over-gauge proxy.

    The GP is trained on one flux-weighted UPPER quantile (`q_hi`) of exit-z
    per grid cell (`_upper_envelope_cells`), not the raw per-ray exit points
    or their mean. The per-ray mean is biased low at the detector footprint by
    a radial opacity-sorting artifact (near-vertical, high-flux rays are rare
    and get outvoted by oblique low-opacity ones); the upper envelope tracks
    the true maximum overburden per column and is robust to that artifact.

    Absolute height is on an ASSUMED inverse-density scale `a`
    (opacity-units per metre): this campaign's 2.2 m position baseline does
    not determine it. Only the fitted SHAPE, at that assumed scale, is a
    genuine data-driven result. See `SurfaceResult.note`.
    """
    pts, pos_index, dz, sky_flat = exit_points(sol, cfg, a=a)
    if pts.shape[0] == 0:
        raise ValueError("no finite, positive-opacity sky pixels to fit a surface from")

    gx, gy = _footprint_grid(pts[:, :2], cell_m)
    nx, ny = len(gx), len(gy)
    inside = ((pts[:, 0] >= gx[0]) & (pts[:, 0] <= gx[-1])
              & (pts[:, 1] >= gy[0]) & (pts[:, 1] <= gy[-1]))
    X = pts[inside, :2]; z = pts[inside, 2]
    dz_in = dz[inside]; flat_in = sky_flat[inside]; pos_in = pos_index[inside]

    pos_pose = _positions(cfg); pids_sorted = sorted(pos_pose)
    index = float(getattr(sol, "flux_index", 2.0))

    ray_counts = None
    heteroscedastic = False
    if sky_counts is not None:
        ray_counts = np.array([sky_counts[pids_sorted[p]][f]
                               for p, f in zip(pos_in, flat_in)], dtype=float)
        heteroscedastic = True

    Xc, yc, noise_c = _upper_envelope_cells(
        X, z, dz_in, gx, gy, index=index, q_hi=q_hi, min_count=min_count,
        ray_counts=ray_counts)
    if Xc.shape[0] < 3:
        raise ValueError("too few populated cells to fit a surface; lower min_count")

    # cells are already few (<= nx*ny); the max_points guard rarely triggers.
    if Xc.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(Xc.shape[0], max_points, replace=False)
        sel.sort(); Xc, yc, noise_c = Xc[sel], yc[sel], noise_c[sel]

    mean = float(np.average(yc))
    hypers = fit_hyperparams(Xc, yc, noise_c, mean, n_restarts=n_restarts)
    noise_var = noise_c + (hypers.noise_floor * hypers.signal_std) ** 2

    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xstar = np.stack([GX.ravel(), GY.ravel()], axis=1)
    mstar, sstar = predict(Xc, yc, noise_var, mean, Xstar, hypers)

    covered = sstar < cov_tau * hypers.signal_std
    H = np.where(covered, mstar, np.nan).reshape(nx, ny)
    sigma = np.where(covered, sstar, np.nan).reshape(nx, ny)
    support = np.where(covered, 1.0 - sstar / hypers.signal_std, np.nan).reshape(nx, ny)

    pred_cell, _ = predict(Xc, yc, noise_var, mean, Xc, hypers)
    ve = variance_explained(yc, pred_cell)

    detectors = [{"id": pid, "x": pos_pose[pid].x, "y": pos_pose[pid].y,
                  "z": pos_pose[pid].z} for pid in pids_sorted]
    cx = float(np.mean([d["x"] for d in detectors]))
    cy = float(np.mean([d["y"] for d in detectors]))
    radii = np.sqrt((Xc[:, 0] - cx) ** 2 + (Xc[:, 1] - cy) ** 2)
    coverage_radius_m = float(np.percentile(radii, 95))
    coverage_frac = float(np.mean(covered))
    n_rays = int(inside.sum())

    note = (
        f"Height is on an ASSUMED inverse-density scale a={a:g} (opacity-units per "
        "metre) -- this campaign's 2.2 m baseline does NOT determine it; only the "
        f"fitted SHAPE at that scale is a genuine result. Estimator: per-cell "
        f"flux-weighted (cos^{index:g}) UPPER quantile q_hi={q_hi:g} of exit-z "
        "(tracks the maximum overburden per column; robust to the radial "
        "opacity-sorting artifact that otherwise dips the detector footprints). "
        f"GP: Matern-5/2, length-scale={hypers.length_scale:.2f} m (ML-II), "
        f"{'heteroscedastic (counts)' if heteroscedastic else 'geometry-only (no counts; pass --run)'}. "
        f"sigma is the posterior std. {coverage_frac:.0%} of the grid within "
        f"~{coverage_radius_m:.1f} m of the detector centroid is data-supported; the "
        f"rest is NaN. variance_explained={ve:.3f} over {Xc.shape[0]} cells "
        f"({n_rays} rays). Opacity gauge zero-point not corrected (deferred)."
    )
    return SurfaceResult(H=H, sigma=sigma, support=support, gx=gx, gy=gy, a=a,
                         variance_explained=ve, coverage_frac=coverage_frac,
                         coverage_radius_m=coverage_radius_m, n_rays=n_rays,
                         detectors=detectors, scale_assumed=True, note=note,
                         length_scale_m=hypers.length_scale,
                         signal_std=hypers.signal_std, noise_floor=hypers.noise_floor)
