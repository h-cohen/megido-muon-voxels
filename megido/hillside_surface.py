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
    sky_flat : (N,) int ndarray, the ray's flat sky-bin index into `sol.sky`.

    Note: the exit z already carries the a*dz factor (z = pose.z + a*lam*dz), so
    the height-space noise model downstream needs |z - pose.z|, NOT dz separately.
    dz is therefore intentionally not returned -- see fit_surface's noise model.
    """
    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)

    centers = sol.sky.centers
    sx_grid, sy_grid = np.meshgrid(centers, centers, indexing="ij")
    sx_flat = sx_grid.ravel()
    sy_flat = sy_grid.ravel()

    pts_list, idx_list, flat_list = [], [], []
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
        flat_list.append(np.nonzero(mask)[0])

    if not pts_list:
        z0 = np.zeros((0,), dtype=int)
        return np.zeros((0, 3)), z0, z0
    return (np.concatenate(pts_list), np.concatenate(idx_list),
            np.concatenate(flat_list))


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
                 n_restarts: int = 3, max_points: int = 1000) -> SurfaceResult:
    """Fit the hillside height field from a baseline solution's opacity maps.

    A Matern-5/2 Gaussian process (`megido.hillside_gp`) fit by ML-II,
    replacing the earlier bilinear + Laplacian linear solve. `sigma` is the
    GP's real posterior standard deviation per grid cell -- a genuine error
    bar, not a bootstrap-over-gauge proxy.

    Absolute height is on an ASSUMED inverse-density scale `a`
    (opacity-units per metre): this campaign's 2.2 m position baseline does
    not determine it. Only the fitted SHAPE, at that assumed scale, is a
    genuine data-driven result. See `SurfaceResult.note`.
    """
    pts, pos_index, sky_flat = exit_points(sol, cfg, a=a)
    if pts.shape[0] == 0:
        raise ValueError("no finite, positive-opacity sky pixels to fit a surface from")

    gx, gy = _footprint_grid(pts[:, :2], cell_m)
    nx, ny = len(gx), len(gy)
    inside = ((pts[:, 0] >= gx[0]) & (pts[:, 0] <= gx[-1])
              & (pts[:, 1] >= gy[0]) & (pts[:, 1] <= gy[-1]))
    X = pts[inside, :2]
    z = pts[inside, 2]
    flat_in = sky_flat[inside]
    pos_in = pos_index[inside]

    # Exact-GP cost is O(N^3) per marginal-likelihood evaluation, so a full
    # campaign's ~3k in-grid rays make each ML-II fit minutes. The height field
    # is smooth and the footprint is small, so a seeded uniform subsample of the
    # rays carries the same shape at a fraction of the cost. n_used is reported
    # honestly; the coverage grid still spans the full footprint.
    n_total = int(len(z))
    subsampled = n_total > max_points
    if subsampled:
        sel = np.random.default_rng(0).choice(n_total, size=max_points, replace=False)
        sel.sort()
        X = X[sel]; z = z[sel]
        flat_in = flat_in[sel]; pos_in = pos_in[sel]

    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)

    # Height-space observation noise sigma_z (spec 3.3: sigma_z = a * dz * sigma_lambda,
    # with opacity error sigma_lambda = lambda / sqrt(N) from Poisson counting).
    # `heights` = |z - pose.z| = a * lambda * dz already carries the a*dz factor,
    # so heights / sqrt(N) IS a * dz * sigma_lambda -- do NOT multiply by dz again.
    heights = np.abs(z - np.array([pos_pose[pids_sorted[p]].z for p in pos_in]))
    if sky_counts is not None:
        N = np.array([max(1.0, sky_counts[pids_sorted[p]][f])
                      for p, f in zip(pos_in, flat_in)])
        sigma_z = heights / np.sqrt(N) + 0.02 * heights   # Poisson + small rel floor
        heteroscedastic = True
    else:
        sigma_z = 0.05 * heights + 0.02 * np.median(heights)  # geometry-only leverage
        heteroscedastic = False
    noise_base = sigma_z ** 2 + 1e-6

    mean = float(np.average(z))
    hypers = fit_hyperparams(X, z, noise_base, mean, n_restarts=n_restarts)
    noise_var = noise_base + (hypers.noise_floor * hypers.signal_std) ** 2

    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xstar = np.stack([GX.ravel(), GY.ravel()], axis=1)
    mstar, sstar = predict(X, z, noise_var, mean, Xstar, hypers)

    # coverage: a node whose posterior std is still ~ the prior std learned
    # nothing from data -> unconstrained -> NaN (never 0).
    covered = sstar < cov_tau * hypers.signal_std
    H = np.where(covered, mstar, np.nan).reshape(nx, ny)
    sigma = np.where(covered, sstar, np.nan).reshape(nx, ny)
    support = np.where(covered, 1.0 - sstar / hypers.signal_std, np.nan).reshape(nx, ny)

    pred_train, _ = predict(X, z, noise_var, mean, X, hypers)
    ve = variance_explained(z, pred_train)

    detectors = [{"id": pid, "x": pos_pose[pid].x, "y": pos_pose[pid].y,
                  "z": pos_pose[pid].z} for pid in pids_sorted]
    cx = float(np.mean([d["x"] for d in detectors]))
    cy = float(np.mean([d["y"] for d in detectors]))
    radii = np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2)
    coverage_radius_m = float(np.percentile(radii, 95))
    coverage_frac = float(np.mean(covered))

    note = (
        f"Height is on an ASSUMED inverse-density scale a={a:g} (opacity-units "
        "per metre) -- this campaign's 2.2 m baseline does NOT determine it; only "
        f"the fitted SHAPE at that scale is a genuine result. GP: Matern-5/2, "
        f"length-scale={hypers.length_scale:.2f} m (ML-II), "
        f"{'heteroscedastic Poisson+leverage' if heteroscedastic else 'geometry-only (no counts; pass --run)'} "
        f"noise. sigma is the posterior std (a real per-cell error bar). "
        f"{coverage_frac:.0%} of the grid within ~{coverage_radius_m:.1f} m of the "
        f"detector centroid is data-supported; the rest is NaN (unconstrained). "
        f"variance_explained={ve:.3f} over {len(z)} rays"
        + (f" (seeded subsample of {n_total})." if subsampled else ".")
    )
    return SurfaceResult(H=H, sigma=sigma, support=support, gx=gx, gy=gy, a=a,
                         variance_explained=ve, coverage_frac=coverage_frac,
                         coverage_radius_m=coverage_radius_m, n_rays=int(len(z)),
                         detectors=detectors, scale_assumed=True, note=note,
                         length_scale_m=hypers.length_scale,
                         signal_std=hypers.signal_std,
                         noise_floor=hypers.noise_floor)
