"""Ray-space check of the fitted hillside surface.

Treat the fitted surface H(x, y) as the top of a uniform solid of density 1/a
(the same assumption that placed its exit points), ray-trace every measured
sky direction from each detector to where it leaves the rock, and compare the
predicted opacity t*/a with what was measured. A goodness-of-fit in the space
the data actually lives in; the per-cell VE of the surface fit is not that.

Because H scales ~linearly with a about each detector, t* scales by a and the
prediction is ~scale-free: this checks the surface's SHAPE (measured), not its
assumed scale. Unpredictable rays (surface unknown / off-grid / never crossed /
detector not below the surface) are NaN, never 0, and excluded from every
metric.

t* is the FIRST exit: a ray that leaves the terrain and re-enters it further
out is under-predicted. That is the same single-exit assumption `exit_points`
makes when it places the surface, so the check is consistent with the fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.hillside_surface import _positions


def _bilinear(H, gx, gy, x, y) -> np.ndarray:
    H = np.asarray(H, float)
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = H.shape
    out = np.full(x.shape, np.nan)
    inside = (x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])
    if not inside.any():
        return out
    xi, yi = x[inside], y[inside]
    i0 = np.clip(np.searchsorted(gx, xi, side="right") - 1, 0, nx - 2)
    j0 = np.clip(np.searchsorted(gy, yi, side="right") - 1, 0, ny - 2)
    tx = (xi - gx[i0]) / (gx[i0 + 1] - gx[i0])
    ty = (yi - gy[j0]) / (gy[j0 + 1] - gy[j0])
    h00, h10 = H[i0, j0], H[i0 + 1, j0]
    h01, h11 = H[i0, j0 + 1], H[i0 + 1, j0 + 1]
    # a NaN corner poisons the sample even at zero weight (0*NaN = NaN): intended
    out[inside] = ((1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10
                   + (1 - tx) * ty * h01 + tx * ty * h11)
    return out


def ray_exit_distance(origin, dirs, H, gx, gy, *, t_max=None, step=None,
                      n_bisect: int = 30) -> np.ndarray:
    o = np.asarray(origin, float)
    d = np.atleast_2d(np.asarray(dirs, float))
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    H = np.asarray(H, float)
    n = d.shape[0]
    t_out = np.full(n, np.nan)
    h_here = _bilinear(H, gx, gy, np.array([o[0]]), np.array([o[1]]))[0]
    if not np.isfinite(h_here) or o[2] >= h_here:
        return t_out
    if step is None:
        step = 0.25 * min(float(np.min(np.diff(gx))), float(np.min(np.diff(gy))))
    if t_max is None:
        span = float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]))
        top = float(np.nanmax(H)) if np.isfinite(H).any() else o[2]
        t_max = span + abs(top - o[2]) + step
    active = np.ones(n, bool)
    for k in range(1, int(np.ceil(t_max / step)) + 1):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        t = k * step
        p = o[None, :] + t * d[idx]
        h = _bilinear(H, gx, gy, p[:, 0], p[:, 1])
        unknown = ~np.isfinite(h)
        active[idx[unknown]] = False            # stays NaN: surface unknown
        cross = np.isfinite(h) & (p[:, 2] >= h)
        ci = idx[cross]
        if ci.size:
            lo = np.full(ci.size, (k - 1) * step)
            hi = np.full(ci.size, t)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                pm = o[None, :] + mid[:, None] * d[ci]
                hm = _bilinear(H, gx, gy, pm[:, 0], pm[:, 1])
                # NaN -> False -> treat as below. Both bracket ends are finite,
                # so a NaN here means the segment grazes a NaN-cornered cell at
                # the edge of the unknown mask; the result stays within one
                # march step (cell/4) of the crossing rather than going NaN.
                above = pm[:, 2] >= hm
                hi = np.where(above, mid, hi)
                lo = np.where(above, lo, mid)
            t_out[ci] = hi
            active[ci] = False
    return t_out


@dataclass(frozen=True)
class RayCheck:
    residual: dict
    predicted: dict
    ray_ve: float
    ray_rms: float
    n_checked: int
    a: float


def surface_ray_check(sol, cfg, result, *, a=None) -> RayCheck:
    a = float(result.a if a is None else a)
    H = np.asarray(result.H, float)
    gx = np.asarray(result.gx, float); gy = np.asarray(result.gy, float)
    centers = np.asarray(sol.sky.centers, float)
    nb = centers.size
    SX, SY = np.meshgrid(centers, centers, indexing="ij")
    sx, sy = SX.ravel(), SY.ravel()
    norm = np.sqrt(1.0 + sx ** 2 + sy ** 2)
    dirs = np.stack([sx / norm, sy / norm, 1.0 / norm], axis=1)

    residual, predicted, meas, pred = {}, {}, [], []
    for pid, pose in sorted(_positions(cfg).items()):
        lam = np.asarray(sol.normalized_opacity(pid), float)
        live = np.isfinite(lam)
        t = np.full(lam.shape, np.nan)
        if live.any():
            t[live] = ray_exit_distance((pose.x, pose.y, pose.z), dirs[live], H, gx, gy)
        p = t / a
        r = lam - p
        ok = np.isfinite(r)
        residual[pid] = np.where(ok, r, np.nan).reshape(nb, nb)
        predicted[pid] = np.where(np.isfinite(p), p, np.nan).reshape(nb, nb)
        meas.append(lam[ok]); pred.append(p[ok])

    m = np.concatenate(meas) if meas else np.zeros(0)
    q = np.concatenate(pred) if pred else np.zeros(0)
    if m.size == 0:
        return RayCheck(residual, predicted, float("nan"), float("nan"), 0, a)
    rr = m - q
    denom = float(np.sum((m - m.mean()) ** 2))
    ve = float(1.0 - np.sum(rr ** 2) / denom) if denom > 0 else float("nan")
    return RayCheck(residual, predicted, ve, float(np.sqrt(np.mean(rr ** 2))),
                    int(m.size), a)
