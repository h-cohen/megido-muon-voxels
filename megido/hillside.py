"""Data-driven hillside (overburden surface) extraction from the muon flux.

Each Phase-2 sky-opacity pixel is an in-rock path length up to a density
scale; the exit point p + L*d̂ is a point on the hill surface z=H(x,y). See
docs/superpowers/specs/...-design.md §11.

`fit_scale`/`overlap_disagreement` below fit that density scale by P0/P1
parallax overlap consistency, and they remain valid on clean/synthetic
geometry (see their tests). But on the real 2.2 m baseline, the controller
found the overlap disagreement is monotone in the scale `a` — no interior
minimum — so a parallax fit there is meaningless, not merely noisy.
`fit_hillside` therefore does NOT call `fit_scale`: it reports the surface
at a stated convention scale (`a_nom`, default 1 opacity-unit = 1 m) and
flags `scale_determined=False` on every result. The lateral SHAPE is what
this geometry constrains; every result carries a per-cell bootstrap band on
that shape, and a P0/P1 shape-agreement diagnostic that does not set scale.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from megido.baseline import position_ids


def ray_dirs(tan_xy: np.ndarray) -> np.ndarray:
    """N×2 (tx,ty) -> N×3 unit sky-frame ray directions (tx,ty,1)/|.|."""
    tan_xy = np.asarray(tan_xy, dtype=np.float64)
    v = np.column_stack([tan_xy[:, 0], tan_xy[:, 1], np.ones(len(tan_xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def surface_points(p: np.ndarray, L: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Exit points p + L*d̂ (N×3) for detector position p and path lengths L."""
    p = np.asarray(p, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return p[None, :] + L[:, None] * dirs


def grid_surface(points: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    """Bin points into (x,y) cells; return (H mean-z NaN-if-empty, count)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
    inside = (x >= xedges[0]) & (x <= xedges[-1]) & (y >= yedges[0]) & (y <= yedges[-1])
    zsum = np.zeros((nx, ny)); cnt = np.zeros((nx, ny))
    np.add.at(zsum, (ix[inside], iy[inside]), z[inside])
    np.add.at(cnt, (ix[inside], iy[inside]), 1.0)
    H = np.full((nx, ny), np.nan)
    nz = cnt > 0
    H[nz] = zsum[nz] / cnt[nz]
    return H, cnt


def _cloud_H(img, a, b, xedges, yedges, L_max=60.0):
    """Map one position's (tan,lam,p) image to a gridded surface at scale a, offset b.

    Path lengths are clipped to (0, L_max]; a runaway L (opacity outlier, edge
    noise) is dropped rather than allowed to plant a bogus far-away point.
    """
    d = ray_dirs(img["tan"])
    L = a * np.asarray(img["lam"], dtype=np.float64) + b
    ok = np.isfinite(L) & (L > 0) & (L <= L_max)
    pts = surface_points(np.asarray(img["p"], float), L[ok], d[ok])
    return grid_surface(pts, xedges, yedges)


def overlap_disagreement(a, db, images, xedges, yedges, L_max=60.0):
    """Count-weighted RMS of (H_P0 - H_P1) over cells both positions populate.

    Each shared cell is weighted by min(count_P0, count_P1): a cell backed by
    many agreeing rays in both clouds dominates, while a cell that only a
    single stray outlier ray reaches barely moves the objective.
    """
    H0, c0 = _cloud_H(images["pos0"], a, 0.0, xedges, yedges, L_max=L_max)
    H1, c1 = _cloud_H(images["pos1"], a, db, xedges, yedges, L_max=L_max)
    both = (c0 > 0) & (c1 > 0)
    if both.sum() == 0:
        return np.inf, 0
    diff = H0[both] - H1[both]
    w = np.minimum(c0[both], c1[both])
    rms = float(np.sqrt(np.sum(w * diff ** 2) / np.sum(w)))
    return rms, int(both.sum())


def fit_scale(images, xedges, yedges, *, a0=1.0, db0=0.0, L_max=60.0):
    """Fit shared density scale a and P1 relative offset db by P0/P1 overlap consistency."""
    def obj(params):
        a, db = params
        if a <= 1e-6:
            return 1e6
        rms, n = overlap_disagreement(a, db, images, xedges, yedges, L_max=L_max)
        if n < 15:
            return 1e6            # too little overlap: reject
        return rms
    res = minimize(obj, [a0, db0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    a, db = res.x
    rms, n = overlap_disagreement(a, db, images, xedges, yedges, L_max=L_max)
    return {"a": float(a), "db": float(db), "disagreement": rms, "n_overlap": n}


def _sky_centers_flat(sky) -> np.ndarray:
    """N×2 (tx,ty) per flat sky pixel, whatever shape `sky.centers` takes.

    The Task-3 test double exposes `centers()` as a method already returning
    the flat N×2 array (its convenience). The real `SkyGrid.centers` is a
    property giving the 1-D per-axis bin centers; `opacity[position]` is flat
    in `i*n_bins+j` order (see `SkyGrid.bin_index`), so the meshgrid here uses
    `indexing="ij"` to match.
    """
    c = sky.centers
    pts = np.asarray(c() if callable(c) else c, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts
    tx, ty = np.meshgrid(pts, pts, indexing="ij")
    return np.column_stack([tx.ravel(), ty.ravel()])


def _group_positions(cfg):
    """One (position_id, pose) per distinct position, in first-seen order.

    A real SiteConfig has several exposures (P0/T20a/T20b/P1) sharing two
    positions (pos0/pos1); each position contributes exactly one opacity
    image and one detector world pose, never one per exposure. `Exposure` has
    no `.position` field — position is derived from pose translation via the
    canonical `megido.baseline.position_ids` (two tilts at the same spot are
    the same position), the same grouping the S2 solve itself uses.
    """
    pid_by_exposure = position_ids(cfg)
    poses = {}
    order = []
    for exp in cfg.exposures:
        pid = pid_by_exposure[exp.id]
        if pid not in poses:
            poses[pid] = exp.pose
            order.append(pid)
    return order, poses


def _build_images(sol, positions, poses, quantile: float) -> dict:
    tan = None
    images = {}
    for pid in positions:
        if tan is None:
            tan = _sky_centers_flat(sol.sky)
        lam = np.asarray(sol.normalized_opacity(pid, transparent_quantile=quantile),
                         dtype=np.float64)
        p = poses[pid]
        images[pid] = {"tan": tan, "lam": lam, "p": np.array([p.x, p.y, p.z], float)}
    return images


def _build_edges(positions, poses, footprint_m, cell_m):
    xs = [poses[pid].x for pid in positions]
    ys = [poses[pid].y for pid in positions]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    if footprint_m is None:
        footprint_m = 14.0
    n = max(int(round(2 * footprint_m / cell_m)), 1)
    xedges = cx - footprint_m + cell_m * np.arange(n + 1)
    yedges = cy - footprint_m + cell_m * np.arange(n + 1)
    return xedges, yedges


def _combined_cloud(images, a, db, xedges, yedges, L_max=60.0):
    """Combined H/count from BOTH clouds, plus each position's own-only H.

    pos0 carries offset 0 and pos1 carries the fitted relative offset `db`,
    matching the convention `fit_scale`/`overlap_disagreement` use.
    """
    offsets = {"pos0": 0.0, "pos1": db}
    all_pts = []
    per_pos = {}
    for pid, img in images.items():
        b = offsets.get(pid, 0.0)
        d = ray_dirs(img["tan"])
        L = a * np.asarray(img["lam"], dtype=np.float64) + b
        ok = np.isfinite(L) & (L > 0) & (L <= L_max)
        pts = surface_points(np.asarray(img["p"], float), L[ok], d[ok])
        all_pts.append(pts)
        per_pos[pid] = grid_surface(pts, xedges, yedges)
    combined = (np.concatenate(all_pts, axis=0) if all_pts
                else np.zeros((0, 3)))
    H, count = grid_surface(combined, xedges, yedges)
    return H, count, per_pos


def _overlap_agreement(per_pos, H) -> float:
    if "pos0" not in per_pos or "pos1" not in per_pos:
        return float("nan")
    H0, c0 = per_pos["pos0"]
    H1, c1 = per_pos["pos1"]
    both = (c0 > 0) & (c1 > 0)
    if both.sum() == 0:
        return float("nan")
    diff = H0[both] - H1[both]
    rms = float(np.sqrt(np.mean(diff ** 2)))
    finite_H = H[np.isfinite(H)]
    scale = float(np.ptp(finite_H)) if finite_H.size else 0.0
    if scale <= 0:
        return float("nan")
    return rms / scale


def _predict_lambda(img, a, db_for_pid, H, xedges, yedges, L_max=60.0,
                     n_iter=25, damping=0.5):
    """Self-consistency check: re-find where each ray hits the fitted H(x,y)
    and compare the implied opacity to the measured one (mirrors the
    damped fixed-point ray march used to build the synthetic test hill)."""
    p = np.asarray(img["p"], dtype=np.float64)
    d = ray_dirs(img["tan"])
    lam = np.asarray(img["lam"], dtype=np.float64)
    L0 = a * lam + db_for_pid
    ok = np.isfinite(L0) & (L0 > 0) & (L0 <= L_max)
    L = np.where(ok, np.clip(L0, 0.1, L_max), 5.0).astype(np.float64)
    nx, ny = H.shape

    def H_at(x, y):
        ix = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
        iy = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
        return H[ix, iy]

    for _ in range(n_iter):
        s = p[None, :] + L[:, None] * d
        Hs = H_at(s[:, 0], s[:, 1])
        valid = np.isfinite(Hs)
        L_step = np.where(valid,
                           np.clip(np.where(valid, Hs, 0.0) / np.clip(d[:, 2], 1e-3, None),
                                   0.1, L_max),
                           L)
        L = (1 - damping) * L + damping * L_step

    lam_pred = (L - db_for_pid) / a
    valid = ok & np.isfinite(lam_pred)
    return lam[valid], lam_pred[valid]


def _data_residual(images, a, db, H, xedges, yedges, L_max=60.0) -> float:
    offsets = {"pos0": 0.0, "pos1": db}
    obs, pred = [], []
    for pid, img in images.items():
        o, pr = _predict_lambda(img, a, offsets.get(pid, 0.0), H, xedges, yedges, L_max=L_max)
        obs.append(o)
        pred.append(pr)
    if not obs:
        return float("nan")
    obs = np.concatenate(obs)
    pred = np.concatenate(pred)
    if obs.size == 0:
        return float("nan")
    scale = float(np.std(obs))
    rms = float(np.sqrt(np.mean((pred - obs) ** 2)))
    if scale <= 0:
        return rms
    return rms / scale


@dataclass
class HillsideResult:
    """The overburden-surface fit, with its bootstrap uncertainty band.

    `H`/`sigma`/`count` are nx×ny; NaN off the hill (nowhere a ray landed).

    The real 2.2 m baseline turned out NOT to pin the density scale: on real
    data the P0/P1 overlap disagreement is monotone in `a` (no interior
    minimum), so fitting `a` from parallax (`fit_scale`) is meaningless here
    — verified by the controller against real S2 output. `a`/`db` are
    therefore a stated CONVENTION scale (1 opacity-unit = 1 m by default),
    not a fit; `scale_determined` is always False, and `height_confidence`
    says so explicitly. `shape_consistency` (P0/P1 overlap RMS ÷ H relief) is
    reported as a diagnostic of how well the two positions' *shapes* agree —
    it does not set the scale. `sigma` is the per-cell 1-sigma spread of `H`
    under the gauge bootstrap (varying `normalized_opacity`'s
    `transparent_quantile`) at fixed (a, db) — the honest lateral-shape
    uncertainty this geometry does have a handle on.
    """
    H: np.ndarray
    sigma: np.ndarray
    count: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    a: float
    db: float
    disagreement: float
    data_residual: float
    overlap_agreement: float
    height_confidence: str
    shape_consistency: float = float("nan")
    scale_determined: bool = False
    detectors: list = field(default_factory=list)


def fit_hillside(sol, cfg, *, a_nom=1.0, footprint_m=None, cell_m=0.5,
                  n_boot=8, quantiles=None, db=0.0) -> HillsideResult:
    """Build the overburden surface H(x,y) from both positions' opacity images,
    at a stated convention scale.

    Groups `cfg.exposures` by `.position` (a real config has several
    exposures sharing two positions — one image and one world pose per
    position, never per exposure). Does NOT fit the density scale from
    parallax: on the real 2.2 m baseline the P0/P1 overlap disagreement is
    monotone in `a`, so `fit_scale` has no interior minimum to find there.
    Instead uses the stated `a_nom`/`db` (default: 1 opacity-unit = 1 m, no
    relative offset), builds the combined two-cloud height field at that
    scale, reports P0/P1 shape agreement as a diagnostic (not a scale fit),
    and bootstraps the Phase-2 gauge (`transparent_quantile`) at fixed
    (a_nom, db) to get a per-cell lateral-shape uncertainty band.
    """
    positions, poses = _group_positions(cfg)
    xedges, yedges = _build_edges(positions, poses, footprint_m, cell_m)

    images = _build_images(sol, positions, poses, quantile=0.05)
    a = float(a_nom)

    H, count, per_pos = _combined_cloud(images, a, db, xedges, yedges)
    overlap_agreement = _overlap_agreement(per_pos, H)
    data_residual = _data_residual(images, a, db, H, xedges, yedges)

    disagreement, _n_overlap = overlap_disagreement(a, db, images, xedges, yedges)
    finite_H = H[np.isfinite(H)]
    relief = float(np.ptp(finite_H)) if finite_H.size else 0.0
    shape_consistency = (disagreement / relief) if relief > 0 and np.isfinite(disagreement) \
        else float("nan")

    if quantiles is None:
        quantiles = [0.02, 0.05, 0.1, 0.2]
    H_boot = np.full((n_boot,) + H.shape, np.nan)
    for k in range(n_boot):
        q = quantiles[k % len(quantiles)]
        images_q = _build_images(sol, positions, poses, quantile=q)
        H_q, _, _ = _combined_cloud(images_q, a, db, xedges, yedges)
        H_boot[k] = H_q

    sigma = np.full(H.shape, np.nan)
    m = np.isfinite(H)
    n_finite = np.sum(np.isfinite(H_boot), axis=0)
    enough = m & (n_finite >= 2)
    if enough.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sigma[enough] = np.nanstd(H_boot[:, enough], axis=0)
    sigma[m & (n_finite < 2)] = 0.0

    sc_str = f"{shape_consistency:.2f}" if np.isfinite(shape_consistency) else "n/a"
    height_confidence = (
        "absolute height NOT determined by 2.2 m parallax (overlap "
        "disagreement is monotone in scale on real data, no interior "
        "minimum); shown at convention scale (1 opacity-unit = 1 m), true "
        "height proportional to 1/rho (rho assumed, external); lateral "
        f"shape constrained; P0/P1 shape agreement (overlap RMS/relief) = {sc_str}"
    )

    detectors = [{"id": pid, "x": poses[pid].x, "y": poses[pid].y, "z": poses[pid].z}
                 for pid in positions]

    return HillsideResult(
        H=H, sigma=sigma, count=count, xedges=xedges, yedges=yedges,
        a=a, db=float(db), disagreement=disagreement,
        data_residual=data_residual, overlap_agreement=overlap_agreement,
        height_confidence=height_confidence, shape_consistency=shape_consistency,
        scale_determined=False, detectors=detectors,
    )
