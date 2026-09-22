"""Flux-edge silhouette: the hill's angular outline, not its height.

`megido/hillside.py` tried to invert Phase-2 opacity into a height SURFACE
z=H(x,y) via parallax. Real-data diagnosis showed the P0/P1 overlap
disagreement that surface fit relies on is monotone in the density scale `a`
-- no interior minimum -- so on the real 2.2 m baseline that inversion is not
meaningful, not merely noisy (see `hillside.py` module docstring).

What the two-position flux DOES resolve well is the boundary between thick
overburden and near-open sky: a hillside silhouette, as elevation-angle vs
azimuth seen from each detector position. This is a purely ANGULAR outline --
it carries no absolute distance or height, because the 2.2 m baseline cannot
supply that (see `megido/resolution.py`, CLAUDE.md). What it does carry is a
self-consistency check: the same physical horizon should trace the same
ridgeline (within noise) as seen from both positions, and that P0/P1
agreement is the honest cross-check this module reports.

Do not read `topography.csv` or `calibration_open_sky.csv` -- this campaign's
phantom ground truth and its (unused) open-sky run are deliberately excluded
from a pipeline that has no open-sky calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.baseline import position_ids


def sky_to_azel(sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sky-frame tangents (sx, sy) -> (azimuth deg [0,360), elevation deg).

    Ray direction is d = (sx, sy, 1)/|.|. Azimuth is the bearing in the
    (sx, sy) plane; elevation is the angle above the horizon, so zenith
    (sx=sy=0) is elevation 90 and grazing (large tangent) tends to 0.
    """
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    az = np.degrees(np.arctan2(sy, sx)) % 360.0
    r2 = sx ** 2 + sy ** 2
    elev = np.degrees(np.arcsin(1.0 / np.sqrt(1.0 + r2)))
    return az, elev


def hill_threshold(lam: np.ndarray, *, frac: float = 0.5, hi_q: float = 0.9) -> float:
    """A level between open sky (normalized opacity pinned near 0) and hill.

    Set as a fraction of a high quantile of the finite, positive opacity
    values, so it sits well above the open-sky floor without depending on
    the hill's absolute (gauge-pinned, per CLAUDE.md) opacity level.
    """
    lam = np.asarray(lam, dtype=np.float64)
    finite_pos = lam[np.isfinite(lam) & (lam > 0)]
    if finite_pos.size == 0:
        return 0.0
    return float(frac * np.quantile(finite_pos, hi_q))


def extract_edge(image: np.ndarray, sky, *, level: float) -> np.ndarray:
    """(sx, sy) centres of hill-mask cells that border a non-hill cell.

    `image` is opacity reshaped to (n_bins, n_bins) in the SkyGrid flat order
    i*n_bins+j, so image[i, j] corresponds to sx=sky.centers[i],
    sy=sky.centers[j]. An off-grid neighbour counts as "not in mask".
    """
    image = np.asarray(image, dtype=np.float64)
    mask = np.isfinite(image) & (image > level)

    neighbor_up = np.zeros_like(mask)
    neighbor_up[1:, :] = mask[:-1, :]
    neighbor_down = np.zeros_like(mask)
    neighbor_down[:-1, :] = mask[1:, :]
    neighbor_left = np.zeros_like(mask)
    neighbor_left[:, 1:] = mask[:, :-1]
    neighbor_right = np.zeros_like(mask)
    neighbor_right[:, :-1] = mask[:, 1:]

    has_non_mask_neighbor = ~neighbor_up | ~neighbor_down | ~neighbor_left | ~neighbor_right
    edge_mask = mask & has_non_mask_neighbor

    centers = sky.centers
    sxm, sym = np.meshgrid(centers, centers, indexing="ij")
    return np.column_stack([sxm[edge_mask], sym[edge_mask]])


def ridgeline(az_deg: np.ndarray, elev_deg: np.ndarray, *, n_az: int = 72
             ) -> tuple[np.ndarray, np.ndarray]:
    """Per-azimuth-bin median elevation of edge points; empty bins are NaN."""
    az_deg = np.asarray(az_deg, dtype=np.float64)
    elev_deg = np.asarray(elev_deg, dtype=np.float64)
    edges = np.linspace(0.0, 360.0, n_az + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ridge = np.full(n_az, np.nan)
    if az_deg.size == 0:
        return centers, ridge
    idx = np.clip(np.digitize(az_deg, edges) - 1, 0, n_az - 1)
    for b in range(n_az):
        sel = idx == b
        if sel.any():
            ridge[b] = float(np.median(elev_deg[sel]))
    return centers, ridge


@dataclass(frozen=True)
class SilhouetteResult:
    per_pos: dict
    agreement: float
    detectors: list


def _group_positions_with_pose(cfg):
    """One (position_id, first exposure) per distinct position, first-seen order.

    Position is TRANSLATION only (see `megido.baseline.position_ids`);
    `Exposure` has no `.position` field. The real config has 4 exposures over
    2 positions (P0/T20a/T20b share a position, P1 is the other), so this
    must collapse to 2, not 4.
    """
    pid_by_exposure = position_ids(cfg)
    first_exposure = {}
    order = []
    for exp in cfg.exposures:
        pid = pid_by_exposure[exp.id]
        if pid not in first_exposure:
            first_exposure[pid] = exp
            order.append(pid)
    return order, first_exposure


def extract_silhouette(sol, cfg, *, frac: float = 0.5, hi_q: float = 0.9,
                       n_az: int = 72) -> SilhouetteResult:
    """Per-position flux-edge silhouette (ridgeline elevation vs azimuth).

    Honesty note: this is an angular outline as seen from each detector, not
    a height map -- the 2.2 m baseline cannot supply absolute distance
    (CLAUDE.md, `megido/resolution.py`). `agreement` (RMS of the ridgeline
    difference between the first two positions, over azimuth bins where both
    are finite) is the cross-check that stands in for the height fidelity
    this geometry cannot deliver.
    """
    order, first_exposure = _group_positions_with_pose(cfg)
    sky = sol.sky
    n = sky.n_bins

    per_pos = {}
    detectors = []
    for pid in order:
        exp = first_exposure[pid]
        lam = sol.normalized_opacity(pid)
        image = np.asarray(lam, dtype=np.float64).reshape(n, n)
        level = hill_threshold(lam, frac=frac, hi_q=hi_q)
        edge = extract_edge(image, sky, level=level)
        if edge.shape[0] == 0:
            edge_sx = np.array([])
            edge_sy = np.array([])
            az = np.array([])
            elev = np.array([])
        else:
            edge_sx = edge[:, 0]
            edge_sy = edge[:, 1]
            az, elev = sky_to_azel(edge_sx, edge_sy)
        ridge_az, ridge_elev = ridgeline(az, elev, n_az=n_az)

        per_pos[pid] = {
            "edge_sx": edge_sx,
            "edge_sy": edge_sy,
            "az": az,
            "elev": elev,
            "ridge_az": ridge_az,
            "ridge_elev": ridge_elev,
            "level": level,
            "n_edge": int(edge.shape[0]),
        }
        detectors.append({
            "id": pid,
            "x": float(exp.pose.x),
            "y": float(exp.pose.y),
            "z": float(exp.pose.z),
        })

    agreement = float("nan")
    if len(order) >= 2:
        r0 = per_pos[order[0]]["ridge_elev"]
        r1 = per_pos[order[1]]["ridge_elev"]
        both = np.isfinite(r0) & np.isfinite(r1)
        if both.sum() >= 3:
            diff = r0[both] - r1[both]
            agreement = float(np.sqrt(np.mean(diff ** 2)))

    return SilhouetteResult(per_pos=per_pos, agreement=agreement, detectors=detectors)
