"""What this geometry can resolve, computed before any data is fitted.

Spec section 7.4 makes uncertainty a deliverable rather than an appendix, and in
a limited-angle campaign with few viewpoints the dominant error is depth. These
are closed-form geometric statements: they depend on where the detector stood
and how finely angles are binned, not on the counts, so they are the honest
prior on what any reconstruction of this campaign can possibly mean.

The Megiddo campaign has ONE baseline: P0 and T20 share a spot (two tilts, one
translation), so only P0 <-> P1 at 2.2 m carries parallax.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from megido.config import SiteConfig
from megido.fitdata import position_origins
from megido.forward import ForwardModel


def depth_resolution(z_m: float, baseline_m: float, sigma_t: float) -> float:
    """Depth uncertainty of a feature at height z, seen from two views b apart.

    Two rays converging at z from detectors b apart differ in tangent by
    Delta_t = b / z, so dz = (z^2 / b) * d(Delta_t). Each view contributes
    sigma_t, adding in quadrature:  dz = sqrt(2) * sigma_t * z^2 / b.
    """
    if baseline_m <= 0:
        return float("inf")
    return math.sqrt(2.0) * sigma_t * z_m**2 / baseline_m


def alias_period(z_m: float, baseline_m: float, feature_pitch_m: float) -> float:
    """Height spacing of false solutions for a periodic structure (spec 7.4).

    A pattern of pitch p at height z reprojects onto itself when the depth shifts
    by p * z / b. The cafeteria campaign had 5.8 m of margin against this; a
    synthetic test there aliased at 0.9 m.
    """
    if baseline_m <= 0:
        return float("inf")
    return feature_pitch_m * z_m / baseline_m


def position_baselines(cfg: SiteConfig) -> dict[tuple[str, str], float]:
    """Horizontal separation of every pair of POSITIONS (not exposures)."""
    origins = position_origins(cfg)
    out: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(sorted(origins), 2):
        pa, pb = origins[a], origins[b]
        out[(a, b)] = float(math.dist(pa, pb))
    return out


def views_per_voxel(fwd: ForwardModel) -> np.ndarray:
    """How many distinct positions have a ray through each voxel.

    This is the map the spec's viewer calls the exposure-contribution layer: it
    separates where the answer is data from where it is prior. A voxel seen by
    one position carries no depth information at all — whatever the solver puts
    there came from the regulariser, not from a measurement.
    """
    A = fwd.A.tocsr()
    seen = np.zeros((len(fwd.rows.position_ids), A.shape[1]), dtype=bool)
    for i in range(len(fwd.rows.position_ids)):
        rows = np.nonzero(fwd.rows.pos_of_row == i)[0]
        if rows.size:
            block = A[rows]
            seen[i, np.unique(block.indices)] = True
    return seen.sum(axis=0).reshape(fwd.grid.shape).astype(np.int16)


def rays_per_voxel(fwd: ForwardModel) -> np.ndarray:
    """How many measured directions (rows) cross each voxel.

    Finer than `views_per_voxel`: near the grid edge a voxel can be seen by a
    position yet crossed by only one of its rays. Such a voxel is that ray's
    private degree of freedom -- under non-negativity the solver parks the
    ray's noise there as mass (the cafeteria "bright outer shell",
    docs/cafeteria-run.md). The viewer's coverage gate reads this map.
    """
    A = fwd.A.tocsc()
    return np.diff(A.indptr).reshape(fwd.grid.shape).astype(np.int32)


def campaign_resolution(cfg: SiteConfig, *, sigma_t: float,
                        feature_pitch_m: float) -> dict:
    """Closed-form resolution report for a configured campaign.

    `sigma_t` is the angular bin width of the sky grid (0.05 for this campaign);
    `feature_pitch_m` is the scale of structure being looked for, still an open
    item in spec section 11 — pass the value being assumed and it is recorded in
    the report rather than buried.
    """
    baselines = position_baselines(cfg)
    max_b = max(baselines.values(), default=0.0)
    z0, z1 = float(cfg.volume.z_min_m), float(cfg.volume.z_max_m)
    zmid = 0.5 * (z0 + z1)

    dz = {"z_min": depth_resolution(z0, max_b, sigma_t),
          "z_mid": depth_resolution(zmid, max_b, sigma_t),
          "z_max": depth_resolution(z1, max_b, sigma_t)}
    # "Resolved" means the depth error at mid-range is smaller than the voxel
    # grid's own spacing. Anything coarser cannot place a surface within a
    # single voxel, so the solver's z-placement is prior, not data, however
    # fine the grid was drawn.
    spacing = float(cfg.volume.spacing_m)
    resolved = dz["z_mid"] < spacing

    verdict = (
        f"depth RESOLVED at mid-range: dz = {dz['z_mid']:.2f} m is finer than "
        f"the {spacing:.2f} m voxel spacing"
        if resolved else
        f"depth NOT resolved: dz = {dz['z_mid']:.2f} m at z = {zmid:.1f} m is "
        f"{dz['z_mid'] / spacing:.0f}x the {spacing:.2f} m voxel spacing. "
        f"Lateral structure is still measured; the height of that structure is "
        f"set by the regulariser, not by the data."
    )

    return {
        "n_positions": len(position_origins(cfg)),
        "baselines_m": {f"{a}-{b}": round(v, 4) for (a, b), v in baselines.items()},
        "max_baseline_m": max_b,
        "sigma_t": sigma_t,
        "z_range_m": (z0, z1),
        "depth_resolution_m": dz,
        "feature_pitch_m": feature_pitch_m,
        "alias_period_m": alias_period(zmid, max_b, feature_pitch_m),
        "depth_resolved": bool(resolved),
        "verdict": verdict,
    }


def format_resolution(report: dict) -> str:
    lines = [
        f"positions            {report['n_positions']}",
        f"max baseline         {report['max_baseline_m']:.3f} m",
        f"angular bin sigma_t  {report['sigma_t']:.4f}",
        f"z range              {report['z_range_m'][0]:.2f} - "
        f"{report['z_range_m'][1]:.2f} m",
        "depth resolution     " + "  ".join(
            f"{k}={v:.2f} m" for k, v in report["depth_resolution_m"].items()),
        f"alias period         {report['alias_period_m']:.2f} m "
        f"(assuming {report['feature_pitch_m']:.2f} m feature pitch)",
        "",
        report["verdict"],
    ]
    if report["baselines_m"]:
        lines.insert(1, "  " + "  ".join(
            f"{k}={v:.3f} m" for k, v in sorted(report["baselines_m"].items())))
    return "\n".join(lines)
