"""Phase 2 for a campaign that HAS an open-sky run: divide by it.

Megiddo has no open-sky calibration, so `megido.baseline` separates detector
response from absorption with the tilt campaign. A campaign with a clear-sky
run of the same detector (the TAU cafeteria) does not need that: the response
and the cosmic flux shape are both in the sky run, and they cancel in the
per-bin ratio

    T(d) = n_pos(d) / (scale * n_sky(d)),    lambda = -ln(max(T, t_floor))

(the cafeteria project's calibration, `muontomo.calibration`). The ratio is
formed in the DETECTOR frame, where the response lives, then scattered to the
world-frame sky grid through each exposure's pose, exactly like the joint
solve's opacity.

The absolute level is still not measured: `scale` (the live-time / rate ratio
between runs) is set by the `norm_quantile` convention and is degenerate with
lambda's zero point, so the solution is gauge-pinned to median zero like the
joint solve and consumed through `normalized_opacity`. Only differences and
shape are meaningful -- the same honest limit as Megiddo.

The result is an ordinary `BaselineSolution` so reconstruct / export /
hillside run unchanged. It carries no smooth detector correction (coeffs are
zero): the sky run IS the response.
"""
from __future__ import annotations

import numpy as np

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, position_ids
from megido.basis import make_smooth_basis
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.sky import SkyGrid, detector_to_sky, make_sky_grid

MIN_SKY_COUNTS = 25.0    # cafeteria CalibrationConfig.min_sky
NORM_QUANTILE = 0.95     # cafeteria CalibrationConfig.norm_quantile
T_FLOOR = 0.05           # cafeteria clip; caps lambda at -ln(0.05) ~ 3.0


def _accumulate(grid: AnalysisGrid, cfg: SiteConfig, geom: DetectorGeometry,
                sky: SkyGrid, min_sky: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per position: (observed position counts, reference sky counts) per sky bin."""
    if cfg.sky_reference is None:
        raise ValueError("config has no sky_reference; use megido.baseline.solve_baseline")
    sky_id = cfg.sky_reference.id
    if sky_id not in grid.counts:
        raise ValueError(f"analysis grid has no {sky_id!r} counts; ingest the sky reference")
    n_sky = grid.counts[sky_id].astype(np.float64)
    tx, ty = grid.tan_mesh()
    acc = geometric_acceptance(tx, ty, geom)
    positions = position_ids(cfg)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for eid, counts in grid.counts.items():
        if eid == sky_id:
            continue
        exp = cfg.exposure(eid)
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acc > 0) & on_sky & in_grid & (n_sky >= min_sky)
        obs, ref = out.setdefault(positions[eid], (np.zeros(sky.flat_size),
                                                   np.zeros(sky.flat_size)))
        np.add.at(obs, flat[live], counts[live].astype(np.float64))
        np.add.at(ref, flat[live], n_sky[live])
    return out


def solve_skyref(grid: AnalysisGrid, cfg: SiteConfig, *,
                 geom: DetectorGeometry | None = None,
                 sky: SkyGrid | None = None,
                 min_sky: float = MIN_SKY_COUNTS,
                 norm_quantile: float = NORM_QUANTILE,
                 t_floor: float = T_FLOOR,
                 flux_index: float = 2.0,
                 **_solver_kwargs) -> BaselineSolution:
    """Sky-ratio opacity per position. Extra kwargs (e.g. `n_iter` from the
    bootstrap drivers) are accepted and ignored: this solve is closed-form.

    `flux_index` is not used by the ratio (flux cancels); it is carried only
    because the hillside surface's flux weighting reads it off the solution.
    """
    geom = geom or DetectorGeometry.for_site(cfg)
    sky = sky or make_sky_grid()
    basis = make_smooth_basis()

    opacity: dict[str, np.ndarray] = {}
    norms: dict[str, float] = {}
    for pid, (obs, ref) in sorted(_accumulate(grid, cfg, geom, sky, min_sky).items()):
        seen = ref > 0
        lam = np.full(sky.flat_size, np.nan)
        if seen.any():
            ratio = obs[seen] / ref[seen]
            scale = float(np.quantile(ratio, norm_quantile))
            scale = scale if scale > 0 else 1.0
            lam[seen] = -np.log(np.clip(ratio / scale, t_floor, None))
            lam[seen] -= np.median(lam[seen])     # same internal gauge as solve_baseline
            norms[pid] = scale
        opacity[pid] = lam

    return BaselineSolution(coeffs=np.zeros(basis.n_coeff), opacity=opacity, norms=norms,
                            flux_index=flux_index, nll_history=[], grid=grid, sky=sky,
                            basis=basis)


def skyref_sigma(grid: AnalysisGrid, cfg: SiteConfig, *,
                 geom: DetectorGeometry | None = None, sky: SkyGrid | None = None,
                 min_sky: float = MIN_SKY_COUNTS) -> dict[str, np.ndarray]:
    """Analytic Poisson sigma of lambda: sqrt(1/n_pos + 1/n_sky) per sky bin."""
    geom = geom or DetectorGeometry.for_site(cfg)
    sky = sky or make_sky_grid()
    out = {}
    for pid, (obs, ref) in _accumulate(grid, cfg, geom, sky, min_sky).items():
        s = np.full(sky.flat_size, np.nan)
        seen = ref > 0
        s[seen] = np.sqrt(1.0 / np.maximum(obs[seen], 1.0) + 1.0 / ref[seen])
        out[pid] = s
    return out
