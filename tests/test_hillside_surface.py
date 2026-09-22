import numpy as np

from megido.config import Exposure, Pose, SiteConfig, Binning
from megido.hillside_surface import (
    bilinear_matrix,
    fit_surface,
    laplacian_matrix,
    solve_surface,
    variance_explained,
)
from megido.sky import make_sky_grid


def test_bilinear_matrix_partition_of_unity():
    gx = np.linspace(0, 10, 11)
    gy = np.linspace(0, 10, 11)
    B, support = bilinear_matrix(np.array([[2.5, 3.5], [0.0, 0.0]]), gx, gy)
    assert abs(B.sum(axis=1)[0, 0] - 1.0) < 1e-9  # weights sum to 1
    assert B.shape[1] == 121


def test_solve_recovers_a_plane():
    gx = np.linspace(0, 10, 11)
    gy = np.linspace(0, 10, 11)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Htrue = (2 * GX + 3 * GY + 1).ravel()
    # sample the plane at random points
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.5, 9.5, (500, 2))
    z = 2 * xy[:, 0] + 3 * xy[:, 1] + 1
    B, support = bilinear_matrix(xy, gx, gy)
    lap = laplacian_matrix(11, 11)
    h = solve_surface(B, z, np.ones(len(z)), lap, mu=0.01)
    m = support > 0.5
    assert np.corrcoef(h[m], Htrue[m])[0, 1] > 0.999  # plane recovered where supported
    assert variance_explained(B, z, np.ones(len(z)), h) > 0.99


def test_laplacian_penalises_curvature_not_planes():
    lap = laplacian_matrix(6, 6)
    GX, GY = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    plane = (3 * GX + 2 * GY).ravel().astype(float)
    assert np.allclose(lap @ plane, 0, atol=1e-9)  # a plane has zero 2nd-difference (interior)


# --- Task 2: fit_surface driver -------------------------------------------

_H0 = 8.0     # metres above the detectors, at "assumed" scale a=1
_AMP = 3.0    # bump height above h0
_CX, _CY = 1.1, 0.0  # crown, between the two positions
_SIGMA = 3.0


def _true_hill(x, y):
    r2 = (x - _CX) ** 2 + (y - _CY) ** 2
    return _H0 + _AMP * np.exp(-r2 / (2 * _SIGMA ** 2))


def _hill_term(x, y):
    r2 = (x - _CX) ** 2 + (y - _CY) ** 2
    return _AMP * np.exp(-r2 / (2 * _SIGMA ** 2))


def _path_length_to_hill(ox, oy, oz, dx, dy, dz, n_iter=80):
    """Ray-march (test-side ONLY, to generate synthetic ground truth -- this
    is not part of the inversion) the distance t along (dx, dy, dz) from
    (ox, oy, oz) to the known Gaussian hill surface, by Newton's method."""
    t = np.maximum((_H0 - oz) / dz, 0.05)
    for _ in range(n_iter):
        x = ox + t * dx
        y = oy + t * dy
        hterm = _hill_term(x, y)
        h = _H0 + hterm
        f = oz + t * dz - h
        dhdx = -hterm * (x - _CX) / _SIGMA ** 2
        dhdy = -hterm * (y - _CY) / _SIGMA ** 2
        fprime = dz - dhdx * dx - dhdy * dy
        fprime = np.where(np.abs(fprime) < 1e-9, 1e-9, fprime)
        t = np.maximum(t - f / fprime, 1e-6)
    return t


def _fake_cfg():
    from pathlib import Path
    pose0 = Pose(x=0.0, y=0.0, z=0.0, tilt_deg=0.0, az_deg=0.0)
    pose1 = Pose(x=2.2, y=0.0, z=0.0, tilt_deg=0.0, az_deg=0.0)
    exposures = (
        Exposure(id="P0", run_ids=(1,), pose=pose0),
        Exposure(id="P1", run_ids=(2,), pose=pose1),
    )
    return SiteConfig(
        site="test-hill", data_dir=Path("."), frame_origin="", x_axis_bearing_deg=0.0,
        exposures=exposures, binning=Binning(),
    )


class _FakeSol:
    """Minimal stand-in for BaselineSolution: only `.sky` and
    `.normalized_opacity` are used by exit_points/fit_surface.

    `normalized_opacity` here carries the raw path length to the known hill
    at `transparent_quantile=0.05` exactly (so the recovery gate is a clean
    test of the geometry/fitting, not of the quantile-shift arithmetic), and
    a small quantile-dependent offset otherwise, standing in for Phase 2's
    real per-quantile gauge freedom -- enough to make the bootstrap over
    `transparent_quantile` move the fit (sigma > 0) without the clip-to-zero
    pileup a full per-pixel quantile subtraction would put at the detector.
    """

    def __init__(self, sky, raw_lam_by_pos):
        self.sky = sky
        self._raw = raw_lam_by_pos

    def normalized_opacity(self, position_id_, transparent_quantile=0.05):
        lam = np.array(self._raw[position_id_], dtype=np.float64)
        offset = (transparent_quantile - 0.05) * 0.5
        return np.clip(lam - offset, 0.0, None)


def _build_fake_sol_from_hill(t_max=1.0, n_bins=40):
    """Build a fake sol whose (pre-gauge) opacity IS the true path length to
    the known hill, computed through the real sky sampling at the real
    two-position geometry -- so a correct inversion at a=1 should recover the
    hill's shape."""
    sky = make_sky_grid(t_max=t_max, n_bins=n_bins)
    centers = sky.centers
    sx_grid, sy_grid = np.meshgrid(centers, centers, indexing="ij")
    sx = sx_grid.ravel()
    sy = sy_grid.ravel()
    norm = np.sqrt(sx ** 2 + sy ** 2 + 1.0)
    dx, dy, dz = sx / norm, sy / norm, 1.0 / norm

    poses = {"pos0": (0.0, 0.0, 0.0), "pos1": (2.2, 0.0, 0.0)}
    raw = {}
    for pid, (ox, oy, oz) in poses.items():
        raw[pid] = _path_length_to_hill(ox, oy, oz, dx, dy, dz)
    return _FakeSol(sky, raw)


def test_synthetic_hill_recovery_gate():
    """Load-bearing gate: a known hill, sampled through the real sky grid and
    real two-position geometry, must be recovered by fit_surface at a=1."""
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()

    result = fit_surface(sol, cfg, a=1.0, cell_m=0.5, mu=0.3, n_boot=2)

    covered = ~np.isnan(result.H)
    assert covered.any() and (~covered).any()  # partial coverage: some NaN, some filled

    GX, GY = np.meshgrid(result.gx, result.gy, indexing="ij")
    Htrue = _true_hill(GX, GY)

    corr = np.corrcoef(result.H[covered], Htrue[covered])[0, 1]
    assert corr > 0.9, f"shape correlation too low: {corr}"
    assert result.variance_explained > 0.9

    # crown recovered within a couple of grid cells
    fitted_flat = np.where(covered, result.H, -np.inf)
    i_fit, j_fit = np.unravel_index(np.argmax(fitted_flat), fitted_flat.shape)
    crown_x_fit, crown_y_fit = result.gx[i_fit], result.gy[j_fit]
    assert abs(crown_x_fit - _CX) < 2.5
    assert abs(crown_y_fit - _CY) < 2.5

    assert result.scale_assumed is True
    assert "assumed" in result.note.lower() or "ASSUMED" in result.note


def test_bootstrap_sigma_nonzero_and_masked_with_coverage():
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()

    result = fit_surface(sol, cfg, a=1.0, cell_m=0.5, mu=0.3, n_boot=6,
                          quantiles=[0.02, 0.05, 0.1, 0.2])

    covered = ~np.isnan(result.H)
    assert np.nanmax(result.sigma) > 0
    # sigma is NaN exactly where H is NaN
    assert np.array_equal(np.isnan(result.sigma), np.isnan(result.H))
    assert covered.any()


def test_solve_stability_with_unsupported_corner_does_not_raise():
    """A grid node that gets zero bilinear support and no Laplacian coupling
    (an isolated corner beyond the data footprint) must not make the solve
    singular; the ridge term keeps it well-posed and the node is masked."""
    gx = np.linspace(0.0, 10.0, 11)
    gy = np.linspace(0.0, 10.0, 11)
    rng = np.random.default_rng(1)
    # points only in the lower-left, leaving the (10, 10) corner unsupported
    xy = rng.uniform(0.5, 4.5, (200, 2))
    z = 2 * xy[:, 0] + 1 * xy[:, 1] + 5

    from megido.hillside_surface import _solve_with_ridge

    B, support = bilinear_matrix(xy, gx, gy)
    lap = laplacian_matrix(len(gx), len(gy))
    w = np.ones(len(z))

    h = _solve_with_ridge(B, z, w, lap, mu=0.3, ridge=1e-6)
    assert np.all(np.isfinite(h))  # did not raise / did not produce NaNs or infs

    corner_flat = (len(gx) - 1) * len(gy) + (len(gy) - 1)
    assert support[corner_flat] == 0.0  # confirms the corner really is unsupported
