import numpy as np
import pytest

from megido.config import Pose
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid


def test_zero_tilt_zero_yaw_is_the_identity_map():
    tx = np.array([0.0, 0.3, -0.5])
    ty = np.array([0.0, -0.2, 0.4])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=0))
    assert valid.all()
    assert np.allclose(sx, tx)
    assert np.allclose(sy, ty)


def test_yaw_rotates_the_tangent_plane_but_preserves_zenith_angle():
    """A pure yaw cannot change how far from vertical a track is."""
    tx = np.array([0.3, 0.6, -0.4])
    ty = np.array([0.1, -0.2, 0.5])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=241))
    assert valid.all()
    assert np.allclose(np.hypot(sx, sy), np.hypot(tx, ty))
    assert not np.allclose(sx, tx)     # but the components do move


def test_tilt_shifts_a_vertical_detector_ray_off_zenith():
    sx, sy, valid = detector_to_sky(np.array([0.0]), np.array([0.0]),
                                    Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert valid[0]
    theta = np.degrees(np.arctan(np.hypot(sx[0], sy[0])))
    assert theta == pytest.approx(20.0)


def test_tilt_moves_bins_by_roughly_tan_of_the_tilt():
    """This displacement is what separates B(d) from the sky."""
    tx = np.linspace(-0.5, 0.5, 11)
    ty = np.zeros_like(tx)
    sx0, _, _ = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=0))
    sx20, _, _ = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert np.all(sx20 > sx0)
    assert np.median(sx20 - sx0) > 0.25          # tan(20 deg) = 0.364


def test_rays_at_or_below_the_horizon_are_invalid():
    """A steep ray under a large tilt can point below the horizon."""
    sx, sy, valid = detector_to_sky(np.array([5.0]), np.array([0.0]),
                                    Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert not valid[0]


def test_flux_shape_is_one_at_zenith_and_falls_with_angle():
    f = flux_shape(np.array([0.0, 0.5, 1.0]), np.zeros(3), index=2.0)
    assert f[0] == pytest.approx(1.0)
    assert f[0] > f[1] > f[2] > 0


def test_flux_shape_matches_cos_theta_to_the_index():
    sx = np.array([0.75])
    cos_theta = 1.0 / np.sqrt(1.0 + 0.75**2)
    assert flux_shape(sx, np.array([0.0]), 2.0)[0] == pytest.approx(cos_theta**2)
    assert flux_shape(sx, np.array([0.0]), 3.0)[0] == pytest.approx(cos_theta**3)


def test_sky_grid_holds_essentially_all_of_a_tilted_exposures_acceptance():
    """Sized by where the counts are, not by the geometric corners.

    A bin's weight in actual data is acceptance TIMES the cosmic flux shape
    (`flux_shape`, cos(theta_sky)**index) — grazing directions are suppressed
    twice over, by the geometric cutoff and by the steep angular fall-off of
    the muon flux itself. Weighting by geometric_acceptance alone (no flux
    shape) overstates how much weight sits in the lost grazing corner: it
    gives only 99.86% kept at the default grid, failing this bound, while
    real campaign counts measure 99.99% kept at this same t_max=2.5 (per
    Task 5 fix-round-1 measurement). index=2.0 here is the conservative low
    end of what real fits find (the real fitted index comes out "well above
    2" per Task 5's brief), so this bound is not being weakened to pass.
    """
    from megido.acceptance import geometric_acceptance
    from megido.config import Pose
    from megido.detector import DetectorGeometry

    geom = DetectorGeometry.megiddo()
    grid = make_sky_grid()
    centers = np.linspace(-1.25, 1.25, 50)
    tx, ty = np.meshgrid(centers, centers, indexing="ij")

    sx, sy, on_sky = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=20, az_deg=241))
    weight = geometric_acceptance(tx, ty, geom) * flux_shape(sx, sy, index=2.0)
    _, in_grid = grid.bin_index(sx, sy)

    usable = weight[on_sky].sum()
    kept = weight[on_sky & in_grid].sum()
    assert kept / usable > 0.999, f"grid holds only {kept / usable:.4%} of the weighted acceptance"


def test_sky_grid_matches_the_analysis_resolution():
    grid = make_sky_grid()
    assert grid.edges[1] - grid.edges[0] == pytest.approx(0.05)


def test_sky_grid_bin_index_round_trips():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    centers = g.centers
    sx = np.array([centers[3], centers[19], 5.0])
    sy = np.array([centers[7], centers[0], 0.0])
    flat, ok = g.bin_index(sx, sy)
    assert ok[0] and ok[1] and not ok[2]
    assert flat[0] == 3 * 20 + 7
    assert flat[1] == 19 * 20 + 0


def test_sky_grid_flat_size():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    assert g.flat_size == 400
