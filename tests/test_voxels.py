import numpy as np
import pytest

from megido.config import Volume
from megido.voxels import VoxelGrid, auto_grid


def test_grid_reports_its_size_and_centers():
    g = VoxelGrid(origin=(0.0, -1.0, 2.0), spacing=0.5, shape=(4, 2, 3))
    assert g.n_voxels == 24
    np.testing.assert_allclose(g.axis_centers(0), [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(g.axis_centers(1), [-0.75, -0.25])
    assert g.extent(2) == (2.0, 3.5)


def test_grid_key_separates_different_grids():
    a = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.5, shape=(2, 2, 2))
    b = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.25, shape=(2, 2, 2))
    assert a.key() != b.key()


def test_auto_grid_covers_every_ray_footprint():
    """A ray of tangent t leaving x=x0 reaches x0 + t*z at height z, and the
    bundle adds half an aperture. The grid must contain the union over poses."""
    vol = Volume(z_min_m=1.0, z_max_m=5.0, spacing_m=0.5)
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (2.2, 0.0, 0.0)}
    g = auto_grid(vol, origins, t_reach=1.0)

    x0, x1 = g.extent(0)
    assert x0 <= -5.0 - 0.192 + 1e-9     # pos0, t = -1 at z = 5, minus half aperture
    assert x1 >= 2.2 + 5.0 + 0.192 - 1e-9
    assert g.extent(2) == pytest.approx((1.0, 5.0), abs=0.5)


def test_auto_grid_honours_an_explicit_xy_box():
    vol = Volume(z_min_m=1.0, z_max_m=3.0, spacing_m=0.5,
                 xy_m=((-2.0, 2.0), (-1.0, 1.0)))
    g = auto_grid(vol, {"pos0": (0.0, 0.0, 0.0)}, t_reach=10.0)
    assert g.origin[0] == pytest.approx(-2.0)
    assert g.shape[:2] == (8, 4)


def test_auto_grid_rejects_an_inverted_z_range():
    vol = Volume(z_min_m=5.0, z_max_m=1.0, spacing_m=0.5)
    with pytest.raises(ValueError, match="z_max_m"):
        auto_grid(vol, {"pos0": (0.0, 0.0, 0.0)}, t_reach=1.0)


def test_aperture_pad_matches_the_detector():
    from megido.detector import DetectorGeometry
    from megido.voxels import _APERTURE_PAD_M

    assert _APERTURE_PAD_M == pytest.approx(DetectorGeometry.megiddo().aperture_m)
