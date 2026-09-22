import numpy as np
from megido.hillside import ray_dirs, surface_points, grid_surface


def test_ray_dirs_unit_and_upward():
    d = ray_dirs(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -0.5]]))
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[0], [0, 0, 1])          # zenith
    assert (d[:, 2] > 0).all()                    # all upward


def test_surface_points_place_along_ray():
    p = np.array([2.2, 0.0, 0.0])
    dirs = ray_dirs(np.array([[0.0, 0.0]]))       # straight up
    L = np.array([5.0])
    s = surface_points(p, L, dirs)
    assert np.allclose(s[0], [2.2, 0.0, 5.0])     # 5 m straight up from P1


def test_grid_surface_bins_mean_z():
    pts = np.array([[0.1, 0.1, 2.0], [0.15, 0.12, 4.0], [0.9, 0.9, 9.0]])
    xe = np.array([0.0, 0.5, 1.0]); ye = np.array([0.0, 0.5, 1.0])
    H, cnt = grid_surface(pts, xe, ye)
    assert cnt[0, 0] == 2 and cnt[1, 1] == 1
    assert np.isclose(H[0, 0], 3.0)               # mean of 2.0, 4.0
    assert np.isclose(H[1, 1], 9.0)
    assert np.isnan(H[0, 1])                        # empty cell -> NaN
