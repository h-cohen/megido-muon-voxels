import numpy as np
from megido.hillside import ray_dirs, surface_points, grid_surface, fit_scale


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


def _synthetic_hill(tx, ty):
    # a smooth bump surface H(x,y), metres
    return 8.0 * np.exp(-((tx) ** 2 + (ty) ** 2) / (2 * 6.0 ** 2))


def _forward(p, a_true, xy_extent=12.0, n=40):
    # emit sky pixels that hit a known hill; return (tan, lam) with lam=L/a_true
    g = np.linspace(-1.2, 1.2, n)
    tx, ty = np.meshgrid(g, g); tan = np.column_stack([tx.ravel(), ty.ravel()])
    d = ray_dirs(tan)
    # solve for L so that (p + L d)_z == H((p+Ld)_xy): damped fixed-point iterate.
    # Undamped iteration diverges (limit-cycles) for grazing rays near the
    # steepest part of the hill flank, where |dH/dL| > 1; a 0.4 relaxation
    # factor keeps the map contractive everywhere on this grid.
    L = np.full(len(tan), 5.0)
    for _ in range(200):
        s = p[None, :] + L[:, None] * d
        Ht = _synthetic_hill(s[:, 0], s[:, 1])
        L_step = np.clip(Ht / d[:, 2], 0.1, 40.0)
        L = 0.6 * L + 0.4 * L_step
    lam = L / a_true          # opacity = L/a_true (a_true = 1/rho)
    return tan, lam, L


def test_fit_scale_recovers_density_from_parallax():
    a_true = 0.5
    xe = np.linspace(-14.0, 14.0, 57); ye = np.linspace(-14.0, 14.0, 57)
    t0, l0, _ = _forward(np.array([0.0, 0.0, 0.0]), a_true)
    t1, l1, _ = _forward(np.array([2.2, 0.0, 0.0]), a_true)
    images = {"pos0": {"tan": t0, "lam": l0, "p": np.array([0.0, 0, 0])},
              "pos1": {"tan": t1, "lam": l1, "p": np.array([2.2, 0, 0])}}
    out = fit_scale(images, xe, ye, a0=1.0)
    assert abs(out["a"] - a_true) / a_true < 0.15   # scale recovered within 15%
    assert out["n_overlap"] > 20
