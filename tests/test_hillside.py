import numpy as np
from megido.hillside import ray_dirs, surface_points, grid_surface, fit_scale, fit_hillside, HillsideResult


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


def test_fit_scale_robust_to_outlier_opacity_pixels():
    # Same clean parallax setup, but ~5% of the near-vertical (low-tan) pixels
    # in each cloud are runaway opacity outliers (lam * 10) — a bad-channel /
    # saturation pattern a real S2 opacity image can carry, concentrated where
    # muon flux (and so sample count) is highest. Low-tan outliers stay near
    # (x,y)=0 regardless of the trial scale a, so they land inside the grid
    # and inside populated cells for every a the optimizer tries — they don't
    # get filtered out simply by falling off the edge of the grid the way a
    # high-tan outlier would.
    #
    # An unweighted, unclipped RMS objective (the pre-fix version) is pulled
    # far off by these: verified separately that it lands near a=0.99
    # (rel. err ~99%) on this exact injected data. The count-weighted,
    # L-clipped objective here should still recover a_true almost exactly,
    # since a single-ray outlier cell is outweighed by many-ray good cells
    # and the most extreme reconstructed lengths are dropped by the L_max
    # clip.
    rng = np.random.default_rng(1)
    a_true = 0.5
    xe = np.linspace(-14.0, 14.0, 57); ye = np.linspace(-14.0, 14.0, 57)
    t0, l0, _ = _forward(np.array([0.0, 0.0, 0.0]), a_true)
    t1, l1, _ = _forward(np.array([2.2, 0.0, 0.0]), a_true)

    def _inject_central_outliers(tan, lam, frac=0.05, mult=10.0):
        lam = lam.copy()
        r = np.linalg.norm(tan, axis=1)
        low_tan = np.argsort(r)[: 3 * int(frac * len(lam))]
        idx = rng.choice(low_tan, size=int(frac * len(lam)), replace=False)
        lam[idx] *= mult
        return lam

    l0_noisy = _inject_central_outliers(t0, l0)
    l1_noisy = _inject_central_outliers(t1, l1)
    images = {"pos0": {"tan": t0, "lam": l0_noisy, "p": np.array([0.0, 0, 0])},
              "pos1": {"tan": t1, "lam": l1_noisy, "p": np.array([2.2, 0, 0])}}
    out = fit_scale(images, xe, ye, a0=1.0)
    assert abs(out["a"] - a_true) / a_true < 0.20   # still recovered despite outliers
    assert out["n_overlap"] > 20


class _FakeSky:
    n_bins = 40
    def centers(self):
        g = np.linspace(-1.2, 1.2, 40)
        tx, ty = np.meshgrid(g, g)
        return np.column_stack([tx.ravel(), ty.ravel()])


class _FakeSol:
    """Minimal BaselineSolution stand-in emitting a known hill's opacity."""
    def __init__(self, a_true):
        self.sky = _FakeSky()
        self._a = a_true
        self._lam = {}
        for pid, p in (("pos0", [0, 0, 0]), ("pos1", [2.2, 0, 0])):
            _, lam, _ = _forward(np.array(p, float), a_true, n=40)
            self._lam[pid] = lam

    def normalized_opacity(self, pid, transparent_quantile=0.05):
        return self._lam[pid]


class _FakeCfg:
    class _E:
        def __init__(s, pid, x): s.position, s.pose = pid, type("P", (), {"x": x, "y": 0.0, "z": 0.0})
    exposures = [_E("pos0", 0.0), _E("pos1", 2.2)]


def test_fit_hillside_recovers_lateral_shape():
    a_true = 0.5
    res = fit_hillside(_FakeSol(a_true), _FakeCfg(), footprint_m=14.0, cell_m=0.5, n_boot=4)
    assert isinstance(res, HillsideResult)
    truth = _synthetic_hill(*np.meshgrid(
        0.5 * (res.xedges[:-1] + res.xedges[1:]),
        0.5 * (res.yedges[:-1] + res.yedges[1:]), indexing="ij"))
    m = np.isfinite(res.H)
    # lateral SHAPE recovered (correlation), the honest claim
    corr = np.corrcoef(res.H[m], truth[m])[0, 1]
    assert corr > 0.85
    # the fit reproduces its own data
    assert res.data_residual < 0.2
    # uncertainty is populated where the surface is
    assert np.isfinite(res.sigma[m]).all() and (res.sigma[m] >= 0).all()
    assert "height" in res.height_confidence.lower()
