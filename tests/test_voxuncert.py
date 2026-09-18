import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.config import load_site_config
from megido.voxuncert import BootstrapResult, systematic_map, voxel_bootstrap

CONFIG = """
site: t
data_dir: /tmp
binning: {t_max: 1.25, n_bins: 20}
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
reconstruction: {algorithm: tv, n_iter: 20, tv_alpha: 0.01}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 0}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _grid(seed=0):
    """A small analysis grid with plausible counts for all three exposures."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1.25, 1.25, 21)
    tx = 0.5 * (edges[:-1] + edges[1:])
    shape = np.exp(-(tx[:, None] ** 2 + tx[None, :] ** 2))
    counts = {eid: rng.poisson(2000 * shape).astype(np.int64)
              for eid in ("P0", "T20", "P1")}
    return AnalysisGrid(edges=edges, counts=counts)


def test_bootstrap_returns_a_mean_and_sigma_on_the_solve_grid(tmp_path):
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=3,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    assert isinstance(r, BootstrapResult)
    assert r.mean.shape == r.grid.shape
    assert r.sigma.shape == r.grid.shape
    assert r.n_replicas == 3
    assert np.all(r.sigma >= 0.0)


def test_bootstrap_pins_the_nominal_grid(tmp_path):
    """Every replica must share ONE lattice, fixed from the nominal counts.

    solve_voxels otherwise auto-derives its grid from t_reach(), which shifts
    with which sky bins survive Poisson resampling; a replica landing on a
    different shape would break np.stack and desync downstream SNR/views.
    """
    from megido.baseline import solve_baseline
    from megido.reconstruct import solve_voxels

    cfg = _cfg(tmp_path)
    g = _grid()
    nominal_sol = solve_baseline(g, cfg, n_iter=30)
    nominal_grid = solve_voxels(nominal_sol, cfg, cache_dir=None,
                                holdouts=False)["full"].grid

    r = voxel_bootstrap(g, cfg, n_replicas=3, cache_dir=None,
                        solve_kwargs={"n_iter": 30})

    assert r.grid.origin == nominal_grid.origin
    assert r.grid.spacing == nominal_grid.spacing
    assert r.grid.shape == nominal_grid.shape
    assert r.mean.shape == nominal_grid.shape
    assert r.sigma.shape == nominal_grid.shape


def test_bootstrap_sigma_is_not_identically_zero(tmp_path):
    """Zero sigma everywhere means the replicas were not actually resampled."""
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=4,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    assert r.sigma.max() > 0.0


def test_bootstrap_is_reproducible_for_a_fixed_seed(tmp_path):
    kw = dict(n_replicas=3, cache_dir=None, solve_kwargs={"n_iter": 30})
    a = voxel_bootstrap(_grid(), _cfg(tmp_path), seed=5, **kw)
    b = voxel_bootstrap(_grid(), _cfg(tmp_path), seed=5, **kw)
    np.testing.assert_allclose(a.sigma, b.sigma)


def test_snr_is_nan_where_sigma_is_zero_not_infinite(tmp_path):
    r = BootstrapResult(mean=np.ones((2, 2, 2)), sigma=np.zeros((2, 2, 2)),
                        grid=None, n_replicas=3)
    assert np.all(np.isnan(r.snr()))


def test_bootstrap_saves_and_reloads(tmp_path):
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=3,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    p = tmp_path / "u.npz"
    r.save(p)
    d = np.load(p)
    np.testing.assert_allclose(d["sigma"], r.sigma.astype(np.float32))
    assert int(d["n_replicas"]) == 3


def test_systematic_map_is_nonzero_when_the_gauge_choice_matters(tmp_path):
    """A zero systematic map here would not mean the gauge is harmless; it
    would mean the probe is additive and normalized_opacity cancelled it."""
    from megido.baseline import solve_baseline

    cfg = _cfg(tmp_path)
    sol = solve_baseline(_grid(), cfg, n_iter=40)
    sysmap = systematic_map(sol, cfg, cache_dir=None)
    assert np.isfinite(sysmap).all()
    assert np.abs(sysmap).max() > 0.0


def test_systematic_map_is_zero_when_both_quantiles_agree(tmp_path):
    from megido.baseline import solve_baseline

    cfg = _cfg(tmp_path)
    sol = solve_baseline(_grid(), cfg, n_iter=40)
    same = systematic_map(sol, cfg, base_quantile=0.05, alt_quantile=0.05,
                          cache_dir=None)
    np.testing.assert_allclose(same, 0.0, atol=1e-12)
