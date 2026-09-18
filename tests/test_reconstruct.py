import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.reconstruct import VoxelSolution, solve_voxels
from megido.sky import SkyGrid
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
reconstruction: {algorithm: tv, n_iter: 30, tv_alpha: 0.01}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""

N_SKY = 12


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _solution(seed=0):
    rng = np.random.default_rng(seed)
    sky = SkyGrid(edges=np.linspace(-2.5, 2.5, N_SKY + 1))
    opacity = {}
    for pid in ("pos0", "pos1"):
        lam = np.full(N_SKY * N_SKY, np.nan)
        live = rng.random(N_SKY * N_SKY) < 0.7
        lam[live] = rng.uniform(0.0, 2.0, int(live.sum()))
        opacity[pid] = lam
    return BaselineSolution(
        coeffs=np.zeros(1), opacity=opacity, norms={}, flux_index=2.0,
        nll_history=[0.0], grid=AnalysisGrid(edges=np.linspace(-1, 1, 3), counts={}),
        sky=sky, basis=make_smooth_basis(t_max=1.25, n_per_axis=2))


def test_solve_produces_a_full_fit_and_one_holdout_per_position(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert set(fits) == {"full", "holdout_pos0", "holdout_pos1"}


def test_the_volume_is_nonnegative_and_correctly_shaped(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    v = fits["full"]
    assert v.rho.shape == (v.grid.n_voxels,)
    assert v.rho3().shape == v.grid.shape
    assert v.rho.min() >= 0.0


def test_a_holdout_fit_uses_only_its_own_position(tmp_path):
    """holdout_pos0 is TRAINED ON pos0 alone — it is the single-view fit whose
    disagreement with the other view is the cross-validation signal."""
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert fits["holdout_pos0"].info["n_rows_used"] < fits["full"].info["n_rows_used"]
    assert (fits["holdout_pos0"].info["n_rows_used"]
            + fits["holdout_pos1"].info["n_rows_used"]
            == fits["full"].info["n_rows_used"])


def test_holdouts_can_be_skipped(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None, holdouts=False)
    assert set(fits) == {"full"}


def test_offsets_are_recorded_per_position(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert set(fits["full"].offsets) == {"pos0", "pos1"}


def test_save_and_load_round_trip(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    p = tmp_path / "out" / "volume_full.npz"
    fits["full"].save(p)

    back = VoxelSolution.load(p)
    np.testing.assert_allclose(back.rho, fits["full"].rho)
    assert back.grid == fits["full"].grid
    assert back.offsets == pytest.approx(fits["full"].offsets)
    assert back.position_ids == fits["full"].position_ids


def test_the_saved_volume_records_the_inversion_version(tmp_path):
    from megido.raycast import INVERSION_VERSION

    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    p = tmp_path / "v.npz"
    fits["full"].save(p)
    assert VoxelSolution.load(p).version == INVERSION_VERSION


def test_sigma_weights_are_threaded_through(tmp_path):
    """A position whose sigma is enormous should barely move the fit."""
    sol = _solution()
    big = {"pos0": np.full(N_SKY * N_SKY, 1e6),
           "pos1": np.full(N_SKY * N_SKY, 0.1)}
    weighted = solve_voxels(sol, _cfg(tmp_path), sigma=big, cache_dir=None,
                            holdouts=False)["full"]
    only1 = solve_voxels(sol, _cfg(tmp_path), cache_dir=None,
                         holdouts=False)["full"]
    assert not np.allclose(weighted.rho, only1.rho)


def test_the_volume_reproduces_its_own_measurements_better_than_a_zero_volume(tmp_path):
    """The weakest honest fidelity claim, and the one that catches a matrix
    whose rays point the wrong way: the fit must beat doing nothing."""
    from megido.fitdata import build_fit_data
    from megido.forward import build_forward_model

    sol, cfg = _solution(), _cfg(tmp_path)
    fits = solve_voxels(sol, cfg, cache_dir=None, holdouts=False)
    data = build_fit_data(sol, cfg)
    fwd = build_forward_model(data.rows, cfg, cache_dir=None)

    v = fits["full"]
    pred = fwd.predict(v.rho, v.offsets)
    flat = np.mean((data.lam - np.mean(data.lam)) ** 2)
    assert np.mean((data.lam - pred) ** 2) < flat
