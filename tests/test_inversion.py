import numpy as np
import pytest
from scipy import sparse

from megido.config import Reconstruction
from megido.fitdata import FitData, RowIndex
from megido.forward import ForwardModel
from megido.inversion import solve, sirt, sirt_tv
from megido.voxels import VoxelGrid


def _toy(n_rows=40, shape=(4, 4, 2), seed=0):
    """A small well-posed random system with a known nonnegative solution."""
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(shape))
    A = sparse.csr_matrix(rng.random((n_rows, n_vox)) * (rng.random((n_rows, n_vox)) < 0.5))
    truth = np.abs(rng.normal(size=n_vox))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.arange(n_rows) % 2,
                    sx=np.zeros(n_rows), sy=np.zeros(n_rows),
                    sky_flat=np.arange(n_rows))
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape),
                       rows=rows)
    lam = A @ truth
    return fwd, truth, FitData(lam=lam, w=np.ones(n_rows), rows=rows)


def test_sirt_recovers_a_noiseless_solution():
    fwd, truth, data = _toy()
    rc = Reconstruction(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x, info = sirt(fwd, data, rc)
    assert np.corrcoef(x, truth)[0, 1] > 0.95
    assert info["chi2_history"][-1] < info["chi2_history"][0]


def test_sirt_never_returns_a_negative_voxel():
    fwd, truth, data = _toy()
    rng = np.random.default_rng(1)
    noisy = FitData(lam=data.lam + rng.normal(scale=0.5, size=data.lam.size),
                    w=data.w, rows=data.rows)
    x, _ = sirt(fwd, noisy, Reconstruction(algorithm="sirt", n_iter=200,
                                           chi2_target=1e-12, nonneg=True))
    assert x.min() >= 0.0


def test_sirt_stops_early_at_the_discrepancy_target():
    fwd, truth, data = _toy()
    x, info = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=5000,
                                             chi2_target=1e-2))
    assert info["n_iter_used"] < 5000


def test_zero_weight_rows_do_not_influence_the_fit():
    fwd, truth, data = _toy()
    poisoned = FitData(lam=data.lam.copy(), w=data.w.copy(), rows=data.rows)
    poisoned.lam[:5] = 1e6
    kept = np.ones(data.rows.n_rows, dtype=bool)
    kept[:5] = False

    rc = Reconstruction(algorithm="sirt", n_iter=300, chi2_target=1e-12)
    a, _ = sirt(fwd, data.restricted(kept), rc)
    b, _ = sirt(fwd, poisoned.restricted(kept), rc)
    np.testing.assert_allclose(a, b)


def test_offsets_absorb_a_constant_shift_per_position():
    """lambda has a free additive constant per position: Phase 2's gauge is
    pinned per position, so an overall level is not measured.

    The comparison is between the shifted and unshifted fits, not against zero.
    The unshifted fit already carries nonzero offsets absorbing the mean of
    A @ truth, so the absolute offsets say nothing; only their DIFFERENCE should
    track the injected shift.
    """
    fwd, truth, data = _toy()
    shift = np.where(data.rows.pos_of_row == 0, 0.7, -0.4)
    shifted = FitData(lam=data.lam + shift, w=data.w, rows=data.rows)

    rc = Reconstruction(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x0, base = sirt(fwd, data, rc)
    x1, moved = sirt(fwd, shifted, rc)

    assert moved["offsets"]["pos0"] - base["offsets"]["pos0"] == pytest.approx(0.7, abs=0.15)
    assert moved["offsets"]["pos1"] - base["offsets"]["pos1"] == pytest.approx(-0.4, abs=0.15)
    assert np.corrcoef(x1, truth)[0, 1] > 0.9


def test_tv_denoises_a_piecewise_constant_volume_better_than_plain_sirt():
    rng = np.random.default_rng(3)
    shape = (8, 8, 4)
    n_vox = int(np.prod(shape))
    truth3 = np.zeros(shape)
    truth3[2:6, 2:6, 1:3] = 1.0
    truth = truth3.ravel()

    A = sparse.csr_matrix(rng.random((300, n_vox)) * (rng.random((300, n_vox)) < 0.3))
    rows = RowIndex(position_ids=("pos0",), pos_of_row=np.zeros(300, dtype=int),
                    sx=np.zeros(300), sy=np.zeros(300), sky_flat=np.arange(300))
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape),
                       rows=rows)
    lam = A @ truth + rng.normal(scale=0.4, size=300)
    data = FitData(lam=lam, w=np.ones(300), rows=rows)

    plain, _ = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=300,
                                              chi2_target=1e-12))
    tv, _ = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=300,
                                              tv_alpha=0.001, tv_z_weight=0.5))
    assert np.linalg.norm(tv - truth) < np.linalg.norm(plain - truth)


def test_tv_returns_the_best_iterate_not_the_last():
    fwd, truth, data = _toy()
    x, info = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=120,
                                                tv_alpha=0.01))
    assert info["best_chi2"] <= min(info["chi2_history"])


def test_tv_with_zero_alpha_tracks_plain_sirt():
    """With tv_alpha = 0 the proximal step is just the nonnegativity clip, so the
    two solvers follow the same trajectory.

    They do NOT end on the same array: chi2 is evaluated before each update, so
    sirt_tv's best iterate is at most the one after n_iter - 1 updates while sirt
    returns the one after n_iter. The gap is one sweep, by construction.
    """
    fwd, truth, data = _toy()
    a, _ = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=50,
                                          chi2_target=-1.0))
    b, _ = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=50,
                                             tv_alpha=0.0))
    assert np.corrcoef(a, b)[0, 1] > 0.999
    assert np.linalg.norm(b - a) < 0.05 * np.linalg.norm(a)


def test_solve_dispatches_on_the_algorithm_name():
    fwd, truth, data = _toy()
    x, _ = solve(fwd, data, Reconstruction(algorithm="tv", n_iter=20))
    assert x.shape == (fwd.grid.n_voxels,)
    with pytest.raises(ValueError, match="unknown algorithm"):
        solve(fwd, data, Reconstruction(algorithm="mlem", n_iter=5))


def test_an_all_zero_weight_fit_returns_zeros_rather_than_dividing_by_zero():
    fwd, truth, data = _toy()
    dead = data.restricted(np.zeros(data.rows.n_rows, dtype=bool))
    x, info = sirt(fwd, dead, Reconstruction(algorithm="sirt", n_iter=10,
                                             chi2_target=-1.0))
    assert np.all(x == 0.0)
    assert np.all(np.isfinite(list(info["offsets"].values())))
