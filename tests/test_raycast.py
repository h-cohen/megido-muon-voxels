import numpy as np
import pytest
from scipy import sparse

from megido.fitdata import RowIndex
from megido.raycast import bundle_offsets, build_system_matrix
from megido.voxels import VoxelGrid


def _rows(sx, sy, pos=None) -> RowIndex:
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    pos = np.zeros(sx.size, dtype=np.int64) if pos is None else np.asarray(pos)
    return RowIndex(position_ids=("pos0", "pos1")[: pos.max() + 1],
                    pos_of_row=pos, sx=sx, sy=sy,
                    sky_flat=np.arange(sx.size, dtype=np.int64))


ORIGINS = {"pos0": (0.0, 0.0, 0.0), "pos1": (2.0, 0.0, 0.0)}


def test_bundle_offsets_are_perpendicular_to_their_ray():
    d = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.3, -0.4, np.sqrt(1 - 0.25)]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    offs = bundle_offsets(d, aperture_m=0.384, n_sub=3)

    assert offs.shape == (3, 9, 3)
    np.testing.assert_allclose(np.einsum("rsk,rk->rs", offs, d), 0.0, atol=1e-12)


def test_bundle_offsets_span_the_aperture_and_are_centred():
    d = np.array([[0.0, 0.0, 1.0]])
    offs = bundle_offsets(d, aperture_m=0.4, n_sub=4)
    np.testing.assert_allclose(offs.mean(axis=1), 0.0, atol=1e-12)
    # 4 sub-rays at (+-3/8, +-1/8) * 0.4 -> extreme coordinate 0.15
    assert np.abs(offs).max() == pytest.approx(0.15)


def test_a_single_vertical_ray_has_path_length_equal_to_the_slab():
    """A vertical pinhole ray through a 2 m slab must deposit exactly 2 m."""
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.shape == (1, grid.n_voxels)
    assert A.sum() == pytest.approx(2.0, rel=1e-6)


def test_a_tilted_ray_is_longer_by_one_over_cos_theta():
    grid = VoxelGrid(origin=(-8.0, -8.0, 1.0), spacing=0.5, shape=(32, 32, 4))
    A = build_system_matrix(_rows([1.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.sum() == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-3)


def test_the_vertical_ray_lands_in_the_column_above_its_origin():
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1).toarray().reshape(grid.shape)
    hit = np.nonzero(A.sum(axis=2))
    assert (int(hit[0][0]), int(hit[1][0])) == (2, 2)   # x, y just above 0.0


def test_rows_start_at_their_own_position():
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(12, 4, 4))
    rows = _rows([0.0, 0.0], [0.0, 0.0], pos=[0, 1])
    A = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.0, n_sub=1)
    dense = A.toarray().reshape(2, *grid.shape)
    assert int(np.nonzero(dense[0].sum(axis=(1, 2)))[0][0]) == 2    # x = 0.0
    assert int(np.nonzero(dense[1].sum(axis=(1, 2)))[0][0]) == 6    # x = 2.0


def test_a_wide_bundle_spreads_across_more_voxels_than_a_pinhole():
    grid = VoxelGrid(origin=(-2.0, -2.0, 1.0), spacing=0.1, shape=(40, 40, 10))
    pin = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                              aperture_m=0.0, n_sub=1)
    wide = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                               aperture_m=0.8, n_sub=4)
    assert wide.nnz > 4 * pin.nnz
    # total path length is conserved: the bundle averages, it does not multiply
    assert wide.sum() == pytest.approx(pin.sum(), rel=1e-6)


def test_rays_that_miss_the_grid_produce_an_empty_row_not_an_error():
    grid = VoxelGrid(origin=(10.0, 10.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.nnz == 0


def test_the_matrix_is_cached_on_disk_and_reused(tmp_path):
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0, 0.2], [0.0, -0.1])
    first = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2,
                                cache_dir=tmp_path)
    cached = sorted(tmp_path.glob("A_*.npz"))
    assert len(cached) == 1

    second = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2,
                                 cache_dir=tmp_path)
    assert (first != second).nnz == 0
    assert sorted(tmp_path.glob("A_*.npz")) == cached


def test_the_cache_key_separates_different_geometry(tmp_path):
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0], [0.0])
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.2, n_sub=2, cache_dir=tmp_path)
    build_system_matrix(_rows([0.5], [0.0]), ORIGINS, grid, aperture_m=0.1, n_sub=2,
                        cache_dir=tmp_path)
    assert len(sorted(tmp_path.glob("A_*.npz"))) == 3


def test_the_cache_key_includes_the_inversion_version(tmp_path, monkeypatch):
    """A change to ray-casting logic must not serve a stale matrix — the Phase 1
    cache defect, which shipped pre-dither artifacts after the fix landed."""
    import megido.raycast as R

    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0], [0.0])
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    monkeypatch.setattr(R, "INVERSION_VERSION", R.INVERSION_VERSION + 1)
    R.build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    assert len(sorted(tmp_path.glob("A_*.npz"))) == 2


def test_matrix_is_csr_and_finite():
    grid = VoxelGrid(origin=(-2.0, -2.0, 1.0), spacing=0.25, shape=(16, 16, 8))
    rows = _rows(np.linspace(-0.5, 0.5, 7), np.zeros(7))
    A = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.384, n_sub=3)
    assert sparse.isspmatrix_csr(A)
    assert np.all(np.isfinite(A.data))
    assert A.data.min() > 0.0
