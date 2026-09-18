import numpy as np
import pytest

from megido.config import Volume, load_site_config
from megido.fitdata import RowIndex
from megido.forward import ForwardModel, build_forward_model
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _rows():
    sx = np.array([0.0, 0.2, 0.0, -0.2])
    sy = np.array([0.0, 0.0, 0.1, 0.1])
    return RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.array([0, 0, 1, 1]),
                    sx=sx, sy=sy, sky_flat=np.array([0, 1, 2, 3]))


def test_build_sizes_the_grid_from_the_rows(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    assert fwd.A.shape == (4, fwd.grid.n_voxels)
    assert fwd.grid.extent(2) == pytest.approx((1.0, 3.0))
    assert fwd.n_rows == 4


def test_predict_of_a_uniform_volume_is_density_times_path_length(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x = np.full(fwd.grid.n_voxels, 0.3)
    np.testing.assert_allclose(fwd.predict(x), 0.3 * np.asarray(fwd.A.sum(axis=1)).ravel())


def test_predict_adds_the_per_position_offsets(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x = np.zeros(fwd.grid.n_voxels)
    got = fwd.predict(x, offsets={"pos0": 0.5, "pos1": -0.25})
    np.testing.assert_allclose(got, [0.5, 0.5, -0.25, -0.25])


def test_predict_accepts_a_3d_volume(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x3 = np.full(fwd.grid.shape, 0.2)
    np.testing.assert_allclose(fwd.predict(x3), fwd.predict(x3.ravel()))


def test_an_explicit_grid_overrides_the_automatic_one(tmp_path):
    g = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    fwd = build_forward_model(_rows(), _cfg(tmp_path), grid=g, cache_dir=None)
    assert fwd.grid is g


def test_to_sky_image_places_values_at_their_sky_bins(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    img = fwd.to_sky_image(np.array([1.0, 2.0, 3.0, 4.0]), "pos1", n_sky=2)
    assert img.shape == (2, 2)
    np.testing.assert_allclose(img.ravel(), [np.nan, np.nan, 3.0, 4.0])


def test_the_matrix_is_reused_from_cache_across_calls(tmp_path):
    cache = tmp_path / "cache"
    a = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=cache)
    b = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=cache)
    assert (a.A != b.A).nnz == 0
    assert len(list(cache.glob("A_*.npz"))) == 1
