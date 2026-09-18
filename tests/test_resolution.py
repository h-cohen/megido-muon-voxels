import numpy as np
import pytest
from scipy import sparse

from megido.config import load_site_config
from megido.fitdata import RowIndex
from megido.forward import ForwardModel
from megido.resolution import (alias_period, campaign_resolution,
                               depth_resolution, format_resolution,
                               position_baselines, views_per_voxel)
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 2.0, z_max_m: 10.0, spacing_m: 0.25}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def test_depth_resolution_grows_with_the_square_of_distance():
    a = depth_resolution(5.0, baseline_m=2.2, sigma_t=0.05)
    b = depth_resolution(10.0, baseline_m=2.2, sigma_t=0.05)
    assert b == pytest.approx(4 * a, rel=1e-9)


def test_depth_resolution_improves_with_a_longer_baseline():
    assert (depth_resolution(8.0, baseline_m=5.0, sigma_t=0.05)
            < depth_resolution(8.0, baseline_m=2.2, sigma_t=0.05))


def test_depth_resolution_matches_the_closed_form():
    """Two rays converging at z from detectors b apart: Delta_t = b/z, so
    dz = z^2/b * d(Delta_t), and the two views' angular errors add in quadrature."""
    z, b, s = 7.0, 2.2, 0.05
    assert depth_resolution(z, b, s) == pytest.approx(np.sqrt(2) * s * z**2 / b)


def test_a_zero_baseline_has_no_depth_resolution():
    assert depth_resolution(7.0, baseline_m=0.0, sigma_t=0.05) == float("inf")


def test_alias_period_is_the_spec_formula():
    assert alias_period(7.0, baseline_m=2.2, feature_pitch_m=1.0) == pytest.approx(
        1.0 * 7.0 / 2.2)


def test_baselines_are_measured_between_positions_not_exposures(tmp_path):
    """P0 and T20 share a spot: two tilts, one position, zero baseline."""
    b = position_baselines(_cfg(tmp_path))
    assert b == {("pos0", "pos1"): pytest.approx(2.2)}


def test_views_per_voxel_counts_distinct_positions():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 1, 1))
    # row 0 (pos0) hits voxel 0; rows 1 and 2 (pos1) both hit voxel 0; row 3 hits voxel 1
    A = sparse.csr_matrix(np.array([[1.0, 0.0],
                                    [2.0, 0.0],
                                    [0.5, 0.0],
                                    [0.0, 1.0]]))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.array([0, 1, 1, 1]),
                    sx=np.zeros(4), sy=np.zeros(4), sky_flat=np.arange(4))
    v = views_per_voxel(ForwardModel(A=A, grid=grid, rows=rows))
    assert v.shape == grid.shape
    assert v.ravel().tolist() == [2, 1]


def test_campaign_resolution_reports_the_megiddo_numbers(tmp_path):
    r = campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == pytest.approx(2.2)
    assert r["n_positions"] == 2
    assert set(r["depth_resolution_m"]) == {"z_min", "z_mid", "z_max"}
    # 10 m away on a 2.2 m baseline is hopeless and the report must say so
    assert r["depth_resolution_m"]["z_max"] > 2.0
    assert r["z_range_m"] == pytest.approx((2.0, 10.0))


def test_campaign_resolution_flags_when_depth_is_unresolved(tmp_path):
    r = campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["depth_resolved"] is False
    assert "depth" in r["verdict"].lower()


def test_a_single_position_campaign_has_no_depth_information(tmp_path):
    p = tmp_path / "one.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {z_min_m: 2.0, z_max_m: 10.0, spacing_m: 0.25}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n")
    r = campaign_resolution(load_site_config(p), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == 0.0
    assert r["depth_resolution_m"]["z_mid"] == float("inf")
    assert r["depth_resolved"] is False


def test_the_report_formats_without_raising(tmp_path):
    text = format_resolution(
        campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0))
    assert "baseline" in text.lower()
    assert "2.2" in text
