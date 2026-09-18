import numpy as np
import pytest

from megido.backproject import backproject_plane, plane_axes
from megido.config import load_site_config
from megido.fitdata import FitData, RowIndex

CONFIG = """
site: t
data_dir: /tmp
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 4.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _data(sx, sy, lam, pos):
    sx, sy, lam = map(np.asarray, (sx, sy, lam))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.asarray(pos), sx=sx, sy=sy,
                    sky_flat=np.arange(sx.size))
    return FitData(lam=lam.astype(float), w=np.ones(sx.size), rows=rows)


def test_a_vertical_ray_lands_directly_above_its_detector(tmp_path):
    data = _data([0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [0, 1])
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ys = np.array([-1.0, 0.0, 1.0])
    per, _ = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)

    i0 = int(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)) // ys.size)
    i1 = int(np.nanargmax(np.nan_to_num(per["pos1"], nan=-np.inf)) // ys.size)
    assert xs[i0] == pytest.approx(0.0)
    assert xs[i1] == pytest.approx(4.0)


def test_a_tilted_ray_lands_at_the_lever_arm_offset(tmp_path):
    """A ray of tangent 0.4 from z = 0 reaches x = 0.4 * 5 = 2 m at z = 5."""
    data = _data([0.4], [0.0], [3.0], [0])
    xs = np.linspace(-1.0, 5.0, 13)
    ys = np.array([-0.5, 0.0, 0.5])
    per, _ = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    hit = np.unravel_index(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)),
                           per["pos0"].shape)
    assert xs[hit[0]] == pytest.approx(2.0, abs=0.3)


def test_uncovered_pixels_are_nan_not_zero(tmp_path):
    data = _data([0.0], [0.0], [1.0], [0])
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    per, mean = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    assert np.isnan(per["pos1"]).all()
    assert np.isnan(mean).any()


def test_the_mean_ignores_positions_with_no_coverage(tmp_path):
    data = _data([0.0, 0.05, -0.05], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0, 0, 0])
    xs = np.linspace(-1.0, 1.0, 9)
    ys = np.linspace(-1.0, 1.0, 9)
    _, mean = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    covered = mean[np.isfinite(mean)]
    assert covered.size > 0
    np.testing.assert_allclose(covered, 2.0, atol=1e-6)


def test_plane_axes_spans_every_footprint(tmp_path):
    xs, ys = plane_axes(_cfg(tmp_path), z_m=5.0, t_reach=1.0, res_m=0.5)
    assert xs[0] <= -5.0 and xs[-1] >= 9.0
    assert np.allclose(np.diff(xs), 0.5)
