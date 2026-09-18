import numpy as np
import pytest

from megido.baseline import BaselineSolution
from megido.basis import make_smooth_basis
from megido.angular import AnalysisGrid
from megido.config import load_site_config
from megido.fitdata import RowIndex, build_fit_data, position_origins
from megido.sky import SkyGrid


CONFIG = """
site: t
data_dir: /tmp
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 0}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _solution(opacity: dict[str, np.ndarray], n_bins: int) -> BaselineSolution:
    sky = SkyGrid(edges=np.linspace(-2.5, 2.5, n_bins + 1))
    return BaselineSolution(
        coeffs=np.zeros(1), opacity=opacity, norms={}, flux_index=2.0,
        nll_history=[0.0], grid=AnalysisGrid(edges=np.linspace(-1, 1, 3), counts={}),
        sky=sky, basis=make_smooth_basis(t_max=1.25, n_per_axis=2),
    )


def test_positions_share_an_origin_when_they_share_a_translation(tmp_path):
    origins = position_origins(_cfg(tmp_path))
    assert origins == {"pos0": (0.0, 0.0, 0.0), "pos1": (2.2, 0.0, 0.0)}


def test_only_finite_opacity_becomes_a_row(tmp_path):
    n = 4
    lam0 = np.full(n * n, np.nan)
    lam0[[0, 5, 10]] = [0.5, 1.0, 1.5]
    lam1 = np.full(n * n, np.nan)
    lam1[[5]] = [2.0]
    sol = _solution({"pos0": lam0, "pos1": lam1}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    assert data.rows.n_rows == 4
    assert data.rows.position_ids == ("pos0", "pos1")
    assert data.rows.mask_for("pos0").sum() == 3
    assert data.rows.mask_for("pos1").sum() == 1
    assert np.all(np.isfinite(data.lam))


def test_rows_carry_the_physical_opacity_not_the_gauge_pinned_map(tmp_path):
    """normalized_opacity shifts the 5th percentile to zero and clips; the raw
    map is median-pinned and half negative, so it must never reach the rows."""
    n = 4
    lam = np.full(n * n, np.nan)
    lam[:8] = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    assert data.lam.min() >= 0.0
    np.testing.assert_allclose(data.lam.max(), sol.normalized_opacity("pos0")[7])


def test_the_transparent_quantile_changes_the_gauge(tmp_path):
    """The gauge convention is a choice, and voxuncert varies it to price that
    choice. Adding a constant to the opacity map would NOT change the answer —
    normalized_opacity subtracts its own quantile — so the knob has to be the
    quantile itself."""
    n = 4
    lam = np.full(n * n, np.nan)
    lam[:8] = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    sol = _solution({"pos0": lam}, n)

    a = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0, transparent_quantile=0.05)
    b = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0, transparent_quantile=0.25)
    assert not np.allclose(a.lam, b.lam)

    shifted = _solution({"pos0": lam + 7.0}, n)
    c = build_fit_data(shifted, _cfg(tmp_path), max_tan=10.0)
    np.testing.assert_allclose(a.lam, c.lam)   # shift-invariant, as documented


def test_rows_outside_max_tan_are_dropped(tmp_path):
    """The sky grid runs to |t| = 2.5 to catch stray counts. Directions past the
    detector's own acceptance carry almost no tracks and would stretch the voxel
    grid enormously, so the caller caps them."""
    n = 10
    sol = _solution({"pos0": np.zeros(n * n)}, n)

    wide = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    narrow = build_fit_data(sol, _cfg(tmp_path), max_tan=0.5)
    assert narrow.rows.n_rows < wide.rows.n_rows
    assert np.abs(narrow.rows.sx).max() <= 0.5
    assert np.abs(narrow.rows.sy).max() <= 0.5


def test_weights_come_from_sigma_when_given(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1]] = [1.0, 2.0]
    sig = np.full(n * n, np.nan)
    sig[[0, 1]] = [0.5, 0.25]
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), sigma={"pos0": sig}, max_tan=10.0)
    np.testing.assert_allclose(np.sort(data.w), np.sort([1 / 0.25, 1 / 0.0625]))


def test_a_row_with_no_usable_sigma_is_dropped_not_given_infinite_weight(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1]] = [1.0, 2.0]
    sig = np.full(n * n, np.nan)
    sig[[0, 1]] = [0.5, 0.0]        # zero sigma would be an infinite weight
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), sigma={"pos0": sig}, max_tan=10.0)
    assert data.rows.n_rows == 1
    assert np.all(np.isfinite(data.w))


def test_uniform_weights_when_no_sigma_is_supplied(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1, 2]] = [1.0, 2.0, 3.0]
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    np.testing.assert_allclose(data.w, 1.0)


def test_restricted_zeroes_weights_and_keeps_the_row_layout(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1, 2]] = [1.0, 2.0, 3.0]
    sol = _solution({"pos0": lam}, n)
    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)

    keep = np.array([True, False, True])
    r = data.restricted(keep)
    assert r.rows is data.rows
    np.testing.assert_allclose(r.lam, data.lam)
    np.testing.assert_allclose(r.w, [1.0, 0.0, 1.0])


def test_t_reach_is_the_largest_live_tangent(tmp_path):
    n = 10
    sol = _solution({"pos0": np.zeros(n * n)}, n)
    data = build_fit_data(sol, _cfg(tmp_path), max_tan=0.8)
    assert data.rows.t_reach() == pytest.approx(
        max(np.abs(data.rows.sx).max(), np.abs(data.rows.sy).max()))


def test_row_index_key_changes_with_the_rows(tmp_path):
    # n=6 over [-2.5, 2.5] puts bin centers at +/-0.4167, +/-1.25, +/-2.083 --
    # nothing between 0 and 0.4167 -- so max_tan must cross a tier boundary to
    # change the row set; 1.3 pulls in the +/-1.25 tier that 1.0 excludes.
    n = 6
    a = build_fit_data(_solution({"pos0": np.zeros(n * n)}, n), _cfg(tmp_path), max_tan=1.0)
    b = build_fit_data(_solution({"pos0": np.zeros(n * n)}, n), _cfg(tmp_path), max_tan=1.3)
    assert a.rows.key() != b.rows.key()


def test_an_empty_solution_is_an_error_not_a_zero_row_matrix(tmp_path):
    n = 4
    sol = _solution({"pos0": np.full(n * n, np.nan)}, n)
    with pytest.raises(ValueError, match="no constrained"):
        build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
