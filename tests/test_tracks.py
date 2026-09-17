import numpy as np
import pytest

from megido.detector import DetectorGeometry
from megido.hits import REJECT_NO_HIT, REJECT_OK, LayerHits
from megido.tracks import Tracks, fit_tracks


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def _hits(positions, rejects=None):
    """positions: [n, 4] in ASIC order 0..3."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    rej = np.full((n, 4), REJECT_OK, dtype=np.uint8) if rejects is None else np.asarray(rejects, np.uint8)
    return LayerHits(position_cm=pos, bar_lo=np.zeros((n, 4), np.int16),
                     charge_lo=np.ones((n, 4)), charge_hi=np.zeros((n, 4)), reject=rej)


def test_vertical_track_has_zero_slope(geom):
    # ASIC order 0,1,2,3 -> coords y,y,x,x at z 6.2, 37.7, 31.5, 0.0
    hits = _hits([[19.2, 19.2, 19.2, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.valid[0]
    assert t.tan_x[0] == pytest.approx(0.0)
    assert t.tan_y[0] == pytest.approx(0.0)
    assert t.theta_deg()[0] == pytest.approx(0.0)


def test_known_x_slope_is_recovered(geom):
    """X is ASIC 3 (z=0) then ASIC 2 (z=31.5); dz = 31.5."""
    dx = 6.3
    hits = _hits([[19.2, 19.2, 19.2 + dx, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_x[0] == pytest.approx(dx / 31.5)
    assert t.tan_y[0] == pytest.approx(0.0)


def test_known_y_slope_is_recovered(geom):
    """Y is ASIC 0 (z=6.2) then ASIC 1 (z=37.7); dz = 31.5."""
    dy = -3.15
    hits = _hits([[19.2, 19.2 + dy, 19.2, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_y[0] == pytest.approx(dy / 31.5)
    assert t.tan_x[0] == pytest.approx(0.0)


def test_bottom_positions_come_from_the_lower_layers(geom):
    hits = _hits([[11.0, 19.2, 19.2, 5.0]])
    t = fit_tracks(hits, geom)
    assert t.x_bot_cm[0] == pytest.approx(5.0)    # ASIC 3, z=0
    assert t.y_bot_cm[0] == pytest.approx(11.0)   # ASIC 0, z=6.2


def test_event_with_a_rejected_layer_is_invalid(geom):
    rej = np.full((1, 4), REJECT_OK, np.uint8)
    rej[0, 2] = REJECT_NO_HIT
    hits = _hits([[19.2, 19.2, np.nan, 19.2]], rejects=rej)
    t = fit_tracks(hits, geom)
    assert not t.valid[0]
    assert np.isnan(t.tan_x[0])


def test_theta_and_phi_for_a_diagonal_track(geom):
    hits = _hits([[19.2, 19.2 + 31.5, 19.2 + 31.5, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_x[0] == pytest.approx(1.0)
    assert t.tan_y[0] == pytest.approx(1.0)
    assert t.theta_deg()[0] == pytest.approx(np.degrees(np.arctan(np.sqrt(2.0))))
    assert t.phi_deg()[0] == pytest.approx(45.0)


def test_n_valid_counts_only_complete_events(geom):
    rej = np.full((3, 4), REJECT_OK, np.uint8)
    rej[1, 0] = REJECT_NO_HIT
    hits = _hits([[19.2] * 4, [19.2] * 4, [19.2] * 4], rejects=rej)
    t = fit_tracks(hits, geom)
    assert t.n_valid == 2
