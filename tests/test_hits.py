import numpy as np
import pytest

from megido.calib import ChannelCalibration
from megido.detector import BarGeometry, DetectorGeometry
from megido.hits import (REJECT_NON_ADJACENT, REJECT_NO_HIT, REJECT_OK,
                         REJECT_TOO_MANY, REJECT_UNMAPPED, find_hits,
                         hit_position_cm)
from megido.reader import EventChunk


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


@pytest.fixture
def flat_cal():
    """Pedestal 2200, sigma 40, unit gain on every channel."""
    return ChannelCalibration(
        pedestal=np.full((4, 32), 2200.0),
        noise_sigma=np.full((4, 32), 40.0),
        mpv=np.full((4, 32), 4000.0),
        gain=np.ones((4, 32)),
        n_hits=np.zeros((4, 32), dtype=np.int64),
    )


# --- the position convention -------------------------------------------

def test_equal_charge_lands_midway_between_bar_centres():
    b = BarGeometry()
    x = hit_position_cm(5, 1000.0, 1000.0, b)
    assert x == pytest.approx(b.bar_center_cm(5) + b.pitch_cm / 2)


def test_all_charge_in_low_bar_lands_on_its_centre():
    b = BarGeometry()
    assert hit_position_cm(5, 1000.0, 0.0, b) == pytest.approx(b.bar_center_cm(5))


def test_all_charge_in_high_bar_lands_on_the_next_centre():
    b = BarGeometry()
    assert hit_position_cm(5, 0.0, 1000.0, b) == pytest.approx(b.bar_center_cm(6))


def test_position_is_monotonic_in_charge_fraction():
    b = BarGeometry()
    xs = [hit_position_cm(5, 1000.0 - q, q, b) for q in np.linspace(0, 1000, 11)]
    assert all(x2 > x1 for x1, x2 in zip(xs, xs[1:]))


# --- clustering ---------------------------------------------------------

def _chunk(spec, n_events=1):
    """spec: {(asic, channel): charge}. Hits are charges above 2320 ADC."""
    hit = np.zeros((n_events, 4, 32), dtype=bool)
    charge = np.full((n_events, 4, 32), 2200, dtype=np.int32)
    for (a, c), q in spec.items():
        hit[:, a, c] = True
        charge[:, a, c] = q
    return EventChunk(hit=hit, charge=charge)


def test_single_mapped_hit_is_accepted(geom, flat_cal):
    ch = geom.bar_to_channel(0, 7)
    hits = find_hits(_chunk({(0, ch): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_OK
    assert hits.bar_lo[0, 0] == 7
    assert hits.position_cm[0, 0] == pytest.approx(geom.bar.bar_center_cm(7))


def test_two_adjacent_bars_are_accepted_and_interpolated(geom, flat_cal):
    c7, c8 = geom.bar_to_channel(0, 7), geom.bar_to_channel(0, 8)
    hits = find_hits(_chunk({(0, c7): 5200, (0, c8): 5200}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_OK
    assert hits.bar_lo[0, 0] == 7
    expected = geom.bar.bar_center_cm(7) + geom.bar.pitch_cm / 2
    assert hits.position_cm[0, 0] == pytest.approx(expected)


def test_two_non_adjacent_bars_are_rejected(geom, flat_cal):
    c3, c15 = geom.bar_to_channel(0, 3), geom.bar_to_channel(0, 15)
    hits = find_hits(_chunk({(0, c3): 6000, (0, c15): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_NON_ADJACENT
    assert np.isnan(hits.position_cm[0, 0])


def test_three_hits_are_rejected(geom, flat_cal):
    chans = [geom.bar_to_channel(0, b) for b in (7, 8, 9)]
    hits = find_hits(_chunk({(0, c): 6000 for c in chans}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_TOO_MANY


def test_hit_on_an_unmapped_channel_is_rejected(geom, flat_cal):
    unmapped = geom.unmapped_channels(0)[0]
    hits = find_hits(_chunk({(0, unmapped): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_UNMAPPED


def test_no_hit_layer_is_rejected(geom, flat_cal):
    hits = find_hits(_chunk({}), geom, flat_cal)
    assert (hits.reject[0] == REJECT_NO_HIT).all()


def test_charge_below_threshold_does_not_count_as_a_hit(geom, flat_cal):
    """HIT flag set but charge inside pedestal+3sigma: not a real hit."""
    ch = geom.bar_to_channel(0, 7)
    hits = find_hits(_chunk({(0, ch): 2250}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_NO_HIT


def test_gain_correction_is_applied(geom, flat_cal):
    """Halving one bar's gain must shift the interpolated position toward it."""
    c7, c8 = geom.bar_to_channel(0, 7), geom.bar_to_channel(0, 8)
    chunk = _chunk({(0, c7): 5200, (0, c8): 5200})
    gain = flat_cal.gain.copy()
    gain[0, c8] = 2.0     # bar 8 reads low, so its charge is scaled up
    cal2 = ChannelCalibration(flat_cal.pedestal, flat_cal.noise_sigma,
                              flat_cal.mpv, gain, flat_cal.n_hits)
    plain = find_hits(chunk, geom, flat_cal).position_cm[0, 0]
    boosted = find_hits(chunk, geom, cal2).position_cm[0, 0]
    assert boosted > plain
