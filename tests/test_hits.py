import numpy as np
import pytest

from megido.calib import ChannelCalibration
from megido.detector import BarGeometry, DetectorGeometry
from megido.hits import (REJECT_DEAD_CHANNEL, REJECT_NON_ADJACENT,
                         REJECT_NO_HIT, REJECT_OK, REJECT_TOO_MANY,
                         REJECT_UNMAPPED, find_hits, hit_position_cm)
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
        n_hits=np.full((4, 32), 1000, dtype=np.int64),
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
    # dither=False: this test is about classification (accept/bar/reject), not
    # about the sub-bar position, which is deliberately smeared by default —
    # see test_dither_disabled_reproduces_the_bar_centre for that behaviour.
    hits = find_hits(_chunk({(0, ch): 6000}), geom, flat_cal, dither=False)
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


def test_hit_on_a_dead_channel_is_rejected(geom, flat_cal):
    """A channel with no valid MPV has gain=1.0, i.e. uncorrected charge."""
    ch = geom.bar_to_channel(0, 7)
    mpv = flat_cal.mpv.copy()
    mpv[0, ch] = np.nan                      # too few hits for a valid MPV
    cal = ChannelCalibration(flat_cal.pedestal, flat_cal.noise_sigma, mpv,
                             flat_cal.gain, flat_cal.n_hits)
    assert cal.dead()[0, ch]
    hits = find_hits(_chunk({(0, ch): 6000}), geom, cal)
    assert hits.reject[0, 0] == REJECT_DEAD_CHANNEL


def test_unmapped_outranks_dead_in_reason_precedence(geom, flat_cal):
    """Precedence must be deterministic: Task 8 attributes loss per cause."""
    unmapped = geom.unmapped_channels(0)[0]
    mpv = flat_cal.mpv.copy()
    mpv[0, unmapped] = np.nan                # both unmapped AND dead
    cal = ChannelCalibration(flat_cal.pedestal, flat_cal.noise_sigma, mpv,
                             flat_cal.gain, flat_cal.n_hits)
    hits = find_hits(_chunk({(0, unmapped): 6000}), geom, cal)
    assert hits.reject[0, 0] == REJECT_UNMAPPED


def test_dead_outranks_too_many_bars(geom, flat_cal):
    chans = [geom.bar_to_channel(0, b) for b in (7, 8, 9)]
    mpv = flat_cal.mpv.copy()
    mpv[0, chans[0]] = np.nan
    cal = ChannelCalibration(flat_cal.pedestal, flat_cal.noise_sigma, mpv,
                             flat_cal.gain, flat_cal.n_hits)
    hits = find_hits(_chunk({(0, c): 6000 for c in chans}), geom, cal)
    assert hits.reject[0, 0] == REJECT_DEAD_CHANNEL


# --- single-bar dither ---------------------------------------------------

def test_dither_disabled_reproduces_the_bar_centre(geom, flat_cal):
    ch = geom.bar_to_channel(0, 7)
    hits = find_hits(_chunk({(0, ch): 6000}), geom, flat_cal, dither=False)
    assert hits.position_cm[0, 0] == pytest.approx(geom.bar.bar_center_cm(7))


def test_dither_leaves_two_bar_clusters_untouched(geom, flat_cal):
    """Charge sharing genuinely locates these, so they must not be smeared."""
    c7, c8 = geom.bar_to_channel(0, 7), geom.bar_to_channel(0, 8)
    chunk = _chunk({(0, c7): 5200, (0, c8): 5200})
    plain = find_hits(chunk, geom, flat_cal, dither=False).position_cm[0, 0]
    dithered = find_hits(chunk, geom, flat_cal, dither=True).position_cm[0, 0]
    assert dithered == pytest.approx(plain)


def test_dither_spreads_single_bar_positions_within_the_measured_window(geom, flat_cal):
    ch = geom.bar_to_channel(0, 7)
    n = 4000
    hits = find_hits(_chunk({(0, ch): 6000}, n_events=n), geom, flat_cal, dither=True)
    pos = hits.position_cm[:, 0]
    centre = geom.bar.bar_center_cm(7)

    # every layer here is single-bar, so f_single == 1 and the window is one pitch
    assert np.ptp(pos) > 0.5 * geom.bar.pitch_cm, "positions must actually spread"
    assert np.abs(pos - centre).max() <= 0.5 * geom.bar.pitch_cm + 1e-9
    assert np.mean(pos) == pytest.approx(centre, abs=0.05), "dither must be unbiased"


def test_dither_is_reproducible_for_a_fixed_seed(geom, flat_cal):
    ch = geom.bar_to_channel(0, 7)
    chunk = _chunk({(0, ch): 6000}, n_events=200)
    a = find_hits(chunk, geom, flat_cal, dither=True, seed=7).position_cm
    b = find_hits(chunk, geom, flat_cal, dither=True, seed=7).position_cm
    c = find_hits(chunk, geom, flat_cal, dither=True, seed=8).position_cm
    # equal_nan=True: only asic 0 has any hit in this chunk, so the other
    # three columns are NaN in every run and must compare equal as such.
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)
