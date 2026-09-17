import numpy as np
import pytest

from megido.calib import (ChannelCalibration, calibrate, histogram_mpv,
                          sigma_clipped_pedestal)
from megido.reader import EventChunk


def test_sigma_clipped_pedestal_recovers_a_clean_gaussian():
    rng = np.random.default_rng(0)
    v = rng.normal(2200.0, 40.0, size=50_000)
    mean, sigma = sigma_clipped_pedestal(v)
    assert mean == pytest.approx(2200.0, abs=1.0)
    assert sigma == pytest.approx(40.0, rel=0.05)


def test_sigma_clipped_pedestal_rejects_a_signal_tail():
    """A high-side tail must not drag the pedestal up - that is why we clip."""
    rng = np.random.default_rng(1)
    core = rng.normal(2200.0, 40.0, size=50_000)
    tail = rng.normal(7000.0, 1500.0, size=5_000)
    mean, _ = sigma_clipped_pedestal(np.concatenate([core, tail]))
    assert mean == pytest.approx(2200.0, abs=5.0)
    assert np.mean(np.concatenate([core, tail])) > 2500  # the naive mean fails


def test_histogram_mpv_finds_the_landau_peak_not_the_mean():
    from scipy.stats import moyal
    rng = np.random.default_rng(2)
    v = moyal.rvs(loc=4000.0, scale=600.0, size=200_000, random_state=rng)
    mpv = histogram_mpv(v)
    assert mpv == pytest.approx(4000.0, rel=0.05)
    assert mpv < np.mean(v)   # the Landau tail pulls the mean above the MPV


def _synthetic_chunk(n=20_000, seed=0):
    """Every channel: pedestal 2200+-40; hit channels add a Moyal on top."""
    from scipy.stats import moyal
    rng = np.random.default_rng(seed)
    charge = rng.normal(2200.0, 40.0, size=(n, 4, 32))
    hit = rng.random((n, 4, 32)) < 0.05
    extra = moyal.rvs(loc=4000.0, scale=600.0, size=(n, 4, 32), random_state=rng)
    charge[hit] += extra[hit]
    return EventChunk(hit=hit, charge=charge.astype(np.int32))


def test_calibrate_recovers_pedestal_and_mpv():
    cal = calibrate([_synthetic_chunk()])
    assert cal.pedestal.shape == (4, 32)
    assert np.allclose(cal.pedestal, 2200.0, atol=5.0)
    assert np.allclose(cal.noise_sigma, 40.0, rtol=0.15)
    # mpv is pedestal-subtracted
    assert np.allclose(cal.mpv, 4000.0, rtol=0.10)


def test_gain_is_normalised_to_the_median_channel():
    cal = calibrate([_synthetic_chunk()])
    assert np.median(cal.gain) == pytest.approx(1.0, abs=0.02)
    assert np.all(cal.gain > 0)


def test_threshold_is_pedestal_plus_n_sigma():
    cal = calibrate([_synthetic_chunk()])
    thr = cal.threshold(n_sigma=3.0)
    assert np.allclose(thr, cal.pedestal + 3.0 * cal.noise_sigma)


def test_calibration_roundtrips_through_disk(tmp_path):
    cal = calibrate([_synthetic_chunk()])
    p = tmp_path / "cal.npz"
    cal.save(p)
    back = ChannelCalibration.load(p)
    assert np.allclose(back.pedestal, cal.pedestal)
    assert np.allclose(back.gain, cal.gain)
    assert np.array_equal(back.n_hits, cal.n_hits)
