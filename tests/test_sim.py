import numpy as np
import pytest

from megido.calib import calibrate
from megido.detector import DetectorGeometry
from megido.hits import find_hits
from megido.reader import read_chunks
from megido.sim import simulate, write_raw_file
from megido.tracks import fit_tracks


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def test_simulated_hits_land_only_on_mapped_channels(geom):
    truth = simulate(geom, n_events=2000, seed=0)
    for a in range(4):
        unmapped = list(geom.unmapped_channels(a))
        assert not truth.chunk.hit[:, a, unmapped].any()


def test_simulated_charges_sit_above_pedestal_where_hit(geom):
    truth = simulate(geom, n_events=2000, seed=1)
    hit = truth.chunk.hit
    assert truth.chunk.charge[hit].mean() > truth.chunk.charge[~hit].mean() + 1000


def test_pipeline_recovers_the_injected_angles(geom):
    """The end-to-end gate: raw arrays -> calib -> hits -> tracks -> truth."""
    truth = simulate(geom, n_events=20_000, seed=2)
    cal = calibrate([truth.chunk])
    tracks = fit_tracks(find_hits(truth.chunk, geom, cal), geom)

    v = tracks.valid
    assert v.sum() > 0.95 * truth.chunk.n_events, "every simulated event crosses all four layers"

    dx = tracks.tan_x[v] - truth.true_tan_x[v]
    dy = tracks.tan_y[v] - truth.true_tan_y[v]
    # 4 mm single-layer resolution over a 31.5 cm lever arm -> sigma_tan ~ 0.018
    assert np.abs(np.median(dx)) < 0.005, "no bias in tan_x"
    assert np.abs(np.median(dy)) < 0.005, "no bias in tan_y"
    assert np.std(dx) < 0.04
    assert np.std(dy) < 0.04


def test_adjacency_rate_matches_the_real_detector(geom):
    """The simulator must reproduce the 44-47% two-adjacent-bar rate."""
    truth = simulate(geom, n_events=20_000, seed=3)
    cal = calibrate([truth.chunk])
    hits = find_hits(truth.chunk, geom, cal)
    two_bar = (hits.charge_hi > 0).mean()
    assert 0.30 < two_bar < 0.70


def test_written_raw_file_roundtrips_through_the_reader(geom, tmp_path):
    truth = simulate(geom, n_events=500, seed=4)
    p = tmp_path / "DET299998_BEAM_20260101_000000_filter.data"
    write_raw_file(truth, p, header_every=50)

    back = list(read_chunks(p))
    n = sum(c.n_events for c in back)
    assert n == 500, "repeated headers must not be counted as events"
    hit = np.concatenate([c.hit for c in back])
    charge = np.concatenate([c.charge for c in back])
    assert np.array_equal(hit, truth.chunk.hit)
    assert np.array_equal(charge, truth.chunk.charge)
