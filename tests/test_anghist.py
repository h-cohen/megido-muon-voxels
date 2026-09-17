import json

import numpy as np
import pytest

from megido.anghist import AngularHist, histogram_tracks, load_counts, save_counts
from megido.config import Binning
from megido.tracks import Tracks


def _tracks(tx, ty):
    tx = np.asarray(tx, float)
    ty = np.asarray(ty, float)
    return Tracks(tan_x=tx, tan_y=ty, x_bot_cm=np.zeros_like(tx),
                  y_bot_cm=np.zeros_like(tx), valid=np.isfinite(tx) & np.isfinite(ty))


def test_binning_shape_and_edges():
    h = histogram_tracks(_tracks([0.0], [0.0]), Binning())
    assert h.values.shape == (500, 500)
    assert h.values.dtype == np.int64
    assert len(h.xedges) == 501
    assert h.xedges[0] == pytest.approx(-1.25)
    assert h.xedges[-1] == pytest.approx(1.25)


def test_a_track_lands_in_the_expected_bin():
    """np.digitize follows the same left-closed convention as np.histogram2d."""
    b = Binning()
    for tx, ty in [(0.0, 0.0), (0.3125, -0.4375)]:   # on an edge, then mid-bin
        h = histogram_tracks(_tracks([tx], [ty]), b)
        assert h.total == 1
        i = np.digitize(tx, h.xedges) - 1
        j = np.digitize(ty, h.yedges) - 1
        assert h.values[i, j] == 1, f"({tx}, {ty}) landed in the wrong bin"
        assert h.xedges[i] <= tx < h.xedges[i + 1]
        assert h.yedges[j] <= ty < h.yedges[j + 1]


def test_tan_x_runs_along_axis_zero():
    b = Binning()
    h = histogram_tracks(_tracks([0.5], [-0.5]), b)
    i, j = np.unravel_index(int(np.argmax(h.values)), h.values.shape)
    assert h.xedges[i] <= 0.5 < h.xedges[i + 1]
    assert h.yedges[j] <= -0.5 < h.yedges[j + 1]


def test_invalid_tracks_are_excluded():
    h = histogram_tracks(_tracks([0.0, np.nan, 0.1], [0.0, 0.0, np.nan]), Binning())
    assert h.total == 1


def test_tracks_outside_the_range_are_dropped_not_clipped():
    h = histogram_tracks(_tracks([5.0, 0.0], [0.0, 0.0]), Binning())
    assert h.total == 1, "an out-of-range track must not pile up on the edge bin"


def test_counts_are_integers_and_additive():
    b = Binning()
    a = histogram_tracks(_tracks([0.0, 0.0], [0.0, 0.0]), b)
    c = histogram_tracks(_tracks([0.0], [0.0]), b)
    s = a + c
    assert s.total == 3
    assert s.values.dtype == np.int64


def test_save_and_load_roundtrip(tmp_path):
    b = Binning()
    h = histogram_tracks(_tracks([0.1, -0.2], [0.3, 0.4]), b)
    p = save_counts(h, tmp_path, "P0", meta={"exposure": "P0", "n_files": 21})
    assert p.name == "counts_P0.npz"

    back = load_counts(p)
    assert np.array_equal(back.values, h.values)
    assert np.allclose(back.xedges, h.xedges)
    assert back.name == "txty"

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["exposures"]["P0"]["n_files"] == 21


def test_saved_npz_has_the_load_phantom_dir_keys(tmp_path):
    """cafeteria io.load_phantom_dir requires exactly these array names."""
    h = histogram_tracks(_tracks([0.0], [0.0]), Binning())
    p = save_counts(h, tmp_path, "P0", meta={})
    with np.load(p) as d:
        assert set(d.files) >= {"values", "xedges", "yedges", "name"}


def test_dither_removes_the_position_quantisation_comb():
    """Single-bar hits in both layers quantise tan to steps of 1.6/31.5 = 0.0508.

    Built directly rather than through the simulator, whose linear charge-sharing
    model yields ~93% two-bar clusters and so barely exercises this. Real data is
    ~46% single-bar, which is what produces the measured 3x teeth.

    This test fails if the dither is removed.
    """
    import numpy as np
    from megido.calib import ChannelCalibration
    from megido.config import Binning
    from megido.detector import DetectorGeometry
    from megido.hits import find_hits
    from megido.reader import EventChunk
    from megido.tracks import fit_tracks

    geom = DetectorGeometry.megiddo()
    rng = np.random.default_rng(3)
    n = 30_000

    cal = ChannelCalibration(
        pedestal=np.full((4, 32), 2200.0),
        noise_sigma=np.full((4, 32), 40.0),
        mpv=np.full((4, 32), 1750.0),
        gain=np.ones((4, 32)),
        n_hits=np.full((4, 32), 10_000, dtype=np.int64),
    )

    # every layer gets a single-bar cluster at a random bar: the pure comb case
    hit = np.zeros((n, 4, 32), dtype=bool)
    charge = np.full((n, 4, 32), 2200, dtype=np.int32)
    for a in range(4):
        bars = rng.integers(0, geom.bar.n_bars, size=n)
        chans = np.array([geom.bar_to_channel(a, int(b)) for b in bars])
        hit[np.arange(n), a, chans] = True
        charge[np.arange(n), a, chans] = 2200 + 1750
    chunk = EventChunk(hit=hit, charge=charge)

    def comb_strength(dither):
        tracks = fit_tracks(find_hits(chunk, geom, cal, dither=dither), geom)
        h = histogram_tracks(tracks, Binning())
        profile = h.values.sum(axis=0).astype(float)
        detrended = profile - np.convolve(profile, np.ones(41) / 41, mode="same")
        core = detrended[len(profile) // 4: 3 * len(profile) // 4]
        ac = np.correlate(core, core, "full")
        ac = ac[len(ac) // 2:]
        return float(ac[10] / ac[0]) if ac[0] > 0 else 0.0

    undithered = comb_strength(dither=False)
    dithered = comb_strength(dither=True)

    assert undithered > 0.3, f"the comb must be clearly present without dither (got {undithered:.3f})"
    assert dithered < 0.5 * undithered, (
        f"dither must substantially suppress the comb "
        f"(undithered {undithered:.3f}, dithered {dithered:.3f})"
    )
