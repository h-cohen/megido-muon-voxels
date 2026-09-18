import numpy as np
import pytest

from megido.anghist import AngularHist
from megido.angular import AnalysisGrid, load_analysis_grid, rebin


def _hist(n=500, fill=1):
    edges = np.linspace(-1.25, 1.25, n + 1)
    return AngularHist(values=np.full((n, n), fill, dtype=np.int64),
                       xedges=edges, yedges=edges)


def test_rebin_preserves_total_counts():
    h = _hist()
    r = rebin(h, 10)
    assert r.values.shape == (50, 50)
    assert r.values.sum() == h.values.sum()
    assert r.values.dtype == np.int64


def test_rebin_edges_are_a_subsample_of_the_originals():
    h = _hist()
    r = rebin(h, 10)
    assert len(r.xedges) == 51
    assert r.xedges[0] == pytest.approx(-1.25)
    assert r.xedges[-1] == pytest.approx(1.25)
    assert r.xedges[1] - r.xedges[0] == pytest.approx(0.05)


def test_rebin_sums_the_right_neighbours():
    """A single count must land in the bin containing its original bin."""
    h = _hist(fill=0)
    values = np.array(h.values)
    values[0, 0] = 7          # first original bin
    values[499, 499] = 3      # last original bin
    h = AngularHist(values=values, xedges=h.xedges, yedges=h.yedges)
    r = rebin(h, 10)
    assert r.values[0, 0] == 7
    assert r.values[49, 49] == 3
    assert r.values.sum() == 10


def test_rebin_rejects_a_non_divisor_factor():
    with pytest.raises(ValueError, match="divide"):
        rebin(_hist(), 7)


def test_grid_centres_and_mesh():
    g = AnalysisGrid(edges=np.linspace(-1.25, 1.25, 51),
                     counts={"P0": np.zeros((50, 50), np.int64)})
    assert g.n_bins == 50
    assert g.centers[0] == pytest.approx(-1.225)
    tx, ty = g.tan_mesh()
    assert tx.shape == ty.shape == (50, 50)
    # tan_x varies down axis 0, matching np.histogram2d(tan_x, tan_y)
    assert tx[0, 0] == pytest.approx(tx[0, 49])
    assert ty[0, 0] == pytest.approx(ty[49, 0])
    assert tx[1, 0] > tx[0, 0]


def test_occupied_is_the_union_over_exposures():
    a = np.zeros((50, 50), np.int64); a[10, 10] = 5
    b = np.zeros((50, 50), np.int64); b[20, 20] = 5
    g = AnalysisGrid(edges=np.linspace(-1.25, 1.25, 51), counts={"A": a, "B": b})
    occ = g.occupied()
    assert occ[10, 10] and occ[20, 20]
    assert occ.sum() == 2


def test_load_analysis_grid_reads_phase1_artifacts(tmp_path):
    from megido.anghist import save_counts
    for eid in ("P0", "P1"):
        save_counts(_hist(fill=2), tmp_path, eid, meta={})
    g = load_analysis_grid(tmp_path, ["P0", "P1"], factor=10)
    assert set(g.counts) == {"P0", "P1"}
    assert g.counts["P0"].shape == (50, 50)
    assert g.counts["P0"].sum() == 500 * 500 * 2
    assert g.counts["P0"].dtype == np.int64


def test_load_analysis_grid_rejects_mismatched_binning(tmp_path):
    from megido.anghist import save_counts
    save_counts(_hist(n=500), tmp_path, "P0", meta={})
    save_counts(_hist(n=400), tmp_path, "P1", meta={})
    with pytest.raises(ValueError, match="binning"):
        load_analysis_grid(tmp_path, ["P0", "P1"], factor=10)
