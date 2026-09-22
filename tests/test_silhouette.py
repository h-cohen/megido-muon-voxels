import numpy as np

from megido.sky import make_sky_grid
from megido.silhouette import (
    sky_to_azel,
    hill_threshold,
    extract_edge,
    ridgeline,
    extract_silhouette,
    SilhouetteResult,
)


def test_sky_to_azel_zenith():
    az, elev = sky_to_azel(np.array([0.0]), np.array([0.0]))
    assert np.isclose(elev[0], 90.0)


def test_sky_to_azel_known_points():
    az, elev = sky_to_azel(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.isclose(az[0], 0.0)
    assert np.isclose(elev[0], 45.0)  # sqrt(1+1)=sqrt2, arcsin(1/sqrt2)=45
    assert np.isclose(az[1], 90.0)
    assert np.isclose(elev[1], 45.0)


def test_sky_to_azel_vectorized_shapes():
    sx = np.linspace(-1, 1, 20)
    sy = np.linspace(-1, 1, 20)
    az, elev = sky_to_azel(sx, sy)
    assert az.shape == (20,)
    assert elev.shape == (20,)
    assert np.all(elev >= 0) and np.all(elev <= 90)


def test_hill_threshold_between_open_sky_and_hill():
    lam = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, np.nan])
    level = hill_threshold(lam, frac=0.5, hi_q=0.9)
    assert 0.0 < level < 5.0


def test_extract_edge_finds_boundary_of_a_block():
    # 6x6 grid, a 2x2 high-opacity block in the middle (rows/cols 2-3).
    image = np.zeros((6, 6))
    image[2:4, 2:4] = 10.0

    class _Sky:
        centers = np.arange(6, dtype=np.float64)

    edge = extract_edge(image, _Sky(), level=1.0)
    # every cell of the 2x2 block is on its own boundary (all 4 cells border
    # the surrounding zero background)
    assert edge.shape[0] == 4
    pts = {tuple(p) for p in edge}
    assert pts == {(2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)}


def test_extract_edge_interior_cell_not_on_edge():
    # 7x7: a hill block (rows/cols 1-5) surrounded by a ring of measured
    # open sky (row/col 0 and 6). A true interior hill cell (all 4
    # neighbours also hill) is not an edge; a hill cell touching the
    # open-sky ring is.
    image = np.zeros((7, 7))
    image[1:6, 1:6] = 10.0
    class _Sky:
        centers = np.arange(7, dtype=np.float64)
    edge = extract_edge(image, _Sky(), level=1.0)
    pts = {tuple(p) for p in edge}
    assert (3.0, 3.0) not in pts   # true interior, all neighbours are hill
    assert (1.0, 1.0) in pts       # borders the measured open-sky ring


def test_extract_edge_off_grid_and_nan_neighbours_do_not_create_edges():
    # A hill cell fully surrounded by NaN (unconstrained sky, no measurement)
    # must NOT be an edge -- off-grid and NaN neighbours never count as
    # "open sky". A separate hill cell that borders real, measured open sky
    # (finite, <= level) must be an edge. This must fail against the old
    # "off-grid/NaN counts as not-mask" definition, which flags the isolated
    # NaN-surrounded cell as a (spurious) edge too.
    image = np.full((5, 5), np.nan)
    image[1, 1] = 10.0   # isolated hill cell, neighbours all NaN
    image[3, 3] = 10.0   # hill cell bordering measured open sky
    image[3, 4] = 0.0    # measured, constrained open sky (finite, <= level)

    class _Sky:
        centers = np.arange(5, dtype=np.float64)

    edge = extract_edge(image, _Sky(), level=1.0)
    pts = {tuple(p) for p in edge}
    assert (3.0, 3.0) in pts
    assert (1.0, 1.0) not in pts


def test_ridgeline_recovers_known_bins_and_nan_for_empty():
    az = np.array([1.0, 3.0, 91.0, 92.0, 93.0])
    elev = np.array([10.0, 20.0, 50.0, 60.0, 70.0])
    centers, ridge = ridgeline(az, elev, n_az=4)  # bins: 0-90,90-180,180-270,270-360
    assert centers.shape == (4,)
    assert np.isclose(ridge[0], 15.0)   # median of 10,20
    assert np.isclose(ridge[1], 60.0)   # median of 50,60,70
    assert np.isnan(ridge[2])
    assert np.isnan(ridge[3])


class _E:
    def __init__(self, id_, x, y=0.0, z=0.0):
        self.id = id_
        self.pose = type("P", (), {"x": x, "y": y, "z": z})


class _FakeCfg:
    def __init__(self, exposures):
        self.exposures = exposures


class _FakeSol:
    """Minimal BaselineSolution stand-in on a real SkyGrid.

    normalized_opacity is HIGH (hill) where elevation < E0(azimuth) for a
    smooth known ridgeline E0(az), LOW (open sky) elsewhere -- so the
    edge -> ridgeline pipeline can be checked against ground truth.
    """
    def __init__(self, sky, e0_fn):
        self.sky = sky
        n = sky.n_bins
        centers = sky.centers
        sx, sy = np.meshgrid(centers, centers, indexing="ij")
        from megido.silhouette import sky_to_azel
        az, elev = sky_to_azel(sx, sy)
        e0 = e0_fn(az)
        # hill (high opacity) toward the crown (elev above the ridgeline);
        # open sky (low opacity) off the flanks (elev below it) -- this keeps
        # the masked region away from the sky grid's outer border (which sits
        # at low elevation), so the only mask/non-mask transition is the true
        # ridgeline, not a grid-edge artifact.
        lam = np.where(elev > e0, 10.0, 0.0)
        self._lam = {"pos0": lam.ravel(), "pos1": lam.ravel()}

    def normalized_opacity(self, pid, transparent_quantile=0.05):
        return self._lam[pid]


def _e0(az_deg):
    return 45.0 + 15.0 * np.sin(np.radians(az_deg))


def test_extract_silhouette_recovers_known_ridgeline():
    sky = make_sky_grid()
    sol = _FakeSol(sky, _e0)
    cfg = _FakeCfg([
        _E("P0", 0.0, 0.0, 0.0),
        _E("T20a", 0.0, 0.0, 0.0),
        _E("T20b", 0.0, 0.0, 0.0),
        _E("P1", 2.2, 0.0, 0.0),
    ])

    res = extract_silhouette(sol, cfg, n_az=72)
    assert isinstance(res, SilhouetteResult)
    assert set(res.per_pos.keys()) == {"pos0", "pos1"}
    assert len(res.detectors) == 2

    d = res.per_pos["pos0"]
    assert d["n_edge"] > 0
    assert 0.0 < d["az_coverage"] <= 1.0
    ridge_az, ridge_elev = d["ridge_az"], d["ridge_elev"]
    truth = _e0(ridge_az)
    m = np.isfinite(ridge_elev)
    assert m.sum() > 20
    corr = np.corrcoef(ridge_elev[m], truth[m])[0, 1]
    rms = np.sqrt(np.mean((ridge_elev[m] - truth[m]) ** 2))
    assert corr > 0.8 or rms < 5.0


def test_extract_silhouette_agreement_between_positions():
    sky = make_sky_grid()
    sol = _FakeSol(sky, _e0)
    cfg = _FakeCfg([
        _E("P0", 0.0, 0.0, 0.0),
        _E("P1", 2.2, 0.0, 0.0),
    ])
    res = extract_silhouette(sol, cfg, n_az=72)
    # both positions see the same known silhouette (fake sol reuses it) ->
    # agreement should be small and finite.
    assert np.isfinite(res.agreement)
    assert res.agreement < 5.0
