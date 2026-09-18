import pytest
from megido.detector import BarGeometry, DetectorGeometry


def test_bar_geometry_derived_values():
    b = BarGeometry()
    assert b.n_bars == 23
    assert b.base_cm == pytest.approx(3.2)
    assert b.pitch_cm == pytest.approx(1.6)
    assert b.active_width_cm == pytest.approx(38.4)
    # side must be consistent with base and height: sqrt((base/2)^2 + height^2)
    assert b.side_cm == pytest.approx(2.3345, abs=1e-3)


def test_bar_centers_are_evenly_spaced():
    b = BarGeometry()
    assert b.bar_center_cm(0) == pytest.approx(1.6)
    assert b.bar_center_cm(22) == pytest.approx(36.8)
    for i in range(22):
        assert b.bar_center_cm(i + 1) - b.bar_center_cm(i) == pytest.approx(1.6)


def test_each_asic_maps_23_distinct_channels():
    g = DetectorGeometry.megiddo()
    for asic in range(4):
        chans = g.asic_channels[asic]
        assert len(chans) == 23
        assert len(set(chans)) == 23
        assert all(0 <= c <= 31 for c in chans)


def test_channel_bar_roundtrip():
    g = DetectorGeometry.megiddo()
    for asic in range(4):
        for bar in range(23):
            ch = g.bar_to_channel(asic, bar)
            assert g.channel_to_bar(asic, ch) == bar


def test_unmapped_channels_return_none():
    g = DetectorGeometry.megiddo()
    # ASIC 0 uses 23 of 32; the other 9 must be unmapped
    unmapped = [c for c in range(32) if g.channel_to_bar(0, c) is None]
    assert len(unmapped) == 9
    assert sorted(unmapped) == [1, 2, 5, 6, 9, 10, 12, 13, 14]


def test_known_map_entries():
    g = DetectorGeometry.megiddo()
    # detector_info.txt: "channel 28 of ASIC_0 connected to bar number 1"
    assert g.bar_to_channel(0, 0) == 28
    assert g.bar_to_channel(0, 1) == 30
    assert g.bar_to_channel(3, 0) == 12


def test_layer_assignment_matches_detector_info():
    g = DetectorGeometry.megiddo()
    # asic_to_layer = [1, 3, 2, 0]; layers bottom-up are X, Y, X, Y
    assert (g.layer_of_asic(0), g.coord_of_asic(0), g.z_of_asic(0)) == (1, "y", 6.2)
    assert (g.layer_of_asic(1), g.coord_of_asic(1), g.z_of_asic(1)) == (3, "y", 37.7)
    assert (g.layer_of_asic(2), g.coord_of_asic(2), g.z_of_asic(2)) == (2, "x", 31.5)
    assert (g.layer_of_asic(3), g.coord_of_asic(3), g.z_of_asic(3)) == (0, "x", 0.0)


def test_coordinate_pairs_and_lever_arms():
    g = DetectorGeometry.megiddo()
    assert g.asics_for_coord("x") == (3, 2)   # ascending z: z=0 then z=31.5
    assert g.asics_for_coord("y") == (0, 1)   # ascending z: z=6.2 then z=37.7
    assert g.dz_cm("x") == pytest.approx(31.5)
    assert g.dz_cm("y") == pytest.approx(31.5)


def test_max_tan_is_geometric_acceptance_limit():
    g = DetectorGeometry.megiddo()
    # active_width / dz = 38.4 / 31.5
    assert g.max_tan() == pytest.approx(38.4 / 31.5, rel=1e-6)
    assert g.max_tan() > 1.0   # the spec's inherited +-1 crop would lose real data


def test_aperture_is_the_active_width_in_metres():
    from megido.detector import DetectorGeometry

    geom = DetectorGeometry.megiddo()
    assert geom.aperture_m == pytest.approx(geom.bar.active_width_cm / 100.0)
    assert geom.aperture_m == pytest.approx(0.384)
