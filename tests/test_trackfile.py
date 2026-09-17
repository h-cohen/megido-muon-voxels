import numpy as np
import pyarrow.parquet as pq
import pytest

from megido.detector import DetectorGeometry
from megido.hits import REJECT_NO_HIT, REJECT_OK, LayerHits
from megido.trackfile import TRACK_SCHEMA, open_writer, tracks_to_table


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def _hits(n=2, reject=None):
    rej = np.full((n, 4), REJECT_OK, np.uint8) if reject is None else reject
    return LayerHits(
        position_cm=np.full((n, 4), 19.2),
        bar_lo=np.full((n, 4), 11, np.int16),
        charge_lo=np.full((n, 4), 3000.0),
        charge_hi=np.full((n, 4), 1000.0),
        reject=rej,
    )


def test_schema_matches_layers_fit_calibration(geom):
    t = tracks_to_table(_hits(), geom)
    assert tuple(t.column_names) == TRACK_SCHEMA


def test_one_row_per_layer_per_valid_event(geom):
    t = tracks_to_table(_hits(n=3), geom)
    assert t.num_rows == 3 * 4


def test_invalid_events_are_excluded(geom):
    rej = np.full((3, 4), REJECT_OK, np.uint8)
    rej[1, 2] = REJECT_NO_HIT
    t = tracks_to_table(_hits(n=3, reject=rej), geom)
    assert t.num_rows == 2 * 4


def test_bar_index_is_the_physical_bar_not_the_channel(geom):
    """The whole point of the map: bar_index must never be a DAQ channel."""
    t = tracks_to_table(_hits(), geom).to_pydict()
    assert set(t["bar_index"]) == {11}
    # channel 11 maps to a different bar on every ASIC, so a channel would leak through
    assert geom.channel_to_bar(0, 11) != 11


def test_z_layer_uses_supplied_geometry(geom):
    t = tracks_to_table(_hits(n=1), geom).to_pydict()
    assert sorted(set(t["z_layer"])) == [0.0, 6.2, 31.5, 37.7]


def test_layer_column_is_the_layer_index_not_the_asic(geom):
    t = tracks_to_table(_hits(n=1), geom).to_pydict()
    assert sorted(set(t["layer"])) == [0, 1, 2, 3]


def test_track_ids_are_unique_and_offset(geom):
    t = tracks_to_table(_hits(n=2), geom, first_track_id=100).to_pydict()
    assert sorted(set(t["track_id"])) == [100, 101]


def test_roundtrip_through_parquet(tmp_path, geom):
    p = tmp_path / "tracks_P0.parquet"
    table = tracks_to_table(_hits(n=5), geom)
    w = open_writer(p, table.schema)
    w.write_table(table)
    w.close()

    back = pq.read_table(p)
    assert back.num_rows == 20
    assert tuple(back.column_names) == TRACK_SCHEMA
