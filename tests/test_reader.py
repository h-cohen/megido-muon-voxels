import numpy as np
import pytest

from megido.reader import CHARGE_COLUMNS, HIT_COLUMNS, EventChunk, read_chunks


HEADER = ";".join(
    ["ID_CLUSTER", "CLUSTER_RUN_Timecode_ns", "CLUSTER_Timecode_ns", "NEventsInCluster"]
    + [f"{pre}_{a}" for a in range(4)
       for pre in ("ASIC", "EventCounter", "RUN_EventTimeCodeLSB", "RUN_EventTimecode_ns",
                   "T0_to_Event_Timecode", "T0_to_Event_Timecode_ns", "Trigger_ID",
                   "Validation_ID", "Flags")]
    + [f"HIT_{a}_{c}" for a in range(4) for c in range(32)]
    + [f"CHARGE_HG_{a}_{c}" for a in range(4) for c in range(32)]
    + [f"CHARGE_LG_{a}_{c}" for a in range(4) for c in range(32)]
)


def _row(hit_map, charge_map):
    """hit_map/charge_map: {(asic, channel): value}. Everything else zero."""
    vals = ["1", "1000", "1000", "4"] + ["0"] * 36
    vals += [str(int(hit_map.get((a, c), 0))) for a in range(4) for c in range(32)]
    vals += [str(int(charge_map.get((a, c), 2200))) for a in range(4) for c in range(32)]
    vals += ["0"] * 128
    return ";".join(vals)


@pytest.fixture
def fixture_file(tmp_path):
    """Two data rows, a REPEATED header between them, CRLF line endings."""
    p = tmp_path / "DET299999_BEAM_20260101_000000_filter.data"
    lines = [
        HEADER,
        _row({(0, 28): 1, (1, 3): 1}, {(0, 28): 5000, (1, 3): 6000}),
        HEADER,  # DAQ re-arm: header appears again mid-file
        _row({(2, 19): 1, (3, 12): 1}, {(2, 19): 7000, (3, 12): 8000}),
    ]
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    return p


def test_column_name_lists():
    assert len(HIT_COLUMNS) == 128
    assert len(CHARGE_COLUMNS) == 128
    assert HIT_COLUMNS[0] == "HIT_0_0"
    assert HIT_COLUMNS[-1] == "HIT_3_31"
    assert CHARGE_COLUMNS[0] == "CHARGE_HG_0_0"


def test_repeated_headers_are_dropped(fixture_file):
    chunks = list(read_chunks(fixture_file))
    total = sum(c.n_events for c in chunks)
    assert total == 2, "the repeated header row must not be parsed as an event"


def test_shapes_and_dtypes(fixture_file):
    c = next(iter(read_chunks(fixture_file)))
    assert c.hit.shape == (c.n_events, 4, 32)
    assert c.charge.shape == (c.n_events, 4, 32)
    assert c.hit.dtype == np.bool_
    assert c.charge.dtype == np.int32


def test_values_land_in_the_right_asic_and_channel(fixture_file):
    chunks = list(read_chunks(fixture_file))
    hit = np.concatenate([c.hit for c in chunks])
    charge = np.concatenate([c.charge for c in chunks])

    assert hit[0, 0, 28] and hit[0, 1, 3]
    assert hit[0].sum() == 2
    assert charge[0, 0, 28] == 5000
    assert charge[0, 1, 3] == 6000

    assert hit[1, 2, 19] and hit[1, 3, 12]
    assert charge[1, 3, 12] == 8000
    # unhit channels carry pedestal, not zero
    assert charge[1, 0, 0] == 2200


def test_chunking_splits_without_loss(fixture_file):
    chunks = list(read_chunks(fixture_file, chunksize=1))
    assert len(chunks) == 2
    assert all(c.n_events == 1 for c in chunks)
