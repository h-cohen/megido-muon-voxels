"""tracks_<exp>.parquet - the 7-column schema layers_fit_calibration consumes.

Emitting this schema means its self-supervised S-curve corrector (LOLO) plugs in
with no integration work: it requires exactly
{track_id, layer, bar_index, x_formula, charge_n, charge_N, z_layer}.

bar_index is the PHYSICAL bar from the channel map, never a DAQ channel number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from megido.detector import DetectorGeometry
from megido.hits import REJECT_OK, LayerHits

TRACK_SCHEMA: tuple[str, ...] = (
    "track_id", "layer", "bar_index", "x_formula", "charge_n", "charge_N", "z_layer",
)


def tracks_to_table(hits: LayerHits, geom: DetectorGeometry,
                    first_track_id: int = 0) -> pa.Table:
    valid = (hits.reject == REJECT_OK).all(axis=1)
    ev = np.flatnonzero(valid)
    n = ev.size

    track_id = np.repeat(first_track_id + np.arange(n, dtype=np.int64), 4)
    asics = np.tile(np.arange(4), n)

    layer = np.array([geom.layer_of_asic(int(a)) for a in asics], dtype=np.int16)
    z_layer = np.array([geom.z_of_asic(int(a)) for a in asics], dtype=np.float64)

    bar_index = hits.bar_lo[ev].reshape(-1).astype(np.int16)
    x_formula = hits.position_cm[ev].reshape(-1).astype(np.float64)
    charge_n = hits.charge_lo[ev].reshape(-1).astype(np.float64)
    charge_N = hits.charge_hi[ev].reshape(-1).astype(np.float64)

    return pa.table({
        "track_id": track_id,
        "layer": layer,
        "bar_index": bar_index,
        "x_formula": x_formula,
        "charge_n": charge_n,
        "charge_N": charge_N,
        "z_layer": z_layer,
    })


def open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema, compression="zstd")
