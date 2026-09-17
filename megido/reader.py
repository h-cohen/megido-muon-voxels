"""Streaming reader for CAEN DT5550W cluster dumps.

The files are semicolon-delimited ASCII with CRLF endings and 424 columns. The
header line repeats roughly every 165 rows (the DAQ re-arms), so those rows must
be dropped rather than parsed. Only HIT_* and CHARGE_HG_* are needed here.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_ASICS = 4
N_CHANNELS = 32

HIT_COLUMNS: list[str] = [f"HIT_{a}_{c}" for a in range(N_ASICS) for c in range(N_CHANNELS)]
CHARGE_COLUMNS: list[str] = [f"CHARGE_HG_{a}_{c}" for a in range(N_ASICS) for c in range(N_CHANNELS)]

_ID = "ID_CLUSTER"


@dataclass(frozen=True)
class EventChunk:
    hit: np.ndarray      # bool  [n_events, 4, 32]
    charge: np.ndarray   # int32 [n_events, 4, 32]

    @property
    def n_events(self) -> int:
        return int(self.hit.shape[0])


def read_chunks(path: Path, chunksize: int = 50_000) -> Iterator[EventChunk]:
    """Yield EventChunks. Repeated header rows are dropped.

    chunksize is deliberately modest. The repeated header rows force dtype=str,
    which materialises one Python string object per cell, so memory scales as
    chunksize x 257 columns. At 200,000 rows that reached 9 GB resident and had
    to be killed; 50,000 keeps the parse under about 1.5 GB. Raise it only if
    you have measured the headroom.
    """
    usecols = [_ID] + HIT_COLUMNS + CHARGE_COLUMNS
    reader = pd.read_csv(
        path,
        sep=";",
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,          # repeated header rows make numeric dtypes fail
        engine="c",
        low_memory=False,
    )
    for frame in reader:
        frame = frame[frame[_ID] != _ID]
        if frame.empty:
            continue
        hit = frame[HIT_COLUMNS].to_numpy(dtype=np.int8).reshape(-1, N_ASICS, N_CHANNELS)
        charge = frame[CHARGE_COLUMNS].to_numpy(dtype=np.int32).reshape(-1, N_ASICS, N_CHANNELS)
        yield EventChunk(hit=hit.astype(bool), charge=charge)


def count_events(path: Path) -> int:
    return sum(c.n_events for c in read_chunks(path))
