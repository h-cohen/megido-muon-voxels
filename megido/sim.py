"""Synthetic events through the real geometry and the real channel map.

Used as the ground-truth gate: the full chain must recover injected angles
within resolution, and must reproduce the measured two-adjacent-bar rate.

    Known infidelity, deliberate: charge is split linearly in the crossing
    fraction, which yields roughly 93% two-adjacent-bar clusters against about
    53% measured on real data (DET200084). Real triangular bars do not share
    light linearly with position — a muon crossing near a bar's thick centre
    deposits almost all of it in that one bar. This simulator is a pipeline
    gate, not a detector-response model: do not use it to predict acceptance,
    to tune the adjacency threshold, or to estimate cluster-topology rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import moyal

from megido.detector import DetectorGeometry
from megido.reader import CHARGE_COLUMNS, HIT_COLUMNS, EventChunk

_CLUSTER_COLS = ["ID_CLUSTER", "CLUSTER_RUN_Timecode_ns", "CLUSTER_Timecode_ns",
                 "NEventsInCluster"]
_ASIC_COLS = [f"{pre}_{a}" for a in range(4)
              for pre in ("ASIC", "EventCounter", "RUN_EventTimeCodeLSB",
                          "RUN_EventTimecode_ns", "T0_to_Event_Timecode",
                          "T0_to_Event_Timecode_ns", "Trigger_ID", "Validation_ID",
                          "Flags")]
_LG_COLS = [f"CHARGE_LG_{a}_{c}" for a in range(4) for c in range(32)]


@dataclass(frozen=True)
class TruthEvents:
    chunk: EventChunk
    true_tan_x: np.ndarray
    true_tan_y: np.ndarray


def _deposit(bar_geom, pos_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a crossing position into (bar_lo, fraction_in_high_bar).

    Inverse of megido.hits.hit_position_cm.
    """
    pitch = bar_geom.pitch_cm
    u = pos_cm / pitch - 1.0            # continuous bar coordinate
    bar_lo = np.floor(u).astype(int)
    frac = u - bar_lo
    return bar_lo, frac


def simulate(geom: DetectorGeometry, n_events: int = 20_000, seed: int = 0,
             pedestal: float = 2200.0, noise: float = 40.0,
             mpv: float = 4000.0,   # arbitrary charge scale, NOT the measured ~1900 ADC MPV
             max_tan: float | None = None) -> TruthEvents:
    """Throw muons through the real geometry and the real channel map.

    Every generated event crosses all four layers inside the active area, which
    is what the detector's four-fold coincidence trigger selects. The resulting
    angular distribution is therefore acceptance-shaped rather than raw cos^2 —
    correct for a detector-response gate, but not an open-sky flux model.
    """
    rng = np.random.default_rng(seed)
    bar = geom.bar
    if max_tan is None:
        max_tan = 0.6 * geom.max_tan()

    def _crosses_all_layers(tx, ty, x0, y0):
        ok = np.ones(tx.shape, dtype=bool)
        for asic in range(4):
            z = geom.z_of_asic(asic)
            pos = (x0 + tx * z) if geom.coord_of_asic(asic) == "x" else (y0 + ty * z)
            bar_lo, _ = _deposit(bar, pos)
            ok &= (bar_lo >= 0) & (bar_lo < bar.n_bars - 1)
        return ok

    tan_x = np.empty(n_events)
    tan_y = np.empty(n_events)
    x0 = np.empty(n_events)
    y0 = np.empty(n_events)

    filled = 0
    while filled < n_events:
        batch = max(4 * (n_events - filled), 1024)

        tx = rng.uniform(-max_tan, max_tan, size=batch)
        ty = rng.uniform(-max_tan, max_tan, size=batch)
        cos_t = 1.0 / np.sqrt(1.0 + tx**2 + ty**2)
        keep = rng.random(batch) < cos_t**2          # cos^2 zenith weighting
        tx, ty = tx[keep], ty[keep]

        bx = rng.uniform(0.0, bar.active_width_cm, size=tx.size)
        by = rng.uniform(0.0, bar.active_width_cm, size=tx.size)

        ok = _crosses_all_layers(tx, ty, bx, by)
        tx, ty, bx, by = tx[ok], ty[ok], bx[ok], by[ok]

        take = min(tx.size, n_events - filled)
        if take == 0:
            continue
        sl = slice(filled, filled + take)
        tan_x[sl], tan_y[sl] = tx[:take], ty[:take]
        x0[sl], y0[sl] = bx[:take], by[:take]
        filled += take

    hit = np.zeros((n_events, 4, 32), dtype=bool)
    charge = rng.normal(pedestal, noise, size=(n_events, 4, 32))

    for asic in range(4):
        z = geom.z_of_asic(asic)
        coord = geom.coord_of_asic(asic)
        pos = (x0 + tan_x * z) if coord == "x" else (y0 + tan_y * z)

        bar_lo, frac = _deposit(bar, pos)
        amplitude = moyal.rvs(loc=mpv, scale=mpv / 7.0, size=n_events, random_state=rng)
        amplitude = np.clip(amplitude, mpv * 0.3, mpv * 5.0)

        for i in range(n_events):
            b = int(bar_lo[i])
            f = float(frac[i])
            q_hi = amplitude[i] * f
            q_lo = amplitude[i] * (1.0 - f)
            if q_lo > 4.0 * noise:
                charge[i, asic, geom.bar_to_channel(asic, b)] += q_lo
                hit[i, asic, geom.bar_to_channel(asic, b)] = True
            if q_hi > 4.0 * noise:
                charge[i, asic, geom.bar_to_channel(asic, b + 1)] += q_hi
                hit[i, asic, geom.bar_to_channel(asic, b + 1)] = True

    chunk = EventChunk(hit=hit, charge=np.rint(charge).astype(np.int32))
    return TruthEvents(chunk=chunk, true_tan_x=tan_x, true_tan_y=tan_y)


def write_raw_file(truth: TruthEvents, path: Path, header_every: int = 165) -> None:
    """Write a file in the real raw format, including repeated header rows."""
    columns = _CLUSTER_COLS + _ASIC_COLS + HIT_COLUMNS + CHARGE_COLUMNS + _LG_COLS
    header = ";".join(columns)

    n = truth.chunk.n_events
    hit = truth.chunk.hit.reshape(n, -1).astype(np.int8)
    charge = truth.chunk.charge.reshape(n, -1)
    zeros_cluster = np.zeros((n, len(_CLUSTER_COLS) + len(_ASIC_COLS)), dtype=np.int64)
    zeros_cluster[:, 3] = 4          # NEventsInCluster
    zeros_lg = np.zeros((n, len(_LG_COLS)), dtype=np.int64)

    rows = np.concatenate([zeros_cluster, hit, charge, zeros_lg], axis=1)

    lines: list[str] = []
    for i in range(n):
        if i % header_every == 0:
            lines.append(header)
        lines.append(";".join(map(str, rows[i].tolist())))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode())
