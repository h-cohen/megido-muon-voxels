"""Hit finding: threshold, cluster, map to bars, interpolate sub-bar position.

Everything here works in BAR index. The channel->bar map is a folded permutation
(spec 1.3); assuming channel number == bar number scores 0.2% adjacency against
44-47% for the real map, so the conversion is never optional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.calib import ChannelCalibration
from megido.detector import BarGeometry, DetectorGeometry
from megido.reader import EventChunk

REJECT_OK = 0
REJECT_NO_HIT = 1
REJECT_UNMAPPED = 2
REJECT_TOO_MANY = 3
REJECT_NON_ADJACENT = 4
REJECT_DEAD_CHANNEL = 5

REJECT_NAMES = {
    REJECT_OK: "ok",
    REJECT_NO_HIT: "no_hit",
    REJECT_UNMAPPED: "unmapped_channel",
    REJECT_TOO_MANY: "too_many_bars",
    REJECT_NON_ADJACENT: "non_adjacent_bars",
    REJECT_DEAD_CHANNEL: "dead_channel",
}


@dataclass(frozen=True)
class LayerHits:
    position_cm: np.ndarray
    bar_lo: np.ndarray
    charge_lo: np.ndarray
    charge_hi: np.ndarray
    reject: np.ndarray


def hit_position_cm(bar_lo: int, q_lo: float, q_hi: float, bar: BarGeometry) -> float:
    """Sub-bar position from charge sharing between bars (bar_lo, bar_lo+1).

    f = q_hi / (q_lo + q_hi) interpolates between the two bar centres, so
    f=0 -> centre of bar_lo and f=1 -> centre of bar_lo+1.
    """
    total = q_lo + q_hi
    if total <= 0:
        return float("nan")
    f = q_hi / total
    return bar.bar_center_cm(bar_lo) + bar.pitch_cm * f


def find_hits(chunk: EventChunk, geom: DetectorGeometry,
              cal: ChannelCalibration, n_sigma: float = 3.0) -> LayerHits:
    """Threshold, cluster, map to bars, and interpolate sub-bar position.

    Rejection precedence, highest first — Task 8 accounts for loss per cause, so
    a cluster failing several checks is attributed to exactly one:
        unmapped_channel  > dead_channel > too_many_bars > non_adjacent_bars
    An unmapped channel outranks a dead one because it has no bar at all, so no
    position could be formed for it even with a perfect calibration.
    """
    n = chunk.n_events
    position = np.full((n, 4), np.nan)
    bar_lo = np.full((n, 4), -1, dtype=np.int16)
    charge_lo = np.zeros((n, 4))
    charge_hi = np.zeros((n, 4))
    reject = np.full((n, 4), REJECT_NO_HIT, dtype=np.uint8)

    threshold = cal.threshold(n_sigma)
    # Pre-build channel -> bar lookup as an array for vectorised use.
    ch2bar = np.full((4, 32), -1, dtype=np.int16)
    for a in range(4):
        for b in range(geom.bar.n_bars):
            ch2bar[a, geom.bar_to_channel(a, b)] = b
    dead = cal.dead()

    above = chunk.hit & (chunk.charge > threshold[None, :, :])
    corrected = (chunk.charge.astype(np.float64) - cal.pedestal[None, :, :]) * cal.gain[None, :, :]

    for a in range(4):
        fired = above[:, a, :]
        counts = fired.sum(axis=1)

        for ev in np.flatnonzero(counts > 0):
            chans = np.flatnonzero(fired[ev])
            bars = ch2bar[a, chans]
            if np.any(bars < 0):
                reject[ev, a] = REJECT_UNMAPPED
                continue
            if dead[a, chans].any():
                reject[ev, a] = REJECT_DEAD_CHANNEL
                continue
            if bars.size > 2:
                reject[ev, a] = REJECT_TOO_MANY
                continue

            order = np.argsort(bars)
            bars = bars[order]
            q = corrected[ev, a, chans[order]]

            if bars.size == 1:
                reject[ev, a] = REJECT_OK
                bar_lo[ev, a] = bars[0]
                charge_lo[ev, a] = q[0]
                position[ev, a] = geom.bar.bar_center_cm(int(bars[0]))
            elif bars[1] - bars[0] == 1:
                reject[ev, a] = REJECT_OK
                bar_lo[ev, a] = bars[0]
                charge_lo[ev, a] = q[0]
                charge_hi[ev, a] = q[1]
                position[ev, a] = hit_position_cm(int(bars[0]), q[0], q[1], geom.bar)
            else:
                reject[ev, a] = REJECT_NON_ADJACENT

    return LayerHits(position_cm=position, bar_lo=bar_lo, charge_lo=charge_lo,
                     charge_hi=charge_hi, reject=reject)
