"""S0-exp: per-channel pedestal and gain, computed per exposure.

Runs per exposure rather than once per detector so DAQ drift (such as the
unexplained 22% rate change on 6 Aug 2026) is caught rather than absorbed.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.reader import EventChunk

N_ASICS, N_CHANNELS = 4, 32


def sigma_clipped_pedestal(values: np.ndarray, n_sigma: float = 3.0,
                           n_iter: int = 5) -> tuple[float, float]:
    """Mean and sigma of the pedestal core, with the signal tail clipped away."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    keep = np.ones(v.shape, dtype=bool)
    mean = float(np.mean(v))
    sigma = float(np.std(v))
    for _ in range(n_iter):
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(v - mean) < n_sigma * sigma
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
        mean = float(np.mean(v[keep]))
        sigma = float(np.std(v[keep]))
    return mean, sigma


def histogram_mpv(values: np.ndarray, bins: int = 200, smooth: int = 15) -> float:
    """Most-probable value of a Landau-like charge spectrum.

    Muon deposits follow a Landau with a long delta-ray tail, so the mean is
    biased high and unstable between channels. The MPV is the stable estimator.

    Three refinements over a plain peak-bin search, each measured:
      - the histogram range is bounded at the 90th percentile, because a Landau's
        tail maximum grows with sample count and would otherwise make the bin
        width depend on how much data a channel happened to collect;
      - the counts are smoothed, because argmax of a Poisson-noisy histogram
        wanders across the flat region near a smooth maximum;
      - a parabola through the peak and its neighbours gives sub-bin resolution.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 50:
        return float("nan")

    lo, hi = np.percentile(v, [0.0, 90.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    counts = counts.astype(np.float64)
    if smooth > 1:
        counts = np.convolve(counts, np.ones(smooth) / smooth, mode="same")

    i = int(np.argmax(counts))
    offset = 0.0
    if 0 < i < len(counts) - 1:
        curvature = counts[i - 1] - 2.0 * counts[i] + counts[i + 1]
        if curvature != 0.0:
            offset = float(np.clip(0.5 * (counts[i - 1] - counts[i + 1]) / curvature, -1.0, 1.0))

    bin_width = edges[1] - edges[0]
    return float(0.5 * (edges[i] + edges[i + 1]) + offset * bin_width)


@dataclass(frozen=True)
class ChannelCalibration:
    pedestal: np.ndarray
    noise_sigma: np.ndarray
    mpv: np.ndarray
    gain: np.ndarray
    n_hits: np.ndarray

    def threshold(self, n_sigma: float = 3.0) -> np.ndarray:
        return self.pedestal + n_sigma * self.noise_sigma

    def dead(self) -> np.ndarray:
        """Boolean [4, 32] mask of channels with no usable calibration.

        `gain` falls back to 1.0 for these so downstream arithmetic stays finite,
        which would otherwise make a dead channel look perfectly normal to the
        >10% gain-deviation flagging. This mask is how a caller tells them apart.
        """
        return (self.n_hits == 0) | ~np.isfinite(self.mpv)

    def flagged(self, tolerance: float = 0.10) -> np.ndarray:
        """Boolean [4, 32] mask of channels whose gain deviates beyond `tolerance`.

        Spec 4.2's per-exposure quality flag. `gain` is median(mpv)/mpv, so a
        healthy channel sits near 1.0. Dead channels are excluded — they are
        reported by `dead()` and their gain is a placeholder, not a measurement.

        On real data this flags a large fraction of channels (53 of 92 mapped
        bars on DET200084), which is genuine detector structure rather than a
        calibration failure: per-ASIC MPV medians run 1698 / 1854 / 1906 / 2463.
        """
        deviation = np.abs(self.gain - 1.0)
        return (deviation > tolerance) & ~self.dead()

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, pedestal=self.pedestal, noise_sigma=self.noise_sigma,
            mpv=self.mpv, gain=self.gain, n_hits=self.n_hits,
        )

    @staticmethod
    def load(path: Path) -> "ChannelCalibration":
        d = np.load(path)
        return ChannelCalibration(
            pedestal=d["pedestal"], noise_sigma=d["noise_sigma"],
            mpv=d["mpv"], gain=d["gain"], n_hits=d["n_hits"],
        )


def calibrate(chunks: Iterable[EventChunk], max_samples: int = 400_000) -> ChannelCalibration:
    """Accumulate per-channel HIT=0 and HIT=1 charge samples, then fit."""
    ped_samples: list[list[np.ndarray]] = [[[] for _ in range(N_CHANNELS)] for _ in range(N_ASICS)]
    sig_samples: list[list[np.ndarray]] = [[[] for _ in range(N_CHANNELS)] for _ in range(N_ASICS)]
    n_hits = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)
    n_ped = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)

    for chunk in chunks:
        for a in range(N_ASICS):
            for c in range(N_CHANNELS):
                h = chunk.hit[:, a, c]
                q = chunk.charge[:, a, c]
                n_hits[a, c] += int(h.sum())
                n_ped[a, c] += int((~h).sum())
                if n_ped[a, c] <= max_samples:
                    ped_samples[a][c].append(q[~h])
                if n_hits[a, c] <= max_samples:
                    sig_samples[a][c].append(q[h])

    pedestal = np.full((N_ASICS, N_CHANNELS), np.nan)
    noise = np.full((N_ASICS, N_CHANNELS), np.nan)
    mpv = np.full((N_ASICS, N_CHANNELS), np.nan)

    for a in range(N_ASICS):
        for c in range(N_CHANNELS):
            ped = np.concatenate(ped_samples[a][c]) if ped_samples[a][c] else np.empty(0)
            sig = np.concatenate(sig_samples[a][c]) if sig_samples[a][c] else np.empty(0)
            pedestal[a, c], noise[a, c] = sigma_clipped_pedestal(ped)
            peak = histogram_mpv(sig)
            mpv[a, c] = peak - pedestal[a, c] if np.isfinite(peak) else np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.nanmedian(mpv)
        gain = ref / mpv
    gain = np.where(np.isfinite(gain) & (gain > 0), gain, 1.0)

    return ChannelCalibration(pedestal=pedestal, noise_sigma=noise, mpv=mpv,
                              gain=gain, n_hits=n_hits)
