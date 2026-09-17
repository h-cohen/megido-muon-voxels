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


def histogram_mpv(values: np.ndarray, bins: int = 200) -> float:
    """Most-probable value: the peak bin centre.

    Muon deposits follow a Landau with a long delta-ray tail, so the mean is
    both biased high and unstable between channels. The MPV is the stable
    estimator and is what the calibration normalises on.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 50:
        return float("nan")
    counts, edges = np.histogram(v, bins=bins)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


@dataclass(frozen=True)
class ChannelCalibration:
    pedestal: np.ndarray
    noise_sigma: np.ndarray
    mpv: np.ndarray
    gain: np.ndarray
    n_hits: np.ndarray

    def threshold(self, n_sigma: float = 3.0) -> np.ndarray:
        return self.pedestal + n_sigma * self.noise_sigma

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
