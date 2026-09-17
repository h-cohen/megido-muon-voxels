"""Per-exposure orchestration with a content-addressed artifact cache.

6.2 GB of ASCII makes full reprocessing unacceptable for a one-exposure
addition, so artifacts are keyed by the input file list, their sizes and mtimes,
and the config fields that actually affect the output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.anghist import AngularHist, histogram_tracks, save_counts
from megido.calib import calibrate
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.hits import find_hits
from megido.reader import read_chunks
from megido.trackfile import open_writer, tracks_to_table
from megido.tracks import fit_tracks

# Bump whenever a change alters reconstruction OUTPUT for unchanged input:
# hit finding, position reconstruction, track fitting, calibration or binning.
# The artifact cache keys on this, so a stale bump silently serves old results.
RECONSTRUCTION_VERSION = 2


@dataclass(frozen=True)
class ExposureResult:
    exposure_id: str
    key: str
    n_events: int
    n_valid: int
    counts_path: Path
    tracks_path: Path
    cached: bool


def exposure_key(cfg: SiteConfig, eid: str) -> str:
    """Hash the exposure id, pose, binning, input files, and RECONSTRUCTION_VERSION.

    RECONSTRUCTION_VERSION must be bumped whenever a code change alters
    reconstruction output for unchanged input, so this reads the module-level
    constant at call time rather than any value captured earlier.
    """
    exp = cfg.exposure(eid)
    parts = [eid, repr(exp.pose), cfg.binning.t_max, cfg.binning.n_bins, RECONSTRUCTION_VERSION]
    for f in cfg.files_for(eid):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    blob = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def process_exposure(cfg: SiteConfig, eid: str, out_dir: Path,
                     geom: DetectorGeometry | None = None,
                     force: bool = False,
                     chunksize: int = 50_000) -> ExposureResult:
    geom = geom or DetectorGeometry.megiddo()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = exposure_key(cfg, eid)
    stamp = out_dir / f".key_{eid}.json"
    counts_path = out_dir / f"counts_{eid}.npz"
    tracks_path = out_dir / f"tracks_{eid}.parquet"

    if not force and stamp.exists() and counts_path.exists() and tracks_path.exists():
        prev = json.loads(stamp.read_text())
        if prev.get("key") == key:
            return ExposureResult(eid, key, prev["n_events"], prev["n_valid"],
                                  counts_path, tracks_path, cached=True)

    files = cfg.files_for(eid)
    if not files:
        raise FileNotFoundError(f"exposure {eid!r} has no data files in {cfg.data_dir}")

    # Pass 1: calibration needs the whole exposure before hits can be found.
    cal = calibrate(chunk for f in files for chunk in read_chunks(f, chunksize=chunksize))

    # Pass 2: hits, tracks, artifacts.
    edges = cfg.binning.edges()
    total = AngularHist(values=np.zeros((cfg.binning.n_bins, cfg.binning.n_bins), np.int64),
                        xedges=edges, yedges=edges)
    n_events = n_valid = 0
    writer = None
    chunk_index = 0
    try:
        for f in files:
            for chunk in read_chunks(f, chunksize=chunksize):
                # A running counter, not a hash of file/offset: ingest only needs to
                # avoid reusing the same dither pattern across chunks in one run, and
                # a fixed input set always yields the same file/chunk order, so this
                # counter (and hence the whole pipeline's output) is reproducible.
                hits = find_hits(chunk, geom, cal, seed=chunk_index)
                chunk_index += 1
                tracks = fit_tracks(hits, geom)
                total = total + histogram_tracks(tracks, cfg.binning)

                table = tracks_to_table(hits, geom, first_track_id=n_valid)
                if writer is None:
                    writer = open_writer(tracks_path, table.schema)
                if table.num_rows:
                    writer.write_table(table)

                n_events += chunk.n_events
                n_valid += tracks.n_valid
    finally:
        if writer is not None:
            writer.close()

    cal.save(out_dir / f"calib_{eid}.npz")
    save_counts(total, out_dir, eid, meta={
        "exposure": eid,
        "n_files": len(files),
        "n_events": n_events,
        "n_valid_tracks": n_valid,
        "pose": cfg.exposure(eid).pose.__dict__,
        "norm_group": cfg.exposure(eid).norm_group,
    })
    stamp.write_text(json.dumps({"key": key, "n_events": n_events, "n_valid": n_valid}))

    return ExposureResult(eid, key, n_events, n_valid, counts_path, tracks_path, cached=False)


def process_all(cfg: SiteConfig, out_dir: Path, force: bool = False,
                chunksize: int = 50_000) -> list[ExposureResult]:
    return [process_exposure(cfg, e.id, out_dir, force=force, chunksize=chunksize)
            for e in cfg.exposures]
