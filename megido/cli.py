"""python -m megido.cli validate|ingest"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from megido import validate as V
from megido.calib import calibrate
from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.pipeline import process_all
from megido.reader import EventChunk, read_chunks

def _cmd_validate(args) -> int:
    cfg = load_site_config(args.config)
    geom = DetectorGeometry.megiddo()
    files = cfg.files_for(args.exposure)
    if not files:
        print(f"no files for exposure {args.exposure!r}")
        return 1

    chunks = list(itertools.islice(read_chunks(files[0]), args.max_chunks))
    cal = calibrate(chunks)

    merged = EventChunk(hit=np.concatenate([c.hit for c in chunks]),
                        charge=np.concatenate([c.charge for c in chunks]))

    checks = V.run_all(merged, geom, cal)
    print(f"Exposure {args.exposure}  file {files[0].name}  events {merged.n_events}\n")
    print(V.format_report(checks))

    n_dead = int(cal.dead().sum())
    print(f"\ndead channels: {n_dead} of 128")
    if n_dead:
        asics, chans = np.nonzero(cal.dead())
        pairs = ", ".join(f"({a},{c})" for a, c in zip(asics, chans))
        print(f"  {pairs}")

    n_flagged = int(cal.flagged().sum())
    print(f"gain-flagged channels (>10% from median): {n_flagged} of 128")

    return 0 if all(c.passed for c in checks) else 1


def _cmd_ingest(args) -> int:
    cfg = load_site_config(args.config)
    for r in process_all(cfg, Path(args.out), force=args.force):
        tag = "cached" if r.cached else "built"
        rate = r.n_valid / r.n_events if r.n_events else 0.0
        print(f"{r.exposure_id:6s} {tag:6s} key={r.key}  events={r.n_events:>9,}  "
              f"valid={r.n_valid:>9,} ({rate:.1%})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="megido")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="S0-det checks on the supplied constants")
    v.add_argument("--config", default="configs/megido.yaml")
    v.add_argument("--exposure", default="P0")
    v.add_argument("--max-chunks", type=int, default=1)
    v.set_defaults(func=_cmd_validate)

    i = sub.add_parser("ingest", help="S0-exp + S1 for every exposure")
    i.add_argument("--config", default="configs/megido.yaml")
    i.add_argument("--out", default="runs/ingest")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=_cmd_ingest)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
