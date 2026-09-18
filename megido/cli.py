"""python -m megido.cli validate|ingest"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from megido import validate as V
from megido.angular import load_analysis_grid
from megido.baseline import solve_baseline
from megido.calib import calibrate
from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.pipeline import process_all
from megido.reader import EventChunk, read_chunks
from megido.validate2 import format_report2, leave_one_out, nll_per_bin_check


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
    for r in process_all(cfg, Path(args.out), force=args.force, chunksize=args.chunksize):
        tag = "cached" if r.cached else "built"
        rate = r.n_valid / r.n_events if r.n_events else 0.0
        print(f"{r.exposure_id:6s} {tag:6s} key={r.key}  events={r.n_events:>9,}  "
              f"valid={r.n_valid:>9,} ({rate:.1%})")
    return 0


def _cmd_solve(args) -> int:
    cfg = load_site_config(args.config)
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}")
        return 1

    exposure_ids = [e.id for e in cfg.exposures]
    missing = [e for e in exposure_ids if not (run_dir / f"counts_{e}.npz").exists()]
    if missing:
        print(f"missing counts artifacts for: {', '.join(missing)}")
        return 1

    grid = load_analysis_grid(run_dir, exposure_ids, factor=args.rebin)
    sol = solve_baseline(grid, cfg, n_iter=args.iters)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sol.save(out / "baseline.npz")

    print(f"flux index      {sol.flux_index:.3f}")
    print(f"NLL             {sol.nll_history[-1]:.1f} after {len(sol.nll_history)} iterations")
    print("normalizations  " + "  ".join(f"{g}={v:.4g}" for g, v in sorted(sol.norms.items())))
    for pid in sorted(sol.opacity):
        lam = sol.opacity[pid]
        seen = np.isfinite(lam)
        norm = sol.normalized_opacity(pid)
        print(f"opacity {pid}: {int(seen.sum())} sky bins constrained")
        print(f"    gauge-pinned (median 0, internal): "
              f"p5..p95 {np.nanpercentile(lam, 5):+.4f}..{np.nanpercentile(lam, 95):+.4f}")
        print(f"    referenced to the most transparent direction (physical): "
              f"median {np.nanmedian(norm):.4f}  p95 {np.nanpercentile(norm, 95):.4f}  "
              f"max {np.nanmax(norm):.4f}")

    checks = [nll_per_bin_check(sol, grid)]
    checks += leave_one_out(grid, cfg, n_iter=args.iters)
    print()
    print(format_report2(checks))
    return 0 if all(c.passed for c in checks) else 1


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
    i.add_argument("--chunksize", type=int, default=50_000,
                   help="rows per parsed chunk; see megido.reader.read_chunks docstring")
    i.set_defaults(func=_cmd_ingest)

    s = sub.add_parser("solve", help="Phase 2 baseline and opacity solve")
    s.add_argument("--config", default="configs/megido.yaml")
    s.add_argument("--run", default="runs/ingest")
    s.add_argument("--out", default="runs/solve")
    s.add_argument("--rebin", type=int, default=10)
    s.add_argument("--iters", type=int, default=5000,
                   help="outer iterations; the alternating solve converges "
                        "slowly, see solve_baseline")
    s.set_defaults(func=_cmd_solve)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
