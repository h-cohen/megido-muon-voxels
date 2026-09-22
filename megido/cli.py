"""python -m megido.cli validate|ingest|solve|reconstruct|export|compare"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from megido import validate as V
from megido.angular import load_analysis_grid
from megido.backproject import backproject_plane, plane_axes
from megido.baseline import BaselineSolution, solve_baseline
from megido.calib import calibrate
from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.fitdata import build_fit_data
from megido.forward import build_forward_model
from megido.pipeline import process_all
from megido.reader import EventChunk, read_chunks
from megido.reconstruct import VoxelSolution, solve_voxels
from megido.resolution import campaign_resolution, format_resolution, views_per_voxel
from megido.silhouette import extract_silhouette
from megido.validate2 import format_report2, leave_one_out, nll_per_bin_check, opacity_uncertainty
from megido.volexport import _json_safe, compare_volumes, export_volume
from megido.voxuncert import systematic_map, voxel_bootstrap


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


_SKY_SIGMA_T = 0.05     # Phase 2 sky grid bin width; see megido.sky.make_sky_grid


def _cmd_reconstruct(args) -> int:
    cfg = load_site_config(args.config)
    baseline = Path(args.solve) / "baseline.npz"
    if not baseline.exists():
        print(f"no baseline.npz under {args.solve}; run `megido solve` first")
        return 1
    if args.bootstrap and not args.run:
        print("--bootstrap needs --run: replicas are resampled from the ingested "
              "counts, and the saved baseline does not carry them")
        return 1

    sol = BaselineSolution.load(baseline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = campaign_resolution(cfg, sigma_t=_SKY_SIGMA_T,
                              feature_pitch_m=max(2.0, 4 * cfg.volume.spacing_m))
    print(format_resolution(res))
    print()

    sigma = None
    counts_grid = None
    if args.run:
        exposure_ids = [e.id for e in cfg.exposures]
        counts_grid = load_analysis_grid(Path(args.run), exposure_ids, factor=args.rebin)
        if args.bootstrap:
            print(f"bootstrapping sky opacity sigma ({args.bootstrap} replicas)...",
                  flush=True)
            sigma = opacity_uncertainty(counts_grid, cfg, n_replicas=args.bootstrap,
                                        n_iter=args.iters)

    fits = solve_voxels(sol, cfg, sigma=sigma, cache_dir=args.cache,
                        holdouts=not args.no_holdouts)
    for tag, v in fits.items():
        v.save(out / f"volume_{tag}.npz")

    full = fits["full"]
    data = build_fit_data(sol, cfg, sigma=sigma)
    fwd = build_forward_model(data.rows, cfg, cache_dir=args.cache)
    views = views_per_voxel(fwd)
    np.save(out / "views.npy", views)

    print(f"grid            {full.grid.shape} at {full.grid.spacing:.3f} m, "
          f"origin {tuple(round(v, 2) for v in full.grid.origin)}")
    print(f"rows            {data.rows.n_rows} over "
          f"{len(data.rows.position_ids)} positions")
    print(f"chi2            {full.info.get('best_chi2', float('nan')):.4f}")
    print("offsets         " + "  ".join(f"{k}={v:+.4f}"
                                         for k, v in sorted(full.offsets.items())))
    rho = full.rho3()
    print(f"opacity density median {np.median(rho[rho > 0]) if (rho > 0).any() else 0:.4f}"
          f"  p95 {np.percentile(rho, 95):.4f}  max {rho.max():.4f} 1/m")

    if args.bootstrap and counts_grid is not None:
        boot = voxel_bootstrap(counts_grid, cfg, n_replicas=args.bootstrap,
                               cache_dir=args.cache,
                               solve_kwargs={"n_iter": args.iters})
        boot.save(out / "uncertainty.npz")
        snr = boot.snr()
        n_grid = views.size
        # views==0 voxels sit outside every position's footprint (edge padding);
        # they get stable near-zero TV-driven values and score high "SNR" while
        # measuring nothing. Restrict the SNR claim to voxels an actual ray
        # crossed, and separately to voxels crossed by both positions (the only
        # ones with any depth information at all).
        seen1 = np.isfinite(snr) & (views >= 1)
        seen2 = np.isfinite(snr) & (views >= 2)
        n1, n2 = int(seen1.sum()), int(seen2.sum())
        hi1 = int((np.abs(snr[seen1]) > 3).sum())
        hi2 = int((np.abs(snr[seen2]) > 3).sum())
        print(f"bootstrap       {args.bootstrap} replicas")
        print(f"  viewed>=1     {n1} of {n_grid} voxels ({n1 / n_grid:.1%} of grid); "
              f"{hi1} of {max(n1, 1)} above SNR 3 ({hi1 / max(n1, 1):.1%})")
        print(f"  viewed==2     {n2} of {n_grid} voxels ({n2 / n_grid:.1%} of grid, "
              f"depth-informative); {hi2} of {max(n2, 1)} above SNR 3 ({hi2 / max(n2, 1):.1%})")

    if not args.no_systematic:
        sysmap = systematic_map(sol, cfg, cache_dir=args.cache)
        np.save(out / "systematic.npy", sysmap.astype(np.float32))
        print(f"gauge systematic  max |delta rho| {np.abs(sysmap).max():.4f} 1/m "
              f"({np.abs(sysmap).max() / max(rho.max(), 1e-12):.1%} of the peak)")

    if args.backproject_z is not None:
        # The plane's pixel pitch is matched to where the rays actually land, not
        # to the voxel spacing. Rays leave the sky grid at ~0.05 tan spacing, so
        # at height z they land ~0.05*z apart; a finer plane than that is mostly
        # empty pixels (a nearest-scatter anchor at voxel pitch fills only ~40%).
        bp_res = max(full.grid.spacing, 0.05 * args.backproject_z)
        xs, ys = plane_axes(cfg, args.backproject_z, data.rows.t_reach(),
                            res_m=bp_res)
        _, mean = backproject_plane(data, cfg, args.backproject_z, xs, ys)
        np.save(out / "backprojection.npy", mean.astype(np.float32))
        print(f"backprojection  plane z={args.backproject_z:.2f} m, "
              f"{mean.shape} at {bp_res:.2f} m")

    print(f"\nwritten to {out}")
    return 0


def _cmd_export(args) -> int:
    cfg = load_site_config(args.config)
    run = Path(args.run)
    if not (run / "volume_full.npz").exists():
        print(f"no volume_full.npz under {run}; run `megido reconstruct` first")
        return 1
    p = export_volume(run, cfg, out_dir=args.out)
    print(f"exported {p} (+ meta.json)")
    return 0


def _cmd_view(args) -> int:
    import webbrowser

    from megido.viewerbuild import build

    out = build(Path("viewer"))
    print(f"viewer built: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _cmd_compare(args) -> int:
    paths = [Path(args.a) / "volume_full.npz", Path(args.b) / "volume_full.npz"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("missing volume_full.npz: " + ", ".join(missing))
        return 1
    a, b = (VoxelSolution.load(p) for p in paths)
    r = compare_volumes(a, b)
    print(f"correlation     {r['corr']:.6f}")
    print(f"rms delta       {r['rms_delta']:.6g}  (scale {r['scale']:.6g})")
    print(f"max |delta|     {r['max_abs_delta']:.6g}")
    print(f"total mass      {r['mass_change_frac']:+.3%}")
    print(f"verdict         {r['verdict']}")
    return 0


_HONESTY_NOTE = (
    "Angular silhouette (ridgeline elevation vs azimuth) per detector position; "
    "absolute distance/height NOT determined by the 2.2 m parallax; measured "
    "only where constrained open sky borders the hill (see az_coverage)."
)


def _cmd_hillside(args) -> int:
    cfg = load_site_config(args.config)
    baseline = Path(args.solve) / "baseline.npz"
    if not baseline.exists():
        print(f"no baseline.npz under {args.solve}; run `megido solve` first")
        return 1

    sol = BaselineSolution.load(baseline)
    result = extract_silhouette(sol, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "honesty_note": _HONESTY_NOTE,
        "agreement": result.agreement,
        "detectors": result.detectors,
        "per_pos": result.per_pos,
    }
    (out / "hill_silhouette.json").write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n")

    for pid in sorted(result.per_pos):
        p = result.per_pos[pid]
        finite = np.isfinite(p["ridge_elev"])
        if finite.any():
            lo, hi = np.min(p["ridge_elev"][finite]), np.max(p["ridge_elev"][finite])
            span = f"{lo:.1f}..{hi:.1f} deg"
        else:
            span = "n/a"
        print(f"{pid}: n_edge={p['n_edge']}  az_coverage={p['az_coverage']:.1%}  "
              f"ridge elev {span}")
    print(f"P0/P1 agreement (RMS ridge elevation, deg): {result.agreement:.3f}")

    png_path = out / "hill_silhouette.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped hill_silhouette.png")
        return 0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for pid in sorted(result.per_pos):
        p = result.per_pos[pid]
        finite = np.isfinite(p["ridge_elev"])
        if not finite.any():
            continue
        theta = np.radians(p["ridge_az"][finite])
        r = p["ridge_elev"][finite]
        ax.plot(theta, r, marker="o", markersize=3, linestyle="-", label=pid)

    for det in result.detectors:
        az = np.degrees(np.arctan2(det["y"], det["x"])) % 360.0
        ax.plot(np.radians(az), 90.0, marker="^", markersize=8,
                linestyle="none", label=f"{det['id']} ref az")

    ax.set_rlabel_position(135)
    ax.set_title("Megiddo hillside silhouette (ridgeline elevation vs azimuth)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.15), fontsize=8)

    caption = (f"angular only, no absolute distance; "
               f"P0/P1 agreement {result.agreement:.2f} deg; " +
               ", ".join(f"{pid} az_coverage={result.per_pos[pid]['az_coverage']:.0%}"
                         for pid in sorted(result.per_pos)))
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8, wrap=True)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"written {out / 'hill_silhouette.json'} and {png_path}")
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

    r = sub.add_parser("reconstruct", help="Phase 3 voxel inversion")
    r.add_argument("--config", default="configs/megido.yaml")
    r.add_argument("--solve", default="runs/solve", help="directory holding baseline.npz")
    r.add_argument("--run", default=None,
                   help="ingest directory; required for --bootstrap")
    r.add_argument("--out", default="runs/voxels")
    r.add_argument("--cache", default="runs/.cache")
    r.add_argument("--rebin", type=int, default=10)
    r.add_argument("--iters", type=int, default=5000,
                   help="baseline iterations per bootstrap replica")
    r.add_argument("--bootstrap", type=int, default=0,
                   help="Poisson replicas for per-voxel sigma; 0 disables")
    r.add_argument("--no-holdouts", action="store_true")
    r.add_argument("--no-systematic", action="store_true")
    r.add_argument("--backproject-z", type=float, default=None,
                   help="also write a model-free backprojection at this height (m)")
    r.set_defaults(func=_cmd_reconstruct)

    e = sub.add_parser("export", help="write volume.npy + meta.json for the viewer")
    e.add_argument("--config", default="configs/megido.yaml")
    e.add_argument("--run", default="runs/voxels")
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_export)

    vw = sub.add_parser("view", help="build and open the S5 viewer")
    vw.add_argument("--no-open", action="store_true",
                     help="build viewer/dist/index.html but don't open a browser")
    vw.set_defaults(func=_cmd_view)

    c = sub.add_parser("compare", help="what changed between two reconstructions")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.set_defaults(func=_cmd_compare)

    h = sub.add_parser("hillside", help="flux-edge hillside silhouette (S4)")
    h.add_argument("--config", default="configs/megido.yaml")
    h.add_argument("--solve", default="runs/solve", help="directory holding baseline.npz")
    h.add_argument("--out", default="runs/voxels")
    h.set_defaults(func=_cmd_hillside)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
