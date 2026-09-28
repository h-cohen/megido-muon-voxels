# megido-muon-voxels

Muon tomography from a detector placed at several positions and tilts inside a
cavern. From the muon rate in each angular bin, the pipeline reconstructs:

- a **3D voxel opacity field** of the rock above (lateral structure; depth is
  not resolved by this geometry);
- the **hillside surface** `H(x,y)`, a Gaussian-process fit to the hill's flux
  edge with posterior σ, checked in ray space and out-of-sample;
- one self-contained **HTML viewer** for both, with honest uncertainty.

There is no open-sky calibration run at Megiddo. Detector response and rock
absorption are separated using the tilts alone. A second campaign (TAU
cafeteria, ROOT histograms plus an open-sky run) uses the same pipeline:
[`docs/cafeteria-run.md`](docs/cafeteria-run.md). For the physics, limits and
conventions, see [`CLAUDE.md`](CLAUDE.md) and
`docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run playwright install chromium   # headless browser for the viewer tests
```

## Execution

Everything a run needs is in one **site config** (`configs/*.yaml`) plus the
data folder it points at. Each stage reads the previous stage's output folder.
All commands take `--config`, which defaults to `configs/megido.yaml`.

### 1. Site config

| Key | Meaning |
|---|---|
| `site` | name (`megido`, `cafeteria`) |
| `data_dir` | folder holding the raw `DET<run>_*.data` files or the ROOT histograms |
| `frame.origin`, `frame.x_axis_bearing_deg` | the exposure id at the origin, and the compass bearing of site +x |
| `binning.t_max`, `binning.n_bins` | angular grid in tan θ: ±t_max, n_bins per axis (1.25, 500) |
| `volume.z_min_m`, `z_max_m`, `spacing_m` | the voxel lattice: height range and cubic voxel size (1–12 m, 0.25 m) |
| `volume.xy_m` | optional fixed lateral box `[[x0,x1],[y0,y1]]`; default is the ray footprint |
| `volume.viewer_crop_xy_m` | optional initial viewer clip box; display only, the solve box is unchanged |
| `volume.n_aperture_sub` | sub-rays per axis across the detector aperture (4) |
| `reconstruction.algorithm` | `tv` (default) or `sirt` |
| `reconstruction.n_iter`, `tv_alpha`, `tv_z_weight` | iterations, TV strength (fraction of p95), relative z smoothing |
| `reconstruction.coverage_damping` | pull each voxel toward 0 ∝ 1/coverage (0 = off); removes the noise shell |
| `reconstruction.nonneg`, `chi2_target`, `seed` | non-negativity, SIRT stop, RNG seed |
| `exposures[]` | one block per detector placement (below) |
| `sky_reference` | optional open-sky run `{id, root_file}`: the solve divides by it instead of the tilt-based joint solve |
| `detector` | optional measured geometry of a detector other than the Megiddo unit `{active_width_cm, layer_dz_cm, source}` |

Each exposure block has:
- **`id`**.
- **`pose`**: `{x, y, z}` in metres in the site frame, plus `tilt_deg` from zenith and `az_deg`, the bearing of the bar axis.
- **Data source, either:**
  - `runs: DET200084-DET200104`: a raw run range. Missing run ids are skipped.
  - `root_file:`: a pre-binned ROOT `txty` histogram, with `root_hist` as its name.
- **Optional:**
  - `pose_sigma: {xy, ang}`;
  - `norm_group`: exposures sharing a rate normalisation; defaults to the id;
  - `note`.

**Adding a datapoint** needs no code change: put the files in `data_dir`, append
one exposure block, and re-run from `ingest`. Poses and tunables live only in
the config. The two shipped configs show both styles:

- **`configs/megido.yaml`**: raw `.data` runs, four exposures (P0, two 20° tilts, and P1 translated 2.2 m), no sky run.
- **`configs/cafeteria.yaml`**: ROOT histograms for two positions plus `sky_reference`, a measured `detector` block and a `viewer_crop_xy_m`. The file's comments record why each tuned value was chosen.

### 2. Run the pipeline

Megiddo (the defaults write to `runs/`):

```bash
M="uv run python -m megido.cli"
$M validate --exposure P0                       # S0-det: falsify the detector constants (raw .data only)
$M ingest                                       # S0-exp + S1 -> runs/ingest
$M solve                                        # S2 -> runs/solve
$M reconstruct --run runs/ingest --bootstrap 8  # S3/S4 -> runs/voxels
$M export                                       # viewer files -> runs/voxels
$M hillside --run runs/ingest                   # S6 hillside surface -> runs/voxels
$M view                                         # build + open viewer; "Load run" -> runs/voxels
```

Cafeteria (`ingest` switches to the ROOT reader and `solve` to the sky-run
reference automatically, from the config):

```bash
C=configs/cafeteria.yaml; R=runs/cafeteria
$M ingest      --config $C --out $R/ingest
$M solve       --config $C --run $R/ingest --out $R/solve
$M reconstruct --config $C --solve $R/solve --run $R/ingest --out $R/voxels --cache $R/.cache --bootstrap 8 --backproject-z 7.0
$M export      --config $C --run $R/voxels
$M hillside    --config $C --surface-a 48 --solve $R/solve --run $R/ingest --out $R/voxels
$M view                                          # "Load run" -> runs/cafeteria/voxels
```

A new campaign follows the same pattern: copy a config, point `data_dir` at
its data, give every stage the same `--config`, and use its own `runs/<name>/`
folders.

To compare two reconstructions, run `$M compare --a runs/A/voxels --b runs/B/voxels`. In the
viewer, "Load compare" shows B minus A.

### 3. What each stage writes

| Stage | Folder | Files |
|---|---|---|
| `ingest` | `runs/ingest` | `counts_<exp>.npz` (angular histograms), `calib_<exp>.npz` and `tracks_<exp>.parquet` (raw runs only), `meta.json` (per-exposure summary, live times) |
| `solve` | `runs/solve` | `baseline.npz` (detector response, per-position opacity, gauge) |
| `reconstruct` | `runs/voxels` | `volume_full.npz`, `volume_holdout_<pos>.npz`, `views.npy`, `rays.npy` (rays per voxel), `uncertainty.npz` (with `--bootstrap`), `systematic.npy`, `backprojection.npy` (with `--backproject-z`) |
| `export` | `runs/voxels` | `volume.npy`, `sigma.npy` / `snr.npy` (from the bootstrap), `meta.json` (grid, layers, detectors, resolution verdict) |
| `hillside` | `runs/voxels` | `hill_silhouette.json/.png`, `hill_surface.npy`, `hill_surface_sigma.npy`, `hill_residual_grid.npy`, `hill_surface_meta.json` (fit, ray check, cross-position check), `hill_surface.png`, `hill_residual.png` |

Point the viewer at the folder that holds `meta.json`. Optional files (σ,
SNR, rays, hillside) switch their controls on when present.

### 4. Options

| Command | Options (default) |
|---|---|
| all | `--config` (`configs/megido.yaml`) |
| `validate` | `--exposure` (`P0`), `--max-chunks` (`1`) |
| `ingest` | `--out` (`runs/ingest`), `--force` (rebuild despite the cache), `--chunksize` (`50000`; keep it bounded, the full ingest is ~6 GB) |
| `solve` | `--run` (`runs/ingest`), `--out` (`runs/solve`), `--rebin` angular rebin (`10`), `--iters` (`5000`) |
| `reconstruct` | `--solve` (`runs/solve`), `--run` ingest dir (required for `--bootstrap`), `--out` (`runs/voxels`), `--cache` (`runs/.cache`), `--rebin` (`10`, match the solve), `--iters` per bootstrap replica (`5000`), `--bootstrap N` Poisson replicas for per-voxel σ (`0` = off), `--no-holdouts` (skip the leave-one-position-out solves), `--no-systematic`, `--backproject-z Z` (extra model-free backprojection at height Z m) |
| `export` | `--run` (`runs/voxels`), `--out` (same as `--run`) |
| `hillside` | `--solve` (`runs/solve`), `--run` ingest dir, which enables Poisson weights (`runs/ingest`), `--out` (`runs/voxels`), `--rebin` (`10`, match the solve), `--surface-a` assumed inverse-density scale (`8.0`; sets absolute height, not shape), `--surface-cell` m (`1.0`), `--surface-qhi` upper-envelope quantile (`0.85`), `--surface-min-count` (`8`), `--surface-max-points` GP rays (`1000`), `--surface-restarts` (`3`) |
| `view` | `--no-open` (build `viewer/dist/index.html` only) |
| `compare` | `--a`, `--b` (two voxel run dirs, required) |

**Caching:**
- `ingest` artifacts are keyed on the exposure, pose, binning, input files and
  `RECONSTRUCTION_VERSION`, so a changed config rebuilds only what it affects.
- `reconstruct` caches forward models in `--cache`, keyed with `INVERSION_VERSION`.
- Bump the version when changing reconstruction logic.

**Viewer controls** are display-only:
- fog or voxel-cube render, opacity, colormap and window;
- clip box, plane or slice;
- σ / coverage / SNR gates;
- the hillside surface, coloured by σ or by ray residual;
- clipping the volume above the surface;
- detectors and saved views.

## Tests

```bash
uv run pytest -q        # Python + headless-browser viewer tests (run one session at a time)
node --test viewer/test/*.test.mjs   # viewer JS unit tests
```
