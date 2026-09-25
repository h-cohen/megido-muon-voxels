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
```

## Execution

Stages run in order. Each stage reads the previous stage's output directory.
Every command takes `--config` (default `configs/megido.yaml`; the cafeteria
campaign uses `configs/cafeteria.yaml`).

```bash
M="uv run python -m megido.cli"

$M validate                    # S0-det: falsify the supplied detector constants
$M ingest                      # S0-exp + S1: raw .data -> runs/ingest (angular histograms)
$M solve                       # S2: detector response + per-position opacity -> runs/solve
$M reconstruct --run runs/ingest --bootstrap 8   # S3/S4: voxel inversion + sigma -> runs/voxels
$M export                      # volume.npy + meta.json for the viewer
$M hillside --run runs/ingest  # S6: silhouette, GP surface, ray + cross-position checks
$M view                        # build viewer/dist/index.html and open it; Load run -> runs/voxels
$M compare --a runs/A --b runs/B   # what changed between two reconstructions
```

| Command | Options (default) |
|---|---|
| `validate` | `--exposure` (`P0`), `--max-chunks` (`1`) |
| `ingest` | `--out` (`runs/ingest`), `--force` (ignore the cache), `--chunksize` (`50000`; keep it bounded, the full ingest is ~6 GB) |
| `solve` | `--run` (`runs/ingest`), `--out` (`runs/solve`), `--rebin` angular rebin (`10`), `--iters` (`5000`) |
| `reconstruct` | `--solve` (`runs/solve`), `--run` ingest dir (required for `--bootstrap`), `--out` (`runs/voxels`), `--cache` (`runs/.cache`), `--rebin` (`10`), `--iters` per replica (`5000`), `--bootstrap N` Poisson replicas for per-voxel σ (`0` = off), `--no-holdouts`, `--no-systematic`, `--backproject-z Z` extra model-free backprojection at height Z m |
| `export` | `--run` (`runs/voxels`), `--out` (same as `--run`) |
| `hillside` | `--solve` (`runs/solve`), `--run` ingest dir, enables Poisson weights (`runs/ingest`), `--out` (`runs/voxels`), `--rebin` (`10`, match the solve), `--surface-a` assumed inverse-density scale (`8.0`; sets absolute height, not shape), `--surface-cell` m (`1.0`), `--surface-qhi` upper-envelope quantile (`0.85`), `--surface-min-count` (`8`), `--surface-max-points` GP rays (`1000`), `--surface-restarts` (`3`) |
| `view` | `--no-open` (build only) |
| `compare` | `--a`, `--b` (two run dirs, required) |

Viewer controls are all display-only: fog or voxel-cube render, opacity,
colormap and window, clip box/plane/slice, σ / coverage / SNR gates, the
hillside surface (coloured by σ or ray residual), clipping the volume above
the surface, detectors, and saved views.

## Tests

```bash
uv run pytest -q        # Python + headless-browser viewer tests (run one session at a time)
node --test viewer/test/*.test.mjs   # viewer JS unit tests
```
