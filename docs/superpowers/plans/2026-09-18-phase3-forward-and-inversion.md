# Phase 3 — Forward Model and Voxel Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 2 per-position sky opacity maps into a 3D voxel opacity field with honest, first-class uncertainty.

**Architecture:** Phase 2 already produced `lambda_p(sky direction)` — a line integral of opacity density along a ray leaving detector position `p` in a world-frame direction. That is exactly the tomographic measurement, so Phase 3 is `A x = lambda`: build a sparse path-length matrix `A` over a voxel grid, solve for `x >= 0` with SIRT plus an anisotropic-TV proximal step, and quantify what the two-position geometry can and cannot resolve. Ported in structure from `cafeteria_3d_modeling/muontomo`, with the pose rotation removed (Phase 2 already applied it) and the cafeteria's ceiling-beam machinery dropped.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, PyYAML, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` — section 7 (S3/S4), section 9 (incremental execution), section 11 (open items).

**Reference source (read-only, do not import):** `/home/hadar/Cloud/Work/Postdoc/03_projects/cafeteria_3d_modeling/muontomo/` — `geometry.py`, `raycast.py`, `forward.py`, `reconstruct.py`, `backproject.py`, `export.py`, `compare.py`. That project is a separate repo; copy the ideas and the maths, never add it as a dependency.

## Global Constraints

- **Lengths are metres in Phase 3.** Phase 1 and `megido/detector.py` are centimetres. Every Phase 3 module works in metres and converts at the boundary. `Pose.x/y/z` are already metres.
- **Poses live in data, never in code** (spec §3). Any new tunable goes in `configs/megido.yaml`, not as a module-level constant.
- **Never hardcode the aperture.** It is `DetectorGeometry.megiddo().bar.active_width_cm / 100.0 = 0.384 m`. Add a property; do not retype the number.
- **Any change to reconstruction logic must bump a version constant that is part of the artifact cache key** (spec §11, and the Phase 1 defect that motivated it). Phase 3 introduces `INVERSION_VERSION` and it must appear in the system-matrix cache key and in saved volumes.
- **Opacity fed to the solver is `BaselineSolution.normalized_opacity(pid)`**, not `.opacity[pid]`. The raw map is gauge-pinned to median zero and is half negative; it is meaningless as an absolute optical depth.
- **Sky bins with no finite opacity are not rows.** `np.nan` in `sol.opacity[pid]` means "not constrained"; such a bin must never reach the matrix.
- **No `import uproot`, no ROOT, no network access in tests.**
- **Tests must run in well under a minute each.** The real-data run is a CLI invocation in Task 12, not a test.
- Run tests with `uv run pytest`. Commit after every task with the repo's existing trailer convention.

---

## File Structure

| File | Responsibility |
|---|---|
| `megido/config.py` (modify) | Add `Volume` and `Reconstruction` dataclasses, read from new optional YAML blocks. |
| `megido/detector.py` (modify) | Add `aperture_m` property. |
| `megido/voxels.py` (create) | `VoxelGrid` and `auto_grid` — the world-frame voxel lattice. |
| `megido/fitdata.py` (create) | `RowIndex`, `FitData`, `build_fit_data`, `position_origins` — turns a `BaselineSolution` into the measurement vector and defines the row ordering everything else follows. |
| `megido/raycast.py` (create) | `build_system_matrix` — sparse path lengths, ray bundles, SHA disk cache. |
| `megido/forward.py` (create) | `ForwardModel` — `A` plus the bookkeeping to go volume ↔ per-position maps. |
| `megido/inversion.py` (create) | `sirt`, `sirt_tv`, the TV proximal operator. Pure algorithms, no I/O. |
| `megido/reconstruct.py` (create) | `VoxelSolution`, `solve_voxels`, per-position holdout fits. Orchestration and persistence. |
| `megido/phantom.py` (create) | Synthetic truth volumes (from `topography.csv` and from a synthetic cavern), forward projection, round-trip scoring. |
| `megido/resolution.py` (create) | Analytic depth resolution `δz`, the aliasing period, and the along-ray degeneracy check. |
| `megido/voxuncert.py` (create) | Poisson bootstrap over counts → per-voxel σ; the S2 null-space direction as a separate systematic map. |
| `megido/backproject.py` (create) | Model-free single-plane backprojection — the anchor the inversion is judged against. |
| `megido/volexport.py` (create) | `volume.npy` + `meta.json` viewer data contract, and `compare_volumes`. |
| `megido/cli.py` (modify) | `reconstruct`, `export`, `compare` subcommands. |
| `tests/test_voxels.py`, `test_fitdata.py`, `test_raycast.py`, `test_forward.py`, `test_inversion.py`, `test_reconstruct.py`, `test_phantom.py`, `test_resolution.py`, `test_voxuncert.py`, `test_backproject.py`, `test_volexport.py`, `test_cli3.py` | One test module per implementation module. |

---

## Rulings carried into this plan

These resolve conflicts between the spec and what Phase 2 actually built. Each is binding on implementers.

1. **Spec §7.1 says to thread the pose rotation through `bin_directions`. Phase 3 does not.** Phase 2's `detector_to_sky` already applied `Rz(az)·Ry(tilt)`, so `sol.opacity[pid]` is indexed by *world-frame* sky tangents. Re-applying the rotation would rotate twice. Rows are `(position, sky bin)`, and a row's direction is simply `normalize(sx, sy, 1)`. *Cost if wrong: every ray points in the wrong direction and the reconstruction is meaningless — which the Task 7 phantom gate would catch immediately.*
2. **The ray bundle is a square of side `aperture_m` perpendicular to each ray, not the detector's physical aperture plane.** A row aggregates exposures at different tilts, so there is no single aperture orientation to use. Ray-perpendicular is pose-free and correct to first order in tilt. *Cost if wrong: the bundle cross-section is overestimated by `1/cos(tilt)` ≈ 6% at 20°, blurring slightly more than reality.*
3. **`mlem_transmission` is not ported.** It needs an expected open-sky count map, which is precisely what this campaign does not have. *Cost if wrong: we lose a Poisson-exact solver; SIRT-TV on `lambda` with bootstrap weights is the spec's chosen algorithm (§7.3) anyway.*
4. **`layered_fit`, `focus.py` and `beams.py` are not ported.** They fit a thin horizontal ceiling layer and locate periodic ceiling beams — cafeteria-specific. The spec's unknown here is a full voxel field (§12 decision record). The depth question they answered is answered instead by `resolution.py` and the Task 8 gate. *Cost if wrong: no autofocus height estimate; `resolution.py` must therefore carry the alias check §7.4 requires, and it does.*
5. **`uncertainty.py` is not ported verbatim despite spec §1.6 listing it.** Its content is beam positions and autofocus CIs. Only `resample_tmaps`' idea survives: Poisson-resample counts, re-run the whole chain, take the spread. *Cost if wrong: none identified; the spec's actual §7.4 requirements are each implemented.*
6. **The Task 7 phantom uses `topography.csv`, whose geometry is nothing like Megiddo** — three detectors 50 m apart under a 160 m surface, with an open-sky calibration file we must ignore. It is a *code-correctness* gate under favourable conditions, not evidence about Megiddo. Task 8 is the Megiddo-geometry gate and is the one that constrains claims about real results. *Cost if wrong: passing Task 7 alone would be mistaken for "the method works here". Task 8 exists to prevent exactly that.*
7. **Grid spacing defaults to 0.25 m, not the spec's 0.10 m starting value.** The spec's number was reasoned from the cafeteria's 3 m-away ceiling; Megiddo's rock is at roughly 5–15 m (measured λ ≈ 1.5 implies of order 10 m.w.e. of overburden), so a 0.10 m lattice over that range is ~5·10⁷ voxels against ~2·10⁴ measurements. *Cost if wrong: features below 0.5 m are not representable; `resolution.py` reports the achievable resolution so the value can be retuned from evidence, and it is a config field.*

---

### Task 1: Voxel grid and the volume/reconstruction config blocks

**Files:**
- Create: `megido/voxels.py`
- Modify: `megido/config.py` (add `Volume`, `Reconstruction`, wire into `SiteConfig` and `load_site_config`)
- Modify: `megido/detector.py` (add `aperture_m` property to `DetectorGeometry`)
- Modify: `configs/megido.yaml` (add `volume:` and `reconstruction:` blocks)
- Test: `tests/test_voxels.py`, and add cases to `tests/test_config.py`, `tests/test_detector.py`

**Interfaces:**
- Consumes: `megido.config.SiteConfig`, `megido.detector.DetectorGeometry`.
- Produces:
  - `megido.config.Volume(z_min_m: float, z_max_m: float, spacing_m: float, xy_m: tuple | None, n_aperture_sub: int)`
  - `megido.config.Reconstruction(algorithm: str, n_iter: int, nonneg: bool, chi2_target: float, tv_alpha: float, tv_z_weight: float, seed: int)`
  - `SiteConfig.volume: Volume` and `SiteConfig.reconstruction: Reconstruction`, both with defaults so existing configs keep loading
  - `DetectorGeometry.aperture_m -> float`
  - `megido.voxels.VoxelGrid(origin, spacing, shape)` with `n_voxels`, `axis_centers(axis)`, `extent(axis)`, `key()`
  - `megido.voxels.auto_grid(vol: Volume, origins: dict[str, tuple[float, float, float]], t_reach: float, *, aperture_m: float) -> VoxelGrid`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voxels.py`:

```python
import numpy as np
import pytest

from megido.config import Volume
from megido.voxels import VoxelGrid, auto_grid


def test_grid_reports_its_size_and_centers():
    g = VoxelGrid(origin=(0.0, -1.0, 2.0), spacing=0.5, shape=(4, 2, 3))
    assert g.n_voxels == 24
    np.testing.assert_allclose(g.axis_centers(0), [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(g.axis_centers(1), [-0.75, -0.25])
    assert g.extent(2) == (2.0, 3.5)


def test_grid_key_separates_different_grids():
    a = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.5, shape=(2, 2, 2))
    b = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.25, shape=(2, 2, 2))
    assert a.key() != b.key()


def test_auto_grid_covers_every_ray_footprint():
    """A ray of tangent t leaving x=x0 reaches x0 + t*z at height z, and the
    bundle adds half an aperture. The grid must contain the union over poses."""
    vol = Volume(z_min_m=1.0, z_max_m=5.0, spacing_m=0.5)
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (2.2, 0.0, 0.0)}
    g = auto_grid(vol, origins, t_reach=1.0, aperture_m=0.384)

    x0, x1 = g.extent(0)
    assert x0 <= -5.0 - 0.192 + 1e-9     # pos0, t = -1 at z = 5, minus half aperture
    assert x1 >= 2.2 + 5.0 + 0.192 - 1e-9
    assert g.extent(2) == pytest.approx((1.0, 5.0), abs=0.5)


def test_auto_grid_honours_an_explicit_xy_box():
    vol = Volume(z_min_m=1.0, z_max_m=3.0, spacing_m=0.5,
                 xy_m=((-2.0, 2.0), (-1.0, 1.0)))
    g = auto_grid(vol, {"pos0": (0.0, 0.0, 0.0)}, t_reach=10.0, aperture_m=0.384)
    assert g.origin[0] == pytest.approx(-2.0)
    assert g.shape[:2] == (8, 4)


def test_auto_grid_rejects_an_inverted_z_range():
    vol = Volume(z_min_m=5.0, z_max_m=1.0, spacing_m=0.5)
    with pytest.raises(ValueError, match="z_max_m"):
        auto_grid(vol, {"pos0": (0.0, 0.0, 0.0)}, t_reach=1.0, aperture_m=0.384)
```

Add to `tests/test_config.py`:

```python
def test_site_config_supplies_volume_and_reconstruction_defaults(tmp_path):
    """Configs written before Phase 3 must still load."""
    from megido.config import load_site_config

    p = tmp_path / "old.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert cfg.volume.spacing_m == 0.25
    assert cfg.reconstruction.algorithm == "tv"


def test_site_config_reads_volume_and_reconstruction_blocks(tmp_path):
    from megido.config import load_site_config

    p = tmp_path / "new.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {z_min_m: 2.0, z_max_m: 9.0, spacing_m: 0.5, n_aperture_sub: 3}\n"
        "reconstruction: {n_iter: 40, tv_alpha: 0.02}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert (cfg.volume.z_min_m, cfg.volume.z_max_m) == (2.0, 9.0)
    assert cfg.volume.n_aperture_sub == 3
    assert cfg.reconstruction.n_iter == 40
    assert cfg.reconstruction.tv_z_weight == 0.5   # untouched default


def test_volume_xy_box_survives_yaml_round_trip(tmp_path):
    from megido.config import load_site_config

    p = tmp_path / "box.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {xy_m: [[-3.0, 3.0], [-2.0, 2.0]]}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert cfg.volume.xy_m == ((-3.0, 3.0), (-2.0, 2.0))
```

Add to `tests/test_detector.py`:

```python
def test_aperture_is_the_active_width_in_metres():
    from megido.detector import DetectorGeometry

    geom = DetectorGeometry.megiddo()
    assert geom.aperture_m == pytest.approx(geom.bar.active_width_cm / 100.0)
    assert geom.aperture_m == pytest.approx(0.384)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_voxels.py tests/test_config.py tests/test_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.voxels'` and `AttributeError`/`TypeError` on the config and detector cases.

- [ ] **Step 3: Add the detector aperture property**

In `megido/detector.py`, inside `DetectorGeometry`, directly above `max_tan`:

```python
    @property
    def aperture_m(self) -> float:
        """Active transverse width in METRES. Phase 3 works in metres; every
        other length in this module is centimetres."""
        return self.bar.active_width_cm / 100.0
```

- [ ] **Step 4: Add the config dataclasses**

In `megido/config.py`, after the `Binning` dataclass:

```python
@dataclass(frozen=True)
class Volume:
    """The voxel lattice the Phase 3 inversion solves on.

    `spacing_m` defaults to 0.25 rather than the spec's 0.10 starting value:
    Megiddo's rock sits at roughly 5-15 m, not the cafeteria's 3 m ceiling, so a
    0.10 m lattice over that range is about 5e7 unknowns against 2e4
    measurements. megido.resolution reports what is actually resolvable, and
    this is a config field precisely so it can be retuned from that evidence.
    """
    z_min_m: float = 1.0
    z_max_m: float = 12.0
    spacing_m: float = 0.25
    xy_m: tuple | None = None       # ((x0, x1), (y0, y1)); None -> from ray footprints
    n_aperture_sub: int = 4         # sub-rays per axis across the aperture (n^2 total)


@dataclass(frozen=True)
class Reconstruction:
    algorithm: str = "tv"           # "sirt" | "tv"
    n_iter: int = 150
    nonneg: bool = True
    chi2_target: float = 1.0        # discrepancy-principle stop for plain SIRT
    tv_alpha: float = 0.01          # TV threshold as a fraction of x's p95
    tv_z_weight: float = 0.5        # anisotropic TV: relative weight of z gradients
    seed: int = 42
```

Add the two fields to `SiteConfig` (after `binning`), each with a default so pre-Phase-3 configs still load:

```python
    volume: Volume = field(default_factory=Volume)
    reconstruction: Reconstruction = field(default_factory=Reconstruction)
```

In `load_site_config`, before the `return SiteConfig(...)`:

```python
    vol_raw = dict(raw.get("volume", {}))
    if vol_raw.get("xy_m") is not None:
        vol_raw["xy_m"] = tuple(tuple(float(v) for v in pair) for pair in vol_raw["xy_m"])
    volume = Volume(**vol_raw)
    reconstruction = Reconstruction(**raw.get("reconstruction", {}))
```

and pass `volume=volume, reconstruction=reconstruction` to the `SiteConfig(...)` call.

- [ ] **Step 5: Write `megido/voxels.py`**

```python
"""The world-frame voxel lattice the inversion solves on.

World frame = the sky frame Phase 2 already rotated into: z up, lengths in
METRES, origin at the P0 detector (configs/megido.yaml `frame.origin`). Phase 1
and megido.detector work in centimetres; nothing in this module does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import Volume


@dataclass(frozen=True)
class VoxelGrid:
    origin: tuple          # (x0, y0, z0) of the grid CORNER, metres
    spacing: float         # cubic voxel edge, metres
    shape: tuple           # (nx, ny, nz)

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def axis_centers(self, axis: int) -> np.ndarray:
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * self.spacing

    def extent(self, axis: int) -> tuple[float, float]:
        return self.origin[axis], self.origin[axis] + self.shape[axis] * self.spacing

    def key(self) -> str:
        return f"{self.origin}-{self.spacing}-{self.shape}"


def auto_grid(vol: Volume, origins: dict[str, tuple[float, float, float]],
              t_reach: float, *, aperture_m: float) -> VoxelGrid:
    """Lattice covering the union of every position's ray footprint.

    `t_reach` is the largest |tangent| that carries a constrained measurement —
    supplied by the caller from the live rows, NOT the sky grid's nominal edge.
    The sky grid runs to |t| = 2.5 to catch stray counts; sizing the voxel grid
    by that would quadruple it to hold bins nothing constrains.

    `aperture_m` is required rather than defaulted: the ray bundle's half-width
    is what pads the grid, and a default would have to name a specific detector,
    putting site knowledge in a module that otherwise has none.
    """
    if vol.z_max_m <= vol.z_min_m:
        raise ValueError(f"z_max_m ({vol.z_max_m}) must exceed z_min_m ({vol.z_min_m})")
    if not origins:
        raise ValueError("auto_grid needs at least one position origin")

    z0, z1 = float(vol.z_min_m), float(vol.z_max_m)
    if vol.xy_m is not None:
        (x0, x1), (y0, y1) = vol.xy_m
    else:
        # A ray of tangent t leaving (px, py, pz) is at px + t*(z1 - pz) by the
        # top of the grid; the bundle spreads half an aperture either side.
        pad = 0.5 * aperture_m
        xs, ys = [], []
        for px, py, pz in origins.values():
            reach = t_reach * max(z1 - pz, 0.0) + pad
            xs += [px - reach, px + reach]
            ys += [py - reach, py + reach]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    sp = float(vol.spacing_m)
    nx = max(1, int(np.ceil((x1 - x0) / sp)))
    ny = max(1, int(np.ceil((y1 - y0) / sp)))
    nz = max(1, int(np.ceil((z1 - z0) / sp)))
    return VoxelGrid(origin=(float(x0), float(y0), z0), spacing=sp, shape=(nx, ny, nz))
```

- [ ] **Step 6: Add the config blocks to `configs/megido.yaml`**

Insert after the `binning:` block:

```yaml
volume:
  z_min_m: 1.0
  z_max_m: 12.0
  spacing_m: 0.25
  n_aperture_sub: 4
  # xy_m: [[-12.0, 12.0], [-12.0, 12.0]]   # uncomment to pin the lateral box

reconstruction:
  algorithm: tv
  n_iter: 150
  tv_alpha: 0.01
  tv_z_weight: 0.5
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_voxels.py tests/test_config.py tests/test_detector.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite to confirm nothing regressed**

Run: `uv run pytest -q`
Expected: every pre-existing test still passes (176 before this task).

- [ ] **Step 9: Commit**

```bash
git add megido/voxels.py megido/config.py megido/detector.py configs/megido.yaml \
        tests/test_voxels.py tests/test_config.py tests/test_detector.py
git commit -m "feat: voxel grid and the Phase 3 volume/reconstruction config blocks"
```

---

### Task 2: Measurement vector and row index from a BaselineSolution

This task defines the row ordering that every later task follows. `A`, the weights, the holdout masks and the saved per-position maps all key off `RowIndex`.

**Files:**
- Create: `megido/fitdata.py`
- Test: `tests/test_fitdata.py`

**Interfaces:**
- Consumes: `megido.baseline.BaselineSolution` (`.opacity`, `.normalized_opacity(pid)`, `.sky`), `megido.baseline.position_ids`, `megido.config.SiteConfig`, `megido.sky.SkyGrid`.
- Produces:
  - `position_origins(cfg: SiteConfig) -> dict[str, tuple[float, float, float]]`
  - `RowIndex(position_ids: tuple[str, ...], pos_of_row: np.ndarray, sx: np.ndarray, sy: np.ndarray, sky_flat: np.ndarray)` with `n_rows`, `mask_for(pid) -> np.ndarray[bool]`, `t_reach() -> float`, `key() -> str`
  - `FitData(lam: np.ndarray, w: np.ndarray, rows: RowIndex)` with `restricted(keep) -> FitData`
  - `build_fit_data(sol, cfg, *, sigma: dict[str, np.ndarray] | None = None, max_tan: float = 1.25, transparent_quantile: float = 0.05) -> FitData`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fitdata.py`:

```python
import numpy as np
import pytest

from megido.baseline import BaselineSolution
from megido.basis import make_smooth_basis
from megido.angular import AnalysisGrid
from megido.config import load_site_config
from megido.fitdata import RowIndex, build_fit_data, position_origins
from megido.sky import SkyGrid


CONFIG = """
site: t
data_dir: /tmp
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 0}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _solution(opacity: dict[str, np.ndarray], n_bins: int) -> BaselineSolution:
    sky = SkyGrid(edges=np.linspace(-2.5, 2.5, n_bins + 1))
    return BaselineSolution(
        coeffs=np.zeros(1), opacity=opacity, norms={}, flux_index=2.0,
        nll_history=[0.0], grid=AnalysisGrid(edges=np.linspace(-1, 1, 3), counts={}),
        sky=sky, basis=make_smooth_basis(t_max=1.25, n_per_axis=2),
    )


def test_positions_share_an_origin_when_they_share_a_translation(tmp_path):
    origins = position_origins(_cfg(tmp_path))
    assert origins == {"pos0": (0.0, 0.0, 0.0), "pos1": (2.2, 0.0, 0.0)}


def test_only_finite_opacity_becomes_a_row(tmp_path):
    n = 4
    lam0 = np.full(n * n, np.nan)
    lam0[[0, 5, 10]] = [0.5, 1.0, 1.5]
    lam1 = np.full(n * n, np.nan)
    lam1[[5]] = [2.0]
    sol = _solution({"pos0": lam0, "pos1": lam1}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    assert data.rows.n_rows == 4
    assert data.rows.position_ids == ("pos0", "pos1")
    assert data.rows.mask_for("pos0").sum() == 3
    assert data.rows.mask_for("pos1").sum() == 1
    assert np.all(np.isfinite(data.lam))


def test_rows_carry_the_physical_opacity_not_the_gauge_pinned_map(tmp_path):
    """normalized_opacity shifts the 5th percentile to zero and clips; the raw
    map is median-pinned and half negative, so it must never reach the rows."""
    n = 4
    lam = np.full(n * n, np.nan)
    lam[:8] = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    assert data.lam.min() >= 0.0
    np.testing.assert_allclose(data.lam.max(), sol.normalized_opacity("pos0")[7])


def test_the_transparent_quantile_changes_the_gauge(tmp_path):
    """The gauge convention is a choice, and voxuncert varies it to price that
    choice. Adding a constant to the opacity map would NOT change the answer —
    normalized_opacity subtracts its own quantile — so the knob has to be the
    quantile itself."""
    n = 4
    lam = np.full(n * n, np.nan)
    lam[:8] = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    sol = _solution({"pos0": lam}, n)

    a = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0, transparent_quantile=0.05)
    b = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0, transparent_quantile=0.25)
    assert not np.allclose(a.lam, b.lam)

    shifted = _solution({"pos0": lam + 7.0}, n)
    c = build_fit_data(shifted, _cfg(tmp_path), max_tan=10.0)
    np.testing.assert_allclose(a.lam, c.lam)   # shift-invariant, as documented


def test_rows_outside_max_tan_are_dropped(tmp_path):
    """The sky grid runs to |t| = 2.5 to catch stray counts. Directions past the
    detector's own acceptance carry almost no tracks and would stretch the voxel
    grid enormously, so the caller caps them."""
    n = 10
    sol = _solution({"pos0": np.zeros(n * n)}, n)

    wide = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    narrow = build_fit_data(sol, _cfg(tmp_path), max_tan=0.5)
    assert narrow.rows.n_rows < wide.rows.n_rows
    assert np.abs(narrow.rows.sx).max() <= 0.5
    assert np.abs(narrow.rows.sy).max() <= 0.5


def test_weights_come_from_sigma_when_given(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1]] = [1.0, 2.0]
    sig = np.full(n * n, np.nan)
    sig[[0, 1]] = [0.5, 0.25]
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), sigma={"pos0": sig}, max_tan=10.0)
    np.testing.assert_allclose(np.sort(data.w), np.sort([1 / 0.25, 1 / 0.0625]))


def test_a_row_with_no_usable_sigma_is_dropped_not_given_infinite_weight(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1]] = [1.0, 2.0]
    sig = np.full(n * n, np.nan)
    sig[[0, 1]] = [0.5, 0.0]        # zero sigma would be an infinite weight
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), sigma={"pos0": sig}, max_tan=10.0)
    assert data.rows.n_rows == 1
    assert np.all(np.isfinite(data.w))


def test_uniform_weights_when_no_sigma_is_supplied(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1, 2]] = [1.0, 2.0, 3.0]
    sol = _solution({"pos0": lam}, n)

    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
    np.testing.assert_allclose(data.w, 1.0)


def test_restricted_zeroes_weights_and_keeps_the_row_layout(tmp_path):
    n = 4
    lam = np.full(n * n, np.nan)
    lam[[0, 1, 2]] = [1.0, 2.0, 3.0]
    sol = _solution({"pos0": lam}, n)
    data = build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)

    keep = np.array([True, False, True])
    r = data.restricted(keep)
    assert r.rows is data.rows
    np.testing.assert_allclose(r.lam, data.lam)
    np.testing.assert_allclose(r.w, [1.0, 0.0, 1.0])


def test_t_reach_is_the_largest_live_tangent(tmp_path):
    n = 10
    sol = _solution({"pos0": np.zeros(n * n)}, n)
    data = build_fit_data(sol, _cfg(tmp_path), max_tan=0.8)
    assert data.rows.t_reach() == pytest.approx(
        max(np.abs(data.rows.sx).max(), np.abs(data.rows.sy).max()))


def test_row_index_key_changes_with_the_rows(tmp_path):
    # At n = 6 the sky centres are +-0.4167, +-1.25, +-2.083. max_tan must
    # straddle a tier boundary to change the row set at all: 1.0 keeps the
    # innermost tier, 1.3 keeps two. Anything below 0.4167 keeps NOTHING and
    # raises the "no constrained" ValueError.
    n = 6
    a = build_fit_data(_solution({"pos0": np.zeros(n * n)}, n), _cfg(tmp_path), max_tan=1.0)
    b = build_fit_data(_solution({"pos0": np.zeros(n * n)}, n), _cfg(tmp_path), max_tan=1.3)
    assert a.rows.key() != b.rows.key()


def test_an_empty_solution_is_an_error_not_a_zero_row_matrix(tmp_path):
    n = 4
    sol = _solution({"pos0": np.full(n * n, np.nan)}, n)
    with pytest.raises(ValueError, match="no constrained"):
        build_fit_data(sol, _cfg(tmp_path), max_tan=10.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_fitdata.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.fitdata'`.

- [ ] **Step 3: Write `megido/fitdata.py`**

```python
"""The measurement vector the voxel inversion consumes, and its row ordering.

Phase 2 solved for lambda_p(sky direction): the opacity integrated along a ray
leaving detector position p in a world-frame direction. That IS the tomographic
measurement, so Phase 3 needs only to decide which of those directions are
trustworthy enough to invert, in what order, and with what weight.

Rows are (position, sky bin) pairs. NOT (exposure, detector bin): exposures at
the same spot with different tilts already had their counts combined by Phase 2
into one opacity map per position, and it is the position — not the tilt — that
sets where the ray starts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from megido.baseline import BaselineSolution, position_ids
from megido.config import SiteConfig


def position_origins(cfg: SiteConfig) -> dict[str, tuple[float, float, float]]:
    """World-frame ray origin of each position id, in metres.

    Mirrors megido.baseline.position_ids, which assigns pos0, pos1, ... in
    config order by TRANSLATION only. Two tilts at one spot are one position and
    share one origin.
    """
    ids = position_ids(cfg)
    out: dict[str, tuple[float, float, float]] = {}
    for exp in cfg.exposures:
        pid = ids[exp.id]
        origin = (float(exp.pose.x), float(exp.pose.y), float(exp.pose.z))
        if pid in out and out[pid] != origin:
            raise ValueError(f"position {pid} has two origins: {out[pid]} and {origin}")
        out[pid] = origin
    return out


@dataclass(frozen=True)
class RowIndex:
    """Which (position, sky direction) pairs are inverted, and in what order."""
    position_ids: tuple[str, ...]
    pos_of_row: np.ndarray      # [n_rows] index into position_ids
    sx: np.ndarray              # [n_rows] world-frame tangent, x
    sy: np.ndarray              # [n_rows] world-frame tangent, y
    sky_flat: np.ndarray        # [n_rows] flat index into the Phase 2 sky grid

    @property
    def n_rows(self) -> int:
        return int(self.pos_of_row.size)

    def mask_for(self, pid: str) -> np.ndarray:
        return self.pos_of_row == self.position_ids.index(pid)

    def t_reach(self) -> float:
        """Largest |tangent| carrying a measurement — what sizes the voxel grid."""
        return float(max(np.abs(self.sx).max(), np.abs(self.sy).max()))

    def directions(self) -> np.ndarray:
        """[n_rows, 3] unit world-frame direction of each row."""
        d = np.stack([self.sx, self.sy, np.ones_like(self.sx)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(self.position_ids).encode())
        for arr in (self.pos_of_row, self.sx, self.sy, self.sky_flat):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class FitData:
    lam: np.ndarray             # [n_rows] measured optical depth
    w: np.ndarray               # [n_rows] 1/sigma^2, zero on excluded rows
    rows: RowIndex

    def restricted(self, keep: np.ndarray) -> "FitData":
        """Same rows, weights zeroed outside `keep`. The layout is shared so one
        cached system matrix serves the full fit and every holdout fit."""
        return FitData(lam=self.lam, w=np.where(keep, self.w, 0.0), rows=self.rows)


def build_fit_data(sol: BaselineSolution, cfg: SiteConfig, *,
                   sigma: dict[str, np.ndarray] | None = None,
                   max_tan: float = 1.25,
                   transparent_quantile: float = 0.05) -> FitData:
    """Flatten a baseline solution into (lam, w) over live (position, sky) rows.

    `transparent_quantile` is the gauge convention: which quantile of each
    position's opacity map is called open sky. It is a CHOICE, not a
    measurement — Phase 2 leaves one flat direction per position (spec section
    6.4) — and megido.voxuncert.systematic_map varies it to measure how much of
    the answer that choice is responsible for.

    `max_tan` caps the sky tangent. The Phase 2 sky grid deliberately runs to
    |t| = 2.5 so the tilted exposures lose almost no counts off its edge, but
    those far corners hold a handful of tracks and would stretch the voxel grid
    by a factor of four in each lateral axis. 1.25 matches the analysis grid.

    `sigma` is the per-sky-bin bootstrap uncertainty from
    megido.validate2.opacity_uncertainty. Without it every live row gets weight
    one, which is a statement that all directions are equally trusted — untrue,
    and the reason the reconstruct CLI computes sigma by default.
    """
    origins = position_origins(cfg)
    centers = sol.sky.centers
    n = sol.sky.n_bins
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    all_sx = centers[ii].ravel()
    all_sy = centers[jj].ravel()

    ordered = tuple(pid for pid in sorted(sol.opacity) if pid in origins)
    pos_idx, sx, sy, flat, lam, w = [], [], [], [], [], []

    for k, pid in enumerate(ordered):
        lam_p = sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        live = np.isfinite(lam_p)
        live &= (np.abs(all_sx) <= max_tan) & (np.abs(all_sy) <= max_tan)

        if sigma is not None:
            s = np.asarray(sigma.get(pid, np.full(lam_p.shape, np.nan)), dtype=np.float64)
            # A zero or non-finite sigma is not "perfectly measured"; it means the
            # bootstrap said nothing about this bin. Drop it rather than hand the
            # solver an infinite weight that would pin the fit to one direction.
            live &= np.isfinite(s) & (s > 0)
            weights = np.zeros_like(lam_p)
            weights[live] = 1.0 / s[live] ** 2
        else:
            weights = np.where(live, 1.0, 0.0)

        idx = np.nonzero(live)[0]
        pos_idx.append(np.full(idx.size, k, dtype=np.int64))
        sx.append(all_sx[idx])
        sy.append(all_sy[idx])
        flat.append(idx.astype(np.int64))
        lam.append(lam_p[idx])
        w.append(weights[idx])

    rows = RowIndex(
        position_ids=ordered,
        pos_of_row=np.concatenate(pos_idx) if pos_idx else np.zeros(0, dtype=np.int64),
        sx=np.concatenate(sx) if sx else np.zeros(0),
        sy=np.concatenate(sy) if sy else np.zeros(0),
        sky_flat=np.concatenate(flat) if flat else np.zeros(0, dtype=np.int64),
    )
    if rows.n_rows == 0:
        raise ValueError(
            "no constrained sky directions: every opacity value is non-finite, "
            f"or none survives max_tan={max_tan}")
    return FitData(lam=np.concatenate(lam), w=np.concatenate(w), rows=rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_fitdata.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/fitdata.py tests/test_fitdata.py
git commit -m "feat: measurement vector and row index from a baseline solution"
```

---

### Task 3: Sparse path-length system matrix

**Files:**
- Create: `megido/raycast.py`
- Test: `tests/test_raycast.py`

**Interfaces:**
- Consumes: `megido.fitdata.RowIndex`, `megido.voxels.VoxelGrid`.
- Produces:
  - `megido.raycast.INVERSION_VERSION: int`
  - `bundle_offsets(directions: np.ndarray, aperture_m: float, n_sub: int) -> np.ndarray` → `[n_rows, n_sub**2, 3]`
  - `build_system_matrix(rows, origins, grid, *, aperture_m, n_sub=4, cache_dir=None) -> scipy.sparse.csr_matrix` of shape `[rows.n_rows, grid.n_voxels]`, entries in **metres**

- [ ] **Step 1: Write the failing test**

Create `tests/test_raycast.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from megido.fitdata import RowIndex
from megido.raycast import bundle_offsets, build_system_matrix
from megido.voxels import VoxelGrid


def _rows(sx, sy, pos=None) -> RowIndex:
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    pos = np.zeros(sx.size, dtype=np.int64) if pos is None else np.asarray(pos)
    return RowIndex(position_ids=("pos0", "pos1")[: pos.max() + 1],
                    pos_of_row=pos, sx=sx, sy=sy,
                    sky_flat=np.arange(sx.size, dtype=np.int64))


ORIGINS = {"pos0": (0.0, 0.0, 0.0), "pos1": (2.0, 0.0, 0.0)}


def test_bundle_offsets_are_perpendicular_to_their_ray():
    d = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.3, -0.4, np.sqrt(1 - 0.25)]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    offs = bundle_offsets(d, aperture_m=0.384, n_sub=3)

    assert offs.shape == (3, 9, 3)
    np.testing.assert_allclose(np.einsum("rsk,rk->rs", offs, d), 0.0, atol=1e-12)


def test_bundle_offsets_span_the_aperture_and_are_centred():
    d = np.array([[0.0, 0.0, 1.0]])
    offs = bundle_offsets(d, aperture_m=0.4, n_sub=4)
    np.testing.assert_allclose(offs.mean(axis=1), 0.0, atol=1e-12)
    # 4 sub-rays at (+-3/8, +-1/8) * 0.4 -> extreme coordinate 0.15
    assert np.abs(offs).max() == pytest.approx(0.15)


def test_a_single_vertical_ray_has_path_length_equal_to_the_slab():
    """A vertical pinhole ray through a 2 m slab must deposit exactly 2 m."""
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.shape == (1, grid.n_voxels)
    assert A.sum() == pytest.approx(2.0, rel=1e-6)


def test_a_tilted_ray_is_longer_by_one_over_cos_theta():
    grid = VoxelGrid(origin=(-8.0, -8.0, 1.0), spacing=0.5, shape=(32, 32, 4))
    A = build_system_matrix(_rows([1.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.sum() == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-3)


def test_the_vertical_ray_lands_in_the_column_above_its_origin():
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1).toarray().reshape(grid.shape)
    hit = np.nonzero(A.sum(axis=2))
    assert (int(hit[0][0]), int(hit[1][0])) == (2, 2)   # x, y just above 0.0


def test_rows_start_at_their_own_position(): 
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(12, 4, 4))
    rows = _rows([0.0, 0.0], [0.0, 0.0], pos=[0, 1])
    A = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.0, n_sub=1)
    dense = A.toarray().reshape(2, *grid.shape)
    assert int(np.nonzero(dense[0].sum(axis=(1, 2)))[0][0]) == 2    # x = 0.0
    assert int(np.nonzero(dense[1].sum(axis=(1, 2)))[0][0]) == 6    # x = 2.0


def test_a_wide_bundle_spreads_across_more_voxels_than_a_pinhole():
    grid = VoxelGrid(origin=(-2.0, -2.0, 1.0), spacing=0.1, shape=(40, 40, 10))
    pin = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                              aperture_m=0.0, n_sub=1)
    wide = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                               aperture_m=0.8, n_sub=4)
    assert wide.nnz > 4 * pin.nnz
    # total path length is conserved: the bundle averages, it does not multiply
    assert wide.sum() == pytest.approx(pin.sum(), rel=1e-6)


def test_rays_that_miss_the_grid_produce_an_empty_row_not_an_error():
    grid = VoxelGrid(origin=(10.0, 10.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    A = build_system_matrix(_rows([0.0], [0.0]), ORIGINS, grid,
                            aperture_m=0.0, n_sub=1)
    assert A.nnz == 0


def test_the_matrix_is_cached_on_disk_and_reused(tmp_path):
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0, 0.2], [0.0, -0.1])
    first = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2,
                                cache_dir=tmp_path)
    cached = sorted(tmp_path.glob("A_*.npz"))
    assert len(cached) == 1

    second = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2,
                                 cache_dir=tmp_path)
    assert (first != second).nnz == 0
    assert sorted(tmp_path.glob("A_*.npz")) == cached


def test_the_cache_key_separates_different_geometry(tmp_path):
    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0], [0.0])
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.2, n_sub=2, cache_dir=tmp_path)
    build_system_matrix(_rows([0.5], [0.0]), ORIGINS, grid, aperture_m=0.1, n_sub=2,
                        cache_dir=tmp_path)
    assert len(sorted(tmp_path.glob("A_*.npz"))) == 3


def test_the_cache_key_includes_the_inversion_version(tmp_path, monkeypatch):
    """A change to ray-casting logic must not serve a stale matrix — the Phase 1
    cache defect, which shipped pre-dither artifacts after the fix landed."""
    import megido.raycast as R

    grid = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    rows = _rows([0.0], [0.0])
    build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    monkeypatch.setattr(R, "INVERSION_VERSION", R.INVERSION_VERSION + 1)
    R.build_system_matrix(rows, ORIGINS, grid, aperture_m=0.1, n_sub=2, cache_dir=tmp_path)
    assert len(sorted(tmp_path.glob("A_*.npz"))) == 2


def test_matrix_is_csr_and_finite():
    grid = VoxelGrid(origin=(-2.0, -2.0, 1.0), spacing=0.25, shape=(16, 16, 8))
    rows = _rows(np.linspace(-0.5, 0.5, 7), np.zeros(7))
    A = build_system_matrix(rows, ORIGINS, grid, aperture_m=0.384, n_sub=3)
    assert sparse.isspmatrix_csr(A)
    assert np.all(np.isfinite(A.data))
    assert A.data.min() > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_raycast.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.raycast'`.

- [ ] **Step 3: Write `megido/raycast.py`**

```python
"""Sparse system matrix: mean path length of each row's ray bundle per voxel.

A row is one (position, sky direction) pair from megido.fitdata.RowIndex, and
its measurement is lambda = integral of opacity density along that ray. The
detector aperture is 38.4 cm, several voxels wide at the default 25 cm spacing,
so a row is modelled as a bundle of n_sub^2 parallel sub-rays across the
aperture rather than as a pinhole. Entries are path lengths AVERAGED over the
bundle, so a bundle and a pinhole through uniform material predict the same
optical depth.

The pose rotation is deliberately absent. Phase 2's detector_to_sky already
mapped every measurement into the world frame, so a row's direction is just
normalize(sx, sy, 1); rotating again would rotate twice.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from megido.fitdata import RowIndex
from megido.voxels import VoxelGrid

# Bumped whenever ray-casting logic changes. It is part of the cache key: in
# Phase 1 a cache that ignored the code version silently served artifacts built
# before a reconstruction fix, and the same trap is live here.
INVERSION_VERSION = 1

_SAMPLES_PER_VOXEL = 3          # sampling step along a ray = spacing / this
_ROW_BLOCK = 512                # rows processed per vectorised block


def bundle_offsets(directions: np.ndarray, aperture_m: float,
                   n_sub: int) -> np.ndarray:
    """Sub-ray start offsets, [n_rows, n_sub**2, 3], perpendicular to each ray.

    The physical aperture is a plane in the DETECTOR frame, but a row aggregates
    exposures at different tilts, so there is no single aperture orientation to
    use. A square of side `aperture_m` perpendicular to the ray is pose-free and
    correct to first order in tilt (it overestimates the cross-section by
    1/cos(tilt), about 6% at the campaign's 20 degrees).
    """
    d = np.asarray(directions, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    # An orthonormal frame perpendicular to each ray. cross(d, z) degenerates for
    # a vertical ray, so fall back to x there.
    zhat = np.array([0.0, 0.0, 1.0])
    u = np.cross(d, zhat)
    small = np.linalg.norm(u, axis=-1) < 1e-9
    u[small] = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(d, u)

    g = (np.arange(n_sub) + 0.5) / n_sub - 0.5          # centred, in [-0.5, 0.5)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    gx, gy = gx.ravel() * aperture_m, gy.ravel() * aperture_m
    return gx[None, :, None] * u[:, None, :] + gy[None, :, None] * v[:, None, :]


def build_system_matrix(rows: RowIndex,
                        origins: dict[str, tuple[float, float, float]],
                        grid: VoxelGrid, *,
                        aperture_m: float,
                        n_sub: int = 4,
                        cache_dir: str | Path | None = None) -> sparse.csr_matrix:
    """A[row, voxel] in METRES, shape [rows.n_rows, grid.n_voxels]."""
    if cache_dir is not None:
        key = _cache_key(rows, origins, grid, aperture_m, n_sub)
        cache = Path(cache_dir) / f"A_{key}.npz"
        if cache.exists():
            return sparse.load_npz(cache)

    dirs = rows.directions()                                    # [nr, 3]
    offs = bundle_offsets(dirs, aperture_m, n_sub)              # [nr, ns, 3]
    starts = np.array([origins[rows.position_ids[i]] for i in rows.pos_of_row])
    starts = starts[:, None, :] + offs                          # [nr, ns, 3]

    origin = np.asarray(grid.origin, dtype=np.float64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    z0, z1 = grid.extent(2)

    # Entry and exit of the grid's z slab along each ray, measured from its own
    # start. dz > 0 always: a row's direction is normalize(sx, sy, 1).
    dz = dirs[:, 2]
    t_in = (z0 - starts[:, :, 2].mean(axis=1)) / dz
    length = (z1 - z0) / dz
    step = grid.spacing / _SAMPLES_PER_VOXEL
    n_samp = np.maximum(np.ceil(length / step).astype(np.int64), 1)

    nr, ns = rows.n_rows, offs.shape[1]
    r_out, c_out, v_out = [], [], []
    for lo in range(0, nr, _ROW_BLOCK):
        hi = min(lo + _ROW_BLOCK, nr)
        cn = n_samp[lo:hi]
        row_rep = np.repeat(np.arange(lo, hi), cn)                        # [Nt]
        # midpoint sampling fraction along each ray's in-slab segment
        base = np.repeat(np.concatenate([[0], np.cumsum(cn[:-1])]), cn)
        frac = (np.arange(int(cn.sum())) - base + 0.5) / np.repeat(cn, cn)
        t = t_in[row_rep] + frac * length[row_rep]                        # [Nt]

        pts = starts[row_rep] + (t[:, None] * dirs[row_rep])[:, None, :]  # [Nt, ns, 3]
        idx = np.floor((pts - origin) / grid.spacing).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), axis=-1)              # [Nt, ns]
        flat = (idx[..., 0] * grid.shape[1] + idx[..., 1]) * grid.shape[2] + idx[..., 2]
        # Divide by ns: the bundle AVERAGES path length, it does not accumulate.
        dl = length[row_rep] / n_samp[row_rep] / ns

        r_out.append(np.broadcast_to(row_rep[:, None], inside.shape)[inside])
        c_out.append(flat[inside])
        v_out.append(np.broadcast_to(dl[:, None], inside.shape)[inside])

    A = sparse.coo_matrix(
        (np.concatenate(v_out) if v_out else np.zeros(0),
         (np.concatenate(r_out) if r_out else np.zeros(0, dtype=np.int64),
          np.concatenate(c_out) if c_out else np.zeros(0, dtype=np.int64))),
        shape=(nr, grid.n_voxels),
    ).tocsr()
    A.sum_duplicates()

    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, A)
    return A


def _cache_key(rows: RowIndex, origins: dict, grid: VoxelGrid,
               aperture_m: float, n_sub: int) -> str:
    h = hashlib.sha256()
    h.update(f"v{INVERSION_VERSION}|".encode())
    h.update(rows.key().encode())
    for pid in rows.position_ids:
        x, y, z = origins[pid]
        h.update(f"{pid}:{x:.6f},{y:.6f},{z:.6f};".encode())
    h.update(f"{grid.key()}|{aperture_m:.6f}|{n_sub}|{_SAMPLES_PER_VOXEL}".encode())
    return h.hexdigest()[:16]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_raycast.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/raycast.py tests/test_raycast.py
git commit -m "feat: sparse path-length system matrix with ray bundles and a versioned cache"
```

---

### Task 4: ForwardModel — the matrix plus its bookkeeping

**Files:**
- Create: `megido/forward.py`
- Test: `tests/test_forward.py`

**Interfaces:**
- Consumes: `megido.fitdata.RowIndex`, `megido.voxels.VoxelGrid`/`auto_grid`, `megido.raycast.build_system_matrix`, `megido.config.SiteConfig`/`Volume`, `megido.detector.DetectorGeometry`.
- Produces:
  - `ForwardModel(A, grid, rows)` with `n_rows`, `predict(x, offsets=None) -> np.ndarray[n_rows]`, `to_sky_image(values, pid, n_sky) -> np.ndarray[n_sky, n_sky]`
  - `build_forward_model(rows, cfg, *, geom=None, grid=None, cache_dir="runs/.cache") -> ForwardModel`

Note for the implementer: `build_system_matrix` computes each ray's slab-entry point from the bundle's mean z, not per sub-ray. Sub-rays that consequently sample just outside the grid are dropped by the `inside` mask, so the approximation cannot add spurious path length. Do not "fix" it by sampling per sub-ray — that would break the shared sampling layout the block loop depends on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forward.py`:

```python
import numpy as np
import pytest

from megido.config import Volume, load_site_config
from megido.fitdata import RowIndex
from megido.forward import ForwardModel, build_forward_model
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _rows():
    sx = np.array([0.0, 0.2, 0.0, -0.2])
    sy = np.array([0.0, 0.0, 0.1, 0.1])
    return RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.array([0, 0, 1, 1]),
                    sx=sx, sy=sy, sky_flat=np.array([0, 1, 2, 3]))


def test_build_sizes_the_grid_from_the_rows(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    assert fwd.A.shape == (4, fwd.grid.n_voxels)
    assert fwd.grid.extent(2) == pytest.approx((1.0, 3.0))
    assert fwd.n_rows == 4


def test_predict_of_a_uniform_volume_is_density_times_path_length(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x = np.full(fwd.grid.n_voxels, 0.3)
    np.testing.assert_allclose(fwd.predict(x), 0.3 * np.asarray(fwd.A.sum(axis=1)).ravel())


def test_predict_adds_the_per_position_offsets(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x = np.zeros(fwd.grid.n_voxels)
    got = fwd.predict(x, offsets={"pos0": 0.5, "pos1": -0.25})
    np.testing.assert_allclose(got, [0.5, 0.5, -0.25, -0.25])


def test_predict_accepts_a_3d_volume(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    x3 = np.full(fwd.grid.shape, 0.2)
    np.testing.assert_allclose(fwd.predict(x3), fwd.predict(x3.ravel()))


def test_an_explicit_grid_overrides_the_automatic_one(tmp_path):
    g = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    fwd = build_forward_model(_rows(), _cfg(tmp_path), grid=g, cache_dir=None)
    assert fwd.grid is g


def test_to_sky_image_places_values_at_their_sky_bins(tmp_path):
    fwd = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=None)
    img = fwd.to_sky_image(np.array([1.0, 2.0, 3.0, 4.0]), "pos1", n_sky=2)
    assert img.shape == (2, 2)
    np.testing.assert_allclose(img.ravel(), [np.nan, np.nan, 3.0, 4.0])


def test_the_matrix_is_reused_from_cache_across_calls(tmp_path):
    cache = tmp_path / "cache"
    a = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=cache)
    b = build_forward_model(_rows(), _cfg(tmp_path), cache_dir=cache)
    assert (a.A != b.A).nnz == 0
    assert len(list(cache.glob("A_*.npz"))) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_forward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.forward'`.

- [ ] **Step 3: Write `megido/forward.py`**

```python
"""The one forward model shared by inversion, phantom generation and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.fitdata import RowIndex, position_origins
from megido.raycast import build_system_matrix
from megido.voxels import VoxelGrid, auto_grid


@dataclass(frozen=True)
class ForwardModel:
    A: sparse.csr_matrix
    grid: VoxelGrid
    rows: RowIndex

    @property
    def n_rows(self) -> int:
        return self.rows.n_rows

    def predict(self, x: np.ndarray, offsets: dict[str, float] | None = None
                ) -> np.ndarray:
        """Volume (flat or 3D) -> predicted optical depth per row."""
        y = self.A @ np.asarray(x, dtype=np.float64).ravel()
        if offsets:
            add = np.array([offsets.get(pid, 0.0) for pid in self.rows.position_ids])
            y = y + add[self.rows.pos_of_row]
        return y

    def to_sky_image(self, values: np.ndarray, pid: str, n_sky: int) -> np.ndarray:
        """Scatter a per-row quantity back onto one position's sky grid.

        Bins with no row are NaN, never zero: "not measured" and "measured as
        zero opacity" are different statements and the viewer must not conflate
        them.
        """
        img = np.full(n_sky * n_sky, np.nan)
        sel = self.rows.mask_for(pid)
        img[self.rows.sky_flat[sel]] = np.asarray(values)[sel]
        return img.reshape(n_sky, n_sky)


def build_forward_model(rows: RowIndex, cfg: SiteConfig, *,
                        geom: DetectorGeometry | None = None,
                        grid: VoxelGrid | None = None,
                        cache_dir: str | Path | None = "runs/.cache"
                        ) -> ForwardModel:
    geom = geom or DetectorGeometry.megiddo()
    origins = position_origins(cfg)
    if grid is None:
        grid = auto_grid(cfg.volume, origins, t_reach=rows.t_reach(),
                         aperture_m=geom.aperture_m)
    A = build_system_matrix(rows, origins, grid,
                            aperture_m=geom.aperture_m,
                            n_sub=cfg.volume.n_aperture_sub,
                            cache_dir=cache_dir)
    return ForwardModel(A=A, grid=grid, rows=rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_forward.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/forward.py tests/test_forward.py
git commit -m "feat: forward model wrapping the system matrix and its row bookkeeping"
```

---

### Task 5: SIRT and the anisotropic-TV proximal solver

Pure algorithms: no file I/O, no config loading, no site knowledge. Everything here is testable against a hand-built system.

**Files:**
- Create: `megido/inversion.py`
- Test: `tests/test_inversion.py`

**Interfaces:**
- Consumes: `megido.forward.ForwardModel`, `megido.fitdata.FitData`, `megido.config.Reconstruction`.
- Produces:
  - `sirt(fwd, data, rc) -> tuple[np.ndarray, dict]`
  - `sirt_tv(fwd, data, rc) -> tuple[np.ndarray, dict]`
  - `solve(fwd, data, rc) -> tuple[np.ndarray, dict]` dispatching on `rc.algorithm`
  - info dict keys: `offsets` (dict position id → float), `chi2_history` (list), `best_chi2` (float), `n_iter_used` (int)

- [ ] **Step 1: Write the failing test**

Create `tests/test_inversion.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from megido.config import Reconstruction
from megido.fitdata import FitData, RowIndex
from megido.forward import ForwardModel
from megido.inversion import solve, sirt, sirt_tv
from megido.voxels import VoxelGrid


def _toy(n_rows=40, shape=(4, 4, 2), seed=0):
    """A small well-posed random system with a known nonnegative solution."""
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(shape))
    A = sparse.csr_matrix(rng.random((n_rows, n_vox)) * (rng.random((n_rows, n_vox)) < 0.5))
    truth = np.abs(rng.normal(size=n_vox))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.arange(n_rows) % 2,
                    sx=np.zeros(n_rows), sy=np.zeros(n_rows),
                    sky_flat=np.arange(n_rows))
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape),
                       rows=rows)
    lam = A @ truth
    return fwd, truth, FitData(lam=lam, w=np.ones(n_rows), rows=rows)


def test_sirt_recovers_a_noiseless_solution():
    fwd, truth, data = _toy()
    rc = Reconstruction(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x, info = sirt(fwd, data, rc)
    assert np.corrcoef(x, truth)[0, 1] > 0.95
    assert info["chi2_history"][-1] < info["chi2_history"][0]


def test_sirt_never_returns_a_negative_voxel():
    fwd, truth, data = _toy()
    rng = np.random.default_rng(1)
    noisy = FitData(lam=data.lam + rng.normal(scale=0.5, size=data.lam.size),
                    w=data.w, rows=data.rows)
    x, _ = sirt(fwd, noisy, Reconstruction(algorithm="sirt", n_iter=200,
                                           chi2_target=1e-12, nonneg=True))
    assert x.min() >= 0.0


def test_sirt_stops_early_at_the_discrepancy_target():
    fwd, truth, data = _toy()
    x, info = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=5000,
                                             chi2_target=1e-2))
    assert info["n_iter_used"] < 5000


def test_zero_weight_rows_do_not_influence_the_fit():
    fwd, truth, data = _toy()
    poisoned = FitData(lam=data.lam.copy(), w=data.w.copy(), rows=data.rows)
    poisoned.lam[:5] = 1e6
    kept = np.ones(data.rows.n_rows, dtype=bool)
    kept[:5] = False

    rc = Reconstruction(algorithm="sirt", n_iter=300, chi2_target=1e-12)
    a, _ = sirt(fwd, data.restricted(kept), rc)
    b, _ = sirt(fwd, poisoned.restricted(kept), rc)
    np.testing.assert_allclose(a, b)


def test_offsets_absorb_a_constant_shift_per_position():
    """lambda has a free additive constant per position: Phase 2's gauge is
    pinned per position, so an overall level is not measured.

    The comparison is between the shifted and unshifted fits, not against zero.
    The unshifted fit already carries nonzero offsets absorbing the mean of
    A @ truth, so the absolute offsets say nothing; only their DIFFERENCE should
    track the injected shift.
    """
    fwd, truth, data = _toy()
    shift = np.where(data.rows.pos_of_row == 0, 0.7, -0.4)
    shifted = FitData(lam=data.lam + shift, w=data.w, rows=data.rows)

    rc = Reconstruction(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x0, base = sirt(fwd, data, rc)
    x1, moved = sirt(fwd, shifted, rc)

    assert moved["offsets"]["pos0"] - base["offsets"]["pos0"] == pytest.approx(0.7, abs=0.15)
    assert moved["offsets"]["pos1"] - base["offsets"]["pos1"] == pytest.approx(-0.4, abs=0.15)
    assert np.corrcoef(x1, truth)[0, 1] > 0.9


def test_tv_denoises_a_piecewise_constant_volume_better_than_plain_sirt():
    rng = np.random.default_rng(3)
    shape = (8, 8, 4)
    n_vox = int(np.prod(shape))
    truth3 = np.zeros(shape)
    truth3[2:6, 2:6, 1:3] = 1.0
    truth = truth3.ravel()

    A = sparse.csr_matrix(rng.random((300, n_vox)) * (rng.random((300, n_vox)) < 0.3))
    rows = RowIndex(position_ids=("pos0",), pos_of_row=np.zeros(300, dtype=int),
                    sx=np.zeros(300), sy=np.zeros(300), sky_flat=np.arange(300))
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape),
                       rows=rows)
    lam = A @ truth + rng.normal(scale=0.4, size=300)
    data = FitData(lam=lam, w=np.ones(300), rows=rows)

    plain, _ = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=300,
                                              chi2_target=1e-12))
    tv, _ = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=300,
                                              tv_alpha=0.001, tv_z_weight=0.5))
    assert np.linalg.norm(tv - truth) < np.linalg.norm(plain - truth)


def test_tv_returns_the_best_iterate_not_the_last():
    fwd, truth, data = _toy()
    x, info = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=120,
                                                tv_alpha=0.01))
    assert info["best_chi2"] <= min(info["chi2_history"])


def test_tv_with_zero_alpha_tracks_plain_sirt():
    """With tv_alpha = 0 the proximal step is just the nonnegativity clip, so the
    two solvers follow the same trajectory.

    They do NOT end on the same array: chi2 is evaluated before each update, so
    sirt_tv's best iterate is at most the one after n_iter - 1 updates while sirt
    returns the one after n_iter. The gap is one sweep, by construction.
    """
    fwd, truth, data = _toy()
    a, _ = sirt(fwd, data, Reconstruction(algorithm="sirt", n_iter=50,
                                          chi2_target=-1.0))
    b, _ = sirt_tv(fwd, data, Reconstruction(algorithm="tv", n_iter=50,
                                             tv_alpha=0.0))
    assert np.corrcoef(a, b)[0, 1] > 0.999
    assert np.linalg.norm(b - a) < 0.05 * np.linalg.norm(a)


def test_solve_dispatches_on_the_algorithm_name():
    fwd, truth, data = _toy()
    x, _ = solve(fwd, data, Reconstruction(algorithm="tv", n_iter=20))
    assert x.shape == (fwd.grid.n_voxels,)
    with pytest.raises(ValueError, match="unknown algorithm"):
        solve(fwd, data, Reconstruction(algorithm="mlem", n_iter=5))


def test_an_all_zero_weight_fit_returns_zeros_rather_than_dividing_by_zero():
    fwd, truth, data = _toy()
    dead = data.restricted(np.zeros(data.rows.n_rows, dtype=bool))
    x, info = sirt(fwd, dead, Reconstruction(algorithm="sirt", n_iter=10,
                                             chi2_target=-1.0))
    assert np.all(x == 0.0)
    assert np.all(np.isfinite(list(info["offsets"].values())))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_inversion.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.inversion'`.

- [ ] **Step 3: Write `megido/inversion.py`**

```python
"""Voxel inversion algorithms. One linear model, no I/O, no site knowledge.

    lambda = A x + c_p,    x >= 0

x is the voxel opacity density [1/m] and c_p is a free additive constant per
position. c_p is not a fudge: Phase 2 pinned each position's opacity gauge
independently (the median of lambda_p is zero by construction, then shifted to
put the most transparent direction at zero), so the absolute level of lambda is
not a measurement and must be refit here.

Fits are selected by zeroing weights over the shared row layout, so one cached
system matrix serves the full fit and every holdout fit alike.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from megido.config import Reconstruction
from megido.fitdata import FitData
from megido.forward import ForwardModel


def _update_offsets(resid: np.ndarray, w: np.ndarray,
                    pos_of_row: np.ndarray, n_pos: int) -> np.ndarray:
    """Closed-form weighted-mean residual per position (the c_p nuisance)."""
    c = np.zeros(n_pos)
    for i in range(n_pos):
        sel = pos_of_row == i
        sw = w[sel].sum()
        if sw > 0:
            c[i] = float((w[sel] * resid[sel]).sum() / sw)
    return c


def _scalings(A: sparse.csr_matrix, w: np.ndarray):
    """Row and column scalings of the weighted system used by SIRT."""
    Aw = A.multiply(np.sqrt(w)[:, None]).tocsr()
    row_sum = np.asarray(abs(Aw).sum(axis=1)).ravel()
    col_sum = np.asarray(abs(Aw).sum(axis=0)).ravel()
    return 1.0 / np.maximum(row_sum, 1e-12), 1.0 / np.maximum(col_sum, 1e-12)


def _named(c: np.ndarray, position_ids) -> dict[str, float]:
    return {pid: float(v) for pid, v in zip(position_ids, c)}


def sirt(fwd: ForwardModel, data: FitData, rc: Reconstruction
         ) -> tuple[np.ndarray, dict]:
    """Weighted SIRT with nonnegativity and per-position offset refinement.

    Stops at rc.chi2_target by the discrepancy principle: fitting past the noise
    floor is how SIRT turns counting statistics into structure.
    """
    A, lam, w = fwd.A, data.lam, data.w
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    n_used = max(int(np.count_nonzero(w)), 1)

    history: list[float] = []
    chi2 = float("inf")
    k = -1
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if k % 20 == 0:
            history.append(chi2)
        if chi2 <= rc.chi2_target:
            break
        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
    history.append(chi2)
    return x, {"offsets": _named(c, fwd.rows.position_ids),
               "chi2_history": history,
               "best_chi2": float(min(history)),
               "n_iter_used": k + 1}


def _grad3(x3: np.ndarray, z_weight: float) -> np.ndarray:
    """Anisotropic forward differences -> [3, nx, ny, nz], zero-padded at the far edge."""
    g = np.zeros((3,) + x3.shape)
    g[0, :-1] = np.diff(x3, axis=0)
    g[1, :, :-1] = np.diff(x3, axis=1)
    g[2, :, :, :-1] = np.diff(x3, axis=2) * z_weight
    return g


def _div3(p: np.ndarray, z_weight: float) -> np.ndarray:
    """Negative adjoint of _grad3."""
    d = np.zeros(p.shape[1:])
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    d[:, :, :-1] += p[2, :, :, :-1] * z_weight
    d[:, :, 1:] -= p[2, :, :, :-1] * z_weight
    return d


def _prox_tv(v: np.ndarray, gamma: float, zw: float, dual: np.ndarray,
             n_inner: int = 10) -> np.ndarray:
    """Chambolle dual projection for prox of gamma * ||grad .||_1 under x >= 0.

    `dual` is warm-started across outer iterations and mutated in place, which is
    what makes ten inner steps enough.
    """
    if gamma <= 0.0:
        return np.maximum(v, 0.0)
    step = 1.0 / (4.0 * (2.0 + zw * zw))
    for _ in range(n_inner):
        u = np.maximum(v + _div3(dual, zw), 0.0)
        np.clip(dual + step * _grad3(u, zw), -gamma, gamma, out=dual)
    return np.maximum(v + _div3(dual, zw), 0.0)


def sirt_tv(fwd: ForwardModel, data: FitData, rc: Reconstruction
            ) -> tuple[np.ndarray, dict]:
    """SIRT with a per-iteration anisotropic-TV proximal (denoising) step.

    tv_alpha is a fraction of the reconstructed scale (x's p95), so it transfers
    across datasets instead of needing a retune per campaign. Runs the full
    budget and returns the best-chi2 iterate: the TV step keeps it from
    overfitting the way plain SIRT does, so there is no discrepancy stop.
    """
    A, lam, w = fwd.A, data.lam, data.w
    shape = fwd.grid.shape
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    n_used = max(int(np.count_nonzero(w)), 1)
    dual = np.zeros((3,) + shape)

    history: list[float] = []
    best = (float("inf"), x.copy(), c.copy())
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if chi2 < best[0]:
            best = (chi2, x.copy(), c.copy())
        if k % 20 == 0:
            history.append(chi2)

        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
        scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 0.0
        gamma = rc.tv_alpha * max(scale, 1e-9)
        x = _prox_tv(x.reshape(shape), gamma, rc.tv_z_weight, dual).ravel()

    history.append(best[0])
    return best[1], {"offsets": _named(best[2], fwd.rows.position_ids),
                     "chi2_history": history,
                     "best_chi2": float(best[0]),
                     "n_iter_used": rc.n_iter}


SOLVERS = {"sirt": sirt, "tv": sirt_tv}


def solve(fwd: ForwardModel, data: FitData, rc: Reconstruction
          ) -> tuple[np.ndarray, dict]:
    if rc.algorithm not in SOLVERS:
        raise ValueError(f"unknown algorithm {rc.algorithm!r}; have {sorted(SOLVERS)}")
    return SOLVERS[rc.algorithm](fwd, data, rc)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_inversion.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/inversion.py tests/test_inversion.py
git commit -m "feat: SIRT and anisotropic-TV proximal voxel solvers"
```

---

### Task 6: VoxelSolution, solve_voxels, and per-position holdout fits

**Files:**
- Create: `megido/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `megido.baseline.BaselineSolution`, `megido.fitdata.build_fit_data`, `megido.forward.build_forward_model`, `megido.inversion.solve`, `megido.raycast.INVERSION_VERSION`.
- Produces:
  - `VoxelSolution(rho, grid, offsets, position_ids, info, version)` with `rho3() -> np.ndarray[nx,ny,nz]`, `save(path)`, static `load(path)`
  - `solve_voxels(sol, cfg, *, sigma=None, max_tan=1.25, transparent_quantile=0.05, cache_dir="runs/.cache", holdouts=True) -> dict[str, VoxelSolution]` keyed `"full"` and `"holdout_<pid>"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconstruct.py`:

```python
import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.reconstruct import VoxelSolution, solve_voxels
from megido.sky import SkyGrid
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
reconstruction: {algorithm: tv, n_iter: 30, tv_alpha: 0.01}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""

N_SKY = 12


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _solution(seed=0):
    rng = np.random.default_rng(seed)
    sky = SkyGrid(edges=np.linspace(-2.5, 2.5, N_SKY + 1))
    opacity = {}
    for pid in ("pos0", "pos1"):
        lam = np.full(N_SKY * N_SKY, np.nan)
        live = rng.random(N_SKY * N_SKY) < 0.7
        lam[live] = rng.uniform(0.0, 2.0, int(live.sum()))
        opacity[pid] = lam
    return BaselineSolution(
        coeffs=np.zeros(1), opacity=opacity, norms={}, flux_index=2.0,
        nll_history=[0.0], grid=AnalysisGrid(edges=np.linspace(-1, 1, 3), counts={}),
        sky=sky, basis=make_smooth_basis(t_max=1.25, n_per_axis=2))


def test_solve_produces_a_full_fit_and_one_holdout_per_position(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert set(fits) == {"full", "holdout_pos0", "holdout_pos1"}


def test_the_volume_is_nonnegative_and_correctly_shaped(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    v = fits["full"]
    assert v.rho.shape == (v.grid.n_voxels,)
    assert v.rho3().shape == v.grid.shape
    assert v.rho.min() >= 0.0


def test_a_holdout_fit_uses_only_its_own_position(tmp_path):
    """holdout_pos0 is TRAINED ON pos0 alone — it is the single-view fit whose
    disagreement with the other view is the cross-validation signal."""
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert fits["holdout_pos0"].info["n_rows_used"] < fits["full"].info["n_rows_used"]
    assert (fits["holdout_pos0"].info["n_rows_used"]
            + fits["holdout_pos1"].info["n_rows_used"]
            == fits["full"].info["n_rows_used"])


def test_holdouts_can_be_skipped(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None, holdouts=False)
    assert set(fits) == {"full"}


def test_offsets_are_recorded_per_position(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    assert set(fits["full"].offsets) == {"pos0", "pos1"}


def test_save_and_load_round_trip(tmp_path):
    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    p = tmp_path / "out" / "volume_full.npz"
    fits["full"].save(p)

    back = VoxelSolution.load(p)
    np.testing.assert_allclose(back.rho, fits["full"].rho)
    assert back.grid == fits["full"].grid
    assert back.offsets == pytest.approx(fits["full"].offsets)
    assert back.position_ids == fits["full"].position_ids


def test_the_saved_volume_records_the_inversion_version(tmp_path):
    from megido.raycast import INVERSION_VERSION

    fits = solve_voxels(_solution(), _cfg(tmp_path), cache_dir=None)
    p = tmp_path / "v.npz"
    fits["full"].save(p)
    assert VoxelSolution.load(p).version == INVERSION_VERSION


def test_sigma_weights_are_threaded_through(tmp_path):
    """A position whose sigma is enormous should barely move the fit."""
    sol = _solution()
    big = {"pos0": np.full(N_SKY * N_SKY, 1e6),
           "pos1": np.full(N_SKY * N_SKY, 0.1)}
    weighted = solve_voxels(sol, _cfg(tmp_path), sigma=big, cache_dir=None,
                            holdouts=False)["full"]
    only1 = solve_voxels(sol, _cfg(tmp_path), cache_dir=None,
                         holdouts=False)["full"]
    assert not np.allclose(weighted.rho, only1.rho)


def test_the_volume_reproduces_its_own_measurements_better_than_a_zero_volume(tmp_path):
    """The weakest honest fidelity claim, and the one that catches a matrix
    whose rays point the wrong way: the fit must beat doing nothing."""
    from megido.fitdata import build_fit_data
    from megido.forward import build_forward_model

    sol, cfg = _solution(), _cfg(tmp_path)
    fits = solve_voxels(sol, cfg, cache_dir=None, holdouts=False)
    data = build_fit_data(sol, cfg)
    fwd = build_forward_model(data.rows, cfg, cache_dir=None)

    v = fits["full"]
    pred = fwd.predict(v.rho, v.offsets)
    flat = np.mean((data.lam - np.mean(data.lam)) ** 2)
    assert np.mean((data.lam - pred) ** 2) < flat
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_reconstruct.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.reconstruct'`.

- [ ] **Step 3: Write `megido/reconstruct.py`**

```python
"""Orchestration: baseline solution in, voxel volumes out.

Alongside the full fit, one volume per position is produced from that position
ALONE. With two positions 2.2 m apart, agreement between those single-view fits
is the only direct evidence that the depth structure is measured rather than
assumed, so they are produced by default and not as an opt-in diagnostic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from megido.baseline import BaselineSolution
from megido.config import SiteConfig
from megido.fitdata import build_fit_data
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.raycast import INVERSION_VERSION
from megido.voxels import VoxelGrid


@dataclass(frozen=True)
class VoxelSolution:
    rho: np.ndarray                     # [n_voxels] opacity density, 1/m, >= 0
    grid: VoxelGrid
    offsets: dict[str, float]
    position_ids: tuple[str, ...]
    info: dict = field(default_factory=dict, repr=False)
    version: int = INVERSION_VERSION

    def rho3(self) -> np.ndarray:
        return self.rho.reshape(self.grid.shape)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rho=self.rho3().astype(np.float32),
            origin=np.asarray(self.grid.origin, dtype=np.float64),
            spacing=np.asarray(self.grid.spacing, dtype=np.float64),
            shape=np.asarray(self.grid.shape, dtype=np.int64),
            version=np.asarray(self.version, dtype=np.int64),
            meta=np.array(json.dumps({
                "offsets": self.offsets,
                "position_ids": list(self.position_ids),
                "info": self.info,
            })),
        )

    @staticmethod
    def load(path: str | Path) -> "VoxelSolution":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        grid = VoxelGrid(origin=tuple(float(v) for v in d["origin"]),
                         spacing=float(d["spacing"]),
                         shape=tuple(int(v) for v in d["shape"]))
        return VoxelSolution(
            rho=d["rho"].astype(np.float64).ravel(),
            grid=grid,
            offsets={k: float(v) for k, v in meta["offsets"].items()},
            position_ids=tuple(meta["position_ids"]),
            info=meta["info"],
            version=int(d["version"]),
        )


def solve_voxels(sol: BaselineSolution, cfg: SiteConfig, *,
                 sigma: dict[str, np.ndarray] | None = None,
                 max_tan: float = 1.25,
                 transparent_quantile: float = 0.05,
                 cache_dir: str | Path | None = "runs/.cache",
                 holdouts: bool = True) -> dict[str, VoxelSolution]:
    """Full fit plus one single-position fit per position.

    Every fit shares one system matrix; a holdout is a row-weight mask, not a
    rebuild. That is why the matrix cache pays for itself even on a single run.
    """
    data = build_fit_data(sol, cfg, sigma=sigma, max_tan=max_tan,
                          transparent_quantile=transparent_quantile)
    fwd = build_forward_model(data.rows, cfg, cache_dir=cache_dir)
    rc = cfg.reconstruction

    def run(keep: np.ndarray) -> VoxelSolution:
        restricted = data.restricted(keep)
        x, info = solve(fwd, restricted, rc)
        info["n_rows_used"] = int(np.count_nonzero(restricted.w))
        info["algorithm"] = rc.algorithm
        return VoxelSolution(rho=x, grid=fwd.grid,
                             offsets=info.pop("offsets"),
                             position_ids=fwd.rows.position_ids,
                             info=info)

    out = {"full": run(np.ones(data.rows.n_rows, dtype=bool))}
    if holdouts:
        for pid in fwd.rows.position_ids:
            out[f"holdout_{pid}"] = run(fwd.rows.mask_for(pid))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_reconstruct.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add megido/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: voxel solve orchestration with per-position holdout fits"
```

---

### Task 7: Phantom machinery and the code-correctness gate

This is the first end-to-end gate. It answers one question — *is the chain built correctly?* — and it answers it honestly, which for a one-sided muon geometry is subtler than "does it recover the surface."

**What a one-sided geometry can and cannot do (established by measurement, not assumption).** Every ray in this system points along `normalize(sx, sy, 1)` — upward, by construction. Muons only come from above, so there is no view from the side or below, and the inverse problem is limited-angle no matter how many detectors or how wide the baseline. Measured on this exact code: a phantom's measurements are reproduced to `corr(pred, lambda) = 0.998`, and its *lateral* (column-integrated) structure is recovered to `corr = 0.78` — but its *depth* is smeared across the whole z-range (peak-layer mass / total = 0.19 against a truth of 1.0). That is not a defect; it is the null space of a one-sided system. The cafeteria project localized depth with a separate `layered`/autofocus mechanism that this phase deliberately does not port (spec §12 chose a full voxel field). So the honest gate is: **the code reproduces the data, recovers lateral structure, and does not falsely localize depth.** Asserting an interface-RMSE fidelity bound would be asserting something no correct one-sided reconstruction can deliver.

This gate uses `01_data/raw/topography.csv` to *shape* the phantom, per the spec's exit gate. `01_data/raw/calibration_open_sky.csv` sits beside it and MUST be ignored — Megiddo has no open-sky run, and a phantom that used one would validate a pipeline we do not have.

**Files:**
- Modify: `megido/phantom.py` (add `anomaly_from_topography` and `depth_localization`; the six helpers from the prior Task 7 attempt — `load_topography`, `surface_volume`, `sky_rows`, `project`, `interface_height`, `score` — are already present and correct, keep them unchanged)
- Test: `tests/test_phantom.py`

**Interfaces:**
- Consumes: `megido.voxels.VoxelGrid`, `megido.forward.build_forward_model`/`ForwardModel`, `megido.fitdata.FitData`/`RowIndex`, `megido.config.load_site_config`, `megido.inversion.solve`.
- Produces (new):
  - `anomaly_from_topography(grid, xs, ys, z_surface, z_anomaly_m, *, quantile=0.5, density=1.0) -> np.ndarray` — a flat `[n_voxels]` volume with a thin one-layer anomaly at height `z_anomaly_m`, present in every column whose sampled surface exceeds the `quantile` of the surface over the grid. The lateral footprint of the topography, placed at a single known depth.
  - `depth_localization(rho3, grid) -> float` — the fraction of total recovered mass in the single densest z-layer. 1.0 means perfectly localized in depth; ~1/nz means uniformly smeared.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phantom.py` (the helper-level tests plus the gate):

```python
from pathlib import Path

import numpy as np
import pytest

from megido.config import load_site_config
from megido.fitdata import FitData
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.phantom import (anomaly_from_topography, depth_localization,
                            interface_height, load_topography, project,
                            sky_rows, surface_volume)
from megido.voxels import VoxelGrid

TOPO = Path("/home/hadar/Cloud/Work/Postdoc/01_data/raw/topography.csv")


def test_topography_loads_as_a_regular_grid():
    xs, ys, z = load_topography(TOPO)
    assert (xs.size, ys.size) == (41, 41)
    assert z.shape == (41, 41)
    assert xs[0] == pytest.approx(-80.0) and xs[-1] == pytest.approx(80.0)
    assert np.isfinite(z).all()


def test_surface_volume_fills_below_the_surface_only():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 4))
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    z_surface = np.array([[2.0, 2.0], [2.0, 2.0]])
    rho = surface_volume(grid, xs, ys, z_surface, density=0.7).reshape(grid.shape)
    np.testing.assert_allclose(rho[:, :, :2], 0.7)
    np.testing.assert_allclose(rho[:, :, 2:], 0.0)


def test_sky_rows_covers_every_position_and_bin():
    rows = sky_rows(("pos0", "pos1"), t_max=1.0, n_bins=6)
    assert rows.n_rows == 2 * 36
    assert rows.mask_for("pos0").sum() == 36
    assert np.abs(rows.sx).max() < 1.0


def test_project_is_the_forward_model_plus_noise(tmp_path):
    rows = sky_rows(("pos0", "pos1"), t_max=0.8, n_bins=6)
    cfg = _cfg(tmp_path)
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    truth = np.full(fwd.grid.n_voxels, 0.1)

    clean = project(fwd, truth, sigma=0.0)
    np.testing.assert_allclose(clean.lam, fwd.predict(truth))
    assert isinstance(clean, FitData)

    noisy = project(fwd, truth, sigma=0.05, seed=1)
    assert np.std(noisy.lam - clean.lam) == pytest.approx(0.05, rel=0.3)


def test_interface_height_finds_the_top_of_a_filled_column():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 6))
    rho3 = np.zeros(grid.shape)
    rho3[:, :, :3] = 1.0
    np.testing.assert_allclose(interface_height(rho3, grid), 2.5)


def test_anomaly_sits_at_one_layer_and_follows_the_high_ground():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(4, 1, 6))
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0])
    surf = np.array([[1.0], [1.0], [9.0], [9.0]])   # high ground = last two columns
    vol = anomaly_from_topography(grid, xs, ys, surf, z_anomaly_m=3.0,
                                  quantile=0.5, density=1.0).reshape(grid.shape)
    # exactly one z-layer populated, the one containing z=3.0 (index 3)
    assert set(np.nonzero(vol.sum(axis=(0, 1)))[0].tolist()) == {3}
    # only the high-ground columns carry the anomaly
    col = vol.sum(axis=2)[:, 0]
    np.testing.assert_allclose(col > 0, [False, False, True, True])


def test_depth_localization_is_one_for_a_single_layer_and_small_when_smeared():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 5))
    sharp = np.zeros(grid.shape)
    sharp[:, :, 2] = 1.0
    assert depth_localization(sharp, grid) == pytest.approx(1.0)

    flat = np.ones(grid.shape)
    assert depth_localization(flat, grid) == pytest.approx(1.0 / 5)


def _cfg(tmp_path):
    """25-view dense config over a +-30 m grid: generous multi-position baseline.

    This is the FAVOURABLE geometry — as many well-separated views as a phantom
    can have. It gates the CODE. It is still one-sided (every ray points up), so
    even here depth is not localized; that is asserted below, not worked around.
    """
    dets = [(x, y) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)]
    blk = "\n".join(
        f"  - id: D{i}\n    runs: DET{i}-DET{i}\n"
        f"    pose: {{x: {dx}, y: {dy}, z: 0, tilt_deg: 0, az_deg: 0}}"
        for i, (dx, dy) in enumerate(dets))
    p = tmp_path / "phantom.yaml"
    p.write_text(
        "site: phantom\ndata_dir: /tmp\n"
        "volume: {z_min_m: 0.0, z_max_m: 8.0, spacing_m: 1.0, n_aperture_sub: 2,\n"
        "         xy_m: [[-30, 30], [-30, 30]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 400, tv_alpha: 0.002, tv_z_weight: 0.3}\n"
        "exposures:\n" + blk + "\n")
    return load_site_config(p)


@pytest.mark.skipif(not TOPO.exists(), reason="topography.csv not on this machine")
def test_the_code_reproduces_data_recovers_lateral_structure_and_does_not_fake_depth(
        tmp_path, capsys):
    """THE TASK 7 GATE.

    Twenty-five detectors over a +-30 m grid view a thin anomaly whose lateral
    footprint is the topography surface's high ground, placed at a single known
    height. Truth is built and projected at HALF the inversion spacing, so the
    solver never inverts its own discretisation.

    Three assertions, and each is the honest one:
      1. The reconstruction reproduces the measurements (data-space). This is
         what the forward model + solver provably must do, and it is what caught
         the detector-inside-grid ray bug during this task's development.
      2. Lateral (column-integrated) structure is recovered. This is the part of
         the scene a one-sided geometry actually constrains.
      3. Depth is NOT localized. Truth is one layer; the reconstruction must
         smear it, because muons arrive from above only. A gate that let the
         reconstruction claim sharp depth would be rewarding a lie.

    Do NOT add an interface-RMSE fidelity assertion here and do NOT loosen these
    three. If assertion 1 fails, the forward model or solver is broken — a defect
    to find and report, never a threshold to lower.
    """
    cfg = _cfg(tmp_path)
    xs, ys, z = load_topography(TOPO)
    rows = sky_rows(tuple(e.id.replace("D", "pos") for e in cfg.exposures),
                    t_max=1.2, n_bins=25)

    import dataclasses
    fine_cfg = dataclasses.replace(
        cfg, volume=dataclasses.replace(cfg.volume, spacing_m=0.5))
    fine = build_forward_model(rows, fine_cfg, cache_dir=None)
    coarse = build_forward_model(rows, cfg, cache_dir=None)

    truth = anomaly_from_topography(fine.grid, xs, ys, z, z_anomaly_m=4.0,
                                    quantile=0.5, density=1.0)
    data = project(fine, truth, sigma=0.01, seed=1)
    data = FitData(lam=data.lam, w=data.w, rows=rows)

    x, info = solve(coarse, data, cfg.reconstruction)

    pred = coarse.predict(x, info["offsets"])
    data_corr = float(np.corrcoef(pred, data.lam)[0, 1])
    rel_res = float(np.linalg.norm(pred - data.lam)
                    / np.linalg.norm(data.lam - data.lam.mean()))

    truth_c = anomaly_from_topography(coarse.grid, xs, ys, z, z_anomaly_m=4.0)
    cr = x.reshape(coarse.grid.shape).sum(axis=2)
    ct = truth_c.reshape(coarse.grid.shape).sum(axis=2)
    lateral = float(np.corrcoef(cr.ravel(), ct.ravel())[0, 1])
    depth = depth_localization(x.reshape(coarse.grid.shape), coarse.grid)

    print(f"\nTask 7 gate: data corr={data_corr:.4f} rel-res={rel_res:.4f} "
          f"lateral col-corr={lateral:.3f} depth peak/total={depth:.3f} "
          f"(truth depth=1.0)")

    assert data_corr > 0.99          # code reproduces the measurements
    assert rel_res < 0.15
    assert lateral > 0.6             # lateral structure recovered
    assert depth < 0.5               # depth NOT falsely localized
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_phantom.py -q`
Expected: FAIL — `ImportError` on `anomaly_from_topography` / `depth_localization`.

- [ ] **Step 3: Add the two helpers to `megido/phantom.py`**

Append these; leave the existing six helpers untouched:

```python
def anomaly_from_topography(grid: VoxelGrid, xs: np.ndarray, ys: np.ndarray,
                            z_surface: np.ndarray, z_anomaly_m: float, *,
                            quantile: float = 0.5, density: float = 1.0) -> np.ndarray:
    """A thin one-layer density anomaly at height z_anomaly_m.

    The anomaly is present in every voxel column whose sampled surface height
    exceeds the `quantile` of the surface over the grid — so its lateral
    footprint is the topography's high ground, but its DEPTH is a single known
    layer. This is the phantom a one-sided geometry can actually be tested
    against: the lateral pattern is recoverable, and placing it at one depth
    makes the depth null space measurable (a correct reconstruction smears it).
    """
    gx = grid.axis_centers(0)
    gy = grid.axis_centers(1)
    ix = np.abs(xs[None, :] - gx[:, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - gy[:, None]).argmin(axis=1)
    h = z_surface[np.ix_(ix, iy)]                       # [nx, ny]
    footprint = h > np.quantile(h, quantile)

    iz = int(np.argmin(np.abs(grid.axis_centers(2) - z_anomaly_m)))
    rho = np.zeros(grid.shape)
    rho[footprint, iz] = float(density)
    return rho.ravel()


def depth_localization(rho3: np.ndarray, grid: VoxelGrid) -> float:
    """Fraction of total mass in the single densest z-layer.

    1.0 means the reconstruction put everything at one depth; ~1/nz means it
    smeared uniformly. For a one-sided muon geometry a correct reconstruction of
    a single-layer truth scores LOW here — that is the honest signature of the
    depth null space, not a failure.
    """
    zsum = np.asarray(rho3).reshape(grid.shape).sum(axis=(0, 1))
    total = zsum.sum()
    return float(zsum.max() / total) if total > 0 else 0.0
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_phantom.py -q -s`
Expected: PASS, 8 tests. Read the printed gate line — it should read roughly `data corr≈0.998 rel-res≈0.06 lateral col-corr≈0.78 depth peak/total≈0.19`.

- [ ] **Step 5: If the gate fails, diagnose — do not loosen**

Assertion 1 (data corr) failing means the forward model or solver is broken: stop and report which, with the number you got. Assertion 2 (lateral) failing after assertion 1 passes means the geometry is under-sampling — report it; do not lower the floor. Assertion 3 is a ceiling, not a floor: if depth localization comes out *high*, something is wrong (a one-sided geometry should not localize depth), and that too is a report, not a tweak. Record all four printed numbers in your report regardless.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add megido/phantom.py tests/test_phantom.py
git commit -m "feat: phantom machinery and the honest one-sided-geometry code gate"
```

---

### Task 8: Depth resolution, aliasing, and per-voxel view count

Spec §7.4 makes uncertainty a first-class deliverable, and in a limited-angle two-position campaign the dominant term is depth. This task computes what the geometry can resolve *before* any data is fitted, so Task 9's phantom has a prediction to be checked against rather than a number to be admired.

**Files:**
- Create: `megido/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: `megido.config.SiteConfig`/`Volume`, `megido.fitdata.position_origins`, `megido.forward.ForwardModel`.
- Produces:
  - `depth_resolution(z_m, baseline_m, sigma_t) -> float`
  - `alias_period(z_m, baseline_m, feature_pitch_m) -> float`
  - `position_baselines(cfg) -> dict[tuple[str, str], float]`
  - `views_per_voxel(fwd) -> np.ndarray[nx, ny, nz]` — how many distinct positions have a ray through each voxel
  - `campaign_resolution(cfg, *, sigma_t, feature_pitch_m) -> dict`
  - `format_resolution(report: dict) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolution.py`:

```python
import numpy as np
import pytest
from scipy import sparse

from megido.config import load_site_config
from megido.fitdata import RowIndex
from megido.forward import ForwardModel
from megido.resolution import (alias_period, campaign_resolution,
                               depth_resolution, format_resolution,
                               position_baselines, views_per_voxel)
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 2.0, z_max_m: 10.0, spacing_m: 0.25}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def test_depth_resolution_grows_with_the_square_of_distance():
    a = depth_resolution(5.0, baseline_m=2.2, sigma_t=0.05)
    b = depth_resolution(10.0, baseline_m=2.2, sigma_t=0.05)
    assert b == pytest.approx(4 * a, rel=1e-9)


def test_depth_resolution_improves_with_a_longer_baseline():
    assert (depth_resolution(8.0, baseline_m=5.0, sigma_t=0.05)
            < depth_resolution(8.0, baseline_m=2.2, sigma_t=0.05))


def test_depth_resolution_matches_the_closed_form():
    """Two rays converging at z from detectors b apart: Delta_t = b/z, so
    dz = z^2/b * d(Delta_t), and the two views' angular errors add in quadrature."""
    z, b, s = 7.0, 2.2, 0.05
    assert depth_resolution(z, b, s) == pytest.approx(np.sqrt(2) * s * z**2 / b)


def test_a_zero_baseline_has_no_depth_resolution():
    assert depth_resolution(7.0, baseline_m=0.0, sigma_t=0.05) == float("inf")


def test_alias_period_is_the_spec_formula():
    assert alias_period(7.0, baseline_m=2.2, feature_pitch_m=1.0) == pytest.approx(
        1.0 * 7.0 / 2.2)


def test_baselines_are_measured_between_positions_not_exposures(tmp_path):
    """P0 and T20 share a spot: two tilts, one position, zero baseline."""
    b = position_baselines(_cfg(tmp_path))
    assert b == {("pos0", "pos1"): pytest.approx(2.2)}


def test_views_per_voxel_counts_distinct_positions():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 1, 1))
    # row 0 (pos0) hits voxel 0; rows 1 and 2 (pos1) both hit voxel 0; row 3 hits voxel 1
    A = sparse.csr_matrix(np.array([[1.0, 0.0],
                                    [2.0, 0.0],
                                    [0.5, 0.0],
                                    [0.0, 1.0]]))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.array([0, 1, 1, 1]),
                    sx=np.zeros(4), sy=np.zeros(4), sky_flat=np.arange(4))
    v = views_per_voxel(ForwardModel(A=A, grid=grid, rows=rows))
    assert v.shape == grid.shape
    assert v.ravel().tolist() == [2, 1]


def test_campaign_resolution_reports_the_megiddo_numbers(tmp_path):
    r = campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == pytest.approx(2.2)
    assert r["n_positions"] == 2
    assert set(r["depth_resolution_m"]) == {"z_min", "z_mid", "z_max"}
    # 10 m away on a 2.2 m baseline is hopeless and the report must say so
    assert r["depth_resolution_m"]["z_max"] > 2.0
    assert r["z_range_m"] == pytest.approx((2.0, 10.0))


def test_campaign_resolution_flags_when_depth_is_unresolved(tmp_path):
    r = campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["depth_resolved"] is False
    assert "depth" in r["verdict"].lower()


def test_a_single_position_campaign_has_no_depth_information(tmp_path):
    p = tmp_path / "one.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {z_min_m: 2.0, z_max_m: 10.0, spacing_m: 0.25}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n")
    r = campaign_resolution(load_site_config(p), sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == 0.0
    assert r["depth_resolution_m"]["z_mid"] == float("inf")
    assert r["depth_resolved"] is False


def test_the_report_formats_without_raising(tmp_path):
    text = format_resolution(
        campaign_resolution(_cfg(tmp_path), sigma_t=0.05, feature_pitch_m=1.0))
    assert "baseline" in text.lower()
    assert "2.2" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_resolution.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.resolution'`.

- [ ] **Step 3: Write `megido/resolution.py`**

```python
"""What this geometry can resolve, computed before any data is fitted.

Spec section 7.4 makes uncertainty a deliverable rather than an appendix, and in
a limited-angle campaign with few viewpoints the dominant error is depth. These
are closed-form geometric statements: they depend on where the detector stood
and how finely angles are binned, not on the counts, so they are the honest
prior on what any reconstruction of this campaign can possibly mean.

The Megiddo campaign has ONE baseline: P0 and T20 share a spot (two tilts, one
translation), so only P0 <-> P1 at 2.2 m carries parallax.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from megido.config import SiteConfig
from megido.fitdata import position_origins
from megido.forward import ForwardModel


def depth_resolution(z_m: float, baseline_m: float, sigma_t: float) -> float:
    """Depth uncertainty of a feature at height z, seen from two views b apart.

    Two rays converging at z from detectors b apart differ in tangent by
    Delta_t = b / z, so dz = (z^2 / b) * d(Delta_t). Each view contributes
    sigma_t, adding in quadrature:  dz = sqrt(2) * sigma_t * z^2 / b.
    """
    if baseline_m <= 0:
        return float("inf")
    return math.sqrt(2.0) * sigma_t * z_m**2 / baseline_m


def alias_period(z_m: float, baseline_m: float, feature_pitch_m: float) -> float:
    """Height spacing of false solutions for a periodic structure (spec 7.4).

    A pattern of pitch p at height z reprojects onto itself when the depth shifts
    by p * z / b. The cafeteria campaign had 5.8 m of margin against this; a
    synthetic test there aliased at 0.9 m.
    """
    if baseline_m <= 0:
        return float("inf")
    return feature_pitch_m * z_m / baseline_m


def position_baselines(cfg: SiteConfig) -> dict[tuple[str, str], float]:
    """Horizontal separation of every pair of POSITIONS (not exposures)."""
    origins = position_origins(cfg)
    out: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(sorted(origins), 2):
        pa, pb = origins[a], origins[b]
        out[(a, b)] = float(math.dist(pa, pb))
    return out


def views_per_voxel(fwd: ForwardModel) -> np.ndarray:
    """How many distinct positions have a ray through each voxel.

    This is the map the spec's viewer calls the exposure-contribution layer: it
    separates where the answer is data from where it is prior. A voxel seen by
    one position carries no depth information at all — whatever the solver puts
    there came from the regulariser, not from a measurement.
    """
    A = fwd.A.tocsr()
    seen = np.zeros((len(fwd.rows.position_ids), A.shape[1]), dtype=bool)
    for i in range(len(fwd.rows.position_ids)):
        rows = np.nonzero(fwd.rows.pos_of_row == i)[0]
        if rows.size:
            block = A[rows]
            seen[i, np.unique(block.indices)] = True
    return seen.sum(axis=0).reshape(fwd.grid.shape).astype(np.int16)


def campaign_resolution(cfg: SiteConfig, *, sigma_t: float,
                        feature_pitch_m: float) -> dict:
    """Closed-form resolution report for a configured campaign.

    `sigma_t` is the angular bin width of the sky grid (0.05 for this campaign);
    `feature_pitch_m` is the scale of structure being looked for, still an open
    item in spec section 11 — pass the value being assumed and it is recorded in
    the report rather than buried.
    """
    baselines = position_baselines(cfg)
    max_b = max(baselines.values(), default=0.0)
    z0, z1 = float(cfg.volume.z_min_m), float(cfg.volume.z_max_m)
    zmid = 0.5 * (z0 + z1)

    dz = {"z_min": depth_resolution(z0, max_b, sigma_t),
          "z_mid": depth_resolution(zmid, max_b, sigma_t),
          "z_max": depth_resolution(z1, max_b, sigma_t)}
    # "Resolved" means the depth error at mid-range is smaller than a quarter of
    # the range being solved for. Anything coarser cannot place a surface within
    # the volume, whatever the reconstruction looks like.
    resolved = dz["z_mid"] < 0.25 * (z1 - z0)

    verdict = (
        f"depth RESOLVED at mid-range: dz = {dz['z_mid']:.2f} m over a "
        f"{z1 - z0:.1f} m range"
        if resolved else
        f"depth NOT resolved: dz = {dz['z_mid']:.2f} m at z = {zmid:.1f} m is "
        f"{dz['z_mid'] / max(z1 - z0, 1e-9):.0%} of the {z1 - z0:.1f} m range. "
        f"Lateral structure is still measured; the height of that structure is "
        f"set by the regulariser, not by the data."
    )

    return {
        "n_positions": len(position_origins(cfg)),
        "baselines_m": {f"{a}-{b}": round(v, 4) for (a, b), v in baselines.items()},
        "max_baseline_m": max_b,
        "sigma_t": sigma_t,
        "z_range_m": (z0, z1),
        "depth_resolution_m": dz,
        "feature_pitch_m": feature_pitch_m,
        "alias_period_m": alias_period(zmid, max_b, feature_pitch_m),
        "depth_resolved": bool(resolved),
        "verdict": verdict,
    }


def format_resolution(report: dict) -> str:
    lines = [
        f"positions            {report['n_positions']}",
        f"max baseline         {report['max_baseline_m']:.3f} m",
        f"angular bin sigma_t  {report['sigma_t']:.4f}",
        f"z range              {report['z_range_m'][0]:.2f} - "
        f"{report['z_range_m'][1]:.2f} m",
        "depth resolution     " + "  ".join(
            f"{k}={v:.2f} m" for k, v in report["depth_resolution_m"].items()),
        f"alias period         {report['alias_period_m']:.2f} m "
        f"(assuming {report['feature_pitch_m']:.2f} m feature pitch)",
        "",
        report["verdict"],
    ]
    if report["baselines_m"]:
        lines.insert(1, "  " + "  ".join(
            f"{k}={v:.3f} m" for k, v in sorted(report["baselines_m"].items())))
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_resolution.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/resolution.py tests/test_resolution.py
git commit -m "feat: closed-form depth resolution, aliasing, and per-voxel view count"
```

---

### Task 9: The Megiddo-geometry gate — what this campaign can actually claim

Task 7 showed the code works when the geometry is generous. This task runs the same code at the **real campaign poses** and pins down what may honestly be said about the real result. It is the gate that constrains every claim in the Phase 3 report.

The expected outcome is *lateral structure recovered, depth not*. That is not a failure — it is the measurement. What would be a failure is a reconstruction that looks confidently three-dimensional while `resolution.py` says the depth information is absent.

**Files:**
- Modify: `tests/test_phantom.py` (add the Megiddo gate)
- Modify: `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` (record the outcome in §11 open items — "structure scale" now has evidence)

**Interfaces:**
- Consumes everything from Tasks 1–8. Produces no new module.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phantom.py`:

```python
def test_megiddo_geometry_recovers_lateral_structure_but_not_depth(tmp_path, capsys):
    """THE TASK 9 GATE — the honest one.

    Real campaign poses: P0/T20 at the origin (two tilts, ONE position) and P1
    at 2.2 m. One baseline, 2.2 m, against rock several metres away. A slab with
    a lateral notch is injected; the gate asserts that the notch is found in x/y
    and that the slab's HEIGHT is not determined better than the closed-form
    depth resolution allows.

    A reconstruction that beat the analytic dz here would not be good news: with
    truth this close to the TV prior, it would mean the regulariser was
    reproducing the phantom rather than the data doing so.
    """
    import numpy as np

    from megido.config import load_site_config
    from megido.forward import build_forward_model
    from megido.inversion import solve
    from megido.phantom import project, score, sky_rows, surface_volume
    from megido.resolution import campaign_resolution, views_per_voxel

    p = tmp_path / "megiddo.yaml"
    p.write_text(
        "site: megido-phantom\ndata_dir: /tmp\n"
        "volume: {z_min_m: 1.0, z_max_m: 11.0, spacing_m: 0.5, n_aperture_sub: 2,\n"
        "         xy_m: [[-10.0, 12.0], [-10.0, 10.0]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 120, tv_alpha: 0.02, tv_z_weight: 0.5}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n"
        "  - id: T20\n    runs: DET3-DET4\n"
        "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}\n"
        "  - id: P1\n    runs: DET5-DET6\n"
        "    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n")
    cfg = load_site_config(p)

    res = campaign_resolution(cfg, sigma_t=0.05, feature_pitch_m=2.0)
    assert res["max_baseline_m"] == pytest.approx(2.2)
    assert res["n_positions"] == 2

    rows = sky_rows(("pos0", "pos1"), t_max=1.0, n_bins=28)
    fwd = build_forward_model(rows, cfg, cache_dir=None)

    # Truth: rock filling z < 6 m, with a 4 m wide notch cut out in x.
    gx = fwd.grid.axis_centers(0)
    gy = fwd.grid.axis_centers(1)
    xs = gx
    ys = gy
    h = np.full((gx.size, gy.size), 6.0)
    h[(gx > -2.0) & (gx < 2.0), :] = 3.0
    truth = surface_volume(fwd.grid, xs, ys, h, density=0.2)

    data = project(fwd, truth, sigma=0.05, seed=11)
    x, info = solve(fwd, data, cfg.reconstruction)

    # --- lateral: the notch must be found ---
    col_rec = x.reshape(fwd.grid.shape).sum(axis=2)
    col_tru = truth.reshape(fwd.grid.shape).sum(axis=2)
    lateral = float(np.corrcoef(col_rec.ravel(), col_tru.ravel())[0, 1])

    # --- depth: how well is the interface height placed? ---
    s = score(x, truth, fwd.grid)
    dz_pred = res["depth_resolution_m"]["z_mid"]
    views = views_per_voxel(fwd)

    print(f"\nTask 9 gate (Megiddo geometry)"
          f"\n  lateral column corr     {lateral:.3f}"
          f"\n  voxel corr              {s['corr']:.3f}"
          f"\n  interface rmse          {s['interface_rmse_m']:.2f} m"
          f"\n  interface bias          {s['interface_bias_m']:+.2f} m"
          f"\n  analytic dz at z_mid    {dz_pred:.2f} m"
          f"\n  voxels seen by 2 views  {int((views == 2).sum())} of {views.size}"
          f"\n  {res['verdict']}")

    # Lateral structure IS measured.
    assert lateral > 0.6

    # Depth is NOT. The reconstruction must not appear to beat the geometry;
    # if it does, the phantom is being reproduced by the prior, not the data.
    assert s["interface_rmse_m"] > 0.3 * dz_pred

    # And the resolution report must have said so up front.
    assert res["depth_resolved"] is False
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_phantom.py::test_megiddo_geometry_recovers_lateral_structure_but_not_depth -q -s`
Expected: PASS. Read every printed number.

- [ ] **Step 3: If the lateral assertion fails, diagnose before touching the threshold**

A lateral correlation below 0.6 with two positions and a 4 m notch at 6 m means something upstream is wrong — most likely the ray directions (Task 3) or the grid extent (Task 1). Investigate and report; do not lower the number.

If lateral correlation is comfortably above 0.6, tighten it to roughly 90% of the achieved value and re-run.

- [ ] **Step 4: Record the finding in the spec**

In `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` §11, replace the "Structure scale" bullet with:

```markdown
- **Structure scale** — still needed to finalize grid spacing, but no longer
  blocking: Phase 3's Megiddo-geometry phantom (`tests/test_phantom.py`,
  `test_megiddo_geometry_recovers_lateral_structure_but_not_depth`) measured
  what the campaign resolves. Lateral structure is recovered; the HEIGHT of that
  structure is set by the regulariser, not the data, because the campaign has a
  single 2.2 m baseline. `megido.resolution.campaign_resolution` reports the
  closed-form number for any configured volume. A second translated position is
  what would change this — a larger tilt at the same spot would not.
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_phantom.py docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md
git commit -m "test: Megiddo-geometry gate — lateral structure resolved, depth is not"
```

---

### Task 10: Poisson bootstrap uncertainty and the S2 systematic map

Spec §7.4 requires three uncertainty products. Task 8 delivered the analytic `δz` and the alias period. This task delivers the other two: a statistical per-voxel σ from resampled counts, and the Phase 2 null-space direction propagated into the volume as a **separate** systematic map — kept separate because it is not a random error and must not be averaged into one.

**Files:**
- Create: `megido/voxuncert.py`
- Test: `tests/test_voxuncert.py`

**Interfaces:**
- Consumes: `megido.angular.AnalysisGrid`, `megido.baseline.solve_baseline`, `megido.reconstruct.solve_voxels`, `megido.validate2.opacity_uncertainty`.
- Produces:
  - `voxel_bootstrap(grid, cfg, *, n_replicas=8, seed=0, cache_dir=None, solve_kwargs=None) -> BootstrapResult`
  - `BootstrapResult(mean, sigma, grid, n_replicas)` with `snr()` and `save(path)`
  - `systematic_map(sol, cfg, *, base_quantile=0.05, alt_quantile=0.25, cache_dir=None) -> np.ndarray` — the volume difference between two gauge conventions, i.e. how much of the answer is the gauge choice rather than the data

- [ ] **Step 1: Write the failing test**

Create `tests/test_voxuncert.py`:

```python
import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.config import load_site_config
from megido.voxuncert import BootstrapResult, systematic_map, voxel_bootstrap

CONFIG = """
site: t
data_dir: /tmp
binning: {t_max: 1.25, n_bins: 20}
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}
reconstruction: {algorithm: tv, n_iter: 20, tv_alpha: 0.01}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: T20
    runs: DET3-DET4
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 0}
  - id: P1
    runs: DET5-DET6
    pose: {x: 2.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _grid(seed=0):
    """A small analysis grid with plausible counts for all three exposures."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1.25, 1.25, 21)
    tx = 0.5 * (edges[:-1] + edges[1:])
    shape = np.exp(-(tx[:, None] ** 2 + tx[None, :] ** 2))
    counts = {eid: rng.poisson(2000 * shape).astype(np.int64)
              for eid in ("P0", "T20", "P1")}
    return AnalysisGrid(edges=edges, counts=counts)


def test_bootstrap_returns_a_mean_and_sigma_on_the_solve_grid(tmp_path):
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=3,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    assert isinstance(r, BootstrapResult)
    assert r.mean.shape == r.grid.shape
    assert r.sigma.shape == r.grid.shape
    assert r.n_replicas == 3
    assert np.all(r.sigma >= 0.0)


def test_bootstrap_sigma_is_not_identically_zero(tmp_path):
    """Zero sigma everywhere means the replicas were not actually resampled."""
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=4,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    assert r.sigma.max() > 0.0


def test_bootstrap_is_reproducible_for_a_fixed_seed(tmp_path):
    kw = dict(n_replicas=3, cache_dir=None, solve_kwargs={"n_iter": 30})
    a = voxel_bootstrap(_grid(), _cfg(tmp_path), seed=5, **kw)
    b = voxel_bootstrap(_grid(), _cfg(tmp_path), seed=5, **kw)
    np.testing.assert_allclose(a.sigma, b.sigma)


def test_snr_is_nan_where_sigma_is_zero_not_infinite(tmp_path):
    r = BootstrapResult(mean=np.ones((2, 2, 2)), sigma=np.zeros((2, 2, 2)),
                        grid=None, n_replicas=3)
    assert np.all(np.isnan(r.snr()))


def test_bootstrap_saves_and_reloads(tmp_path):
    r = voxel_bootstrap(_grid(), _cfg(tmp_path), n_replicas=3,
                        cache_dir=None, solve_kwargs={"n_iter": 30})
    p = tmp_path / "u.npz"
    r.save(p)
    d = np.load(p)
    np.testing.assert_allclose(d["sigma"], r.sigma.astype(np.float32))
    assert int(d["n_replicas"]) == 3


def test_systematic_map_is_nonzero_when_the_gauge_choice_matters(tmp_path):
    """A zero systematic map here would not mean the gauge is harmless; it
    would mean the probe is additive and normalized_opacity cancelled it."""
    from megido.baseline import solve_baseline

    cfg = _cfg(tmp_path)
    sol = solve_baseline(_grid(), cfg, n_iter=40)
    sysmap = systematic_map(sol, cfg, cache_dir=None)
    assert np.isfinite(sysmap).all()
    assert np.abs(sysmap).max() > 0.0


def test_systematic_map_is_zero_when_both_quantiles_agree(tmp_path):
    from megido.baseline import solve_baseline

    cfg = _cfg(tmp_path)
    sol = solve_baseline(_grid(), cfg, n_iter=40)
    same = systematic_map(sol, cfg, base_quantile=0.05, alt_quantile=0.05,
                          cache_dir=None)
    np.testing.assert_allclose(same, 0.0, atol=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_voxuncert.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.voxuncert'`.

- [ ] **Step 3: Write `megido/voxuncert.py`**

```python
"""Statistical and systematic uncertainty of the voxel volume.

Spec section 7.4 names three products. Two live here:

  * a per-voxel statistical sigma from a Poisson bootstrap over the RAW COUNTS,
    pushed through the entire chain — Phase 2 baseline solve included — so the
    spread carries the baseline's own uncertainty rather than pretending the
    detector response is known;
  * the Phase 2 gauge direction as a SEPARATE systematic map. The absolute level
    of lambda is not measured (spec section 6.4), and which convention is chosen
    moves the volume. That is not a random error and must never be averaged into
    sigma.

The third, the analytic depth resolution, is geometric and lives in
megido.resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.config import SiteConfig
from megido.reconstruct import solve_voxels
from megido.voxels import VoxelGrid


@dataclass(frozen=True)
class BootstrapResult:
    mean: np.ndarray            # [nx, ny, nz]
    sigma: np.ndarray           # [nx, ny, nz]
    grid: VoxelGrid | None
    n_replicas: int
    replica_seeds: tuple[int, ...] = field(default=(), repr=False)

    def snr(self) -> np.ndarray:
        """mean / sigma, NaN where sigma is zero.

        Zero spread across replicas does not mean infinite confidence; it means
        the voxel never moved, which for a voxel no ray reaches is a statement
        about coverage, not about certainty.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.sigma > 0, self.mean / self.sigma, np.nan)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            sigma=self.sigma.astype(np.float32),
            snr=self.snr().astype(np.float32),
            n_replicas=np.asarray(self.n_replicas, dtype=np.int64),
            origin=np.asarray(self.grid.origin if self.grid else (0.0, 0.0, 0.0)),
            spacing=np.asarray(self.grid.spacing if self.grid else 0.0),
        )


def voxel_bootstrap(grid: AnalysisGrid, cfg: SiteConfig, *,
                    n_replicas: int = 8, seed: int = 0,
                    cache_dir: str | Path | None = "runs/.cache",
                    solve_kwargs: dict | None = None) -> BootstrapResult:
    """Poisson-resample the counts and re-run the WHOLE chain per replica.

    Resampling counts and re-solving only the voxels would hold the baseline
    fixed and understate the error: with no open-sky run the detector response
    is fitted from the same counts, so its uncertainty is part of the answer's.
    That makes each replica a full baseline solve, which is why the default
    replica count is small and the CLI exposes it.
    """
    rng = np.random.default_rng(seed)
    kw = dict(solve_kwargs or {})
    stack: list[np.ndarray] = []
    vgrid: VoxelGrid | None = None

    for _ in range(n_replicas):
        counts = {eid: rng.poisson(v).astype(np.int64)
                  for eid, v in grid.counts.items()}
        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=counts), cfg, **kw)
        fits = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False)
        vgrid = fits["full"].grid
        stack.append(fits["full"].rho3())

    arr = np.stack(stack)
    return BootstrapResult(mean=arr.mean(axis=0), sigma=arr.std(axis=0),
                           grid=vgrid, n_replicas=n_replicas)


def systematic_map(sol: BaselineSolution, cfg: SiteConfig, *,
                   base_quantile: float = 0.05,
                   alt_quantile: float = 0.25,
                   cache_dir: str | Path | None = "runs/.cache") -> np.ndarray:
    """How much of the volume comes from the gauge choice rather than the data.

    Phase 2 leaves one flat direction per position: adding a constant to
    lambda_p and rescaling that position's normalization changes nothing
    observable (spec section 6.4). `normalized_opacity` picks a convention — the
    5th percentile of each map is called open sky — and the inversion's
    per-position offsets absorb most, but not all, of a different choice.

    This map is the volume difference between the chosen convention and a
    deliberately different one. It is a systematic, not a sigma: it does not
    shrink with more counts, and it is reported alongside the statistical error
    rather than combined with it.

    The knob is the QUANTILE, not an additive shift. normalized_opacity
    subtracts its own quantile, so adding a constant to every opacity value
    leaves the result bit-identical — an additive probe would report a
    systematic of exactly zero and look reassuring while measuring nothing.
    """
    base = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False,
                        transparent_quantile=base_quantile)["full"]
    alt = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False,
                       transparent_quantile=alt_quantile)["full"]
    return alt.rho3() - base.rho3()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_voxuncert.py -q`
Expected: PASS, 7 tests. If any test exceeds ~30 s, reduce `n_iter` in the fixture config, never the number of replicas below 3.

- [ ] **Step 5: Commit**

```bash
git add megido/voxuncert.py tests/test_voxuncert.py
git commit -m "feat: per-voxel Poisson bootstrap and the gauge-choice systematic map"
```

---

### Task 11: Model-free backprojection

The anchor the inversion is judged against: each measured opacity placed on a horizontal plane along its own ray, with no solver and no regulariser. Whatever appears here is what the detector saw. Anything in the reconstruction that is *not* here came from the prior.

**Files:**
- Create: `megido/backproject.py`
- Test: `tests/test_backproject.py`

**Interfaces:**
- Consumes: `megido.fitdata.FitData`, `megido.fitdata.position_origins`, `megido.config.SiteConfig`.
- Produces:
  - `backproject_plane(data, cfg, z_m, xs, ys) -> tuple[dict[str, np.ndarray], np.ndarray]` — `({pid: grid[nx, ny]}, mean_grid)`, NaN where a position has no coverage
  - `plane_axes(cfg, z_m, t_reach, res_m) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backproject.py`:

```python
import numpy as np
import pytest

from megido.backproject import backproject_plane, plane_axes
from megido.config import load_site_config
from megido.fitdata import FitData, RowIndex

CONFIG = """
site: t
data_dir: /tmp
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
  - id: P1
    runs: DET3-DET4
    pose: {x: 4.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}
"""


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _data(sx, sy, lam, pos):
    sx, sy, lam = map(np.asarray, (sx, sy, lam))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.asarray(pos), sx=sx, sy=sy,
                    sky_flat=np.arange(sx.size))
    return FitData(lam=lam.astype(float), w=np.ones(sx.size), rows=rows)


def test_a_vertical_ray_lands_directly_above_its_detector(tmp_path):
    data = _data([0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [0, 1])
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ys = np.array([-1.0, 0.0, 1.0])
    per, _ = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)

    i0 = int(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)) // ys.size)
    i1 = int(np.nanargmax(np.nan_to_num(per["pos1"], nan=-np.inf)) // ys.size)
    assert xs[i0] == pytest.approx(0.0)
    assert xs[i1] == pytest.approx(4.0)


def test_a_tilted_ray_lands_at_the_lever_arm_offset(tmp_path):
    """A ray of tangent 0.4 from z = 0 reaches x = 0.4 * 5 = 2 m at z = 5."""
    data = _data([0.4], [0.0], [3.0], [0])
    xs = np.linspace(-1.0, 5.0, 13)
    ys = np.array([-0.5, 0.0, 0.5])
    per, _ = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    hit = np.unravel_index(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)),
                           per["pos0"].shape)
    assert xs[hit[0]] == pytest.approx(2.0, abs=0.3)


def test_uncovered_pixels_are_nan_not_zero(tmp_path):
    data = _data([0.0], [0.0], [1.0], [0])
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    per, mean = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    assert np.isnan(per["pos1"]).all()
    assert np.isnan(mean).any()


def test_the_mean_ignores_positions_with_no_coverage(tmp_path):
    data = _data([0.0, 0.05, -0.05], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0, 0, 0])
    xs = np.linspace(-1.0, 1.0, 9)
    ys = np.linspace(-1.0, 1.0, 9)
    _, mean = backproject_plane(data, _cfg(tmp_path), z_m=5.0, xs=xs, ys=ys)
    covered = mean[np.isfinite(mean)]
    assert covered.size > 0
    np.testing.assert_allclose(covered, 2.0, atol=1e-6)


def test_plane_axes_spans_every_footprint(tmp_path):
    xs, ys = plane_axes(_cfg(tmp_path), z_m=5.0, t_reach=1.0, res_m=0.5)
    assert xs[0] <= -5.0 and xs[-1] >= 9.0
    assert np.allclose(np.diff(xs), 0.5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_backproject.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.backproject'`.

- [ ] **Step 3: Write `megido/backproject.py`**

```python
"""Single-plane backprojection of the measured opacity: the model-free view.

Each measured lambda is placed where its own ray crosses a horizontal plane. No
solver, no regulariser, no grid prior. This is what the detector saw, in site
coordinates.

It is the reference the 3D reconstruction is read against: structure present in
the backprojection is data; structure present only in the reconstruction came
from the regulariser and must be described that way.
"""
from __future__ import annotations

import numpy as np

from megido.config import SiteConfig
from megido.fitdata import FitData, position_origins


def plane_axes(cfg: SiteConfig, z_m: float, t_reach: float,
               res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a plane covering every position's footprint at z_m."""
    origins = position_origins(cfg)
    xs_b, ys_b = [], []
    for px, py, pz in origins.values():
        reach = t_reach * max(z_m - pz, 0.0)
        xs_b += [px - reach, px + reach]
        ys_b += [py - reach, py + reach]
    return (np.arange(min(xs_b), max(xs_b) + res_m, res_m),
            np.arange(min(ys_b), max(ys_b) + res_m, res_m))


def backproject_plane(data: FitData, cfg: SiteConfig, z_m: float,
                      xs: np.ndarray, ys: np.ndarray
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Interpolate each position's measured opacity onto the plane z = z_m.

    Returns ({position id: grid[len(xs), len(ys)]}, mean over covered positions).
    Pixels a position does not cover are NaN, never zero: "no ray went there"
    and "no material there" are different statements.
    """
    from scipy.interpolate import griddata

    origins = position_origins(cfg)
    gx, gy = np.meshgrid(np.asarray(xs), np.asarray(ys), indexing="ij")

    per: dict[str, np.ndarray] = {}
    for pid in data.rows.position_ids:
        sel = data.rows.mask_for(pid) & (data.w > 0)
        ox, oy, oz = origins[pid]
        lever = z_m - oz
        if not sel.any() or lever <= 0:
            per[pid] = np.full(gx.shape, np.nan)
            continue
        px = ox + data.rows.sx[sel] * lever
        py = oy + data.rows.sy[sel] * lever
        per[pid] = griddata((px, py), data.lam[sel], (gx, gy), method="linear")

    stack = np.stack([per[pid] for pid in data.rows.position_ids])
    n_ok = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(n_ok > 0, np.nansum(stack, axis=0) / np.maximum(n_ok, 1), np.nan)
    return per, mean
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_backproject.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/backproject.py tests/test_backproject.py
git commit -m "feat: model-free single-plane backprojection of the measured opacity"
```

---

### Task 12: Viewer data contract and run comparison

The `volume.npy` + `meta.json` contract is what keeps Phase 4's viewer decoupled from the solver (spec §8). `compare_volumes` answers the user's stated reason for the whole framework: *"I should be able to add a new datapoint and observe the changes."*

**Files:**
- Create: `megido/volexport.py`
- Test: `tests/test_volexport.py`

**Interfaces:**
- Consumes: `megido.reconstruct.VoxelSolution`, `megido.voxuncert.BootstrapResult`, `megido.config.SiteConfig`, `megido.resolution.campaign_resolution`.
- Produces:
  - `export_volume(run_dir, cfg, *, out_dir=None) -> Path` — writes `volume.npy` and `meta.json`, plus `sigma.npy`, `snr.npy`, `views.npy`, `systematic.npy` and `backprojection.npy` when those artifacts exist in `run_dir`
  - `compare_volumes(a: VoxelSolution, b: VoxelSolution) -> dict` with `corr`, `rms_delta`, `max_abs_delta`, `mass_change_frac`, `verdict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_volexport.py`:

```python
import json

import numpy as np
import pytest

from megido.config import load_site_config
from megido.reconstruct import VoxelSolution
from megido.volexport import compare_volumes, export_volume
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
"""

GRID = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _vol(seed=0) -> VoxelSolution:
    rng = np.random.default_rng(seed)
    return VoxelSolution(rho=np.abs(rng.normal(size=GRID.n_voxels)), grid=GRID,
                         offsets={"pos0": 0.1, "pos1": -0.1},
                         position_ids=("pos0", "pos1"),
                         info={"best_chi2": 1.2, "algorithm": "tv"})


def test_export_writes_the_contract(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")

    out = export_volume(run, _cfg(tmp_path))
    assert (out.parent / "volume.npy").exists()
    meta = json.loads((out.parent / "meta.json").read_text())
    assert meta["shape"] == list(GRID.shape)
    assert meta["axis_order"] == "xyz"
    assert meta["spacing_m"] == pytest.approx(0.5)
    assert meta["origin_m"] == list(GRID.origin)


def test_meta_carries_the_detectors_with_their_tilts(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))

    meta = json.loads((run / "meta.json").read_text())
    ids = {d["id"] for d in meta["detectors"]}
    assert ids == {"P0", "P1"}
    assert all("tilt_deg" in d and "az_deg" in d for d in meta["detectors"])


def test_meta_carries_the_resolution_verdict(tmp_path):
    """The viewer must be able to show what the campaign cannot resolve without
    re-deriving it."""
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))

    meta = json.loads((run / "meta.json").read_text())
    assert meta["resolution"]["depth_resolved"] is False
    assert "max_baseline_m" in meta["resolution"]


def test_optional_layers_are_exported_when_present(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    np.savez_compressed(run / "uncertainty.npz",
                        mean=np.zeros(GRID.shape, dtype=np.float32),
                        sigma=np.ones(GRID.shape, dtype=np.float32),
                        snr=np.ones(GRID.shape, dtype=np.float32),
                        n_replicas=np.asarray(4), origin=np.asarray(GRID.origin),
                        spacing=np.asarray(GRID.spacing))
    np.save(run / "views.npy", np.full(GRID.shape, 2, dtype=np.int16))

    export_volume(run, _cfg(tmp_path))
    meta = json.loads((run / "meta.json").read_text())
    assert (run / "sigma.npy").exists()
    assert (run / "views.npy").exists()
    assert set(meta["layers"]) >= {"volume", "sigma", "snr", "views"}


def test_export_without_optional_layers_still_succeeds(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))
    meta = json.loads((run / "meta.json").read_text())
    assert meta["layers"] == ["volume"]


def test_compare_of_a_volume_with_itself_reports_no_change():
    v = _vol()
    r = compare_volumes(v, v)
    assert r["corr"] == pytest.approx(1.0)
    assert r["rms_delta"] == pytest.approx(0.0)
    assert r["verdict"] == "unchanged"


def test_compare_detects_a_real_change():
    a, b = _vol(0), _vol(1)
    r = compare_volumes(a, b)
    assert r["rms_delta"] > 0
    assert r["verdict"] != "unchanged"


def test_compare_rejects_mismatched_grids():
    a = _vol()
    other = VoxelSolution(rho=np.zeros(8), grid=VoxelGrid((0, 0, 0), 1.0, (2, 2, 2)),
                          offsets={}, position_ids=(), info={})
    with pytest.raises(ValueError, match="grid"):
        compare_volumes(a, other)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_volexport.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.volexport'`.

- [ ] **Step 3: Write `megido/volexport.py`**

```python
"""The viewer data contract, and run-to-run comparison.

volume.npy + meta.json keeps Phase 4's viewer decoupled from the solver: the
viewer reads arrays and metadata, never the solver's internals, so either side
can be rebuilt without touching the other.

meta.json deliberately carries the resolution verdict. A viewer that can render
a crisp isosurface without being able to say that its height is unmeasured would
be a misleading instrument.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from megido.config import SiteConfig
from megido.reconstruct import VoxelSolution
from megido.resolution import campaign_resolution

_SKY_SIGMA_T = 0.05      # the Phase 2 sky grid's angular bin width


def export_volume(run_dir: str | Path, cfg: SiteConfig, *,
                  out_dir: str | Path | None = None) -> Path:
    """Write volume.npy + meta.json (+ any optional layers found) for the viewer."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    vol = VoxelSolution.load(run / "volume_full.npz")
    rho = vol.rho3().astype(np.float32)
    np.save(out / "volume.npy", rho)
    layers = ["volume"]

    unc = run / "uncertainty.npz"
    if unc.exists():
        with np.load(unc) as d:
            for name in ("sigma", "snr"):
                if name in d:
                    np.save(out / f"{name}.npy", d[name].astype(np.float32))
                    layers.append(name)

    for name in ("views", "systematic", "backprojection"):
        src = run / f"{name}.npy"
        if src.exists():
            if out != run:
                np.save(out / f"{name}.npy", np.load(src))
            layers.append(name)

    pos = np.maximum(rho, 0.0)
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0
    res = campaign_resolution(cfg, sigma_t=_SKY_SIGMA_T,
                              feature_pitch_m=max(2.0, 4 * cfg.volume.spacing_m))

    meta = {
        "shape": list(rho.shape),
        "axis_order": "xyz",
        "origin_m": list(vol.grid.origin),
        "spacing_m": vol.grid.spacing,
        "units": "opacity density [1/m]",
        "value_range": [float(rho.min()), float(rho.max())],
        "suggested_iso": [round(0.3 * p99, 6), round(0.6 * p99, 6)],
        "run": run.name,
        "layers": layers,
        "inversion_version": vol.version,
        "offsets": vol.offsets,
        "fit_info": vol.info,
        "detectors": [
            {"id": e.id, "x": e.pose.x, "y": e.pose.y, "z": e.pose.z,
             "tilt_deg": e.pose.tilt_deg, "az_deg": e.pose.az_deg}
            for e in cfg.exposures
        ],
        "resolution": {
            "max_baseline_m": res["max_baseline_m"],
            "n_positions": res["n_positions"],
            "depth_resolution_m": res["depth_resolution_m"],
            "depth_resolved": res["depth_resolved"],
            "verdict": res["verdict"],
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out / "volume.npy"


def compare_volumes(a: VoxelSolution, b: VoxelSolution) -> dict:
    """What changed between two runs.

    `verdict` is deliberately coarse. A delta smaller than a thousandth of the
    volume's own scale is reported as `unchanged` rather than as an improvement:
    with a limited-angle geometry, small deltas are noise far more often than
    they are news.
    """
    if a.grid != b.grid:
        raise ValueError(f"grid mismatch: {a.grid.key()} vs {b.grid.key()}")

    da, db = a.rho, b.rho
    delta = db - da
    scale = max(float(np.percentile(np.abs(da), 95)), 1e-12)
    rms = float(np.sqrt(np.mean(delta**2)))
    corr = (float(np.corrcoef(da, db)[0, 1])
            if da.std() > 0 and db.std() > 0 else float("nan"))
    mass_a, mass_b = float(da.sum()), float(db.sum())

    if rms < 1e-3 * scale:
        verdict = "unchanged"
    elif rms < 1e-1 * scale:
        verdict = "shifted slightly"
    else:
        verdict = "shifted substantially"

    return {
        "corr": corr,
        "rms_delta": rms,
        "max_abs_delta": float(np.abs(delta).max()),
        "mass_change_frac": (mass_b - mass_a) / mass_a if mass_a else float("nan"),
        "scale": scale,
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_volexport.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add megido/volexport.py tests/test_volexport.py
git commit -m "feat: viewer data contract and run-to-run volume comparison"
```

---

### Task 13: CLI, the full real run, and the Phase 3 report

**Files:**
- Modify: `megido/cli.py`
- Create: `tests/test_cli3.py`
- Create: `docs/phase3-reconstruction-report.md`

**Interfaces:**
- Consumes everything above.
- Produces three subcommands:
  - `megido reconstruct --config configs/megido.yaml --solve runs/solve --run runs/ingest --out runs/voxels [--bootstrap N] [--rebin 10] [--no-holdouts] [--no-systematic] [--backproject-z Z]`
  - `megido export --config configs/megido.yaml --run runs/voxels [--out DIR]`
  - `megido compare --a runs/voxels --b runs/voxels_prev`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli3.py`:

```python
import json

import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.baseline import solve_baseline
from megido.cli import main
from megido.config import load_site_config

CONFIG = """
site: t
data_dir: {data_dir}
binning: {{t_max: 1.25, n_bins: 20}}
volume: {{z_min_m: 1.0, z_max_m: 3.0, spacing_m: 0.5, n_aperture_sub: 2}}
reconstruction: {{algorithm: tv, n_iter: 20, tv_alpha: 0.01}}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {{x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}}
  - id: T20
    runs: DET3-DET4
    pose: {{x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}}
  - id: P1
    runs: DET5-DET6
    pose: {{x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}}
"""


@pytest.fixture
def workspace(tmp_path):
    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(CONFIG.format(data_dir=tmp_path))
    cfg = load_site_config(cfg_path)

    rng = np.random.default_rng(0)
    edges = np.linspace(-1.25, 1.25, 21)
    t = 0.5 * (edges[:-1] + edges[1:])
    shape = np.exp(-(t[:, None] ** 2 + t[None, :] ** 2))
    counts = {e.id: rng.poisson(3000 * shape).astype(np.int64) for e in cfg.exposures}

    # These are the exact keys megido.anghist.load_counts reads. Writing
    # "counts"/"edges" instead would fail inside the CLI, not in the code under
    # test.
    ingest = tmp_path / "ingest"
    ingest.mkdir()
    for eid, c in counts.items():
        np.savez_compressed(ingest / f"counts_{eid}.npz", values=c,
                            xedges=edges, yedges=edges, name=np.array("txty"))

    solve_dir = tmp_path / "solve"
    sol = solve_baseline(AnalysisGrid(edges=edges, counts=counts), cfg, n_iter=40)
    sol.save(solve_dir / "baseline.npz")
    return cfg_path, ingest, solve_dir, tmp_path


def test_reconstruct_writes_volumes_and_exits_zero(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    assert rc == 0
    assert (out / "volume_full.npz").exists()
    assert (out / "views.npy").exists()
    text = capsys.readouterr().out
    assert "baseline" in text.lower()
    assert "depth" in text.lower()


def test_reconstruct_writes_holdout_volumes(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
          "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    assert (out / "volume_holdout_pos0.npz").exists()
    assert (out / "volume_holdout_pos1.npz").exists()


def test_reconstruct_with_bootstrap_writes_uncertainty(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    # --rebin 1: the fixture has 20 bins, and the CLI default factor of 10 would
    # collapse the analysis grid to 2x2 and make the baseline solve degenerate.
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--run", str(ingest), "--out", str(out), "--bootstrap", "2",
               "--rebin", "1", "--iters", "30", "--no-systematic"])
    assert rc == 0
    assert (out / "uncertainty.npz").exists()


def test_bootstrap_without_a_run_directory_fails_loudly(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--out", str(tmp / "v"), "--bootstrap", "3", "--rebin", "1"])
    assert rc == 1
    assert "--run" in capsys.readouterr().out


def test_reconstruct_reports_a_missing_baseline(tmp_path, capsys):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG.format(data_dir=tmp_path))
    rc = main(["reconstruct", "--config", str(p), "--solve", str(tmp_path / "nope"),
               "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "baseline.npz" in capsys.readouterr().out


def test_export_produces_the_viewer_contract(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
          "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    rc = main(["export", "--config", str(cfg_path), "--run", str(out)])
    assert rc == 0
    assert (out / "volume.npy").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert "views" in meta["layers"]
    assert meta["resolution"]["depth_resolved"] is False


def test_compare_two_runs(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    a, b = tmp / "va", tmp / "vb"
    for out in (a, b):
        main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
              "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    rc = main(["compare", "--a", str(a), "--b", str(b)])
    assert rc == 0
    assert "unchanged" in capsys.readouterr().out


def test_compare_reports_a_missing_run(tmp_path, capsys):
    rc = main(["compare", "--a", str(tmp_path / "x"), "--b", str(tmp_path / "y")])
    assert rc == 1
    assert "volume_full.npz" in capsys.readouterr().out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cli3.py -q`
Expected: FAIL — `invalid choice: 'reconstruct'`.

- [ ] **Step 3: Add the subcommands to `megido/cli.py`**

Add these imports at the top:

```python
from megido.backproject import backproject_plane, plane_axes
from megido.baseline import BaselineSolution
from megido.fitdata import build_fit_data
from megido.forward import build_forward_model
from megido.reconstruct import VoxelSolution, solve_voxels
from megido.resolution import campaign_resolution, format_resolution, views_per_voxel
from megido.validate2 import opacity_uncertainty
from megido.volexport import compare_volumes, export_volume
from megido.voxuncert import systematic_map, voxel_bootstrap
```

Add the three command functions after `_cmd_solve`:

```python
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
    np.save(out / "views.npy", views_per_voxel(fwd))

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
        ok = np.isfinite(snr)
        print(f"bootstrap       {args.bootstrap} replicas; "
              f"{int((np.abs(snr[ok]) > 3).sum())} of {int(ok.sum())} voxels above SNR 3")

    if not args.no_systematic:
        sysmap = systematic_map(sol, cfg, cache_dir=args.cache)
        np.save(out / "systematic.npy", sysmap.astype(np.float32))
        print(f"gauge systematic  max |delta rho| {np.abs(sysmap).max():.4f} 1/m "
              f"({np.abs(sysmap).max() / max(rho.max(), 1e-12):.1%} of the peak)")

    if args.backproject_z is not None:
        xs, ys = plane_axes(cfg, args.backproject_z, data.rows.t_reach(),
                            res_m=full.grid.spacing)
        _, mean = backproject_plane(data, cfg, args.backproject_z, xs, ys)
        np.save(out / "backprojection.npy", mean.astype(np.float32))
        print(f"backprojection  plane z={args.backproject_z:.2f} m, "
              f"{mean.shape} at {full.grid.spacing:.2f} m")

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
```

Register them in `main`, after the `solve` parser:

```python
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

    c = sub.add_parser("compare", help="what changed between two reconstructions")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.set_defaults(func=_cmd_compare)
```

Update the module docstring to `"""python -m megido.cli validate|ingest|solve|reconstruct|export|compare"""`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cli3.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit the CLI**

```bash
git add megido/cli.py tests/test_cli3.py
git commit -m "feat: reconstruct, export and compare subcommands"
```

- [ ] **Step 7: Run the real campaign end to end**

```bash
cd /home/hadar/Cloud/Work/Postdoc/03_projects/megido-muon-voxels
mkdir -p runs
uv run python -m megido.cli reconstruct \
    --config configs/megido.yaml \
    --solve runs/solve --run runs/ingest --out runs/voxels \
    --bootstrap 6 --backproject-z 6.0 2>&1 | tee runs/phase3_real.log
uv run python -m megido.cli export --config configs/megido.yaml --run runs/voxels
```

If `runs/solve/baseline.npz` is absent, regenerate it first with
`uv run python -m megido.cli solve --config configs/megido.yaml --run runs/ingest --out runs/solve`.

**If the run exceeds about 20 minutes, stop it and report the bottleneck rather than reducing the replica count to make it finish.** The bootstrap re-solves the full Phase 2 baseline per replica by design (see `voxel_bootstrap`); if that is the cost, say so with a measured timing.

- [ ] **Step 8: Write `docs/phase3-reconstruction-report.md`**

Structure it exactly as `docs/phase2-baseline-report.md` is structured — **caveats first, numbers second**. Required sections:

1. **What to be careful about, before the numbers.** Lead with the resolution verdict: one 2.2 m baseline, lateral structure measured, depth set by the regulariser. State the Task 9 phantom result as the evidence.
2. **What was built** — the six-stage chain from `baseline.npz` to `volume.npy`.
3. **Real-campaign numbers** — grid shape and spacing, row count, chi², per-position offsets, opacity density quantiles, bootstrap SNR fractions, gauge-systematic magnitude as a fraction of the peak. Every number copied from `runs/phase3_real.log`, never from memory or from an earlier run.
4. **The two phantom gates** — Task 7's achieved values and Task 9's, each with its thresholds, and an explicit statement of what each does and does not license.
5. **What would change the answer** — a second translated position, and the measured `δz` improvement it would buy at the campaign's depth range. State plainly that a larger tilt at the same spot would not help, because tilt adds no baseline.
6. **Open items carried forward** — the rock material for the multiple-scattering systematic, the surveyed 2.2 m baseline, and the structure scale now that there is phantom evidence for it.

- [ ] **Step 9: Commit the report and the run log**

```bash
git add docs/phase3-reconstruction-report.md runs/phase3_real.log
git commit -m "docs: Phase 3 reconstruction report from the full campaign run"
```

---

## Plan self-review

**Spec coverage.** §7.1 pose/tilt — handled, and deliberately inverted (ruling 1); the rotation is applied once, in Phase 2. §7.2 ray bundles, `spacing/3` sampling, sparse SHA-cached `A` built per position — Task 3. §7.3 SIRT with anisotropic-TV under non-negativity, `tv_alpha` as a fraction of p95, grid spacing as a config field — Tasks 1 and 5. §7.4 all three uncertainty products: Poisson bootstrap (Task 10), the S2 null-space direction as a separate systematic map (Task 10), the analytic `δz` (Task 8), plus the alias check (Task 8). §9.1 per-position matrix caching — Task 3. §9.2 run comparison — Task 12. §9.3 leave-one-out — Task 6's holdout fits. §9.4 the delta layer — `compare_volumes` plus the exported layers; the viewer half is Phase 4. §8's data contract — Task 12. Not covered, by design and stated as rulings: `mlem`, `layered`, `focus`, `beams`, and `uncertainty.py` verbatim.

**Placeholder scan.** Every code step carries the code. Two gate thresholds (Tasks 7 and 9) are written as initial values with an explicit tightening procedure and an explicit prohibition on loosening — that is a measurement protocol, not a placeholder, and the alternative would be inventing numbers for a geometry nobody has run yet.

**Type consistency.** `RowIndex` is constructed in Tasks 2, 3, 5, 7, 8, 11 with the same five fields throughout. `FitData(lam, w, rows)` is consistent everywhere. `solve_voxels` and `build_fit_data` both carry `transparent_quantile` after the Task 10 correction. `VoxelSolution.save`/`load` round-trip the same keys. `campaign_resolution` returns the same dict shape consumed by `format_resolution` (Task 8) and `export_volume` (Task 12). `views_per_voxel` is defined in `resolution.py` (Task 8) and imported by the CLI from there.
