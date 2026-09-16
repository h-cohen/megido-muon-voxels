# Megido muon voxels — design

**Date:** 2026-09-16
**Status:** approved, pending implementation plan
**Deliverable:** 3D voxel opacity reconstruction of the rock surface surrounding the
Megiddo cavern, from multiple muon-detector exposures, presented in an interactive
HTML viewer.

This project uses the same physical detector as `cafeteria_3d_modeling`, but a
different site, a different data format, and a harder inverse problem: **there is no
open-sky calibration run**. The baseline that would normally be measured must instead
be estimated jointly with the scene.

---

## 1. Ground truth established during design

Everything in this section was measured from the data during the design session, not
inherited from documentation. Several inherited constants turned out to be wrong; they
are called out explicitly so they are not silently reintroduced.

### 1.1 Data format

The data is **raw per-event CSV**, not ROOT histograms. `NEW_PROJECT_BRIEF.md` assumed
ROOT and is wrong on this point.

| Property | Value |
|---|---|
| Location | `/home/hadar/Cloud/Work/Postdoc/01_data/processed/megido/` |
| Files | 70 × `DET______BEAM_________filter.data`, 6.2 GB total |
| Format | Semicolon-delimited ASCII, CRLF, CAEN DT5550W cluster dump |
| Columns | 424 = 4 cluster cols + 4 ASICs × (9 header + 32 `HIT` + 32 `CHARGE_HG` + 32 `CHARGE_LG`) |
| Header rows | **Repeat ~every 165 rows** (338× per file). One distinct layout. Parser must skip them |
| Coincidence | Every row has `NEventsInCluster = 4` — files are pre-filtered 4-fold coincidences |
| Rate | ~56 k clusters/day; 74.5 % have 1–2 hits in every layer |
| `HIT_a_c` | Boolean 0/1 |
| `CHARGE_HG` | 14-bit ADC. Pedestal ≈ 2200 (HIT=0 median 2197, p99 4110); hit-channel median 6288, deciles 3717→9766. MIP ≈ +4000 ADC above pedestal |

This is **strictly more information than the per-track tree** that `NEW_PROJECT_BRIEF.md`
§1.3 identified as the single highest-leverage ask. Offline refocusing at arbitrary
height, per-track quality cuts, re-weighting and re-binning are all available.

### 1.2 Inherited constants that are empirically false

| Claim | Source | Reality |
|---|---|---|
| "23 bars/layer, 9 channels unused" | `detector_summary.md:138` | **All 32 channels per ASIC are live**, with comparable occupancy and comparable MIP charge spectra. Confirmed independently on both Megido and cafeteria data |
| `ACTIVE_CHANNELS = 23`; `bar_id = asic*23 + channel` | `cafeteria_flux_analysis/calibrate_bars.py:45,202` | An assumption, never measured. Silently discards 36 of 128 real channels |
| Layer order X, Y, X, Y | `detector_summary.md:31-40`, `layers_fit_calibration/config.py:32` | ASIC-pair mutual information indicates **X, Y, Y, X** (see §1.4) |
| `aperture_m = 0.65` | cafeteria config default | Was wrong there too (real ≈ 0.354 m). Do not inherit — measure |

This is the failure mode `NEW_PROJECT_BRIEF.md` §1.5 warns about: *a wrong instrumental
constant survives indefinitely when nothing independent checks it.* Every detector
constant in this project is measured from data and gated by an independent test.

### 1.3 Channel → bar mapping is a folded permutation, and is unknown

Two-hit coincidence analysis (adjacent bars share charge, so co-occurring channel pairs
reveal adjacency):

- Strongest pairs, identical across all four ASICs: **(3,31), (7,27), (11,23), (15,19)**
  — each sums to 34, each member ≡ 3 (mod 4).
- Channel occupancy has a strict period-4 pattern: `c mod 4 ∈ {0,3}` high, `{1,2}` low.
- The adjacency graph splits into **four disjoint blocks of 8 that never mix**:
  `{0,1,2,3,28,29,30,31}`, `{4,5,6,7,24,25,26,27}`, `{8,9,10,11,20,21,22,23}`,
  `{12,13,14,15,16,17,18,19}` — i.e. block *k* = `{4k..4k+3} ∪ {31-4k..28-4k}`.

Naive greedy seriation of the coincidence graph explains only 65 % of two-hit events and
its weakest forced edge has weight 2 — **not trustworthy**. A proper solve is required
(§4.1).

**Open risk:** four *disconnected* adjacency blocks are not what a single contiguous row
of bars produces — one line of bars gives one connected path. Either a layer is built
from four separate 8-bar modules, or the readout does something not yet understood.
Until §4.1 resolves it, **bars-per-layer and active width are unknown** — neither 23 nor
32 may be assumed.

### 1.4 Layer orientation

Mutual information between ASIC pairs, over single-hit events (layers measuring the same
coordinate must correlate):

| pair | MI | pair | MI |
|---|---|---|---|
| **0–3** | **0.274** | 0–1 | 0.232 |
| **1–2** | **0.247** | 2–3 | 0.245 |
| | | 0–2 | 0.156 |
| | | 1–3 | 0.159 |

Highest {0,3} and {1,2}; lowest {0,2} and {1,3}. Consistent with two identical XY modules
with one flipped — **ASIC 0,3 measure one coordinate; ASIC 1,2 the other**. An X,Y,X,Y
stack predicts the opposite ranking. Evidence is suggestive, not conclusive at this
exposure; §4.1 settles it by track χ².

### 1.5 Exposures

Poses supplied by the user. Azimuth 241° is the **detector yaw** — the compass bearing of
the detector's local +x (bar) axis — constant for the whole campaign; for T20 the tilt
leans toward that same bearing.

| Exposure | Runs | Files | x, y, z (m) | tilt° | az° | Note |
|---|---|---|---|---|---|---|
| **P0** | DET200084–104 | 21 | 0, 0, 0 | 0 | 241 | reference |
| **T20a** | DET200105–116 | 12 | 0, 0, 0 | 20 | 241 | pre-gap |
| **T20b** | DET200119–144 | 26 | 0, 0, 0 | 20 | 241 | post-gap, −22 % rate |
| **P1** | DET200145–155 | 11 | 2.2, 0, 0 | 0 | 241 | parallax baseline 2.2 m |

The campaign geometry is favourable and the two effects are cleanly separated:

- **P0 ↔ P1** — pure translation, 2.2 m baseline → parallax, i.e. depth resolution.
- **P0 ↔ T20** — pure rotation, same position → decorrelates detector response from sky
  direction. This is what makes §5 possible.

File size drops 107 MB → 83 MB at DET200119 (Aug 6), immediately after the Jul 20 – Aug 5
gap, with **no** tilt change. Cause unknown. T20a and T20b are therefore treated as
**independent exposures with independent efficiency normalization** — a 22 % rate change
inside one nominal pose would otherwise corrupt the baseline solve.

### 1.6 What exists to reuse, and what does not

**No raw-data → hits → tracks → angles pipeline exists in Python anywhere on this
machine.** The ROOT `txty` histograms the cafeteria project consumed were produced by
external C++ that is not on disk. Stage S1 is genuinely new code.

From `cafeteria_3d_modeling` (~5460 lines, of which ~5200 are input-format-agnostic; the
entire ROOT dependency is three lazy `import uproot` calls inside `io.py`):

- **Port verbatim:** `geometry`, `raycast`, `forward`, `reconstruct`, `opacity`,
  `metrics/`, `uncertainty`, `backproject`, `export`, `focus`, `beams`, `compare`
- **Port with changes:** `config` (new geometry fields), `calibration` (no sky reference)
- **Drop:** `io.load_root_*`, `selfcal.stage_a` (needs DAQ `XY0*m` histograms),
  `BinningConfig.refocus_origin_m`, `REFOCUS_HEIGHTS_M`
- **Rebuild:** `viewer/`

The **ingest seam**: `io.load_phantom_dir` reads `counts_<id>.npz`
(`values`, `xedges`, `yedges`, `name`) plus `meta.json`. Writing our angular histograms
in that format runs the entire ported pipeline with zero changes to it.

From `obs/`:

- `layers_fit_calibration/geometry.py` (48 lines) — straight-line LSQ track fit; its
  `ax`, `ay` are exactly tan θx, tan θy
- `layers_fit_calibration` as a whole — plugs in as an S-curve corrector if S1 emits its
  7-column schema (`track_id, layer, bar_index, x_formula, charge_n, charge_N, z_layer`)
- `cafeteria_flux_analysis/calibrate_bars.py:107-131,142-170` — Moyal/peak MPV extraction
  and sigma-clipped pedestal estimation. Note it **measures pedestal but never subtracts
  it**; we must subtract
- `muon_flux_analysis/physics.py` — cos²θ open-sky flux and opacity↔transmission helpers

Synthetic ground truth already on disk for phantom validation:
`01_data/raw/topography.csv` (41×41 z(x,y) surface), `detector_config.csv`,
`calibration_open_sky.csv`.

---

## 2. Architecture

Six stages, each writing a content-addressed artifact. Downstream stages consume
artifacts, never upstream code.

```
S0-det  detector calibration   raw .data  -> channel map, bars/pitch, X/Y, layer dz
S0-exp  exposure calibration   raw .data  -> pedestals, per-channel gain, live-time
S1      event reconstruction   raw .data  -> tracks -> per-exposure angular histograms
S2      baseline solve         histograms -> baseline B(d), transmission T
S3      forward model          poses      -> sparse system matrix A
S4      inversion              T, A       -> voxel opacity + uncertainty
S5      viewer                 volume     -> interactive HTML
```

Artifacts are keyed by `sha(inputs + relevant config)`. Re-running S4 with S0 frozen is
cheap; adding an exposure re-runs only that exposure's S0-exp/S1.

### 2.1 Implementation phasing

This design spans the whole system and is **too large for a single implementation plan**.
It decomposes into four phases along the artifact boundaries. Each phase gets its own
implementation plan, and each ends at a gate that must pass before the next begins.

| Phase | Scope | Exit gate |
|---|---|---|
| **1** | S0-det, S0-exp, S1 — raw `.data` to per-exposure angular histograms | Synthetic simulator's injected channel permutation recovered exactly; `counts_<exp>.npz` written for all four exposures |
| **2** | S2 — baseline solve | Synthetic scene + synthetic baseline both recovered from multi-pose data; null space correctly identified |
| **3** | S3, S4 — tilt-aware forward model, inversion, uncertainty | Phantom RMSE gate from `topography.csv`; LOO cross-validation honest |
| **4** | S5 — viewer | Playwright smoke; every control operable on real reconstructed output |

Phase 1 carries nearly all of the project risk, because every inherited detector constant
it depends on has already been shown to be wrong (§1.2) and its central unknown — the
channel permutation — is unsolved (§1.3). It is also the only phase with no reusable prior
art. **Plan and execute Phase 1 first; do not scope Phases 2–4 in detail until it lands**,
since its measured outputs (bars per layer, active width, layer Δz) set parameters those
phases depend on.

---

## 3. Exposure registry

Poses live in data, not code. One file is the entire interface for adding a datapoint.

```yaml
# configs/megido.yaml
site: megido
frame: {origin: P0, x_axis_bearing_deg: 241, units: m}
detector_cal: cal/det_v1          # frozen S0-det artifact

exposures:
  - id: P0
    runs: DET200084-DET200104
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0,  az_deg: 241}
    pose_sigma: {xy: 0.05, ang: 1.0}
  - id: T20a
    runs: DET200105-DET200116
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
    norm_group: T20a
  - id: T20b
    runs: DET200119-DET200144
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
    norm_group: T20b              # independent - unexplained -22% rate
  - id: P1
    runs: DET200145-DET200155
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0,  az_deg: 241}
```

**Adding a datapoint = drop files in the data directory, append one block, re-run.** No
code is touched.

Field semantics, stated explicitly so they cannot be read two ways:

- `runs` — an **inclusive** range of DET ids. Missing ids inside the range (e.g. 117, 118)
  are simply absent files, not an error
- `pose.x/y/z` — detector centre in the site frame, **metres**. `frame.origin` names the
  exposure defining the origin
- `pose.tilt_deg` — angle of the detector normal (stack axis) from zenith
- `pose.az_deg` — compass bearing of the detector's local +x (bar) axis. For a tilted
  exposure the normal leans toward this same bearing
- `pose_sigma.xy` — metres; `pose_sigma.ang` — degrees. Feeds pose self-calibration priors
- `norm_group` — exposures sharing a group share one efficiency normalization. Defaults to
  the exposure `id` when omitted

CLI:

```
python -m megido ingest              # S0-exp + S1, new/changed exposures only
python -m megido solve               # S2 -> S3 -> S4
python -m megido view                # S5
python -m megido compare runs/007 runs/008
```

---

## 4. S0 — calibration

### 4.1 S0-det (detector-level, frozen, versioned)

Runs once; re-run only if hardware changed. Consumes the full P0 bunch (21 files,
~1.2 M events) for statistics.

| Sub-task | Method | Independent gate |
|---|---|---|
| **0a** channel → bar permutation | Two-hit coincidence matrix over all P0 files; seriate into a bar ordering | **Track collinearity**: the correct permutation maximizes the fraction of 4-layer events with a good straight-line fit. Independent of the coincidence statistic used to derive it |
| **0b** bars per layer, bar pitch | Length of the recovered ordering; pitch from hit-position histogram hard edges | Flat-topped occupancy with sharp edges at both ends |
| **0c** X/Y assignment and z-order | Test all three pairings by track χ² | χ² must clearly prefer one; cross-check against §1.4 MI |
| **0d** layer z separation | **Not recorded anywhere.** Derive from angular acceptance cutoff: `\|tan θ\|_max = active_width / Δz` | Cross-check against the "~80 cm" figure in `detector_summary.md` |

**Acceptance gate for the whole stage:** a synthetic detector simulator generates events
through a known, randomly chosen channel permutation; S0-det must recover it exactly.
Nothing downstream runs until this passes.

### 4.2 S0-exp (per exposure, automatic)

Pedestal (sigma-clipped, from HIT=0 population), per-channel gain from Moyal/peak MPV,
live-time from cluster timecodes. Runs per exposure so DAQ drift — such as the Aug 6
event — is caught rather than absorbed into the reconstruction.

Output: `ChannelCalibration{gain, pedestal, noise_sigma, MPV}` per (asic, channel) per
exposure. Flag channels deviating >10 % from the exposure median.

---

## 5. S1 — event reconstruction

New code. Streaming chunked reader; must skip the repeated header line every ~165 rows.

Per event:

1. Per layer, select channels with `HIT = 1` and charge above `pedestal + 3σ`
2. Accept 1 hit, or 2 hits on **physically adjacent bars** (adjacency from the S0-det
   permutation, not from channel number). Reject larger or non-adjacent clusters
3. Sub-bar position `x = a · n / (N + n)` on pedestal-subtracted, gain-corrected charge
4. Straight-line least squares across the four layers → `(ax, ay) = tan θx, tan θy`, plus χ²
5. Cuts: hit in all 4 layers, ≤2 adjacent bars per layer, χ² below threshold

Outputs per exposure:

- `tracks_<exp>.parquet` — schema `track_id, layer, bar_index, x_formula, charge_n,
  charge_N, z_layer`, exactly what `layers_fit_calibration` consumes, so its
  self-supervised S-curve correction plugs in later at no integration cost.
  **`bar_index` is the physical bar index from the S0-det permutation, never the raw DAQ
  channel number** — the two are not equal and conflating them is the failure this project
  exists to avoid
- `counts_<exp>.npz` — angular histogram in the `load_phantom_dir` format. **This is the
  seam into the ported pipeline**

Binning: **400 × 400 over ±1** in tan units, rebinned at analysis time. Not the legacy
800 × 800 over ±2 — acceptance dies well before |t| = 1. Final `rebin` is tuned to the
site's feature scale once known (`NEW_PROJECT_BRIEF.md` §1.2 warns that too coarse a grid
aliased the cafeteria beam pitch).

Integer counts must be preserved un-normalized — `mlem_transmission` and the Poisson
bootstrap both need raw `n`.

---

## 6. S2 — baseline solve (project core)

There is no open-sky run. The standard `T = N_measured / N_expected` is unavailable.

### 6.1 Model

```
N_e(d) = I(theta_sky(R_e . d)) * B(d) * norm_e * t_e
```

- `d` — direction in the **detector** frame
- `B(d) = A_geom(d) * eps(d)` — geometric acceptance × efficiency. A property of the
  **detector**, therefore **shared across all exposures**
- `lambda` — scene opacity, a property of the **world**
- `R_e` — exposure pose rotation, `Rz(yaw) · Ry(tilt)`
- `norm_e` — per-`norm_group` normalization
- `t_e` — live-time

### 6.2 Why rotation separates the unknowns

`B` is fixed in the detector frame; `lambda` is fixed in the world frame. Rotating the
detector sends each detector bin to a different sky direction. For P0 and T20 — same
position, 20° apart — the baseline cancels exactly:

```
lambda(R_P0 . d) - lambda(R_T20 . d) = ln N_T20(d) - ln N_P0(d)
```

This constrains the scene with **zero** knowledge of cosmic flux or detector acceptance.
It is the muon-tomography analogue of recovering a flat field from dithered pointings —
standard practice in astronomical imaging.

### 6.3 Chosen approach: hybrid (approach C)

Joint Poisson-likelihood solve for `{lambda, B, norm_e}`:

- **Data term** — Poisson likelihood on raw counts, all exposures
- **Prior on B** — analytic aperture-overlap model × smooth per-channel efficiency, entered
  as a **weak prior**, not as truth
- **Regularization on lambda** — anisotropic TV, non-negative

Rationale for the hybrid over either pure alternative:

- Pure physics model (A) is fastest and gives absolute ρL, but is **unverifiable** without
  a sky run: any acceptance error becomes structure that looks real. This is exactly how
  the cafeteria `aperture_m` bug survived for months.
- Pure self-calibration (B) cannot produce a fake acceptance artifact, but leaves a null
  space unconstrained and yields relative, not absolute, opacity.
- The hybrid degenerates gracefully in both directions: with one exposure it reduces to A;
  with a perfect model it reduces to B.

**Failure modes are visible rather than silent** — the deciding criterion.

### 6.4 Known null space

Any `lambda` constant on cones about the tilt axis survives the rotation difference. It is
pinned by the flux prior, and — critically — it is **surfaced in the viewer as its own
systematic layer** rather than buried in the result. A user must be able to see which part
of the answer the data constrains and which part the prior supplies.

### 6.5 Validation

Leave-one-exposure-out (§9.3) is the honest cross-validation: a baseline solved on a
subset must predict the held-out exposure.

---

## 7. S3 / S4 — forward model and inversion

Ported largely verbatim. One substantive change.

### 7.1 Pose must carry tilt

`PoseConfig` is currently `x, y, z, yaw_deg` — built for a detector that always looks
straight up. Required changes:

- Add `tilt_deg`, `az_deg` to `PoseConfig`
- Replace the yaw-only `pose_rotation` with `R = Rz(yaw) · Ry(tilt)`
- Thread the full rotation through `bin_directions`, `aperture_offsets`, `auto_grid`

Everything downstream of `geometry.py` is unaffected — it consumes directions, not poses.

### 7.2 Forward model

Per `NEW_PROJECT_BRIEF.md` §1.4, ported unchanged: each angular bin is a bundle of
`n_aperture_sub²` parallel sub-rays across the aperture, not a pinhole; effective aperture
narrows off-axis under the 4-layer coincidence requirement. Voxel path lengths sampled at
`spacing/3`. `A` is sparse, disk-cached by SHA, and **built per pose** so adding an
exposure appends rows rather than rebuilding.

### 7.3 Solver

SIRT with an anisotropic-TV proximal step under non-negativity (`sirt_tv`). The unknown is
a full voxel opacity field — the `tv` algorithm, not `layered`. `tv_alpha` is expressed as
a fraction of `x`'s p95 so it transfers across datasets.

Grid spacing is set by the site's expected feature scale (open, §11). **Starting value
0.10 m**, so implementation is not blocked on that answer; it is a config field and the
phantom gate re-tunes it. The cafeteria project needed 0.08 m to avoid aliasing a 1.7 m
feature pitch at 0.15 m spacing, so treat any value above ~0.15 m as suspect until the
alias check in §7.4 is run.

### 7.4 Uncertainty — a first-class deliverable, not an appendix

With limited-angle geometry and few viewpoints, depth uncertainty is the dominant error:
`δz ≈ z²σ_θ / b`. Required outputs:

- Poisson bootstrap over counts (port `uncertainty.py`; resamples both numerator and
  denominator, holds mask and calibration fixed)
- The S2 null-space direction as a **separate systematic map**
- The analytic `δz` estimate for the achieved baselines

Alias check before trusting any single-minimum autofocus curve: false height solutions for
a periodic structure repeat every `Δz = pitch · z / baseline`. The cafeteria campaign had
5.8 m of margin; a synthetic test aliased at 0.9 m. Compute this for the Megido geometry.

---

## 8. S5 — viewer (ground-up rebuild)

Rebuilt, not ported. The cafeteria viewer is reference only.

**Core upgrade: GPU raymarching of a 3D texture**, replacing marching-cubes meshes.
Transfer function, iso level and clipping become continuous 60 fps controls with no
re-meshing, and soft density renders honestly instead of being forced into a hard surface.

Controls:

- **Layer manager** — every artifact independently toggleable: per-exposure solve,
  combined solve, baseline `B`, uncertainty, systematic null-space, backprojection,
  phantom truth, run-delta
- **Transfer-function editor** — draggable opacity/colour ramp over density, not a fixed
  palette dropdown
- **Clip box and arbitrary clip plane**, plus slice views
- **Exposure-contribution layer** — which exposures constrain each voxel, i.e. where the
  answer is data and where it is prior
- **Uncertainty gate** — hide voxels whose σ exceeds a slider
- Hover voxel readout; live histogram of the current transfer window
- Camera presets, PNG export

Self-contained single HTML, vendored libraries, no network requests, works from `file://`.

**Data contract** stays `volume.npy` (float32 `[nx,ny,nz]`) + `meta.json`, keeping the
viewer decoupled from the solver.

---

## 9. Incremental execution and change observation

### 9.1 Incremental staging

6.2 GB of ASCII makes full reprocessing unacceptable for a one-exposure addition.

- **S0-exp, S1** are per-exposure — a new exposure processes only its own files
- **S3** is per-pose — a new pose appends matrix rows, existing blocks stay cached
- **S2, S4** re-solve globally — the only genuinely global cost, and the cheap end

### 9.2 Run comparison

`compare` reports metric deltas with an IMPROVED / REGRESSED verdict. Ported from
`metrics.REGISTRY`, which carries per-metric noise bands, so a delta inside the noise band
is reported as **noise** rather than as improvement.

### 9.3 Leave-one-exposure-out

Every solve additionally produces the fit with each exposure held out. "What did this new
datapoint buy?" is then answered directly by `all` versus `all − P2`, with no bookkeeping.
The cafeteria pipeline does this for two fixed positions; this generalizes it to N
exposures. The same LOO set is S2's cross-validation (§6.5).

### 9.4 Delta layer

The viewer loads two runs and renders `B − A` as a signed layer, showing exactly which
voxels moved.

---

## 10. Testing strategy

| Level | Test |
|---|---|
| S0-det | Synthetic detector simulator injects a known random channel permutation; S0-det must recover it **exactly**. Hard gate — nothing downstream runs until it passes |
| S1 | Synthetic tracks with known angles; recovered tan θx/θy within resolution. Header-skipping and chunk-boundary handling unit-tested |
| S2 | Synthetic scene + synthetic baseline; both must be recovered from multi-pose data. Null-space direction must be identified correctly |
| S3/S4 | Phantom built from `01_data/raw/topography.csv` as ground-truth surface; RMSE gate |
| S5 | Playwright smoke test — page renders, canvas non-blank, every control operable |
| Cross-stage | Artifact-level regression test per stage; cache-key correctness (changing a config field must invalidate exactly the right artifacts) |

Every algorithm choice is gated against the phantom before being trusted on real data.
This was the cafeteria project's core engineering discipline and it is what caught its
alias-distance and CV-scan-degeneracy problems early.

---

## 11. Open items (do not block implementation)

- **Structure scale** — feature size and expected depth range, needed to finalize grid
  spacing and the `rebin` factor
- **Rock material** — needed to re-derive the multiple-Coulomb-scattering systematic. Do
  not inherit `X0_CONCRETE_M` from the cafeteria project
- **Surveyed baseline** — independent confirmation of the 2.2 m P0↔P1 separation. Absolute
  scale is locked by one surveyed length; angles alone fix only ratios. The cafeteria
  project ran an entire campaign on an unverified self-calibrated baseline
- **Aug 6 rate drop** — cause unknown; handled defensively by splitting T20a/T20b
- **Fiducial marker** — strongly recommended for future campaigns; decouples "is the
  pipeline working" from "what is the scene"

---

## 12. Decision record

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Baseline strategy | Hybrid physics-prior + multi-pose self-calibration | Pure physics model; pure self-calibration | Only option whose failure mode is visible rather than silent |
| Reconstruction unknown | Voxel opacity field | Surface / height-field parametrization | Matches the stated deliverable; makes no shape assumption |
| Angular binning | 400 × 400 over ±1 | Legacy 800 × 800 over ±2 | Acceptance dies well before \|t\| = 1 |
| T20 handling | Split into T20a / T20b | Merge as one exposure | Unexplained 22 % rate change inside one nominal pose |
| Pose storage | YAML registry, data not code | Config dataclass defaults | Adding a datapoint must touch no code |
| Viewer | Ground-up rebuild, GPU raymarching | Port cafeteria marching-cubes viewer | Continuous transfer-function control; honest soft-density rendering |
| Detector constants | Measured from data, independently gated | Inherited from `detector_summary.md` | Three inherited constants already proven false |
