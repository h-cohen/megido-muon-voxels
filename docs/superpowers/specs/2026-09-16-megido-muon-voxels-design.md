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

### 1.2 Authoritative detector constants

Supplied by the detector engineers on 2026-09-17 as
`00_knowledge/literature_notes/muon-detector/{parameters.data, detector_info.txt}`.
These supersede `detector_summary.md`, which is wrong on several points.

| Quantity | Value | Note |
|---|---|---|
| Layers | 4 | |
| Bars per layer | **23** | `detector_summary.md` was right; §1.3 records why we briefly doubted it |
| Bar base width | **3.2 cm** | `detector_summary.md` says 3.3 cm — wrong |
| Bar height | 1.7 cm | |
| Bar side | 2.3345 cm | Consistent: √(1.6² + 1.7²) = 2.335 |
| Bar length | 40 cm | |
| Apex angle `alpha` | 43.08° | |
| **Layer z** | **0, 6.2, 31.5, 37.7 cm** | `detector_summary.md` says "~80 cm detector height" — **wrong**. Station separation is 31.5 cm |
| Layer offsets | 0, 0, 0, 0 | |
| `asic_to_layer` | `[1, 3, 2, 0]` | Layer order bottom-up |
| Layer stack (bottom-up) | **X, Y, X, Y** | ASIC 3 = X_bot, ASIC 0 = Y_bot, ASIC 2 = X_up, ASIC 1 = Y_up |
| Coordinate frame | Origin bottom-left, right-handed, Z up | Rotation about Z = azimuth φ; about X = polar θ |

The values are filed under a `# Detector 3` heading; the user has confirmed the Megiddo
campaign used that unit.

**Layer separation matters more than any other number here.** At 31.5 cm rather than the
assumed 80 cm, the angular acceptance is far wider than the inherited configuration
implies. Derive the acceptance cutoff from data and cross-check it against this value
before trusting either.

### 1.2.1 Constants that must still not be inherited

| Claim | Source | Status |
|---|---|---|
| `ACTIVE_CHANNELS = 23`; `bar_id = asic*23 + channel` | `cafeteria_flux_analysis/calibrate_bars.py:45,202` | **Wrong.** The bar count is right, but the mapping is not the identity — see §1.3 |
| `aperture_m = 0.65` | cafeteria config default | Was wrong there too (real ≈ 0.354 m). Measure, do not inherit |
| "9 channels unused per layer" | `detector_summary.md:138` | True that 9 are unmapped, but they are **not quiet** — see §1.3.1 |

`NEW_PROJECT_BRIEF.md` §1.5 warns that *a wrong instrumental constant survives
indefinitely when nothing independent checks it.* Every constant above is checked against
data, and §1.3 is the worked example of why that discipline pays.

### 1.3 Channel → bar mapping — supplied and validated

The mapping is **not** the identity. It is a folded permutation, different per ASIC,
supplied as four 23-entry lists indexed by bar position:

```
asic0 = [28, 30,  3, 31, 29,  0, 24, 26,  7, 27, 25,  4, 20, 22, 11, 23, 21,  8, 16, 18, 15, 19, 17]
asic1 = [ 3, 31,  2,  0, 28,  1,  7, 27,  6,  4, 24,  5, 11, 23, 10,  8, 20,  9, 15, 19, 14, 12, 16]
asic2 = [19, 15, 18, 16, 12, 17, 23, 11, 22, 20,  8, 21, 27,  7, 26, 24,  4, 25, 31,  3, 30, 28,  0]
asic3 = [12, 14, 19, 15, 13, 16,  8, 10, 23, 11,  9, 20,  4,  6, 27,  7,  5, 24,  0,  2, 31,  3,  1]
```

i.e. `asic0[0] = 28` means bar 1 is read by channel 28. Each list holds 23 distinct
channels, leaving 9 unmapped per ASIC.

**Validation — the acceptance test for any mapping.** Adjacent bars share charge, so
two-hit events must land on *adjacent bar indices*. Measured on DET200084 (56 081 events):

| Interpretation | 2-hit events on adjacent bars |
|---|---|
| Raw channel number (i.e. assuming identity) | 130 — **0.2 %** |
| Supplied map, in bar space | ~17 000 — **44–47 %** on every ASIC |

Chance level for two random distinct bars out of 23 is 8.7 %. The supplied map beats the
identity assumption by ~130× and sits far above chance on all four ASICs. **The map is
correct.** This test is the regression gate: any future mapping change must reproduce
≥40 % adjacency.

Had we shipped `bar_id = asic*23 + channel`, 99.8 % of charge-sharing pairs would have
been mis-assigned and every sub-bar interpolated position would have been noise.

**Why the coincidence structure looked pathological.** Before the map arrived, the
adjacency graph in *channel* space appeared to split into four disjoint blocks of 8
(`{0-3,28-31}`, `{4-7,24-27}`, `{8-11,20-23}`, `{12-15,16-19}`), which no contiguous row
of bars can produce. That structure is entirely an artifact of the fold; it dissolves in
bar space. Recorded here so it is not re-investigated.

### 1.3.1 Unmapped channels are not quiet — open item

The 9 unmapped channels per ASIC are neither dead nor cleanly pedestal:

| | mapped (23 ch) | unmapped (9 ch) |
|---|---|---|
| Share of all hits | 80 % | **20 %** |
| Hits per channel | ~3700 | ~2200 (60 % of mapped) |
| `CHARGE_HG` median | 6098–6286 | 5917–6306 |

A near-identical charge median argues against simple crosstalk. Unexplained, and on the
open question list. It does not block the pipeline: S1 rejects events whose hits fall on
unmapped channels, so the cost is acceptance, not correctness. **Quantify that acceptance
loss** — 20 % of hits is not negligible, and if it is angle-dependent it biases the
baseline solve.

### 1.4 Layer orientation — resolved, and a retracted inference

**Authoritative:** `asic_to_layer = [1, 3, 2, 0]`, layers ordered bottom-up, stack
**X, Y, X, Y**:

| Position | Layer | ASIC | Coordinate | z (cm) |
|---|---|---|---|---|
| top | 3 | 1 | Y_up | 37.7 |
| | 2 | 2 | X_up | 31.5 |
| | 1 | 0 | Y_bot | 6.2 |
| bottom | 0 | 3 | X_bot | 0 |

So ASIC 0 and 1 measure Y; ASIC 2 and 3 measure X.

**Retracted.** During design we inferred X, Y, Y, X from mutual information between ASIC
pairs on single-hit events, reasoning that same-coordinate layers must correlate most:

| pair | MI | pair | MI |
|---|---|---|---|
| 0–3 | 0.274 | **0–1** | **0.232** |
| 1–2 | 0.247 | **2–3** | **0.245** |
| | | 0–2 | 0.156 |
| | | 1–3 | 0.159 |

The true same-coordinate pairs are {0,1} and {2,3} — mid-table, not the top-ranked {0,3}
and {1,2}. **Mutual information did not discriminate orientation on this detector.** The
likely reason is that the two same-coordinate layers sit either side of the drift gap,
so angular spread degrades their correlation below that of adjacent orthogonal layers.

Kept as a recorded negative result: do not re-derive orientation this way, and do not
treat a plausible-looking ranking as evidence without an independent check.

### 1.5 Exposures

Poses supplied by the user. Azimuth 241° is the **detector yaw** — the compass bearing of
the detector's local +x (bar) axis — constant for the whole campaign; for T20 the tilt
leans toward that same bearing.

**Noted conflict, resolved in favour of the campaign log.** `parameters.data` states
*"Y axis is pointed 281 degrees WRT true north"*. The user has confirmed **241° is
correct** for this campaign; the 281° figure refers to a different deployment. Recorded
because a 40° error would rotate the entire reconstruction about the vertical with nothing
downstream flagging it — if a result ever looks rotated, check here first.

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
| **1** | S0-det, S0-exp, S1 — raw `.data` to per-exposure angular histograms | Supplied constants validated (adjacency ≥ 40 %, acceptance cutoff consistent with 31.5 cm); synthetic angles recovered within resolution; `counts_<exp>.npz` written for all four exposures |
| **2** | S2 — baseline solve | Synthetic scene + synthetic baseline both recovered from multi-pose data; null space correctly identified |
| **3** | S3, S4 — tilt-aware forward model, inversion, uncertainty | Phantom RMSE gate from `topography.csv`; LOO cross-validation honest |
| **4** | S5 — viewer | Playwright smoke; every control operable on real reconstructed output |

Phase 1 was the dominant risk while the channel map was unknown. **That risk is now
largely retired** — the engineers supplied the map and the layer geometry (§1.2–§1.4), and
the map validates against data at 44–47 % adjacency versus 0.2 % for the identity
assumption. Phase 1 is now substantial but ordinary work: a streaming reader, hit
clustering, charge-sharing interpolation, track fitting, and a set of falsification tests
on supplied constants.

It remains the only phase with no reusable prior art — no raw-data-to-angles pipeline
exists in Python anywhere on this machine (§1.6). **Plan and execute Phase 1 first**, since
its outputs (validated acceptance cutoff, achieved angular resolution, unmapped-channel
acceptance loss) set parameters Phases 2–4 depend on.

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

All four constants are now **supplied** (§1.2, §1.3, §1.4), so this stage is
*validation*, not discovery. It loads the engineers' values and proves each against data
before anything consumes them.

| Sub-task | Supplied value | Validation against data |
|---|---|---|
| **0a** channel → bar map | Four 23-entry lists (§1.3) | **Adjacency rate ≥ 40 %** of two-hit events on adjacent bar indices, per ASIC. Identity mapping scores 0.2 %, chance 8.7 %. Already passes at 44–47 % |
| **0b** bars per layer, pitch | 23 bars, base 3.2 cm | Hit-position histogram is flat-topped with hard edges; recovered active width consistent with 23 bars at the derived pitch |
| **0c** X/Y assignment, z-order | `asic_to_layer = [1,3,2,0]`, X,Y,X,Y | Track χ² must prefer this pairing over the two alternatives. Note §1.4: mutual information is **not** a valid discriminator here |
| **0d** layer z separation | 0, 6.2, 31.5, 37.7 cm | Angular acceptance cutoff `\|tan θ\|_max = active_width / Δz` must agree with a 31.5 cm station separation. **Disagreement here is a stop condition** — it is the absolute angular scale |
| **0e** unmapped-channel acceptance | — | Quantify the acceptance loss from rejecting events that touch unmapped channels (§1.3.1, ~20 % of hits), and test whether it is angle-dependent |

**Acceptance gate for the whole stage:** a synthetic detector simulator generates events
through the known channel permutation and known layer geometry; S0-det must recover the
injected angles within resolution and reproduce the adjacency rate. Nothing downstream
runs until this passes.

Each validation is a *falsification test on a supplied number*, not a fit. A supplied
constant that fails its test is escalated to the engineers, never silently replaced.

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

1. Per layer, select channels with `HIT = 1` and charge above `pedestal + 3σ`. Translate
   channel → bar index through the §1.3 map for that ASIC. **Reject the event if any hit
   falls on an unmapped channel** — ~20 % of hits, see §1.3.1; record the rejection rate
   per angular bin, since an angle-dependent loss biases S2
2. Accept 1 hit, or 2 hits on **physically adjacent bars** (adjacency in bar index, never
   in channel number). Reject larger or non-adjacent clusters
3. Sub-bar position `x = a · n / (N + n)` on pedestal-subtracted, gain-corrected charge
   for two-bar clusters. **Single-bar clusters are dithered uniformly across a
   self-calibrated window of width `pitch * f_single`** (`f_single` the measured
   single-bar fraction), not assigned the bar centre — a fixed centre value produces a
   position-quantisation comb in the angle that survives detector-frame tilts and mimics
   the Phase 2 baseline signature; see `docs/phase1-validation-report.md` for the
   measurement and fix
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

- **Data-driven hillside extraction from the flux edge — the most promising next
  analysis.** The setup is a detector under a small hill in a long cavern,
  looking up; the hillside's exact shape is unknown and should be recovered from
  the data, not assumed. The voxel solve cannot localise depth (single 2.2 m
  baseline, §7.4), but the hill's SILHOUETTE is a different, well-conditioned
  observable: the overburden is thick toward the crown and thins toward the
  flanks, so a line of sight that grazes out through the hillside crosses from
  heavy absorption to near-open sky over a narrow angular range. That produces a
  **sharp spike / steep gradient in muon flux along the hill edge**, which is a
  LATERAL feature — exactly the class this geometry resolves well (lateral column
  correlation ~0.66, against depth localization ~0.09). Two consequences worth
  pursuing: (1) the edge can be located per position by an edge/gradient detector
  on the reconstructed sky flux rather than by trusting per-voxel depth; (2) a
  sharp edge gives far better parallax than a diffuse field, so the shift of that
  edge between P0 and P1 (2.2 m) can constrain the hillside surface where the
  voxel field cannot — turning the campaign's one weakness (depth) into a
  tractable 1-D problem (edge disparity) on its one strength (lateral contrast).
  This is a Phase 4+ analysis, not part of the S3/S4 voxel inversion; flagged
  here so the "depth not resolved" finding is read as *use a different observable*
  rather than *nothing more is possible*.
- **Structure scale** — still needed to finalize grid spacing, but no longer
  blocking: Phase 3's Megiddo-geometry phantom (`tests/test_phantom.py`,
  `test_megiddo_geometry_recovers_lateral_structure_but_not_depth`) measured
  what the campaign resolves. Lateral structure is recovered (column correlation
  ~0.66); the HEIGHT of that structure is set by the regulariser, not the data,
  because the campaign has a single 2.2 m baseline (depth localization ~0.09,
  analytic dz ~1.4 m at mid-range). `megido.resolution.campaign_resolution`
  reports the closed-form number for any configured volume. A second translated
  position is what would change this — a larger tilt at the same spot would not,
  because tilt adds no baseline.
- **Rock material** — needed to re-derive the multiple-Coulomb-scattering systematic. Do
  not inherit `X0_CONCRETE_M` from the cafeteria project
- **Surveyed baseline** — independent confirmation of the 2.2 m P0↔P1 separation. Absolute
  scale is locked by one surveyed length; angles alone fix only ratios. The cafeteria
  project ran an entire campaign on an unverified self-calibrated baseline
- **Aug 6 rate drop** — cause unknown; handled defensively by splitting T20a/T20b
- **Unmapped-channel hits** (§1.3.1) — the 9 unmapped channels per ASIC fire at 60 % of
  the mapped rate with a near-identical charge median, which argues against simple
  crosstalk. Costs ~20 % acceptance. Not blocking, but worth an answer
- **`_filter` condition** — every row is a 4-fold coincidence; the exact trigger condition
  and its rejection fraction are needed to model acceptance
- **Fiducial marker** — strongly recommended for future campaigns; decouples "is the
  pipeline working" from "what is the scene"
- **Reader memory and cache-versioning are hard constraints, not tuning** — the
  full-scale ingest showed `dtype=str` chunked reads must stay at a bounded chunk size
  (50,000 rows measured safe; 200,000 exhausted a 14 GB machine), and any change to
  reconstruction logic must bump `RECONSTRUCTION_VERSION` or stale cached artifacts will
  be served silently. Future phases that add reconstruction steps inherit both
  constraints; see `docs/phase1-validation-report.md`

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
| Detector constants | Engineers' supplied values, each falsification-tested against data | Inherited from `detector_summary.md`; or fitted from data | `detector_summary.md` is wrong on bar width and layer height; a supplied constant that fails its test is escalated, not silently refitted |
| Channel → bar map | Supplied lookup, gated at ≥40 % bar adjacency | `bar_id = asic*23 + channel` | The identity assumption scores 0.2 % adjacency against 44–47 % for the real map — it would have made every interpolated position noise |
| Layer orientation | `asic_to_layer = [1,3,2,0]`, X,Y,X,Y | X,Y,Y,X inferred from ASIC-pair mutual information | MI ranked the true same-coordinate pairs mid-table. Recorded as a negative result in §1.4 |
| Campaign azimuth | 241° from the campaign log | 281° from `parameters.data` | User confirmed 241°; the file value is a different deployment |
