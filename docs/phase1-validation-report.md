# Phase 1 validation report — S0-det gate on real P0 data

## Command run

```bash
uv run python -m megido.cli validate --config configs/megido.yaml --exposure P0 --max-chunks 1
```

Input: `DET200084_BEAM_20260617_210539_filter.data`, the first run of exposure
P0, 56081 events (one 200,000-row chunk, but the file itself holds only
56081 events).

## Full `format_report` output (verbatim)

```
Exposure P0  file DET200084_BEAM_20260617_210539_filter.data  events 56081

[PASS] adjacency.asic0          measured=0.5329  expected >= 40% (chance 8.7%)
         ASIC 0: 16629/31206 accepted clusters span two adjacent bars
[PASS] adjacency.asic1          measured=0.5356  expected >= 40% (chance 8.7%)
         ASIC 1: 17352/32396 accepted clusters span two adjacent bars
[PASS] adjacency.asic2          measured=0.5388  expected >= 40% (chance 8.7%)
         ASIC 2: 17160/31847 accepted clusters span two adjacent bars
[PASS] adjacency.asic3          measured=0.5553  expected >= 40% (chance 8.7%)
         ASIC 3: 17665/31811 accepted clusters span two adjacent bars
[PASS] coordinate_pairing       measured=0.4549  expected min(same-coord r) > max(cross-coord r)
         same-coordinate r=[0.454, 0.442] [(3, 2), (0, 1)]; cross r=[-0.013, -0.02, -0.027, -0.016] [(0, 2), (0, 3), (1, 2), (1, 3)]
[PASS] active_width             measured=35.2000  expected <= 38.4 cm (23 bars x 1.6 cm pitch)
         hit positions span 35.20 cm (1.60 to 36.80); geometry allows 38.4 cm
[PASS] acceptance_cutoff        measured=1.2478  expected <= 1.219 (= 38.400000000000006 cm / 31.5 cm)
         99.9th percentile of |tan theta| is 1.248, geometric limit 1.219
[PASS] unmapped_loss            measured=0.5620  expected reported only
         31516/56081 events rejected for touching an unmapped channel
[PASS] rejection_breakdown      measured=0.5673  expected reported only
         ok 56.7%; unmapped_channel 31.9%; too_many_bars 5.0%; non_adjacent_bars 6.4%

9/9 checks passed

dead channels: 0 of 128
```

## Gate results vs. thresholds

| Check | Measured | Threshold | Result |
|---|---|---|---|
| `adjacency.asic0` | 0.533 | >= 0.40 (chance 0.087) | PASS |
| `adjacency.asic1` | 0.536 | >= 0.40 (chance 0.087) | PASS |
| `adjacency.asic2` | 0.539 | >= 0.40 (chance 0.087) | PASS |
| `adjacency.asic3` | 0.555 | >= 0.40 (chance 0.087) | PASS |
| `coordinate_pairing` | margin 0.455 | margin > 0 | PASS |
| `active_width` | 35.20 cm (1.60 to 36.80) | <= 38.4 cm x 1.10 = 42.24 cm | PASS |
| `acceptance_cutoff` | 1.248 | <= 1.219 x 1.15 = 1.402 | PASS |

## Acceptance cutoff — Phase 1's designated stop condition

The `acceptance_cutoff` check is Phase 1's stop condition: if it fails, either
the 31.5 cm layer separation or the 38.4 cm active width is wrong, and those
two engineer-supplied constants set the absolute angular scale for every
downstream angle in the pipeline. On real P0 data the 99.9th percentile of
`|tan theta|` measures 1.248 against the purely geometric limit of
38.4 / 31.5 = 1.219 — a 2.4% difference, well inside the 15% tolerance. This
independently confirms both constants directly from real particle tracks
rather than from a specification sheet: a wrong bar count/pitch would show up
as a wrong `active_width`, and a wrong layer separation would shift this ratio
proportionally. Both instead land almost exactly on the expected geometric
value, which is the strongest evidence available that the geometry model is
correct.

## Acceptance budget

Only 24.7% of P0 events have all four layers accepted with a fittable track
(13849 of 56081 events on DET200084 pass full acceptance downstream of the
per-ASIC checks reported here). The dominant loss by far is
`unmapped_channel` at 31.9% of all (event, ASIC) slots — traced to the 9
unmapped channels per ASIC in the channel map. `non_adjacent_bars` (6.4%) and
`too_many_bars` (5.0%) are minor by comparison. Recovering the unmapped
channels would roughly quadruple the usable statistics (24.7% -> ~99% of the
per-ASIC "ok" ceiling of 56.7%), which makes "why do the unmapped channels
fire?" the highest-value open question to hand to the detector engineers.

## Ingest-seam verification

The controller wrote `counts_P0.npz` and `counts_P1.npz` via `save_counts`
and loaded them with `cafeteria_3d_modeling`'s `muontomo.io.load_phantom_dir`.
This built a `Dataset` with sources `['P0', 'P1']`, `txty` arrays of shape
(500, 500) and dtype `int64`, counts preserved exactly (integer in, integer
out), `xedges` running from -1.250 to 1.250 with 501 edges, and a correctly
merged `meta.json` across the two exposures. This confirms Phase 1's output
artifact feeds directly into the reconstruction solver with no changes
required on the solver side.

## Full 70-file ingest

Command:

```bash
python -m megido.cli ingest --config configs/megido.yaml --out runs/ingest
```

Total runtime was about 10 minutes for all 70 files.

| Exposure | Files | Events | Valid tracks | Parquet rows |
|---|---|---|---|---|
| P0 | 21 | 1,172,798 | 292,983 (25.0%) | 1,171,932 |
| T20a | 12 | 672,299 | 168,116 (25.0%) | 672,464 |
| T20b | 26 | 1,132,770 | 284,913 (25.2%) | 1,139,652 |
| P1 | 11 | 484,034 | 122,046 (25.2%) | 488,184 |
| **Total** | **70** | **3,461,901** | **868,058 (25.1%)** | |

Three observations from the full-scale run:

1. Parquet rows are exactly 4x valid tracks in every exposure, confirming the
   per-layer row expansion stays aligned across chunk and file boundaries at
   full scale.
2. 100% of valid tracks fall inside the +/-1.25 tan binning, confirming that
   widening the range from the inherited +/-1.0 was both necessary and
   sufficient.
3. The valid-track fraction is 25.0-25.2% across all four exposures,
   consistent with the 24.7% measured on the single file DET200084. That
   stability across poses and across the unexplained 6 August rate change is
   itself a mild cross-check.

## Per-channel gain structure

Measured on the committed calibration artifacts (`runs/ingest/calib_*.npz`),
restricted to the 92 mapped bars:

| Exposure | hits/channel (median) | mapped MPV median | per-ASIC MPV medians | flagged |
|---|---|---|---|---|
| P0 | 85,824 | 1734 | 1695 / 1716 / 1742 / 1964 | 20 / 92 |
| T20a | 49,048 | 1746 | 1685 / 1707 / 1775 / 2034 | 21 / 92 |
| T20b | 83,162 | 1761 | 1699 / 1764 / 1757 / 2011 | 23 / 92 |
| P1 | 34,957 | 1787 | 1709 / 1796 / 1748 / 1970 | 18 / 92 |

The flagged fraction is stable at 18-23 of 92 across all four exposures,
including across the unexplained 6 August rate change that separates T20a
from T20b, which is a mild independent cross-check that the change did not
alter per-channel gains.

The mapped-bar MPV median is 1734-1787 ADC pedestal-subtracted, consistent
across exposures. This corroborates the Task 4 finding that spec section
1.1's "MIP ~ +4000 ADC" conflated the hit-charge median with the
most-probable value. The prior group's independent calibration of this same
detector reported roughly 1409 ADC subtracted, the same regime.

ASIC 3's ~15% gain offset is per-channel efficiency structure that Phase 2's
baseline model will need to account for.

## Defects found after the gate passed

All nine S0-det checks above pass, and this phase's process included twelve
per-task reviews plus a whole-branch review. None of that caught the three
defects below. Every one of them surfaced only from running the real
pipeline at full scale — none from review, none from the gate. The common
thread in the first defect is the sharpest: the gate checks acceptance rates,
widths, and cutoffs, but no check ever inspected the *fine structure* of the
angular distribution. The new regression test (below) closes that blind
spot; the other two are operational lessons from the full 70-file run.

### 1. Position-quantisation comb in the angle histogram

**Mechanism.** `find_hits` assigned a single-bar cluster the exact bar
centre. When both layers of a coordinate gave single-bar hits, the
reconstructed angle landed on the discrete comb `tan = 1.6 * delta_bar /
31.5`. About 46% of accepted clusters are single-bar, so single-single pairs
are roughly 20% of events per coordinate — enough to leave a visible mark on
the histogram, not enough that any of the nine acceptance/width/cutoff gate
checks would notice.

**Measured before the fix**, on the real P0 angular histogram: teeth at
tan_y = +0.003, +0.053, +0.103 with counts 5558 / 4221 / 3646 against a floor
near 1900 — about 3x. Tooth spacing 0.050, against the predicted 1.6/31.5 =
0.0508. Autocorrelation of the detrended profile peaked at lag 10 bins =
0.0500, matching the tooth spacing exactly.

**Why it nearly fooled us.** The comb is fixed in the *detector* frame, so it
survives a 20 degree tilt unchanged — the profile peak sat at tan_y = +0.023
in all four exposures to three decimals, with peak/median about 2.0. That is
precisely the signature `B(d)`, the Phase 2 baseline, is defined to absorb.
Phase 2 would have fitted it happily and produced a plausible-looking
baseline built on a reconstruction artifact rather than on detector
efficiency structure.

**The fix.** A single-bar hit constrains the crossing only to the region
where that bar collects essentially all the light. Since crossings are
uniform in x and each bar owns one pitch of width, that region has width
`pitch * f_single`, with `f_single` the measured single-bar fraction — so the
dither width is self-calibrated from the data, not a tuned constant.
Positions are now drawn uniformly across it. Two-bar clusters are untouched;
charge sharing genuinely locates those.

**Measured after the fix**, same bins: 2318 / 2036 / 1780 against the same
~1900 floor — flat. The profile is now a smooth monotonic falloff with
angle. Comb autocorrelation at lag 10 is negative for every exposure: P0
-0.2529, T20a -0.2450, T20b -0.3579, P1 -0.3094.

**Regression test.** `tests/test_anghist.py::test_dither_removes_the_position_quantisation_comb`
builds the input directly rather than through the simulator, because the
simulator's linear charge-sharing model yields ~93% two-bar clusters and
barely exercises this path. It measures autocorrelation 0.813 undithered
against -0.050 dithered.

### 2. Reader memory at full-scale chunk size

`dtype=str` is required to survive the repeated header rows in the raw CSV,
but it materialises one Python string per cell. At the old 200,000-row chunk
size that is 51.4 million objects per chunk; the full-scale ingest reached
9 GB resident on a 14 GB machine and had to be killed. Default chunk size is
now 50,000 rows (measured 0.6-0.8 GB resident), with `--chunksize` exposed on
the CLI for machines with more headroom.

### 3. Cache key ignored the code version

`exposure_key` hashed inputs and config but nothing about the reconstruction
code itself, so a re-run after the dither fix would have served the stale
pre-dither cached artifacts as a silent cache hit. The key now includes
`RECONSTRUCTION_VERSION`, currently 2.

## Verdict

Phase 1's exit gate is fully met, with no engineering work outstanding: all 9
S0-det checks pass on real P0 data, including the stop-condition
`acceptance_cutoff` check, which confirms the absolute angular scale to
within 2.4% of the purely geometric prediction; the adjacency, active-width,
and coordinate-pairing checks all pass with comfortable margin; zero dead
channels were found in this chunk; the ingest seam into the downstream
reconstruction solver (`load_phantom_dir`) is verified end to end; and the
full 70-file ingest across all four exposures completed successfully with a
stable ~25% valid-track fraction and exact 4x row expansion throughout.

The ingest artifacts have since been rebuilt with the single-bar dither fix
(§ Defects found after the gate passed) and `RECONSTRUCTION_VERSION = 2`.
The valid-track counts are unchanged — P0 292,983 / T20a 168,116 / T20b
284,913 / P1 122,046, 868,058 total — as expected, since dithering moves a
hit's position within its bar and does not change which clusters are
accepted.

One question remains open, and it is a question for the detector engineers
rather than a gap in this phase's code: the acceptance-budget finding above,
where the nine unmapped channels per ASIC cost roughly three quarters of the
usable statistics. Answering it would not change anything built here, but it
could quadruple the data Phase 2 has to work with, so it is the highest-value
item to escalate before the baseline solve begins.
