# Questions for the detector engineers

**Detector:** 4-layer triangular-scintillator muon tracker, CAEN DT5550W + Citiroc, 4 ASICs × 32 channels
**Campaign:** Megiddo cavern, 17 Jun – 14 Sep 2026, runs DET200084 – DET200155
**Asked by:** Hadar Cohen · 2026-09-16

---

## Why we are asking

We are reconstructing a 3D density map from this campaign's data. To convert a fired
channel into a muon direction we need to know which physical bar each DAQ channel is
connected to, how the bars are spaced, and how far apart the layers sit.

We have analysed the raw `.data` files directly. **Three statements in the detector
documentation we inherited are contradicted by the data**, and one critical number is not
recorded anywhere. We would rather have the answers from you than reverse-engineer them
and get them subtly wrong — a wrong instrumental constant produces a reconstruction that
looks plausible and is silently false.

Answers to §A are blocking. §B and §C would save us weeks. §D is bookkeeping.

---

## A. Bars and channel mapping — **blocking**

### A1. How many scintillator bars are there per layer?

**What the documentation says:** 23 bars per layer, 92 total; "9 channels unused per
layer" because the DAQ provides 32.

**What the data shows:** all 32 channels on all 4 ASICs register hits, at comparable rates
and with comparable muon charge spectra. Channels 23–31 are not dead and do not look like
noise. Sample from ASIC 0, pedestal subtracted:

| Channel | hits (100 k events) | pedestal (HIT=0) | signal median (HIT=1) | p95 |
|---|---|---|---|---|
| 0 | 6868 | 2353 | 6732 | 11549 |
| 22 | 4437 | 2246 | 6060 | 10063 |
| **23** | **9076** | **2324** | **6746** | **11492** |
| **28** | 4747 | 2279 | 6220 | 11031 |
| **31** | 6863 | 2267 | 6449 | 11139 |

This holds on both the Megiddo data and older cafeteria-campaign data, so it is a property
of the detector, not of this site.

> **Question:** How many bars per layer are physically installed and instrumented?
> If the answer is 23, what are channels 23–31 connected to, and why do they see
> muon-like signals?

---

### A2. What is the channel → bar mapping?

**What we need:** for each ASIC, a table mapping DAQ channel number to physical bar
position across the layer. Equivalently, the cabling or connector diagram.

**Why we cannot assume it is the identity:** adjacent bars share charge, so two-hit events
reveal which bars are physically neighbours. The strongest coincidence pairs are *not*
consecutive channel numbers. From 178 420 two-hit events on ASIC 0:

```
11-23: 8501    7-27: 8087    15-19: 7749    3-31: 6385     <- top tier, ~2x the next
2-31: 4808    15-18: 4717   12-14: 4636     9-20: 4630
```

The same four dominant pairs appear on all four ASICs, so the cabling is consistent
between layers. Note each top-tier pair sums to 34, and every member is ≡ 3 (mod 4).

Prior analysis code in our group assumed `bar_id = asic × 23 + channel`. That assumption
was never checked against data and, given the above, is wrong.

> **Question:** Please provide the channel → bar map, or the connector/cabling diagram it
> can be derived from. A 32-entry list per ASIC is ideal.

---

### A3. Why do adjacent-bar coincidences split into four disjoint groups of eight?

Building the neighbour graph from two-hit coincidences, the 32 channels partition into
four blocks that **never** share a coincidence with each other:

```
{ 0,  1,  2,  3, 28, 29, 30, 31}
{ 4,  5,  6,  7, 24, 25, 26, 27}
{ 8,  9, 10, 11, 20, 21, 22, 23}
{12, 13, 14, 15, 16, 17, 18, 19}
```

i.e. block *k* = `{4k … 4k+3} ∪ {31−4k … 28−4k}`.

A single contiguous row of 32 bars would produce **one connected chain**, not four isolated
blocks. Candidate explanations we can think of:

- (a) each layer is built from four separate 8-bar modules, with a physical gap between
  modules that suppresses cross-module charge sharing
- (b) a connector folding that we have mis-modelled
- (c) something else entirely

> **Question:** Which is it? If there are physical gaps between bar groups, we need their
> width — our forward model currently assumes a continuous active area and would place
> every track slightly wrong.

---

### A4. Why does channel occupancy alternate with period 4?

Hit counts follow a strict period-4 pattern on every ASIC: channels ≡ 0 or 3 (mod 4)
collect roughly twice the hits of channels ≡ 1 or 2 (mod 4). ASIC 0, hits per channel:

```
ch:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    3k  2k  2k  3k  5k  2k  2k  4k  4k  2k  2k  4k  3k  1k  2k  4k
ch: 16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
    4k  1k  2k  3k  5k  2k  2k  5k  4k  2k  2k  4k  3k  1k  1k  4k
```

Candidate causes: per-channel discriminator thresholds, per-channel SiPM bias, or
triangular bars alternating apex-up / apex-down so that the mean path length differs
between the two orientations.

> **Question:** Is this expected? Which of the above causes it?

This one matters more than it looks. We have **no open-sky calibration run** for this
campaign, so we must estimate the detector's angular efficiency from the data itself. If
this pattern is a known, characterisable hardware effect, we can model it instead of
fitting it — which materially improves the final reconstruction.

---

## B. Layer geometry — high value

### B1. Layer order and orientation

**What the documentation says:** top-X, top-Y, bottom-X, bottom-Y (i.e. X, Y, X, Y).

**What the data suggests:** mutual information between ASIC pairs on single-hit events —
two layers measuring the *same* coordinate must correlate — ranks as follows:

| pair | MI | pair | MI |
|---|---|---|---|
| **0–3** | **0.274** | 0–1 | 0.232 |
| **1–2** | **0.247** | 2–3 | 0.245 |
| | | 0–2 | 0.156 |
| | | 1–3 | 0.159 |

Highest for {0,3} and {1,2}; lowest for {0,2} and {1,3}. That is the signature of
**X, Y, Y, X** — two identical XY modules with one flipped — not X, Y, X, Y.

> **Question:** What is the physical stacking order top to bottom, which ASIC reads which
> layer, and which coordinate does each layer measure?

---

### B2. What are the layer z positions?

This number is **not recorded in any file, config, or document we have.** The
documentation says "detector height ≈ 80 cm" and "layer separation is adjustable", with no
value.

Without it we cannot convert a hit displacement into an angle at all — it sets the
absolute angular scale of the entire reconstruction.

> **Question:** The z position of each of the four layers, in mm, as configured for this
> campaign. If the spacing was changed at any point between June and September, please
> give the dates.

---

### B3. Bar geometry

> **Question:** Please confirm or correct:
> - bar pitch (centre-to-centre spacing of adjacent bars) — mm
> - triangle base width and height — mm
> - do bars alternate apex-up / apex-down within a layer?
> - total active width of one layer — mm

Documentation says base 33 mm, height 17 mm, bar length 400 mm, active area ≈ 65 × 65 cm.
The pitch enters the sub-bar position formula `x = a·n/(N+n)` directly, so an error here
scales every reconstructed position linearly.

---

## C. Operating conditions — high value

### C1. What does the `_filter` in the filenames do?

Every row in every file has `NEventsInCluster = 4`, so the files appear to be pre-selected
4-fold coincidences.

> **Question:** What exactly is the filter condition, and what fraction of triggers does
> it reject? We need this to model the detector's acceptance correctly.

---

### C2. What changed around 5–6 August 2026?

Event rate dropped about 22 % between run DET200116 (19 Jul) and DET200119 (6 Aug), across
a gap from 20 Jul to 5 Aug. The detector's position and tilt did **not** change at that
point — the last pose change was on 8 Jul.

> **Question:** Was anything altered during that gap — discriminator thresholds, SiPM bias,
> firmware, temperature/HVAC, physical servicing, re-cabling, a bar or SiPM swap?

If the detector was re-cabled, the channel map may differ before and after, and we must
treat the two halves as separate detectors. We are currently handling this defensively by
splitting the period in two, which costs us statistical power.

---

### C3. Per-channel settings

> **Question:** If they were logged, please send the Citiroc discriminator threshold DAC
> settings and the SiPM bias voltage per channel, as used during this campaign.

This would let us build the angular-efficiency model from known hardware settings rather
than fitting it from data — valuable given the missing sky-calibration run.

---

## D. Campaign bookkeeping — low priority

### D1. Missing run IDs

DET200117 and DET200118 are absent, as is data for 11–13 Sep.

> **Question:** Were those runs taken and discarded, or never taken? If discarded, why?

### D2. Surveyed geometry

We are told the third detector position is 2.2 m from the first. Absolute scale in the
reconstruction is fixed by exactly one surveyed length; everything else is a ratio, so an
error here scales the whole 3D result proportionally.

> **Question:** Was that 2.2 m tape-measured or estimated? Is there a surveyed position for
> the detector, and a measured height above the cavern floor?

### D3. Fiducial marker

> **Question:** For future campaigns — would it be feasible to place an object of known
> size at a known position in the field of view? It separates "is the analysis working"
> from "what is the scene", which is currently impossible to do independently.

---

## Summary of what we need most

| Priority | Item | Blocks |
|---|---|---|
| 1 | Channel → bar map (A2) | Everything. No track direction without it |
| 2 | Bars per layer (A1) | Active width, angular acceptance |
| 3 | Layer z positions (B2) | Absolute angular scale |
| 4 | Explanation of the four blocks (A3) | Forward model geometry |
| 5 | Layer order / orientation (B1) | Track fitting |
| 6 | What changed on 5–6 Aug (C2) | Whether 38 runs can be merged |
