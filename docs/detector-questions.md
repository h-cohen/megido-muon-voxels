# Questions for the detector engineers

**Detector:** 4-layer triangular-scintillator muon tracker, CAEN DT5550W + Citiroc, 4 ASICs × 32 channels
**Campaign:** Megiddo cavern, 17 Jun – 14 Sep 2026, runs DET200084 – DET200155
**Asked by:** Hadar Cohen

**2026-09-16** — first list issued.
**2026-09-17** — `parameters.data` + `detector_info.txt` received. Most of the list is
answered; see §0. Four questions remain, plus two new ones raised by the files themselves.

---

## 0. Answered on 2026-09-17 — thank you

The channel→bar lookup, layer assignment and bar geometry resolved the blocking items.
Recorded here so nobody re-answers them.

| Was asked | Answer received |
|---|---|
| Channel → bar map | Four 23-entry lists, one per ASIC, indexed by bar |
| Bars per layer | 23 |
| Why coincidences split into four blocks of 8 | Artifact of the folded map. Dissolves in bar space; no physical gaps |
| Layer order and orientation | `asic_to_layer = [1,3,2,0]`, bottom-up X,Y,X,Y. ASIC 0,1 = Y; ASIC 2,3 = X |
| Layer z positions | 0, 6.2, 31.5, 37.7 cm |
| Bar geometry | base 3.2 cm, height 1.7, side 2.3345, length 40, apex 43.08° |

**We validated the map before using it.** Adjacent bars share charge, so two-hit events
must land on adjacent *bar indices*. On run DET200084 (56 081 events):

| Interpretation | Two-hit events on adjacent bars |
|---|---|
| Assuming channel number = bar number | 130 — **0.2 %** |
| Your map, in bar space | ~17 000 — **44–47 %** on every ASIC |

Chance level is 8.7 %. The map is confirmed correct. For reference, prior analysis code in
our group had assumed `bar_id = asic × 23 + channel`; that would have mis-assigned 99.8 %
of charge-sharing pairs.

Two corrections to note against the older `detector_summary.md` we had been working from:
it gives bar base width as 3.3 cm (actual 3.2) and detector height as ~80 cm (actual layer
span 37.7 cm, station separation 31.5 cm). The second is a large correction — it sets the
angular scale of the whole reconstruction.

---

## 1. Why do the 9 unmapped channels per ASIC still fire?

Your lookup uses 23 of 32 channels per ASIC, leaving 9 unmapped. Those 9 are neither dead
nor cleanly pedestal-only:

| | mapped (23 ch) | unmapped (9 ch) |
|---|---|---|
| Share of all hits | 80 % | **20 %** |
| Hits per channel | ~3700 | ~2200 — 60 % of the mapped rate |
| `CHARGE_HG` median (ADC) | 6098 – 6286 | 5917 – 6306 |

A near-identical charge median is what makes this odd — simple crosstalk or threshold
noise should sit much lower, near the ~2200 ADC pedestal.

> **Question:** Are those 9 channels physically connected to anything? If they are
> genuinely unused inputs, what mechanism puts a full-amplitude signal on them?

We currently reject any event touching an unmapped channel, which costs ~20 % of hits. If
that loss is angle-dependent it biases our efficiency model, which matters because this
campaign has no open-sky calibration run to normalise against.

---

## 2. What exactly does the `_filter` in the filenames do?

Every row in every file has `NEventsInCluster = 4`, so the files appear to be pre-selected
four-fold coincidences.

> **Question:** What is the exact filter condition, and what fraction of triggers does it
> reject? We need it to model the detector's angular acceptance.

---

## 3. What changed around 5–6 August 2026?

Event rate dropped about 22 % between run DET200116 (19 Jul) and DET200119 (6 Aug), across
a gap from 20 Jul to 5 Aug. The detector's position and tilt did **not** change at that
point — the last pose change was on 8 Jul.

> **Question:** Was anything altered during that gap — discriminator thresholds, SiPM bias,
> firmware, temperature or HVAC, physical servicing, re-cabling, a bar or SiPM swap?

If it was re-cabled the channel map may differ across the gap and the two halves are
effectively different detectors. We are currently splitting the period defensively, which
costs statistical power across 38 runs.

---

## 4. Per-channel settings, if they were logged

> **Question:** Citiroc discriminator threshold DAC settings and SiPM bias voltage per
> channel, as used during this campaign.

This would let us build the angular-efficiency model from known hardware settings rather
than fitting it from data — valuable given the missing sky-calibration run, and it may also
explain §1.

---

## 5. Confirmation only — Detector 3, and the 281° azimuth

Two points where the supplied file and our campaign log differ. We have resolved both from
our side and are proceeding on that basis; a one-line confirmation would close them out.

**The unit.** The layer z values in `parameters.data` sit under a `# Detector 3` heading.
Our records say Megiddo used Detector 3, so we are using 0 / 6.2 / 31.5 / 37.7 cm.

> **Confirm:** Megiddo was Detector 3, and its layer spacing was unchanged across
> 17 Jun – 14 Sep.

**The orientation.** `parameters.data` states *"Y axis is pointed 281 degrees WRT true
north"*. Our campaign log records **241°** for every exposure, and we are using 241°.

> **Confirm:** 281° refers to a different installation, not the Megiddo deployment.

Both matter because they are silent failure modes — a wrong layer spacing rescales every
angle, and a 40° azimuth error rotates the whole 3D result about the vertical. Neither
would be flagged by anything downstream.

---

## 6. Surveyed geometry — low priority

We are told the third detector position is 2.2 m from the first. Absolute scale in the
reconstruction is fixed by exactly one surveyed length; everything else is a ratio, so an
error here scales the whole 3D result proportionally.

> **Question:** Was that 2.2 m tape-measured or estimated? Is there a surveyed position for
> the detector, and a measured height above the cavern floor?

Also, for future campaigns: would it be feasible to place an object of known size at a
known position in the field of view? A fiducial separates "is the analysis working" from
"what is the scene", which is currently impossible to establish independently.

---

## Priority

| | Item | Blocks |
|---|---|---|
| 1 | Unmapped channels firing (§1) | ~20 % acceptance, and the efficiency model |
| 2 | The 5–6 Aug change (§3) | Whether 38 runs can be merged |
| 3 | `_filter` condition (§2) | Acceptance model |
| 4 | Per-channel settings (§4) | Would improve the efficiency model |
| 5 | Detector 3 / 241° confirmation (§5) | Confirmation only — we are proceeding |
| 6 | Survey (§6) | Absolute scale of the reconstruction |

Nothing on this list blocks us from starting. The blocking items were answered on
2026-09-17.
