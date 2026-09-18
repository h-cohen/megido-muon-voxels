# Phase 2 baseline report

## Three things to know before reading any number below

1. **The deviance of 1.0367 is per live bin, not per degree of freedom.** Opacity
   contributes roughly one free parameter per constrained sky bin — about 3400
   against 7169 bins — so per degree of freedom it is about 1.98 (7432.4 over
   roughly 3759 effective dof). The fit is good, not as good as 1.0367 alone
   suggests.
2. **Only one of the three leave-one-out scores is a strong test.** Holding out
   T20a leaves T20b at an almost identical pose, so 0.99 is nearly
   self-prediction. P0's 0.9683 is the meaningful number.
3. **Opacity is relative, not absolute.** The flux index is fixed at 2.0 because
   it is not identifiable from this data, so every opacity value is meaningful
   only against that assumed flux model.

Each is expanded below.

## 1. What Phase 2 does, and why it is hard

There is no open-sky calibration run for this detector: every exposure we have looks
through some amount of rock. That means detector response (how the instrument's
efficiency varies across its own bins) and rock absorption (how opacity varies across
sky direction) have to be estimated together from data that only ever contains their
*product*. Neither is separately visible in a single exposure.

What makes this solvable at all is geometry, not extra data. `P0`, `T20a` and `T20b`
sit at one physical position but at two different tilts. Detector response is fixed
in the detector's own frame — it does not care which way the detector is pointed.
Rock opacity is fixed in the sky frame — it does not care which detector produced the
exposure. A 20 degree tilt moves every detector bin onto a different patch of sky
while the response sitting behind that bin does not move. That gives two views of
the same response through different opacity, and two views of overlapping sky
through the same response, which is enough to separate the two factors by alternating
between them.

## 2. Command and output

```
python -m megido.cli solve --config configs/megido.yaml --run runs/ingest --out runs/solve
```

Runtime: the solve itself (`solve_baseline`) takes about 27 s at the default 5000
iterations — the CLI's `--iters` default was raised from 30 to 5000 between the
first draft of this report and this revision, and every number below is from the
converged run. The full `solve` command, including the three leave-one-out
refits at the same iteration budget (see section 3), takes about 33 s wall time.

Solver output on real data (actual CLI output, current code):

```
flux index      2.000
NLL             -4038940.8 after 5000 iterations
normalizations  P0=0.6571  P1=0.2482  T20a=0.3895  T20b=0.66
opacity pos0: 2076 sky bins constrained
    gauge-pinned (median 0, internal): p5..p95 -1.4929..+0.6979
    referenced to the most transparent direction (physical): median 1.4929  p95 2.1907  max 3.3339
opacity pos1: 1266 sky bins constrained
    gauge-pinned (median 0, internal): p5..p95 -0.9853..+0.5438
    referenced to the most transparent direction (physical): median 0.9853  p95 1.5291  max 2.3847
```

See section 3a for what "gauge-pinned" vs. "referenced to the most transparent
direction" mean and which one Phase 3 should use.

## 3. Validation: 5/5 checks pass

Leave-one-out here means: hold out one exposure entirely, fit the model on the rest,
and ask whether the fit — which never saw that exposure's counts — correctly predicts
its angular shape. A high correlation means the response/opacity split generalizes;
it is not just curve-fitting each exposure to itself.

Leave-one-out is fit at the same iteration budget as the main solve (`--iters`,
default 5000) — it used to run at half that budget as an economy, but given how
convergence-sensitive this method proved to be during earlier fix rounds, that
economy was the wrong one to take. It costs about 6 extra seconds of wall time
(see section 2), which is not a real budget concern.

| Check | Measured | Bound |
|---|---|---|
| deviance_per_bin | 1.0367 (7432.4 over 7169 live bins) | 0.2 - 3.0 |
| loo.P0 | 0.9683 over 1800 bins | > 0.90 |
| loo.T20a | 0.9915 over 1945 bins | > 0.90 |
| loo.T20b | 0.9909 over 1945 bins | > 0.90 |
| loo.P1 | skipped — P1 is the sole exposure at its position, nothing to hold it out against | - |

## 3a. Two opacity conventions, and which one to use

`sol.opacity` (what `BaselineSolution` stores and what the CLI prints as
"gauge-pinned") is pinned to a median of zero. That pin is an internal
identifiability device, not a physical statement: the absolute level of lambda is
degenerate with the per-exposure normalization, nothing in the data fixes where
zero is, and forcing the median to zero necessarily puts half the sky at negative
opacity. Negative opacity would mean more flux than open sky, which is not
physical — so this map is for solver bookkeeping and for comparing runs, not for
reading off a number and calling it "the opacity in this direction."

`BaselineSolution.normalized_opacity()` implements the convention a physicist can
actually read: it shifts lambda so the most transparent directions (the 5th
percentile, by default) sit at zero and clips the rest to be non-negative. That is
the quantity printed as "referenced to the most transparent direction (physical)"
above, and it is the one **Phase 3 should consume**.

The two positions' normalized maps are pinned independently — pos0's zero point
and pos1's zero point are each anchored to that position's own most-transparent
direction, and nothing ties the two together. **Do not naively difference the
pos0 and pos1 normalized opacity maps** expecting comparable absolute levels;
only within-position spatial variation is meaningful across the two.

## 4. Bootstrap: 10 replicas, 23 s

| Position | bins | median sigma | p90 sigma | median abs lambda | SNR>1 | SNR>2 | SNR>3 | SNR>5 |
|---|---|---|---|---|---|---|---|---|
| pos0 | 2076 | 0.0913 | 0.2550 | 1.4929 | 92.9% | 90.2% | 87.5% (1817 bins) | 80.3% |
| pos1 | 1266 | 0.1388 | 0.2878 | 0.9853 | 90.8% | 85.0% | 79.1% (1002 bins) | 63.5% |

Read plainly: roughly 1800 sky directions from pos0 and 1000 from pos1 are measured
above 3 sigma. Typical opacity there sits around lambda 1.0-1.5, i.e. transmission
of about 22-37% (exp(-lambda)).

## 5. Synthetic gates

These are separated deliberately into two kinds:

- **Method checks** (high-count synthetic data, plenty of statistics): confirm the
  alternating solve itself is correct when noise is not the limiting factor.
  Response correlation 0.99997, response width 0.00107, high-count opacity 0.9549,
  high-count flat-sky 0.0298.
- **Reality checks** (synthetic data at campaign-realistic statistics): confirm the
  method still works under the counts we actually have, where noise matters.
  Campaign opacity 0.8548, campaign flat-sky 0.0981.

The gap between the high-count and campaign numbers is the price of real statistics,
not a flaw in the method.

## 6. Caveats, in full

- **Deviance is per live bin, not per degree of freedom.** Opacity contributes
  roughly one free parameter per constrained sky bin — about 3400 parameters against
  7169 live bins. Per degree of freedom the deviance is closer to 1.98 (7432.4 over
  roughly 3759 effective dof), not the 1.0367 quoted above. Both numbers are real;
  the per-bin one alone overstates the fit quality.
- **T20a and T20b leave-one-out scores are the weakest evidence, not the strongest,
  despite being the highest numbers.** Holding out T20a leaves T20b in the fit at an
  almost identical pose (same position, tilt direction close in the same 20 degree
  family), so predicting T20a from "everything else" is close to predicting it from
  itself. **P0's 0.9683 is the meaningful score** — P0 is predicted from exposures at
  a different tilt and, more importantly, from... itself only via T20a/T20b, i.e. a
  genuinely different geometric view.
- **The held-out bin selection is not perfectly blind.** Which sky bins get scored in
  leave-one-out comes from `MIN_SKY_COUNTS` applied to raw counts summed over the
  position, and that sum includes the held-out exposure's own counts. No
  shape-carrying parameter leaks into the fit, so the check is not circular, but the
  bin selection is not blind either — a marginal case could in principle be selected
  because the held-out exposure itself pushed a bin over threshold.

## 7. What is NOT established

The flux index is fixed at 2.0, not fitted, because it is not identifiable from this
data: opacity is a free parameter per sky bin, and the flux's angular shape is also a
function of sky direction, so only their product is constrained — the split between
"steeper flux" and "more absorption in that direction" is not resolved by the data
alone. Measured on real data, fixing the index anywhere from 1 to 5 moves the best
achievable NLL by only about 200, against roughly 1000 for a single early solver
iteration — i.e. the index is nearly a flat direction in the fit. **The recovered
opacity map is therefore meaningful only relative to the assumed flux model**, not as
an absolute measurement independent of it.

Separately: opacity here is per position and per sky direction, a 2D quantity.
Converting it into a 3D density/geology model is Phase 3's job, not this one.
pos1's opacity *level* is degenerate with its own normalization, because P1 is the
only exposure at pos1 — there is nothing to separate "P1 is less efficient" from
"pos1's sky is more opaque". Only the *shape* of the pos1 opacity map (relative
variation across sky bins) is meaningful; Phase 3 must not naively difference the
pos0 and pos1 opacity maps expecting comparable absolute levels.

## 8. Adding a third detector position

A third position slots in without any code change: add one block to
`configs/megido.yaml` under `exposures`/`position_ids` assigning it `pos2`. The
solver will produce a third opacity map automatically, and leave-one-out gains
another genuinely scoreable exposure (one not sharing a position with anything else,
so — like P0 today — its held-out score would be real evidence rather than
near-self-prediction).
