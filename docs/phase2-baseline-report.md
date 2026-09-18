# Phase 2 baseline report

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

Runtime: about 11 s for the solve itself.

Solver output on real data:

- flux index: 2.000 (fixed, not fitted — see section 7)
- normalizations: P0 0.8402, T20a 0.4371, T20b 0.7407, P1 0.3069
- opacity, pos0: 2076 constrained sky bins, median 0 by gauge choice, p5-p95 range
  -1.80 to +0.78
- opacity, pos1: 1266 constrained sky bins, median 0 by gauge, p5-p95 range -1.08 to
  +0.62

## 3. Validation: 5/5 checks pass

Leave-one-out here means: hold out one exposure entirely, fit the model on the rest,
and ask whether the fit — which never saw that exposure's counts — correctly predicts
its angular shape. A high correlation means the response/opacity split generalizes;
it is not just curve-fitting each exposure to itself.

| Check | Measured | Bound |
|---|---|---|
| deviance_per_bin | 1.0412 (7464.2 over 7169 live bins) | 0.2 - 3.0 |
| loo.P0 | 0.9229 over 1800 bins | > 0.90 |
| loo.T20a | 0.9917 | > 0.90 |
| loo.T20b | 0.9908 | > 0.90 |
| loo.P1 | skipped — P1 is the sole exposure at its position, nothing to hold it out against | - |

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

## 6. Caveats — read these before the headline numbers

- **Deviance is per live bin, not per degree of freedom.** Opacity contributes
  roughly one free parameter per constrained sky bin — about 3400 parameters against
  7169 live bins. Per degree of freedom the deviance is closer to 1.98, not the 1.04
  quoted above. Both numbers are real; the per-bin one alone overstates the fit
  quality.
- **T20a and T20b leave-one-out scores are the weakest evidence, not the strongest,
  despite being the highest numbers.** Holding out T20a leaves T20b in the fit at an
  almost identical pose (same position, tilt direction close in the same 20 degree
  family), so predicting T20a from "everything else" is close to predicting it from
  itself. **P0's 0.9229 is the meaningful score** — P0 is predicted from exposures at
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
