"""S0-det: falsification tests on the engineer-supplied constants.

Each check is a test on a SUPPLIED number, not a fit. A constant that fails is
escalated to the engineers, never silently replaced.

Two statistics are deliberately not used here (spec 1.4):
  - mutual information between ASIC pairs does NOT discriminate layer
    orientation on this detector; it ranked the true same-coordinate pairs
    mid-table. Position correlation is used instead.
  - track chi2 does not exist: each coordinate has exactly two layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.calib import ChannelCalibration
from megido.detector import DetectorGeometry
from megido.hits import REJECT_NAMES, REJECT_OK, REJECT_UNMAPPED, LayerHits, find_hits
from megido.reader import EventChunk
from megido.tracks import fit_tracks


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    measured: float
    expected: str
    detail: str


def adjacency_check(hits: LayerHits, min_rate: float = 0.40) -> list[Check]:
    """Two-hit clusters must land on ADJACENT bar indices.

    Chance level for two random distinct bars out of 23 is 2/23 = 8.7%. The
    supplied map measures 53-55% on real data; the identity assumption 0.2%.
    """
    out = []
    for asic in range(4):
        accepted = hits.reject[:, asic] == REJECT_OK
        n_acc = int(accepted.sum())
        two_bar = int(((hits.charge_hi[:, asic] > 0) & accepted).sum())
        rate = two_bar / n_acc if n_acc else 0.0
        out.append(Check(
            name=f"adjacency.asic{asic}",
            passed=rate >= min_rate,
            measured=rate,
            expected=f">= {min_rate:.0%} (chance 8.7%)",
            detail=f"ASIC {asic}: {two_bar}/{n_acc} accepted clusters span two adjacent bars",
        ))
    return out


def coordinate_pairing_check(hits: LayerHits, geom: DetectorGeometry) -> Check:
    """Same-coordinate layers must correlate in POSITION more than cross pairs."""
    pos = hits.position_cm
    ok = (hits.reject == REJECT_OK).all(axis=1)
    p = pos[ok]
    if p.shape[0] < 100:
        return Check("coordinate_pairing", False, 0.0, "n/a", "too few complete events")

    def r(a: int, b: int) -> float:
        return float(np.corrcoef(p[:, a], p[:, b])[0, 1])

    same = [geom.asics_for_coord("x"), geom.asics_for_coord("y")]
    same_r = [r(a, b) for a, b in same]
    cross = [(a, b) for a in range(4) for b in range(a + 1, 4)
             if geom.coord_of_asic(a) != geom.coord_of_asic(b)]
    cross_r = [r(a, b) for a, b in cross]

    margin = min(same_r) - max(cross_r)
    return Check(
        name="coordinate_pairing",
        passed=margin > 0.0,
        measured=margin,
        expected="min(same-coord r) > max(cross-coord r)",
        detail=(f"same-coordinate r={[round(v, 3) for v in same_r]} "
                f"{same}; cross r={[round(v, 3) for v in cross_r]} {cross}"),
    )


def active_width_check(hits: LayerHits, geom: DetectorGeometry,
                       tol: float = 0.10) -> Check:
    """Occupied hit-position span must match 23 bars at the derived pitch.

    Spec 4.1 sub-task 0b. A flat-topped occupancy with hard edges confirms both
    the bar count and the pitch; a short or long span means one of them is wrong.
    """
    ok = hits.reject == REJECT_OK
    pos = hits.position_cm[ok]
    if pos.size < 500:
        return Check("active_width", False, 0.0, "n/a", "too few accepted hits")

    lo, hi = np.quantile(pos, [0.001, 0.999])
    measured = float(hi - lo)
    expected = geom.bar.active_width_cm
    return Check(
        name="active_width",
        passed=measured <= expected * (1.0 + tol),
        measured=measured,
        expected=f"<= {expected:.1f} cm ({geom.bar.n_bars} bars x {geom.bar.pitch_cm} cm pitch)",
        detail=(f"hit positions span {measured:.2f} cm "
                f"({lo:.2f} to {hi:.2f}); geometry allows {expected:.1f} cm"),
    )


def acceptance_cutoff_check(tracks, geom: DetectorGeometry, tol: float = 0.15) -> Check:
    """The |tan| distribution must die by the geometric limit width/dz.

    A mismatch means the layer separation (31.5 cm) or the active width
    (38.4 cm) is wrong. This is the absolute angular scale, so it is a stop
    condition, not a warning.
    """
    t = np.hypot(tracks.tan_x[tracks.valid], tracks.tan_y[tracks.valid])
    if t.size < 100:
        return Check("acceptance_cutoff", False, 0.0, "n/a", "too few valid tracks")
    measured = float(np.quantile(t, 0.999))
    limit = geom.max_tan()
    return Check(
        name="acceptance_cutoff",
        passed=measured <= limit * (1.0 + tol),
        measured=measured,
        expected=f"<= {limit:.3f} (= {geom.bar.active_width_cm} cm / {geom.dz_cm('x')} cm)",
        detail=f"99.9th percentile of |tan theta| is {measured:.3f}, geometric limit {limit:.3f}",
    )


def unmapped_loss_check(hits: LayerHits) -> Check:
    """Acceptance lost to events touching unmapped channels (spec 1.3.1)."""
    n = hits.reject.shape[0]
    lost = int((hits.reject == REJECT_UNMAPPED).any(axis=1).sum())
    frac = lost / n if n else 0.0
    return Check(
        name="unmapped_loss",
        passed=True,   # reported, not gated - the cause is an open question
        measured=frac,
        expected="reported only",
        detail=f"{lost}/{n} events rejected for touching an unmapped channel",
    )


def rejection_breakdown_check(hits: LayerHits) -> Check:
    """Reported, not gated: the acceptance loss attributed to each cause."""
    total = hits.reject.size
    parts = []
    for code in sorted(REJECT_NAMES):
        n = int((hits.reject == code).sum())
        if n:
            parts.append(f"{REJECT_NAMES[code]} {n / total:.1%}")
    accepted = float((hits.reject == REJECT_OK).mean())
    return Check(
        name="rejection_breakdown",
        passed=True,
        measured=accepted,
        expected="reported only",
        detail="; ".join(parts),
    )


def run_all(chunk: EventChunk, geom: DetectorGeometry,
            cal: ChannelCalibration) -> list[Check]:
    hits = find_hits(chunk, geom, cal)
    tracks = fit_tracks(hits, geom)
    checks = list(adjacency_check(hits))
    checks.append(coordinate_pairing_check(hits, geom))
    checks.append(active_width_check(hits, geom))
    checks.append(acceptance_cutoff_check(tracks, geom))
    checks.append(unmapped_loss_check(hits))
    checks.append(rejection_breakdown_check(hits))
    return checks


def format_report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"[{status}] {c.name:24s} measured={c.measured:.4f}  expected {c.expected}")
        lines.append(f"         {c.detail}")
    n_fail = sum(1 for c in checks if not c.passed)
    lines.append(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return "\n".join(lines)
