import numpy as np
import pytest

from megido.calib import calibrate
from megido.detector import DetectorGeometry
from megido.hits import LayerHits, REJECT_OK, REJECT_UNMAPPED, find_hits
from megido.sim import simulate
from megido.tracks import Tracks, fit_tracks
from megido.validate import (Check, acceptance_cutoff_check, active_width_check,
                             adjacency_check, coordinate_pairing_check,
                             format_report, rejection_breakdown_check, run_all,
                             unmapped_loss_check)


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


@pytest.fixture
def simulated(geom):
    truth = simulate(geom, n_events=30_000, seed=7)
    cal = calibrate([truth.chunk])
    return truth, cal


def _layer_hits(position_cm, reject=None, charge_hi=None):
    """Hand-built LayerHits, so a check's threshold can be pinned directly."""
    pos = np.asarray(position_cm, dtype=float)
    n = pos.shape[0]
    rej = np.full((n, 4), REJECT_OK, np.uint8) if reject is None else np.asarray(reject, np.uint8)
    hi = np.zeros((n, 4)) if charge_hi is None else np.asarray(charge_hi, dtype=float)
    return LayerHits(position_cm=pos, bar_lo=np.zeros((n, 4), np.int16),
                     charge_lo=np.ones((n, 4)), charge_hi=hi, reject=rej)


def _tracks(tan_x, tan_y):
    tx = np.asarray(tan_x, dtype=float)
    ty = np.asarray(tan_y, dtype=float)
    return Tracks(tan_x=tx, tan_y=ty, x_bot_cm=np.zeros_like(tx),
                  y_bot_cm=np.zeros_like(tx), valid=np.ones(tx.shape, bool))


def test_adjacency_passes_with_the_correct_map(geom, simulated):
    truth, cal = simulated
    checks = adjacency_check(find_hits(truth.chunk, geom, cal))
    assert len(checks) == 4
    assert all(c.passed for c in checks), [c.detail for c in checks]


def test_adjacency_fails_with_a_scrambled_map(geom, simulated):
    """Negative control. A wrong map must fail the gate.

    23 is prime, so index -> (7*index) % 23 is a bijection that sends physically
    adjacent bars 10 apart (7^-1 = 10 mod 23). Every channel stays mapped, so
    unmapped-channel rejection cannot mask the result, but physical adjacency
    is destroyed.
    """
    truth, cal = simulated
    scrambled = {
        a: tuple(geom.asic_channels[a][(7 * i) % 23] for i in range(23))
        for a in range(4)
    }
    wrong = DetectorGeometry(asic_channels=scrambled, asic_to_layer=geom.asic_to_layer,
                             layer_z_cm=geom.layer_z_cm, layer_coord=geom.layer_coord,
                             bar=geom.bar)
    hits = find_hits(truth.chunk, geom=wrong, cal=cal)

    # the control must not be neutered by mass rejection
    assert (hits.reject == REJECT_UNMAPPED).mean() < 0.01, "every hit must still map"

    checks = adjacency_check(hits)
    assert not all(c.passed for c in checks), [c.detail for c in checks]


def test_coordinate_pairing_prefers_the_supplied_assignment(geom, simulated):
    truth, cal = simulated
    c = coordinate_pairing_check(find_hits(truth.chunk, geom, cal), geom)
    assert c.passed, c.detail


def test_active_width_matches_23_bars_at_the_derived_pitch(geom, simulated):
    """Spec 4.1 sub-task 0b: the occupied span must match 23 bars x 1.6 cm pitch."""
    truth, cal = simulated
    c = active_width_check(find_hits(truth.chunk, geom, cal), geom)
    assert c.measured > 0
    # the simulator confines tracks to the middle half, so the span is a lower bound
    assert c.measured <= geom.bar.active_width_cm * 1.10


def test_acceptance_cutoff_matches_geometry(geom, simulated):
    truth, cal = simulated
    tracks = fit_tracks(find_hits(truth.chunk, geom, cal), geom)
    c = acceptance_cutoff_check(tracks, geom)
    assert c.measured > 0
    # the simulator throws inside 0.6 * max_tan, so the cutoff must not EXCEED geometry
    assert c.measured <= geom.max_tan() * 1.05


def test_unmapped_loss_is_zero_in_simulation(geom, simulated):
    truth, cal = simulated
    c = unmapped_loss_check(find_hits(truth.chunk, geom, cal))
    assert c.measured == pytest.approx(0.0, abs=1e-6)


def test_adjacency_threshold_is_pinned_at_the_boundary():
    """The gate must sit at 0.40 — not merely somewhere below the simulator's rate."""
    n = 2000
    for frac, should_pass in [(0.45, True), (0.35, False)]:
        hi = np.zeros((n, 4))
        hi[: int(frac * n), :] = 1.0          # this fraction are two-bar clusters
        hits = _layer_hits(np.full((n, 4), 19.2), charge_hi=hi)
        checks = adjacency_check(hits)
        assert all(c.passed for c in checks) is should_pass, f"frac={frac}"


def test_active_width_fails_when_the_span_exceeds_the_geometry():
    """Catches a tolerance sign flip: too wide a span must FAIL."""
    geom = DetectorGeometry.megiddo()
    rng = np.random.default_rng(0)
    wide = rng.uniform(0.0, 60.0, size=(4000, 4))     # 60 cm across a 38.4 cm detector
    assert not active_width_check(_layer_hits(wide), geom).passed

    narrow = rng.uniform(1.6, 36.8, size=(4000, 4))   # the real span
    assert active_width_check(_layer_hits(narrow), geom).passed


def test_acceptance_cutoff_fails_when_tracks_exceed_the_geometric_limit():
    """Catches a tolerance sign flip: angles beyond width/dz must FAIL."""
    geom = DetectorGeometry.megiddo()
    rng = np.random.default_rng(1)
    n = 5000

    too_steep = rng.uniform(0.0, 2.5, size=n)          # well past max_tan = 1.219
    assert not acceptance_cutoff_check(_tracks(too_steep, np.zeros(n)), geom).passed

    plausible = rng.uniform(0.0, 1.0, size=n)
    assert acceptance_cutoff_check(_tracks(plausible, np.zeros(n)), geom).passed


def test_coordinate_pairing_fails_when_a_cross_pair_correlates_more():
    """If ASIC 0 (y) tracked ASIC 2 (x), the layer assignment would be wrong."""
    geom = DetectorGeometry.megiddo()
    rng = np.random.default_rng(2)
    n = 3000
    a = rng.normal(19.2, 5.0, n)
    pos = np.empty((n, 4))
    pos[:, 0] = a                                  # ASIC 0, y
    pos[:, 1] = rng.normal(19.2, 5.0, n)           # ASIC 1, y — uncorrelated with ASIC 0
    pos[:, 2] = a                                  # ASIC 2, x — perfectly tracks ASIC 0
    pos[:, 3] = rng.normal(19.2, 5.0, n)           # ASIC 3, x — uncorrelated with ASIC 2
    check = coordinate_pairing_check(_layer_hits(pos), geom)
    assert not check.passed, check.detail
    assert check.measured < 0.0, "margin must go negative when a cross pair wins"


def test_rejection_breakdown_names_every_reason_present():
    n = 100
    rej = np.full((n, 4), REJECT_OK, np.uint8)
    rej[:10, 0] = REJECT_UNMAPPED
    check = rejection_breakdown_check(_layer_hits(np.full((n, 4), 19.2), reject=rej))
    assert check.passed and check.expected == "reported only"
    assert "ok" in check.detail and "unmapped_channel" in check.detail
    assert check.measured == pytest.approx(390 / 400)


def test_run_all_returns_checks_and_formats(geom, simulated):
    truth, cal = simulated
    checks = run_all(truth.chunk, geom, cal)
    assert len(checks) >= 7
    assert all(isinstance(c, Check) for c in checks)
    report = format_report(checks)
    assert "PASS" in report or "FAIL" in report
    assert "adjacency" in report
