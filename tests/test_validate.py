import numpy as np
import pytest

from megido.calib import calibrate
from megido.detector import DetectorGeometry
from megido.hits import REJECT_UNMAPPED, find_hits
from megido.sim import simulate
from megido.tracks import fit_tracks
from megido.validate import (Check, acceptance_cutoff_check, active_width_check,
                             adjacency_check, coordinate_pairing_check,
                             format_report, run_all, unmapped_loss_check)


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


@pytest.fixture
def simulated(geom):
    truth = simulate(geom, n_events=30_000, seed=7)
    cal = calibrate([truth.chunk])
    return truth, cal


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


def test_run_all_returns_checks_and_formats(geom, simulated):
    truth, cal = simulated
    checks = run_all(truth.chunk, geom, cal)
    assert len(checks) >= 7
    assert all(isinstance(c, Check) for c in checks)
    report = format_report(checks)
    assert "PASS" in report or "FAIL" in report
    assert "adjacency" in report
