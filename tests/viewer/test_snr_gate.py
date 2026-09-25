"""SNR gate: hide voxels whose bootstrap SNR is below N (default 3).

The corners of the detector's square acceptance are its noisiest
directions; their rays still cross many voxels (so the coverage gate keeps
them) but the bootstrap says those voxels are not significant. Hiding
SNR < 3 leaves the cafeteria beams at full value while the corner streaks
fall to ~0.2 of it (docs/cafeteria-run.md). Display-only; same contract as
the other gates: shader AND hover, window from the kept voxels.

The fixture's snr layer is 1 (gated) for i < nx/2 and 10 elsewhere, so a
gate that does nothing, or gates the wrong half, fails.
"""
from __future__ import annotations

import numpy as np

SHAPE = (6, 5, 4)
_GRID = [(fx / 20, fy / 20) for fx in range(3, 18) for fy in range(3, 18) if fy != 10]


def _run_with_snr(run_fixture, *, bright_gated=False):
    run = run_fixture(shape=SHAPE, layers=("volume", "snr"))
    snr = np.full(SHAPE, 10.0, dtype=np.float32)
    snr[: SHAPE[0] // 2] = 1.0
    np.save(run / "snr.npy", snr)
    if bright_gated:
        vol = np.load(run / "volume.npy")
        vol[snr < 3] = 50.0
        np.save(run / "volume.npy", vol)
    return run, snr


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-top").click()
    page.evaluate("() => window.__viewerState.idleNow()")


def _picks(page):
    return page.evaluate(
        """(pts) => {
            const r = document.querySelector('#gl-canvas').getBoundingClientRect();
            return pts.map(([fx, fy]) =>
                window.__viewerState.pick(r.left + fx * r.width, r.top + fy * r.height));
        }""", _GRID)


def _frame(page):
    page.evaluate("() => window.__viewerState.idleNow()")
    return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")


def test_snr_gate_on_by_default_and_hover_skips_insignificant_voxels(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run, snr = _run_with_snr(run_fixture)
    _load(page, dist_path, run)
    assert page.locator("#snr-gate-enabled").is_checked()
    assert page.locator("#snr-gate-value").input_value() == "3"
    on = [p for p in _picks(page) if p]
    assert on and all(snr[p["i"], p["j"], p["k"]] >= 3 for p in on)

    page.locator("#snr-gate-enabled").uncheck()
    off = [p for p in _picks(page) if p]
    assert any(snr[p["i"], p["j"], p["k"]] < 3 for p in off)
    assert not errors


def test_snr_gate_changes_render_and_threshold_is_live(page, dist_path, run_fixture):
    run, _ = _run_with_snr(run_fixture)
    _load(page, dist_path, run)
    gated = _frame(page)
    page.locator("#snr-gate-enabled").uncheck()
    assert _frame(page) != gated
    page.locator("#snr-gate-enabled").check()
    page.locator("#snr-gate-value").fill("20")        # above every voxel
    page.locator("#snr-gate-value").dispatch_event("input")
    assert all(p is None for p in _picks(page))


def test_snr_gate_sets_the_window_from_kept_voxels(page, dist_path, run_fixture):
    run, _ = _run_with_snr(run_fixture, bright_gated=True)
    _load(page, dist_path, run)
    assert page.evaluate("() => window.__viewerState.window[1]") <= 1.0
    page.locator("#snr-gate-enabled").uncheck()
    assert page.evaluate("() => window.__viewerState.window[1]") > 10.0


def test_snr_gate_disabled_without_an_snr_layer(page, dist_path, run_fixture):
    run = run_fixture(shape=SHAPE)
    _load(page, dist_path, run)
    assert page.locator("#snr-gate-enabled").is_disabled()
    assert any(p for p in _picks(page))
