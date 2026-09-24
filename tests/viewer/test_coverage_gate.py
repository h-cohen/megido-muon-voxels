"""Coverage gate: hide voxels crossed by fewer than N measured rays.

Voxels near the grid edge are crossed by one or two oblique rays that no
other ray shares; under non-negativity the solver parks those rays' noise
there as mass (the "bright outer shell", docs/cafeteria-run.md). The gate
is display-only and marks them "not constrained". It must apply in the
shader AND in hover picking, in the same order as the other gates.

The fixture's rays layer is 1 (gated at the default N=6) for the half of
the volume with i < nx/2 and 10 for the other half, so a gate that does
nothing, or gates the wrong half, fails.
"""
from __future__ import annotations

import numpy as np

SHAPE = (6, 5, 4)
_GRID = [(fx / 20, fy / 20) for fx in range(3, 18) for fy in range(3, 18) if fy != 10]


def _run_with_rays(run_fixture):
    run = run_fixture(shape=SHAPE, layers=("volume", "rays"))
    rays = np.full(SHAPE, 10.0, dtype=np.float32)
    rays[: SHAPE[0] // 2] = 1.0
    np.save(run / "rays.npy", rays)
    return run, rays


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


def test_gate_is_on_by_default_and_hover_never_lands_on_a_gated_voxel(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run, rays = _run_with_rays(run_fixture)
    _load(page, dist_path, run)

    assert page.locator("#coverage-gate-enabled").is_checked()
    assert page.locator("#coverage-gate-value").input_value() == "6"
    on = [p for p in _picks(page) if p]
    assert on and all(rays[p["i"], p["j"], p["k"]] >= 6 for p in on)

    page.locator("#coverage-gate-enabled").uncheck()
    off = [p for p in _picks(page) if p]
    assert any(rays[p["i"], p["j"], p["k"]] < 6 for p in off)
    assert not errors


def test_gate_changes_the_render_and_threshold_is_live(page, dist_path, run_fixture):
    run, _ = _run_with_rays(run_fixture)
    _load(page, dist_path, run)
    gated = _frame(page)
    page.locator("#coverage-gate-enabled").uncheck()
    ungated = _frame(page)
    assert gated != ungated

    page.locator("#coverage-gate-enabled").check()
    page.locator("#coverage-gate-value").fill("20")      # above every voxel: all hidden
    page.locator("#coverage-gate-value").dispatch_event("input")
    assert all(p is None for p in _picks(page))
    assert _frame(page) != gated


def test_gate_is_disabled_without_a_rays_layer(page, dist_path, run_fixture):
    run = run_fixture(shape=SHAPE)
    _load(page, dist_path, run)
    assert page.locator("#coverage-gate-enabled").is_disabled()
    assert any(p for p in _picks(page))


def test_auto_window_is_set_by_the_voxels_the_gate_keeps(page, dist_path, run_fixture):
    """The shell's inflated values must not set the colour scale of what is
    shown: with the gate on, the window comes from ungated voxels only."""
    run, rays = _run_with_rays(run_fixture)
    vol = np.load(run / "volume.npy")
    vol[rays < 6] = 50.0                  # a bright shell, all of it gated
    np.save(run / "volume.npy", vol)
    _load(page, dist_path, run)
    hi_on = page.evaluate("() => window.__viewerState.window[1]")
    assert hi_on <= 1.0                   # the kept voxels are uniform in [0, 1)

    page.locator("#coverage-gate-enabled").uncheck()
    hi_off = page.evaluate("() => window.__viewerState.window[1]")
    assert hi_off > 10.0
