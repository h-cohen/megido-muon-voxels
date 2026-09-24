"""Hover picking matches the renderer (Task 4).

`castHoverRay` must march the SAME box (rayBox), step rule (RAY_STEPS,
minStepVoxels, cubic voxel size) and clip order (clip box, clip plane, sigma
gate, surface clip) as the FRAGMENT_SRC shader march, even though the render
loop may drop to the adaptive-preview step count while interacting.
`state.pick(x, y)` (exposed as `window.__viewerState.pick`) is the test hook
that calls castHoverRay directly, so these tests do not need to simulate
pointermove events.

The `run_fixture(hill=True)` fixture writes a flat hillside surface H = 2.0
over a volume spanning world z in [1, 3] (origin_m z=1, spacing_m=0.5,
shape z=4), so "above the surface" (z > 2.0) is exactly the upper half of the
volume box.
"""
from __future__ import annotations


def _pick_grid(page, points_frac):
    """points_frac: list of (fx, fy) fractions of the canvas bounding rect.
    Returns the list of pick() results (each {i,j,k,value} or null)."""
    return page.evaluate(
        """(pts) => {
            const canvas = document.querySelector('#gl-canvas');
            const rect = canvas.getBoundingClientRect();
            return pts.map(([fx, fy]) =>
                window.__viewerState.pick(rect.left + fx * rect.width, rect.top + fy * rect.height));
        }""",
        points_frac,
    )


def _world_z(meta, k):
    return meta["origin_m"][2] + (k + 0.5) * meta["spacing_m"]


# A small grid of canvas points. Excludes the exact fy=0.5 row: under the
# Front preset (pitch=0, camera.mjs) the orbit camera always looks straight
# at state.camera.target, so the screen's horizontal centre LINE (NDC y=0)
# is a ray with zero vertical deviation -- its world z is EXACTLY constant
# and equal to target.z along its whole length. frameAll() sets target.z to
# the volume's z midpoint, which for this fixture's shape/spacing/origin is
# exactly 2.0 -- the same as the flat H=2.0 surface. That ray is then an
# exact tie (world.z == h, never world.z > h), a genuine geometric
# degeneracy of this fixture/preset combination, not a hover-vs-shader
# mismatch: the same tie is equally present in the GPU-rendered image. Points
# off that exact line are unaffected and exercise the real clip behaviour.
_GRID = [(fx / 20, fy / 20) for fx in range(4, 17) for fy in range(4, 17) if fy != 10]


def test_surface_clip_bounds_hover_picks_to_below_the_surface(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-front").click()
    page.evaluate("() => window.__viewerState.idleNow()")

    meta = page.evaluate("() => window.__viewerState.meta")

    # Surface clip OFF (default): some pick over a small grid of canvas
    # points must land above the surface (world z > 2.0), so the test can
    # actually fail if the clip were (wrongly) always applied.
    assert page.locator("#toggle-surf-clip").is_checked() is False
    picks_off = _pick_grid(page, _GRID)
    zs_off = [_world_z(meta, p["k"]) for p in picks_off if p is not None]
    assert zs_off, "expected at least one pick with surface clip off"
    assert any(z > 2.0 for z in zs_off), zs_off

    # Surface clip ON: every returned pick must be at or below the surface.
    page.locator("#toggle-surf-clip").check()
    page.evaluate("() => window.__viewerState.idleNow()")
    picks_on = _pick_grid(page, _GRID)
    zs_on = [_world_z(meta, p["k"]) for p in picks_on if p is not None]
    assert zs_on, "expected at least one pick with surface clip on"
    assert all(z <= 2.0 + 1e-9 for z in zs_on), zs_on

    assert not errors, errors


def test_sigma_gate_blocks_all_picks_above_threshold(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True, layers=("volume", "sigma"))
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.evaluate("() => window.__viewerState.idleNow()")

    # Gate disabled: picks return voxels somewhere over the grid.
    picks_disabled = _pick_grid(page, _GRID)
    assert any(p is not None for p in picks_disabled)

    # Enable the gate with a threshold below every sigma value (the gate UI
    # value is a FRACTION of state.sigmaMax; 0 is below every sigma in the
    # fixture, which spans 0.1..1.0).
    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0")
    page.locator("#sigma-gate-value").dispatch_event("input")
    page.evaluate("() => window.__viewerState.idleNow()")
    picks_gated = _pick_grid(page, _GRID)
    assert all(p is None for p in picks_gated), picks_gated

    # Disable again: picks return voxels once more.
    page.locator("#sigma-gate-enabled").uncheck()
    page.evaluate("() => window.__viewerState.idleNow()")
    picks_reenabled = _pick_grid(page, _GRID)
    assert any(p is not None for p in picks_reenabled)

    assert not errors, errors
