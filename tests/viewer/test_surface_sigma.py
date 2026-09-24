"""Playwright test for the hillside surface's colour modes.

Covers Task 3 (sigma) and Task 2 of the xval-and-viewer-bundle plan (residual):
the surface mesh gets a per-vertex colour attribute driven by the GP posterior
sigma or the per-node ray residual, a legend shows the active mode's range and
caveat text, a select switches between sigma/residual/flat rendering, and the
caption reports the per-ray VE alongside the per-cell VE.
"""
from __future__ import annotations


def test_sigma_colour_default_and_legend(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    assert page.locator("#hill-colour-legend").is_visible()
    assert "σ" in page.locator("#hill-colour-range").inner_text()
    assert page.evaluate("() => window.__viewerState.hillColourMode") == "sigma"
    caption_text = page.locator("#hill-surface-caveat").inner_text()
    assert "per ray" in caption_text
    assert "a=" in caption_text
    assert not errors, errors


def test_colour_mode_cycle_changes_render_each_time(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    def canvas_data():
        return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")

    sigma_render = canvas_data()

    page.locator("#hill-colour-mode").select_option("residual")
    page.wait_for_timeout(100)
    residual_render = canvas_data()
    assert residual_render != sigma_render
    assert page.evaluate("() => window.__viewerState.hillColourMode") == "residual"
    legend_text = page.locator("#hill-colour-legend").inner_text()
    assert "not separable" in legend_text

    page.locator("#hill-colour-mode").select_option("flat")
    page.wait_for_timeout(100)
    flat_render = canvas_data()
    assert flat_render != residual_render
    assert page.evaluate("() => window.__viewerState.hillColourMode") == "flat"
    assert page.locator("#hill-colour-legend").is_hidden()

    assert not errors, errors


def test_residual_option_disabled_without_residual_file(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True, hill_residual=False)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    residual_option = page.locator("#hill-colour-mode option[value='residual']")
    assert residual_option.is_disabled()
    assert not errors, errors
