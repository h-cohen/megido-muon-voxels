"""Playwright test for the hillside surface's sigma-coloured render mode.

Covers Task 3 of the surface-check-and-viewer-upgrades plan: the surface mesh
gets a per-vertex colour attribute driven by the GP posterior sigma, a legend
shows the robust sigma range, a toggle switches between sigma-coloured and
flat-amber rendering, and the caption reports the per-ray VE alongside the
per-cell VE.
"""
from __future__ import annotations


def test_sigma_colour_toggle_changes_render_and_shows_legend(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    assert page.locator("#hill-sigma-legend").is_visible()
    assert "σ" in page.locator("#hill-sigma-range").inner_text()
    assert page.evaluate("() => window.__viewerState.hillSigmaColor") is True
    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#toggle-hill-sigma").uncheck()
    page.wait_for_timeout(200)
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
    caption_text = page.locator("#hill-surface-caveat").inner_text()
    assert "per ray" in caption_text
    assert "a=" in caption_text
    assert not errors, errors
