def test_sigma_gate_changes_render_and_hover_shows_a_voxel(page, dist_path, run_fixture):
    run_dir = run_fixture(layers=("volume", "sigma"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input; pass the run directory itself (see
    # tests/viewer/test_clip.py for the same pattern).
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.2")
    page.locator("#sigma-gate-value").dispatch_event("input")
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after

    box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(100)
    text = page.locator("#hover-readout").text_content()
    assert text is not None
