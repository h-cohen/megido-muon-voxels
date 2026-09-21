def test_colormap_switch_changes_render(page, dist_path, run_fixture):
    run = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#colormap-select").select_option("inferno")
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
    assert page.evaluate("() => window.__viewerState.transferStops[4].r") is not None
