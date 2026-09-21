def test_layer_panel_lists_every_layer_and_switching_changes_render(page, dist_path, run_fixture):
    run_dir = run_fixture(layers=("volume", "sigma", "views"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input; pass the run directory itself (see
    # tests/viewer/test_camera.py / test_transfer_ui.py for the same pattern).
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    for key in ("volume", "sigma", "views"):
        assert page.locator(f"#layer-{key}").count() == 1

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#layer-views").check()
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
