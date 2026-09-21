def test_save_apply_delete_view(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    # Put the camera in a known state via a preset, then save it under a name.
    page.locator("#camera-preset-top").click()
    saved_yaw = page.evaluate("() => window.__viewerState.camera.yaw")

    page.locator("#view-name").fill("my view")
    page.locator("#save-view-btn").click()

    items = page.locator("#saved-views li")
    assert items.count() == 1

    # Rotate the camera away from the saved state.
    canvas = page.locator("#gl-canvas")
    box = canvas.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 120, cy + 40)
    page.mouse.up()

    rotated_yaw = page.evaluate("() => window.__viewerState.camera.yaw")
    assert abs(rotated_yaw - saved_yaw) > 1e-6

    # Applying the saved view restores the camera.
    items.locator("button", has_text="Apply").click()
    restored_yaw = page.evaluate("() => window.__viewerState.camera.yaw")
    assert abs(restored_yaw - saved_yaw) < 1e-6

    # Deleting empties the list.
    items.locator("button", has_text="Delete").click()
    assert page.locator("#saved-views li").count() == 0
