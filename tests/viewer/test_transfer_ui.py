def test_dragging_a_transfer_stop_changes_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input; pass the run directory itself (see
    # tests/viewer/test_raymarch.py / test_camera.py for the same pattern).
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    box = page.locator("#xfer-canvas").bounding_box()
    mid_x = box["x"] + box["width"] * 0.5
    mid_y = box["y"] + box["height"] * 0.5
    page.mouse.move(mid_x, mid_y)
    page.mouse.down()
    page.mouse.move(mid_x + box["width"] * 0.3, mid_y)
    page.mouse.up()
    page.evaluate("() => window.__viewerState.idleNow()")  # settle preview before comparing
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after

    hist_blank = page.evaluate("""
        () => {
          const c = document.createElement('canvas');
          c.width = document.querySelector('#histogram-canvas').width;
          c.height = document.querySelector('#histogram-canvas').height;
          return c.toDataURL();
        }
    """)
    hist_now = page.evaluate("() => document.querySelector('#histogram-canvas').toDataURL()")
    assert hist_now != hist_blank


def test_dragging_the_histogram_band_edge_changes_the_window(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before_render = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    box = page.locator("#histogram-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.15, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.35, box["y"] + box["height"] / 2)
    page.mouse.up()
    page.evaluate("() => window.__viewerState.idleNow()")  # settle preview before comparing
    after_render = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before_render != after_render

    readout = page.locator("#window-readout").text_content()
    assert readout and readout.strip(), "window readout must be populated after a drag"
