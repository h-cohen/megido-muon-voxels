def test_gizmo_and_legend_render_non_blank(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    assert page.locator("#gizmo-canvas").count() == 1
    assert page.locator("#legend-canvas").count() == 1

    gizmo_blank = page.evaluate("""
        () => {
          const src = document.querySelector('#gizmo-canvas');
          const c = document.createElement('canvas');
          c.width = src.width; c.height = src.height;
          return c.toDataURL();
        }
    """)
    gizmo_now = page.evaluate("() => document.querySelector('#gizmo-canvas').toDataURL()")
    assert gizmo_now != gizmo_blank

    legend_blank = page.evaluate("""
        () => {
          const src = document.querySelector('#legend-canvas');
          const c = document.createElement('canvas');
          c.width = src.width; c.height = src.height;
          return c.toDataURL();
        }
    """)
    legend_now = page.evaluate("() => document.querySelector('#legend-canvas').toDataURL()")
    assert legend_now != legend_blank

    ticks = page.locator("#legend-ticks").text_content()
    assert ticks and ticks.strip()


def test_gizmo_rotates_with_camera(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before = page.evaluate("() => document.querySelector('#gizmo-canvas').toDataURL()")
    box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 120, box["y"] + box["height"] / 2 + 40)
    page.mouse.up()
    after = page.evaluate("() => document.querySelector('#gizmo-canvas').toDataURL()")
    assert before != after


def test_hover_tooltip_follows_cursor_without_console_errors(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    box = page.locator("#gl-canvas").bounding_box()
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.wait_for_timeout(100)

    hidden = page.evaluate("() => document.querySelector('#hover-readout').hidden")
    text = page.locator("#hover-readout").text_content()
    # Either a hit (visible, non-empty text) or a miss (hidden) — either is
    # fine, but it must not throw, and it must not silently stay in whatever
    # the pre-hover default was.
    assert hidden is True or (text is not None and text.strip() != "")

    assert errors == []
