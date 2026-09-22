"""Task 10: combined smoke test on the real campaign output.

This is Phase 4's exit gate: every control from Tasks 3-9 is exercised in one
page session against the real `runs/voxels` reconstruction (Phase 3's actual
output, not a synthetic fixture) with zero console errors.
"""
from pathlib import Path

REAL_RUN = Path(__file__).resolve().parents[2] / "runs" / "voxels"


def test_real_campaign_output_renders_with_every_control_operable(page, dist_path):
    assert (REAL_RUN / "meta.json").exists(), (
        "runs/voxels is Phase 3's real campaign output - regenerate it with "
        "`uv run python -m megido.cli reconstruct ...` (see docs/phase3-reconstruction-report.md) "
        "before running this gate"
    )
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # Directory-picker upload: pass the directory path, not a filtered file
    # list. The app reads files by basename from meta.layers, so the extra
    # .npz files in runs/voxels are harmless (ignored).
    page.locator("#load-run-input").set_input_files(str(REAL_RUN))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready", timeout=15000)

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    canvas_data = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    blank = page.evaluate("""
        () => { const c = document.createElement('canvas');
                c.width = document.querySelector('#gl-canvas').width;
                c.height = document.querySelector('#gl-canvas').height;
                return c.toDataURL(); }
    """)
    assert canvas_data != blank, "canvas must be non-blank on real reconstructed output"

    banner = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner, (
        "the real campaign's honest verdict must reach the viewer unedited"
    )

    def canvas_is_blank(selector):
        data = page.evaluate(f"""
            () => {{
                const src = document.querySelector('{selector}');
                const c = document.createElement('canvas');
                c.width = src.width; c.height = src.height;
                return [src.toDataURL(), c.toDataURL()];
            }}
        """)
        return data[0] == data[1]

    assert not canvas_is_blank("#gizmo-canvas"), "axis gizmo must render"
    assert not canvas_is_blank("#legend-canvas"), "colorbar legend must render"

    # Switch colormap.
    page.locator("#colormap-select").select_option("inferno")

    # Collapse and re-expand a dock section.
    clip_header = page.locator('[data-section="sec-clip"] .section-header')
    clip_header.click()
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "false"
    clip_header.click()
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "true"

    # backprojection.npy is a 2D (58, 50) diagnostic plane, not a volume grid
    # layer (nx*ny*nz elements); loadRun correctly skips it, so it is not
    # asserted here.
    for key in ("volume", "sigma", "snr", "views", "systematic"):
        assert page.locator(f"#layer-{key}").count() == 1, f"missing layer control for {key}"
        page.locator(f"#layer-{key}").check()

    page.locator("#camera-preset-top").click()
    page.locator("#camera-preset-iso").click()

    box = page.locator("#xfer-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
    page.mouse.up()

    hist_box = page.locator("#histogram-canvas").bounding_box()
    page.mouse.move(hist_box["x"] + hist_box["width"] * 0.15, hist_box["y"] + hist_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(hist_box["x"] + hist_box["width"] * 0.35, hist_box["y"] + hist_box["height"] / 2)
    page.mouse.up()
    readout = page.locator("#window-readout").text_content()
    assert readout and readout.strip(), "window readout must reflect the dragged window"

    page.locator("#clip-x-max").fill("0.6")
    page.locator("#clip-x-max").dispatch_event("input")
    page.locator("#clip-plane-enabled").check()
    page.locator("#slice-axis").select_option("z")
    page.locator("#slice-pos").fill("0.5")
    page.locator("#slice-pos").dispatch_event("input")

    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.5")
    page.locator("#sigma-gate-value").dispatch_event("input")

    canvas_box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(canvas_box["x"] + canvas_box["width"] / 2,
                     canvas_box["y"] + canvas_box["height"] / 2)
    page.wait_for_timeout(100)

    # Shortcuts overlay: open with "?", close with Escape. Move focus off the
    # last-touched form control first - the app ignores shortcut keys while
    # an input/select/textarea has focus.
    page.locator("#gl-canvas").click()
    page.keyboard.press("Shift+Slash")  # '?'
    assert page.locator("#overlay-shortcuts").is_visible(), "? must open the shortcuts cheatsheet"
    page.keyboard.press("Escape")
    assert page.locator("#overlay-shortcuts").is_hidden(), "Escape must close the shortcuts cheatsheet"

    # Keyboard layer switch (digit keys).
    page.keyboard.press("2")

    # Save the current view, then apply it back.
    page.locator("#view-name").fill("smoke-test-view")
    page.locator("#save-view-btn").click()
    saved_item = page.locator("#saved-views li", has_text="smoke-test-view")
    assert saved_item.count() == 1, "saved view must appear in the saved-views list"
    saved_item.get_by_role("button", name="Apply").click()

    with page.expect_download():
        page.locator("#export-png-btn").click()

    # Task S6: hillside silhouette fan. hill_silhouette.json is present in
    # the real runs/voxels output (written by `megido.cli hillside`), so the
    # toggle must be enabled and change the render when checked.
    silhouette_toggle = page.locator("#toggle-silhouette")
    assert silhouette_toggle.count() == 1, "missing #toggle-silhouette control"
    assert silhouette_toggle.is_enabled(), (
        "runs/voxels has hill_silhouette.json - #toggle-silhouette must be enabled"
    )

    def gl_canvas_data():
        return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")

    before_silhouette = gl_canvas_data()
    silhouette_toggle.check()
    page.wait_for_timeout(50)
    after_silhouette = gl_canvas_data()
    assert after_silhouette != before_silhouette, (
        "enabling #toggle-silhouette must change the rendered canvas (ridgeline fan drawn)"
    )
    silhouette_toggle.uncheck()
    page.wait_for_timeout(50)
    assert gl_canvas_data() == before_silhouette, (
        "disabling #toggle-silhouette must remove the ridgeline fan again"
    )

    # Phase 5b Task 4: hillside surface mesh (the primary hillside display).
    # hill_surface.npy + hill_surface_meta.json are present in the real
    # runs/voxels output (written by `megido.cli hillside`), so the toggle
    # must be enabled.
    hill_surface_toggle = page.locator("#toggle-hill-surface")
    assert hill_surface_toggle.count() == 1, "missing #toggle-hill-surface control"
    assert hill_surface_toggle.is_enabled(), (
        "runs/voxels has hill_surface.npy - #toggle-hill-surface must be enabled"
    )

    before_hill_surface = gl_canvas_data()
    hill_surface_toggle.uncheck()
    page.wait_for_timeout(50)
    after_uncheck = gl_canvas_data()
    assert after_uncheck != before_hill_surface, (
        "unchecking #toggle-hill-surface must change the rendered canvas "
        "(it defaults to checked when the artifact is present)"
    )
    hill_surface_toggle.check()
    page.wait_for_timeout(50)
    assert gl_canvas_data() == before_hill_surface, (
        "re-checking #toggle-hill-surface must restore the rendered surface"
    )

    # Display-only smoothing slider for the hillside surface: with the
    # surface toggled on, dragging it to a nonzero value must re-render the
    # (visibly smoother) mesh, and dragging back to 0 must restore the raw
    # fitted surface exactly.
    hill_surface_smooth = page.locator("#hill-surface-smooth")
    assert hill_surface_smooth.count() == 1, "missing #hill-surface-smooth control"
    assert hill_surface_smooth.is_enabled(), (
        "runs/voxels has hill_surface.npy - #hill-surface-smooth must be enabled"
    )
    raw_hill_surface = gl_canvas_data()
    hill_surface_smooth.fill("8")
    hill_surface_smooth.dispatch_event("input")
    page.wait_for_timeout(50)
    smoothed_hill_surface = gl_canvas_data()
    assert smoothed_hill_surface != raw_hill_surface, (
        "moving #hill-surface-smooth to a nonzero value must change the rendered mesh"
    )
    hill_surface_smooth.fill("0")
    hill_surface_smooth.dispatch_event("input")
    page.wait_for_timeout(50)
    assert gl_canvas_data() == raw_hill_surface, (
        "returning #hill-surface-smooth to 0 must restore the raw fitted surface"
    )

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"
