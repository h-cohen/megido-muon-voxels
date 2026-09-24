def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")


def _png(page):
    return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")


def test_smooth_and_shading_toggles_each_change_the_render(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    st = page.evaluate("() => ({s: window.__viewerState.smoothSampling, h: window.__viewerState.shading})")
    assert st == {"s": True, "h": True}
    base = _png(page)
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)
    no_shade = _png(page)
    assert no_shade != base
    page.locator("#toggle-smooth").uncheck(); page.wait_for_timeout(150)
    nearest = _png(page)
    assert nearest != no_shade
    assert not errors, errors


_SNAP = """(key) => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  window.__snaps = window.__snaps || {};
  window.__snaps[key] = ctx.getImageData(0, 0, c.width, c.height).data;
}"""

_DIFF = """([a, b]) => {
  const x = window.__snaps[a], y = window.__snaps[b];
  if (x.length !== y.length) throw new Error('size mismatch');
  let s = 0;
  for (let k = 0; k < x.length; k++) s += Math.abs(x[k] - y[k]);
  return s / x.length;
}"""


def test_manual_trilinear_fallback_agrees_with_hardware_linear(page, dist_path, run_fixture):
    """The manual 8-tap texelFetch path only runs on GPUs without
    OES_texture_float_linear, so headless never exercises it by default. Force
    it and require it to render like hardware LINEAR (same maths) and unlike
    NEAREST -- a transposed or half-texel-shifted fallback would fail this."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    if not page.evaluate("() => window.__viewerState.floatLinear"):
        import pytest
        pytest.skip("no OES_texture_float_linear: hardware reference unavailable")
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)
    page.evaluate(_SNAP, "hw")
    page.evaluate("""() => { const s = window.__viewerState;
        s.floatLinear = false; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "manual")
    page.evaluate("""() => { const s = window.__viewerState;
        s.smoothSampling = false; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "nearest")
    d_manual = page.evaluate(_DIFF, ["hw", "manual"])
    d_nearest = page.evaluate(_DIFF, ["hw", "nearest"])
    print(f"\nmean |Δ| per channel: manual vs hw={d_manual:.3f}, nearest vs hw={d_nearest:.3f}")
    assert d_nearest > 1.0, d_nearest                 # the comparison has teeth
    assert d_manual < 0.25 * d_nearest, (d_manual, d_nearest)
    assert not errors, errors


_COUNT_LIT = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const bg = [d[0], d[1], d[2]];
  let n = 0;
  for (let k = 0; k < d.length; k += 4) {
    if (Math.abs(d[k]-bg[0]) + Math.abs(d[k+1]-bg[1]) + Math.abs(d[k+2]-bg[2]) > 24) n++;
  }
  return n;
}"""


def test_opacity_correction_invariance_across_step_length(page, dist_path, run_fixture):
    """The march loop's opacity correction (`1-(1-a)^(stepLen/uRefStep)`) is
    what makes the calibrated look independent of how finely the box is
    sampled. `state.minStepVoxels` (uMinStepVoxels) is a test-only hook to
    force a finer minimum step than the shipped 0.5-voxel default without
    changing uSteps. A missing/wrong exponent would make the image visibly
    denser or sparser as the step shrinks -- this pins that against a
    same-test NEAREST-vs-LINEAR anchor so the tolerance isn't a magic number."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)

    page.evaluate("""() => { const s = window.__viewerState;
        s.minStepVoxels = 0.5; s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "step_5")
    page.evaluate("""() => { const s = window.__viewerState;
        s.minStepVoxels = 0.25; s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "step_25")
    d_step = page.evaluate(_DIFF, ["step_5", "step_25"])

    # Anchor: the same two snapshots' worth of scale, but for a difference we
    # KNOW is large -- NEAREST vs smooth (hardware LINEAR or the manual
    # trilinear fallback) sampling, same camera, same minStepVoxels.
    page.evaluate("""() => { const s = window.__viewerState;
        s.minStepVoxels = 0.5; s.smoothSampling = false; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "nearest")
    page.evaluate("""() => { const s = window.__viewerState;
        s.smoothSampling = true; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    page.evaluate(_SNAP, "linear")
    d_filter = page.evaluate(_DIFF, ["nearest", "linear"])

    print(f"\nmean |Δ| per channel: step 0.5-vs-0.25={d_step:.3f}, nearest-vs-linear={d_filter:.3f}")
    assert d_step < 1.5, d_step
    assert d_step < 0.25 * d_filter, (d_step, d_filter)
    assert not errors, errors


def test_camera_inside_box_renders_non_blank(page, dist_path, run_fixture):
    """tEnter is clamped to >= 0 specifically so a camera placed INSIDE the
    volume box still marches from the eye, not from a (negative) box-entry
    point behind it. A regression here renders a blank frame."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    page.evaluate("""() => { const s = window.__viewerState;
        const m = s.meta;
        s.camera.target = [
          m.origin_m[0] + 0.5 * m.shape[0] * m.spacing_m,
          m.origin_m[1] + 0.5 * m.shape[1] * m.spacing_m,
          m.origin_m[2] + 0.5 * m.shape[2] * m.spacing_m,
        ];
        s.camera.distance = 1e-3;
        s.render();
    }""")
    page.wait_for_timeout(150)
    lit = page.evaluate(_COUNT_LIT)
    print(f"\ncamera-inside-box lit pixels: {lit}")
    assert lit > 0, lit
    assert not errors, errors


def test_axis_preset_robust_to_tiny_camera_perturbation(page, dist_path, run_fixture):
    """Top/Front/Side are exactly axis-aligned, which is exactly where a ray
    direction component can be 0 and the unguarded `1.0/dir` slab division is
    undefined per spec (review finding L3). A perturbation of +1e-4 rad in
    yaw and pitch must not change how much of the volume renders."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    counts = {}
    for preset in ("top", "front", "side"):
        page.locator(f"#camera-preset-{preset}").click()
        page.wait_for_timeout(150)
        lit_axis = page.evaluate(_COUNT_LIT)
        page.evaluate("""() => { const s = window.__viewerState;
            s.camera.yaw += 1e-4; s.camera.pitch += 1e-4; s.render(); }""")
        page.wait_for_timeout(150)
        lit_perturbed = page.evaluate(_COUNT_LIT)
        counts[preset] = (lit_axis, lit_perturbed)
        assert lit_axis > 0, (preset, lit_axis)
        rel_diff = abs(lit_perturbed - lit_axis) / lit_axis
        assert rel_diff < 0.01, (preset, lit_axis, lit_perturbed, rel_diff)
    print(f"\naxis-aligned vs perturbed lit-pixel counts: {counts}")
    assert not errors, errors
