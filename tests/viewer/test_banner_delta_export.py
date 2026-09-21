"""Task 9: resolution banner, run-delta layer, PNG export.

`run_fixture` (see conftest.py) always writes to `tmp_path / "run"`, so
calling it twice in one test collides on the same directory (the second
`run.mkdir()` raises `FileExistsError`). To get two independent run
directories on disk, this test calls `run_fixture` once for run A, then
copies that directory tree to a sibling `run_b` directory for run B rather
than calling the fixture factory again.
"""
import json
import shutil

from megido.viewerbuild import build  # noqa: F401  (dist_path fixture uses this)


def test_banner_delta_and_export(page, dist_path, run_fixture, tmp_path):
    run_a = run_fixture(depth_resolved=False)
    run_b = tmp_path / "run_b"
    shutil.copytree(run_a, run_b)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_a))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    banner_text = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner_text

    page.locator("#load-second-run-input").set_input_files(str(run_b))
    page.wait_for_function("() => window.__viewerState.layerData.has('delta')")
    assert page.locator("#layer-delta").count() == 1
    assert "delta:" in page.locator("#delta-verdict").text_content()

    with page.expect_download() as dl_info:
        page.locator("#export-png-btn").click()
    download = dl_info.value
    assert download.suggested_filename == "megiddo-voxel-view.png"


def test_grid_mismatch_blocks_delta(page, dist_path, run_fixture, tmp_path):
    """Same shape, different spacing_m: viewer must refuse the diff, mirroring
    compare_volumes, which keys the grid on shape + spacing + origin.

    run_fixture always writes to tmp_path / "run", so a second distinct run
    dir (run_c) is made by copying run_a and editing spacing_m in its
    meta.json, rather than calling run_fixture again (that collides on the
    fixed "run" subdir name)."""
    run_a = run_fixture(depth_resolved=False, spacing_m=0.5)
    run_c = tmp_path / "run_c"
    shutil.copytree(run_a, run_c)
    meta_c = json.loads((run_c / "meta.json").read_text())
    meta_c["spacing_m"] = 0.75
    (run_c / "meta.json").write_text(json.dumps(meta_c))

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_a))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    page.locator("#load-second-run-input").set_input_files(str(run_c))
    page.wait_for_function(
        "() => document.querySelector('#delta-verdict').textContent.length > 0"
    )

    assert page.locator("#delta-verdict").text_content() == "grid mismatch: cannot diff"
    assert page.locator("#layer-delta").count() == 0
    assert "delta" not in page.evaluate("Array.from(window.__viewerState.layerData.keys())")
