from pathlib import Path

from megido.viewerbuild import build


def test_build_produces_single_html_with_no_export_or_import(tmp_path):
    viewer_dir = Path(__file__).resolve().parents[1] / "viewer"
    out = build(viewer_dir, out_dir=tmp_path)
    assert out.name == "index.html"
    text = out.read_text()
    assert "<!DOCTYPE html>" in text or "<!doctype html>" in text
    assert "export function" not in text
    assert "export const" not in text
    assert "import {" not in text
    assert "parseNpy" in text
    # Task 3 adds app.mjs (defining initViewer) and re-adds this assertion:
    # assert 'initViewer(document.body)' in text
