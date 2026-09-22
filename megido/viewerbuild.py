"""Concatenates viewer/src/*.mjs into the single-file HTML deliverable.

Node's `node:test` imports these same .mjs files directly (unit tests run
against real ES module semantics); this module strips the export/import
syntax those files need for that and inlines them as one classic <script>
so the shipped page has zero network requests and works from file://.
"""
from __future__ import annotations

import re
from pathlib import Path

# Dependency order: a module may only import from a module earlier in this list.
_MODULE_ORDER = [
    "npy.mjs",
    "mat4.mjs",
    "grid.mjs",
    "transfer.mjs",
    "histogram.mjs",
    "clip.mjs",
    "camera.mjs",
    "layers.mjs",
    "delta.mjs",
    "dock.mjs",
    "colormap.mjs",
    "views.mjs",
    "shortcuts.mjs",
    "markers.mjs",
    "surfacemesh.mjs",
    "app.mjs",
]

_EXPORT_RE = re.compile(r"^export (default )?")
_IMPORT_RE = re.compile(r"^\s*import\s*\{[^}]*\}\s*from\s*['\"][^'\"]+['\"];?\s*$")


def _strip_module_syntax(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if _IMPORT_RE.match(line):
            continue
        out_lines.append(_EXPORT_RE.sub("", line))
    return "\n".join(out_lines)


def build(viewer_dir: Path = Path("viewer"), *, out_dir: Path | None = None) -> Path:
    viewer_dir = Path(viewer_dir)
    src_dir = viewer_dir / "src"
    shell = (viewer_dir / "shell.html").read_text()

    chunks = []
    for name in _MODULE_ORDER:
        path = src_dir / name
        if not path.exists():
            continue
        chunks.append(f"// ---- {name} ----")
        chunks.append(_strip_module_syntax(path.read_text()))
    chunks.append("initViewer(document.body);")
    bundle = "\n".join(chunks)

    page = shell.replace("/* __VIEWER_BUNDLE__ */", bundle)

    out_dir = Path(out_dir) if out_dir is not None else viewer_dir / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(page)
    return out_path
