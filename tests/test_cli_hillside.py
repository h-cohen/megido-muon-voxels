import json
from pathlib import Path

import pytest

from megido.cli import main

REAL_CONFIG = Path("configs/megido.yaml")
REAL_SOLVE = Path("runs/solve")


@pytest.mark.skipif(not REAL_CONFIG.exists() or not (REAL_SOLVE / "baseline.npz").exists(),
                     reason="needs the real configs/megido.yaml + runs/solve/baseline.npz")
def test_hillside_writes_silhouette_json_and_png(tmp_path):
    out = tmp_path / "voxels"
    rc = main(["hillside", "--config", str(REAL_CONFIG), "--solve", str(REAL_SOLVE),
               "--out", str(out)])
    assert rc == 0

    json_path = out / "hill_silhouette.json"
    assert json_path.exists()
    text = json_path.read_text()

    # Must be STRICT valid JSON: bare NaN/Infinity tokens (Python's default
    # json.dumps(allow_nan=True) output) are not valid JSON and JS's
    # JSON.parse throws on them. Empty ridge bins must serialize as `null`.
    assert "NaN" not in text
    assert "Infinity" not in text

    def _reject_constants(c):
        raise ValueError(f"non-finite JSON constant found: {c}")

    payload = json.loads(text, parse_constant=_reject_constants)

    assert "honesty_note" in payload
    assert "distance" in payload["honesty_note"] or "az_coverage" in payload["honesty_note"]
    assert "agreement" in payload
    assert "detectors" in payload
    assert isinstance(payload["detectors"], list)
    assert len(payload["detectors"]) >= 1

    assert "per_pos" in payload
    assert len(payload["per_pos"]) >= 1
    for pid, p in payload["per_pos"].items():
        for key in ("ridge_az", "ridge_elev", "az_coverage", "edge_sx", "edge_sy",
                    "az", "elev", "level", "n_edge"):
            assert key in p, f"{pid} missing {key}"
        assert isinstance(p["ridge_az"], list)
        assert isinstance(p["ridge_elev"], list)
        if p["az_coverage"] < 1.0:
            assert None in p["ridge_elev"], (
                f"{pid} has az_coverage<1 but no null (empty) ridge_elev bins")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out / "hill_silhouette.png").exists()


@pytest.mark.skipif(not REAL_CONFIG.exists() or not (REAL_SOLVE / "baseline.npz").exists(),
                     reason="needs the real configs/megido.yaml + runs/solve/baseline.npz")
def test_hillside_prints_summary(capsys):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rc = main(["hillside", "--config", str(REAL_CONFIG), "--solve", str(REAL_SOLVE),
                   "--out", td])
        assert rc == 0
        text = capsys.readouterr().out
        assert "az_coverage" in text
        assert "agreement" in text.lower()


def test_hillside_reports_missing_baseline(tmp_path, capsys):
    cfg = REAL_CONFIG if REAL_CONFIG.exists() else tmp_path / "site.yaml"
    if not REAL_CONFIG.exists():
        pytest.skip("needs configs/megido.yaml to exercise the argparse path minimally")
    rc = main(["hillside", "--config", str(cfg), "--solve", str(tmp_path / "nope"),
               "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "baseline.npz" in capsys.readouterr().out
