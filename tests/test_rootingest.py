"""Pre-binned ROOT ingest (megido.rootingest) and the site-config pieces it needs."""
from dataclasses import replace

import numpy as np
import pytest
import uproot

from megido.anghist import load_counts
from megido.config import Binning, load_site_config
from megido.detector import DetectorGeometry
from megido.rootingest import ingest_root, read_root_counts

CFG = "configs/cafeteria.yaml"


def _write_th2(path, values, edges):
    with uproot.recreate(path) as f:
        f["txty"] = (values.astype(np.float32), edges, edges)


def test_crop_keeps_each_count_at_its_own_tangent(tmp_path):
    edges = np.linspace(-2, 2, 801)
    v = np.zeros((800, 800))
    v[150, 649] = 7          # tx bin [-1.25, -1.245), ty bin [1.245, 1.25)
    v[0, 0] = 99             # outside +-1.25: dropped
    _write_th2(tmp_path / "a.root", v, edges)
    target = Binning(t_max=1.25, n_bins=500).edges()
    h, total = read_root_counts(tmp_path / "a.root", "txty", target)
    assert h.values.shape == (500, 500)
    assert h.values[0, 499] == 7
    assert h.total == 7 and total == 106


def test_incompatible_binning_is_an_error_not_a_resample(tmp_path):
    edges = np.linspace(-2, 2, 801)
    _write_th2(tmp_path / "a.root", np.ones((800, 800)), edges)
    with pytest.raises(ValueError, match="common edge set"):
        read_root_counts(tmp_path / "a.root", "txty", np.linspace(-1.25, 1.25, 301))


def test_ingest_writes_exposures_and_sky_reference(tmp_path):
    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    cfg = load_site_config(CFG)
    for i, name in enumerate([e.root_file for e in cfg.exposures] + [cfg.sky_reference.root_file]):
        _write_th2(data / name, np.full((800, 800), i + 1.0), edges)
    cfg = replace(cfg, data_dir=data)
    res = ingest_root(cfg, tmp_path / "out")
    assert [r.source_id for r in res] == ["pos0", "pos1", "SKY"]
    sky = load_counts(tmp_path / "out" / "counts_SKY.npz")
    assert sky.values.shape == (500, 500) and (sky.values == 3).all()


def test_cafeteria_config_loads_root_exposures_and_sky():
    cfg = load_site_config(CFG)
    assert [e.id for e in cfg.exposures] == ["pos0", "pos1"]
    assert all(e.root_file and e.run_ids == () for e in cfg.exposures)
    assert cfg.sky_reference.id == "SKY"
    assert cfg.files_for("pos0") == [] or cfg.files_for("pos0")[0].suffix == ".root"


def test_detector_override_sets_aperture_and_acceptance_edge():
    cfg = load_site_config(CFG)
    g = DetectorGeometry.for_site(cfg)
    assert g.aperture_m == pytest.approx(0.35375)
    assert g.max_tan() == pytest.approx(35.375 / 38.9)
    assert g.dz_cm("x") == pytest.approx(38.9) and g.dz_cm("y") == pytest.approx(38.9)


def test_no_override_is_exactly_the_megiddo_unit():
    cfg = load_site_config("configs/megido.yaml")
    assert cfg.detector is None and cfg.sky_reference is None
    assert DetectorGeometry.for_site(cfg) == DetectorGeometry.megiddo()
