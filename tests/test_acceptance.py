import numpy as np
import pytest

from megido.acceptance import geometric_acceptance
from megido.detector import DetectorGeometry


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def test_peaks_on_axis(geom):
    a = geometric_acceptance(np.array([0.0, 0.3, 0.6]), np.zeros(3), geom)
    assert a[0] > a[1] > a[2] > 0


def test_vanishes_beyond_the_geometric_limit(geom):
    limit = geom.max_tan()
    beyond = geometric_acceptance(np.array([limit * 1.01]), np.array([0.0]), geom)
    assert beyond[0] == 0.0
    inside = geometric_acceptance(np.array([limit * 0.99]), np.array([0.0]), geom)
    assert inside[0] > 0.0


def test_limit_matches_active_width_over_layer_separation(geom):
    """The zero crossing IS width/dz — the number Phase 1's gate confirmed."""
    t = np.linspace(0.0, 1.5, 3001)
    a = geometric_acceptance(t, np.zeros_like(t), geom)
    last_nonzero = t[np.flatnonzero(a > 0)[-1]]
    assert last_nonzero == pytest.approx(geom.bar.active_width_cm / geom.dz_cm("x"), abs=2e-3)
    assert last_nonzero == pytest.approx(1.219, abs=2e-3)


def test_symmetric_in_both_coordinates(geom):
    tx = np.array([0.4, -0.4, 0.4, -0.4])
    ty = np.array([0.2, 0.2, -0.2, -0.2])
    a = geometric_acceptance(tx, ty, geom)
    assert np.allclose(a, a[0])


def test_separable_triangles_times_the_solid_angle_factor(geom):
    """Check one value against the closed form, computed independently."""
    w = geom.bar.active_width_cm
    dz = geom.dz_cm("x")
    tx, ty = 0.5, -0.25
    expected = (w - 0.5 * dz) * (w - 0.25 * dz) * (1 + 0.25 + 0.0625) ** -1.5
    got = geometric_acceptance(np.array([tx]), np.array([ty]), geom)[0]
    assert got == pytest.approx(expected)


def test_shape_is_preserved(geom):
    tx, ty = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 5), indexing="ij")
    assert geometric_acceptance(tx, ty, geom).shape == (7, 5)


def test_solid_angle_factor_actually_applied(geom):
    """Without the (1+t^2)^-3/2 factor the profile would be a pure triangle."""
    w, dz = geom.bar.active_width_cm, geom.dz_cm("x")
    t = 0.8
    triangle_only = (w - t * dz) * w
    got = geometric_acceptance(np.array([t]), np.array([0.0]), geom)[0]
    assert got < 0.8 * triangle_only
