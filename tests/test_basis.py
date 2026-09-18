import numpy as np
import pytest

from megido.basis import SmoothBasis, make_smooth_basis


def test_basis_size_is_n_per_axis_squared():
    b = make_smooth_basis(n_per_axis=8)
    assert b.n_coeff == 64
    assert b.centers.shape == (64, 2)


def test_design_matrix_shape():
    b = make_smooth_basis(n_per_axis=4)
    tx = np.linspace(-1, 1, 30)
    ty = np.zeros(30)
    assert b.design(tx, ty).shape == (30, 16)


def test_zero_coefficients_give_a_flat_field():
    b = make_smooth_basis(n_per_axis=4)
    tx, ty = np.linspace(-1, 1, 20), np.zeros(20)
    assert np.allclose(b.evaluate(np.zeros(b.n_coeff), tx, ty), 0.0)


def test_a_single_coefficient_bumps_its_own_neighbourhood_most():
    b = make_smooth_basis(t_max=1.0, n_per_axis=4)
    c = np.zeros(b.n_coeff)
    c[0] = 1.0
    at_centre = b.evaluate(c, b.centers[0, 0:1], b.centers[0, 1:2])[0]
    far = b.evaluate(c, np.array([b.centers[-1, 0]]), np.array([b.centers[-1, 1]]))[0]
    assert at_centre > 0.5
    assert far < 0.05 * at_centre


def test_the_field_is_smooth():
    """Neighbouring evaluation points must not differ wildly — that is the point."""
    b = make_smooth_basis(t_max=1.0, n_per_axis=6)
    rng = np.random.default_rng(0)
    c = rng.normal(0.0, 1.0, b.n_coeff)
    t = np.linspace(-1, 1, 400)
    f = b.evaluate(c, t, np.zeros_like(t))
    steps = np.abs(np.diff(f))
    assert steps.max() < 0.2 * (f.max() - f.min() + 1e-12)


def test_evaluate_equals_design_times_coefficients():
    b = make_smooth_basis(n_per_axis=4)
    rng = np.random.default_rng(1)
    c = rng.normal(size=b.n_coeff)
    tx, ty = rng.uniform(-1, 1, 15), rng.uniform(-1, 1, 15)
    assert np.allclose(b.evaluate(c, tx, ty), b.design(tx, ty) @ c)


def test_sigma_scales_with_centre_spacing():
    """Too small a sigma leaves gaps between basis functions; too large and the
    basis cannot represent anything but a constant."""
    b = make_smooth_basis(t_max=1.0, n_per_axis=5)
    spacing = 2.0 / 4
    assert 0.5 * spacing < b.sigma < 1.5 * spacing


def test_evaluate_accepts_two_dimensional_input():
    b = make_smooth_basis(n_per_axis=4)
    tx, ty = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 5), indexing="ij")
    out = b.evaluate(np.zeros(b.n_coeff), tx, ty)
    assert out.shape == (7, 5)
