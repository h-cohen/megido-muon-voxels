import numpy as np

from megido.hillside_surface import (
    bilinear_matrix,
    laplacian_matrix,
    solve_surface,
    variance_explained,
)


def test_bilinear_matrix_partition_of_unity():
    gx = np.linspace(0, 10, 11)
    gy = np.linspace(0, 10, 11)
    B, support = bilinear_matrix(np.array([[2.5, 3.5], [0.0, 0.0]]), gx, gy)
    assert abs(B.sum(axis=1)[0, 0] - 1.0) < 1e-9  # weights sum to 1
    assert B.shape[1] == 121


def test_solve_recovers_a_plane():
    gx = np.linspace(0, 10, 11)
    gy = np.linspace(0, 10, 11)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Htrue = (2 * GX + 3 * GY + 1).ravel()
    # sample the plane at random points
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.5, 9.5, (500, 2))
    z = 2 * xy[:, 0] + 3 * xy[:, 1] + 1
    B, support = bilinear_matrix(xy, gx, gy)
    lap = laplacian_matrix(11, 11)
    h = solve_surface(B, z, np.ones(len(z)), lap, mu=0.01)
    m = support > 0.5
    assert np.corrcoef(h[m], Htrue[m])[0, 1] > 0.999  # plane recovered where supported
    assert variance_explained(B, z, np.ones(len(z)), h) > 0.99


def test_laplacian_penalises_curvature_not_planes():
    lap = laplacian_matrix(6, 6)
    GX, GY = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    plane = (3 * GX + 2 * GY).ravel().astype(float)
    assert np.allclose(lap @ plane, 0, atol=1e-9)  # a plane has zero 2nd-difference (interior)
