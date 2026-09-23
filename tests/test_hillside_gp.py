import numpy as np
from megido.hillside_gp import matern52, nll, fit_hyperparams, predict, GPHypers


def test_matern52_psd_and_peak():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, (40, 2))
    K = matern52(X, X, length_scale=2.0, signal_var=1.5)
    assert np.allclose(K, K.T, atol=1e-12)              # symmetric
    assert np.allclose(np.diag(K), 1.5, atol=1e-9)      # k(0)=signal_var
    np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))       # PSD (raises if not)
    # decreasing with distance
    k_near = matern52(np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]]), 2.0, 1.5)[0, 0]
    k_far = matern52(np.array([[0.0, 0.0]]), np.array([[4.0, 0.0]]), 2.0, 1.5)[0, 0]
    assert k_near > k_far


def test_ml2_recovers_length_scale():
    # Data sampled from a smooth function with a known length-scale; ML-II
    # should land near it, NOT return a fixed guess.
    rng = np.random.default_rng(1)
    X = rng.uniform(-10, 10, (200, 2))
    true_ls = 3.0
    # draw a smooth field from a Matern prior at true_ls
    K = matern52(X, X, true_ls, 1.0) + 1e-8 * np.eye(len(X))
    z = np.linalg.cholesky(K) @ rng.standard_normal(len(X))
    noise_base = np.full(len(X), 1e-4)
    h = fit_hyperparams(X, z, noise_base, mean=0.0, n_restarts=3)
    assert 0.6 * true_ls < h.length_scale < 1.7 * true_ls, h.length_scale


def test_nll_matches_bruteforce():
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, (12, 2))
    z = rng.standard_normal(12)
    noise_base = np.full(12, 0.05)
    theta = np.log([1.5, 0.8, 0.1])      # ls, signal_std, noise_floor
    got = nll(theta, X, z, noise_base, mean=0.2)
    ls, sf, nf = np.exp(theta)
    K = matern52(X, X, ls, sf ** 2)
    K[np.diag_indices_from(K)] += noise_base + (nf * sf) ** 2
    r = z - 0.2
    sign, logdet = np.linalg.slogdet(K)
    ref = 0.5 * r @ np.linalg.solve(K, r) + 0.5 * logdet + 0.5 * len(z) * np.log(2 * np.pi)
    assert sign > 0
    assert abs(got - ref) < 1e-8


def test_predict_std_grows_away_from_data():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z = np.array([0.0, 0.0, 0.0, 0.0])
    noise = np.full(4, 1e-6)
    h = GPHypers(length_scale=0.5, signal_std=1.0, noise_floor=0.01, nll=0.0)
    near, std_near = predict(X, z, noise, 0.0, np.array([[0.5, 0.5]]), h)
    far, std_far = predict(X, z, noise, 0.0, np.array([[20.0, 20.0]]), h)
    assert std_far[0] > std_near[0]
    assert abs(std_far[0] - 1.0) < 0.05          # -> prior std (signal_std) far away


def test_predict_interpolates_noise_free_data():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
    z = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
    noise = np.full(len(X), 1e-8)
    h = GPHypers(length_scale=1.5, signal_std=2.0, noise_floor=1e-4, nll=0.0)
    m, _ = predict(X, z, noise, float(z.mean()), X, h)
    assert np.allclose(m, z, atol=1e-2)          # near-interpolation at training points
