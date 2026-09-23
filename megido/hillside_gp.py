"""Gaussian-process surface core: Matern-5/2 kernel, ML-II hyperparameters,
heteroscedastic-noise posterior. Hand-rolled on numpy + scipy (no sklearn):
at N ~ 3k the exact GP is a single Cholesky, seconds per fit.

theta parametrisation for the optimiser is log-space:
theta = (log length_scale, log signal_std, log noise_floor_fraction).
Working in logs keeps every hyperparameter positive and scales the
optimisation well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


@dataclass(frozen=True)
class GPHypers:
    length_scale: float
    signal_std: float
    noise_floor: float   # fraction of signal_std added as a homogeneous nugget
    nll: float


def _sqdist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    a2 = np.sum(X1 ** 2, axis=1)[:, None]
    b2 = np.sum(X2 ** 2, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (X1 @ X2.T), 0.0)


def matern52(X1: np.ndarray, X2: np.ndarray, length_scale: float,
             signal_var: float) -> np.ndarray:
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    d = np.sqrt(_sqdist(X1, X2))
    s = np.sqrt(5.0) * d / length_scale
    return signal_var * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def _assemble(theta):
    ls, sf, nf = np.exp(theta)
    return ls, sf, nf


def nll(theta, X, z, noise_base, mean) -> float:
    ls, sf, nf = _assemble(theta)
    n = len(z)
    K = matern52(X, X, ls, sf ** 2)
    K[np.diag_indices_from(K)] += np.asarray(noise_base) + (nf * sf) ** 2
    try:
        c, low = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e25
    r = np.asarray(z, float) - mean
    alpha = cho_solve((c, low), r)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    return float(0.5 * r @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi))


def fit_hyperparams(X, z, noise_base, mean, *, n_restarts: int = 3) -> GPHypers:
    X = np.atleast_2d(np.asarray(X, float))
    z = np.asarray(z, float)
    noise_base = np.asarray(noise_base, float)

    span = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))
    # nearest-neighbour scale as a lower bound on length_scale
    lo_ls = max(span / len(X) ** 0.5 * 0.25, 1e-3)
    hi_ls = max(span, lo_ls * 10.0)
    zstd = float(np.std(z - mean)) + 1e-9
    bounds = [(np.log(lo_ls), np.log(hi_ls)),
              (np.log(zstd * 1e-2), np.log(zstd * 1e2)),
              (np.log(1e-3), np.log(1.0))]

    inits_ls = np.geomspace(lo_ls * 1.5, hi_ls * 0.5, n_restarts)
    best = None
    for i in range(n_restarts):
        theta0 = np.log([inits_ls[i], zstd, 0.1])
        res = minimize(nll, theta0, args=(X, z, noise_base, mean),
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    ls, sf, nf = _assemble(best.x)
    return GPHypers(float(ls), float(sf), float(nf), float(best.fun))


def predict(X, z, noise_var, mean, Xstar, hypers: GPHypers):
    X = np.atleast_2d(np.asarray(X, float))
    Xstar = np.atleast_2d(np.asarray(Xstar, float))
    z = np.asarray(z, float)
    sv = hypers.signal_std ** 2
    K = matern52(X, X, hypers.length_scale, sv)
    K[np.diag_indices_from(K)] += np.asarray(noise_var, float)
    c, low = cho_factor(K, lower=True)
    alpha = cho_solve((c, low), z - mean)
    Ks = matern52(Xstar, X, hypers.length_scale, sv)      # (m, n)
    mean_star = mean + Ks @ alpha
    v = cho_solve((c, low), Ks.T)                          # (n, m)
    var_star = sv - np.sum(Ks * v.T, axis=1)
    std_star = np.sqrt(np.maximum(var_star, 0.0))
    return mean_star, std_star
