"""Joint Poisson solve for the detector baseline and per-position opacity.

There is no open-sky calibration run for this campaign, so the detector's
angular response cannot be measured separately and must be estimated alongside
the scene. What makes that possible is the campaign geometry: P0, T20a and T20b
sit at ONE position with two different tilts, so rotating the detector sends
each bin to a different patch of sky while the response stays put.

Model, for exposure e at position p(e):

    mu_e(d) = norm_e * A_geom(d) * exp(S(d)) * Phi_n(R_e.d) * exp(-lambda_p(R_e.d))

Opacity is indexed by SKY direction and by POSITION. It is not shared across
positions: P0 and P1 are 2.2 m apart and look through different rock for the
same sky direction, and that difference is precisely the parallax Phase 3 needs.

Three of the four blocks have closed-form Poisson updates; only the smooth
coefficients need numerical optimisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.basis import SmoothBasis, make_smooth_basis
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid

# A sky bin seen by only a handful of tracks cannot constrain opacity. Modelling
# it anyway forces a choice between inventing counts and asserting lambda = 0,
# and both distort the fit. The set is computed once from raw counts, so it is
# fixed across iterations and the likelihood stays comparable between them.
MIN_SKY_COUNTS = 5.0
_EPS = 1e-12


def position_ids(cfg: SiteConfig) -> dict[str, str]:
    """Map each exposure id to a short position id, assigned in config order.

    Exposures sharing a position share an opacity map. Position is the
    TRANSLATION only — two tilts at the same spot are the same position, which
    is exactly what makes P0/T20a/T20b able to separate detector from sky.
    """
    seen: dict[tuple[float, float, float], str] = {}
    out: dict[str, str] = {}
    for exp in cfg.exposures:
        key = (round(exp.pose.x, 6), round(exp.pose.y, 6), round(exp.pose.z, 6))
        if key not in seen:
            seen[key] = f"pos{len(seen)}"
        out[exp.id] = seen[key]
    return out


@dataclass(frozen=True)
class _ExposureTerms:
    """Everything about one exposure that does not change during the solve."""
    exposure_id: str
    norm_group: str
    position: str
    counts: np.ndarray        # [n_live] observed, flattened over live bins
    acceptance: np.ndarray    # [n_live] A_geom
    design: np.ndarray        # [n_live, n_coeff]
    sky_flat: np.ndarray      # [n_live] flat sky-bin index
    sky_tan: np.ndarray       # [n_live, 2] sky tangents, for the flux shape
    live: np.ndarray          # [m, m] bool, which analysis bins are used


@dataclass(frozen=True)
class BaselineSolution:
    coeffs: np.ndarray
    opacity: dict[str, np.ndarray]
    norms: dict[str, float]
    flux_index: float
    nll_history: list[float]
    grid: AnalysisGrid
    sky: SkyGrid
    basis: SmoothBasis
    terms: dict[str, _ExposureTerms] = field(default_factory=dict, repr=False)

    def normalized_opacity(self, position_id_: str,
                           transparent_quantile: float = 0.05) -> np.ndarray:
        """Opacity with its zero point pinned to the most transparent direction.

        The absolute level of lambda is degenerate with the per-exposure
        normalization — nothing in the data says where zero is. The convention,
        inherited from the cafeteria project's `norm_quantile`, is that the
        least-absorbed directions are near-open sky, so lambda is shifted to put
        its low quantile at zero and then clipped. Negative opacity would mean
        more flux than open sky, which is not a thing.
        """
        lam = np.array(self.opacity[position_id_], dtype=np.float64)
        seen = np.isfinite(lam)
        if not seen.any():
            return lam
        lam[seen] = np.clip(lam[seen] - np.quantile(lam[seen], transparent_quantile),
                            0.0, None)
        return lam

    def response(self) -> np.ndarray:
        tx, ty = self.grid.tan_mesh()
        geom = DetectorGeometry.megiddo()
        return geometric_acceptance(tx, ty, geom) * np.exp(
            self.basis.evaluate(self.coeffs, tx, ty))

    def predict(self, exposure_id: str) -> np.ndarray:
        t = self.terms[exposure_id]
        lam = self.opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        mu = (self.norms[t.norm_group] * t.acceptance
              * np.exp(t.design @ self.coeffs)
              * flux_shape(t.sky_tan[:, 0], t.sky_tan[:, 1], self.flux_index)
              * np.exp(-lam))
        out = np.zeros(t.live.shape)
        out[t.live] = mu
        return out

    def opacity_image(self, position_id_: str) -> np.ndarray:
        k = self.sky.n_bins
        return self.opacity[position_id_].reshape(k, k)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coeffs": self.coeffs,
            "grid_edges": self.grid.edges,
            "sky_edges": self.sky.edges,
            "basis_centers": self.basis.centers,
            "basis_sigma": np.array(self.basis.sigma),
            "flux_index": np.array(self.flux_index),
            "nll_history": np.array(self.nll_history),
            "meta": np.array(json.dumps({
                "norms": self.norms,
                "positions": sorted(self.opacity),
            })),
        }
        for pid, lam in self.opacity.items():
            payload[f"opacity__{pid}"] = lam
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path: Path) -> "BaselineSolution":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        opacity = {pid: d[f"opacity__{pid}"] for pid in meta["positions"]}
        return BaselineSolution(
            coeffs=d["coeffs"],
            opacity=opacity,
            norms={k: float(v) for k, v in meta["norms"].items()},
            flux_index=float(d["flux_index"]),
            nll_history=[float(x) for x in d["nll_history"]],
            grid=AnalysisGrid(edges=d["grid_edges"], counts={}),
            sky=SkyGrid(edges=d["sky_edges"]),
            basis=SmoothBasis(centers=d["basis_centers"], sigma=float(d["basis_sigma"])),
        )


def position_sky_counts(grid: AnalysisGrid, cfg: SiteConfig, *,
                         geom: DetectorGeometry | None = None,
                         sky: SkyGrid | None = None) -> dict[str, np.ndarray]:
    """Per-position total live observed counts per sky bin, from raw data only.

    Same geometric-acceptance + on-sky + in-grid live mask `_build_terms` uses
    for its own (private) per-position sky-count accumulation, exposed here so
    callers that only have a rebuilt ingest grid -- e.g. `hillside --run`,
    which needs Poisson counts for heteroscedastic GP noise but does not run
    the full baseline solve -- can get the same totals without duplicating the
    mask logic. This is the sum BEFORE the MIN_SKY_COUNTS statistical cut;
    `_build_terms` applies that cut downstream against these same totals.

    `geom`/`sky` default to the real campaign's (`DetectorGeometry.megiddo()`,
    `make_sky_grid()`) -- the CLI's usage -- but `_build_terms` passes its own,
    since `solve_baseline` callers (tests, synthetic scenes) may use a
    different sky/geom and must get the SAME live mask, not a default one.
    """
    geom = geom or DetectorGeometry.megiddo()
    sky = sky or make_sky_grid()
    tx, ty = grid.tan_mesh()
    acceptance_full = geometric_acceptance(tx, ty, geom)
    positions = position_ids(cfg)

    out: dict[str, np.ndarray] = {}
    for eid, counts in grid.counts.items():
        exp = cfg.exposure(eid)
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acceptance_full > 0) & on_sky & in_grid
        acc = out.setdefault(positions[eid], np.zeros(sky.flat_size))
        np.add.at(acc, flat[live], counts[live].astype(np.float64))
    return out


def _build_terms(grid: AnalysisGrid, cfg: SiteConfig, geom: DetectorGeometry,
                 sky: SkyGrid, basis: SmoothBasis) -> dict[str, _ExposureTerms]:
    tx, ty = grid.tan_mesh()
    acceptance_full = geometric_acceptance(tx, ty, geom)
    positions = position_ids(cfg)

    # First pass: geometry-only live mask and sky mapping per exposure, before
    # any statistics-based exclusion (which needs the raw counts of ALL
    # exposures at a position, not just this one).
    raw: dict[str, dict] = {}
    for eid, counts in grid.counts.items():
        exp = cfg.exposure(eid)
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acceptance_full > 0) & on_sky & in_grid
        raw[eid] = dict(exp=exp, counts=counts, sx=sx, sy=sy, flat=flat, live=live)

    # Per-position total observed counts per sky bin, from raw data only, fixed
    # for the whole solve. A sky bin below MIN_SKY_COUNTS is dropped from every
    # exposure at that position: it cannot constrain opacity there regardless
    # of how the smooth correction or norms are set. Delegated to the public
    # `position_sky_counts` helper (DRY), passing THIS call's geom/sky through
    # -- callers of solve_baseline (tests, synthetic scenes) may use a
    # non-default sky/geom and must get the same live mask `raw` used above.
    pos_sky_counts = position_sky_counts(grid, cfg, geom=geom, sky=sky)

    terms: dict[str, _ExposureTerms] = {}
    for eid, r in raw.items():
        exp = r["exp"]
        pid = positions[eid]
        sx, sy, flat, counts = r["sx"], r["sy"], r["flat"], r["counts"]
        sky_ok = pos_sky_counts[pid][flat] >= MIN_SKY_COUNTS
        live = r["live"] & sky_ok

        terms[eid] = _ExposureTerms(
            exposure_id=eid,
            norm_group=exp.norm_group,
            position=pid,
            counts=counts[live].astype(np.float64),
            acceptance=acceptance_full[live],
            design=basis.design(tx[live], ty[live]),
            sky_flat=flat[live],
            sky_tan=np.stack([sx[live], sy[live]], axis=1),
            live=live,
        )
    return terms


def _kernel(t: _ExposureTerms, coeffs: np.ndarray, flux_index: float) -> np.ndarray:
    """Prediction with normalization and absorption both set to one."""
    return (t.acceptance * np.exp(t.design @ coeffs)
            * flux_shape(t.sky_tan[:, 0], t.sky_tan[:, 1], flux_index))


def _update_opacity(terms, coeffs, norms, flux_index, sky: SkyGrid
                    ) -> dict[str, np.ndarray]:
    """Closed-form Poisson MLE: exp(-lambda) = observed / expected."""
    positions = sorted({t.position for t in terms.values()})
    out: dict[str, np.ndarray] = {}
    for pid in positions:
        observed = np.zeros(sky.flat_size)
        expected = np.zeros(sky.flat_size)
        for t in terms.values():
            if t.position != pid:
                continue
            np.add.at(observed, t.sky_flat, t.counts)
            np.add.at(expected, t.sky_flat,
                      norms[t.norm_group] * _kernel(t, coeffs, flux_index))
        seen = expected > _EPS
        lam = np.full(sky.flat_size, np.nan)
        # No clip needed: sparse sky bins were already excluded from `live` in
        # _build_terms (MIN_SKY_COUNTS), so `observed` here is always backed by
        # a real sample, never an invented one.
        ratio = observed[seen] / expected[seen]
        lam[seen] = -np.log(ratio)

        if np.isfinite(lam).any():
            # Gauge fixing. norm_e and lambda_p are exactly degenerate: scaling
            # every norm at a position by c and adding log(c) to that position's
            # opacity leaves every prediction identical. Left free, the
            # alternating updates slide along that direction indefinitely — on
            # real data the norms collapsed toward zero and opacity ran to -13
            # while the fit got steadily worse. Pinning the median to zero
            # removes the flat direction; the immediately following norm update
            # absorbs the shift in closed form, so the fit itself is unchanged.
            lam[np.isfinite(lam)] -= np.nanmedian(lam)

        out[pid] = lam
    return out


def _update_norms(terms, coeffs, opacity, flux_index) -> dict[str, float]:
    """Closed-form Poisson MLE for a multiplicative scale."""
    numer: dict[str, float] = {}
    denom: dict[str, float] = {}
    for t in terms.values():
        lam = opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        k = _kernel(t, coeffs, flux_index) * np.exp(-lam)
        numer[t.norm_group] = numer.get(t.norm_group, 0.0) + float(t.counts.sum())
        denom[t.norm_group] = denom.get(t.norm_group, 0.0) + float(k.sum())
    return {g: (numer[g] / denom[g] if denom[g] > _EPS else 1.0) for g in numer}


def _nll(terms, coeffs, opacity, norms, flux_index) -> float:
    total = 0.0
    for t in terms.values():
        lam = opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        mu = norms[t.norm_group] * _kernel(t, coeffs, flux_index) * np.exp(-lam)
        mu = np.clip(mu, _EPS, None)
        total += float(np.sum(mu - t.counts * np.log(mu)))
    return total


def _update_coeffs(terms, coeffs, opacity, norms, flux_index) -> np.ndarray:
    """L-BFGS-B on the Poisson NLL. Gradient is analytic:
    d/dc sum(mu - n log mu) = design^T (mu - n), since d mu / d c = mu * design."""
    def objective(c):
        value = 0.0
        grad = np.zeros_like(c)
        for t in terms.values():
            lam = opacity[t.position][t.sky_flat]
            lam = np.where(np.isfinite(lam), lam, 0.0)
            mu = norms[t.norm_group] * _kernel(t, c, flux_index) * np.exp(-lam)
            mu = np.clip(mu, _EPS, None)
            value += float(np.sum(mu - t.counts * np.log(mu)))
            grad += t.design.T @ (mu - t.counts)
        return value, grad

    result = minimize(objective, coeffs, jac=True, method="L-BFGS-B",
                      options={"maxiter": 60})
    return result.x


def _update_flux_index(terms, coeffs, opacity, norms, current: float) -> float:
    candidates = np.clip(current + np.linspace(-0.6, 0.6, 13), 0.5, 5.0)
    scores = [_nll(terms, coeffs, opacity, norms, n) for n in candidates]
    return float(candidates[int(np.argmin(scores))])


def solve_baseline(grid: AnalysisGrid, cfg: SiteConfig, *,
                   geom: DetectorGeometry | None = None,
                   sky: SkyGrid | None = None,
                   basis: SmoothBasis | None = None,
                   n_iter: int = 5000, flux_index: float = 2.0,
                   fit_flux_index: bool = False,
                   tol: float = 1e-12) -> BaselineSolution:
    """Joint solve for the smooth detector correction, per-position opacity,
    and per-exposure normalization.

    The flux index is FIXED, not fitted, and 2.0 is the standard sea-level
    cosmic-muon value. It is not identifiable from this data: lambda(s) is free
    per sky bin and Phi_n(s) is a function of the same sky direction, so only
    their product is determined. Measured on the real campaign, fixing the index
    anywhere from 1 to 5 and converging fully moves the best NLL by about 200,
    against roughly 1000 for a single early iteration.

    This means the recovered opacity is meaningful only RELATIVE to the assumed
    flux model — the unavoidable price of having no open-sky calibration run,
    and what spec section 6.3 means by the physics entering as a prior. Passing
    fit_flux_index=True is retained for diagnostics; the value it returns is not
    a measurement.
    """
    geom = geom or DetectorGeometry.megiddo()
    sky = sky or make_sky_grid()
    basis = basis or make_smooth_basis()

    terms = _build_terms(grid, cfg, geom, sky, basis)
    coeffs = np.zeros(basis.n_coeff)
    norms = {t.norm_group: 1.0 for t in terms.values()}
    opacity = _update_opacity(terms, coeffs, norms, flux_index, sky)

    history: list[float] = []
    for _ in range(n_iter):
        opacity = _update_opacity(terms, coeffs, norms, flux_index, sky)
        norms = _update_norms(terms, coeffs, opacity, flux_index)
        coeffs = _update_coeffs(terms, coeffs, opacity, norms, flux_index)
        if fit_flux_index:
            # Diagnostic only. The flux index is not identifiable — lambda is free
            # per sky bin and the flux shape is a function of the same sky
            # direction, so only their product is determined. The value this
            # returns is not a measurement; see the docstring.
            flux_index = _update_flux_index(terms, coeffs, opacity, norms, flux_index)

        history.append(_nll(terms, coeffs, opacity, norms, flux_index))
        # The alternating scheme converges slowly: on a synthetic scene, opacity
        # recovery runs 0.54 -> 0.74 -> 0.90 -> 0.94 -> 0.955 at 100, 300, 600, 1200
        # and 4800 iterations. Stopping early does not merely lose precision, it
        # changes the answer — a 300-iteration run reads as an intrinsic bias floor
        # that is not there. Five thousand iterations cost a few seconds, so the
        # budget is generous and the tolerance deliberately tight.
        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol * abs(history[-1]):
            break

    return BaselineSolution(coeffs=coeffs, opacity=opacity, norms=norms,
                            flux_index=flux_index, nll_history=history,
                            grid=grid, sky=sky, basis=basis, terms=terms)
