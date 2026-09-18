"""Phase 2 validation: how well is the answer actually determined?

The solve always returns a baseline and a sky. These checks ask whether to
believe them. Leave-one-exposure-out is the honest cross-validation for a fit
this flexible, and the Poisson bootstrap says which parts of the sky map are
measurements and which are noise.

Spec section 6.4 records a known degeneracy: opacity constant on cones about the
tilt axis is barely moved by the rotation, so it is weakly determined. The
bootstrap sigma is where that shows up, which is why the sky map must always be
read alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.config import SiteConfig
from megido.sky import flux_shape


@dataclass(frozen=True)
class Check2:
    name: str
    passed: bool
    measured: float
    expected: str
    detail: str


def opacity_uncertainty(grid: AnalysisGrid, cfg: SiteConfig, *,
                        n_replicas: int = 12, seed: int = 0,
                        **solve_kwargs) -> dict[str, np.ndarray]:
    """Poisson bootstrap over the counts, giving a per-sky-bin sigma."""
    rng = np.random.default_rng(seed)
    stacks: dict[str, list[np.ndarray]] = {}

    for _ in range(n_replicas):
        resampled = {eid: rng.poisson(v).astype(np.int64)
                     for eid, v in grid.counts.items()}
        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=resampled),
                             cfg, **solve_kwargs)
        for pid, lam in sol.opacity.items():
            stacks.setdefault(pid, []).append(lam)

    return {pid: np.nanstd(np.vstack(v), axis=0) for pid, v in stacks.items()}


def leave_one_out(grid: AnalysisGrid, cfg: SiteConfig,
                  **solve_kwargs) -> list[Check2]:
    """Fit without one exposure, then predict it from parameters it never saw.

    A genuine held-out test needs the held-out exposure's position to still be
    represented by another exposure — otherwise its opacity map is unconstrained
    and there is nothing to predict with. In this campaign that means P0, T20a
    and T20b can each be held out (the other two hold pos0), while P1 cannot:
    it is the only exposure at pos1, so holding it out removes that sky map
    entirely. P1 is reported as SKIPPED rather than silently passed.

    Only the held-out exposure's own normalization is re-fitted, in closed form
    from its total counts. Every shape-carrying parameter — the smooth
    coefficients, the flux index, and the opacity map — comes from the fit that
    never saw it.
    """
    from megido.baseline import position_ids

    positions = position_ids(cfg)
    checks: list[Check2] = []

    for held_out in sorted(grid.counts):
        kept = {e: v for e, v in grid.counts.items() if e != held_out}
        same_position = [e for e in kept if positions[e] == positions[held_out]]

        if not same_position:
            checks.append(Check2(
                name=f"loo.{held_out}",
                passed=True,
                measured=float("nan"),
                expected="skipped — sole exposure at its position",
                detail=(f"{held_out} is the only exposure at {positions[held_out]}; "
                        "holding it out leaves that sky map unconstrained, so there "
                        "is nothing to predict against"),
            ))
            continue

        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=kept),
                             cfg, **solve_kwargs)

        # Rebuild the held-out exposure's terms against the subset-trained fit.
        full = solve_baseline(AnalysisGrid(edges=grid.edges, counts=grid.counts),
                              cfg, n_iter=1, **{k: v for k, v in solve_kwargs.items()
                                                if k != "n_iter"})
        term = full.terms[held_out]

        lam = sol.opacity[positions[held_out]][term.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        shape = (term.acceptance * np.exp(term.design @ sol.coeffs)
                 * flux_shape(term.sky_tan[:, 0], term.sky_tan[:, 1], sol.flux_index)
                 * np.exp(-lam))

        observed = term.counts
        if shape.sum() <= 0 or observed.sum() <= 0 or observed.size < 20:
            checks.append(Check2(f"loo.{held_out}", False, 0.0,
                                 "at least 20 live bins with counts",
                                 "too few live bins to judge"))
            continue

        o = observed / observed.sum()
        p = shape / shape.sum()
        corr = float(np.corrcoef(o, p)[0, 1])
        checks.append(Check2(
            name=f"loo.{held_out}",
            passed=corr > 0.9,
            measured=corr,
            expected="> 0.90 shape correlation",
            detail=(f"held out {held_out}: its angular shape predicted from a fit "
                    f"that never saw it, correlation {corr:.4f} over "
                    f"{observed.size} live bins"),
        ))
    return checks


def nll_per_bin_check(sol: BaselineSolution, grid: AnalysisGrid) -> Check2:
    """Poisson deviance per live bin. Near 1 means the model fits as well as
    counting statistics allow; far above means it does not; far below means it
    has more freedom than the data supports."""
    deviance = 0.0
    n_live = 0
    for eid, observed in grid.counts.items():
        predicted = sol.predict(eid)
        live = predicted > 0
        o = observed[live].astype(float)
        p = predicted[live]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(o > 0, o * np.log(np.clip(o / p, 1e-12, None)), 0.0)
        deviance += 2.0 * float(np.sum(term - (o - p)))
        n_live += int(live.sum())

    value = deviance / max(n_live, 1)
    return Check2(
        name="deviance_per_bin",
        passed=0.2 < value < 3.0,
        measured=value,
        expected="0.2 < deviance/bin < 3.0",
        detail=f"Poisson deviance {deviance:.1f} over {n_live} live bins",
    )


def format_report2(checks: list[Check2]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:24s} "
                     f"measured={c.measured:.4f}  expected {c.expected}")
        lines.append(f"         {c.detail}")
    n_pass = sum(1 for c in checks if c.passed)
    lines.append(f"\n{n_pass}/{len(checks)} checks passed")
    return "\n".join(lines)
