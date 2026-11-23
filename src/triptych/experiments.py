"""The experiments behind every table and figure.

Each function takes a :class:`Config` and returns plain rows (a list of
dictionaries). Nothing here writes files or draws; see :mod:`triptych.report`.
All randomness comes from seeds held in the config, so two runs give the same
rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from triptych import black_scholes as bs
from triptych.american import lsm_price
from triptych.greeks import finite_difference_greeks, pathwise_greeks
from triptych.monte_carlo import McResult, european_mc, variance_reduction_factor
from triptych.option import Option
from triptych.published import (
    LS_EXERCISE_DATES_PER_YEAR,
    LS_RATE,
    LS_STRIKE,
    LS_TABLE_1,
)
from triptych.tree import crr_european_sum, crr_price

Row = dict[str, float | int | str]

# The running example: the at-the-money one-year put of Longstaff and
# Schwartz, Table 1 (S = 40, sigma = 0.20, T = 1).
BASE = Option(spot=40.0, strike=LS_STRIKE, rate=LS_RATE, vol=0.20, maturity=1.0, kind="put")

# The in-the-money one-year put of the same table, used for American pricing.
AMERICAN_BASE = BASE.with_(spot=36.0)

ESTIMATORS: tuple[tuple[str, bool, bool], ...] = (
    ("plain", False, False),
    ("antithetic", True, False),
    ("control variate", False, True),
    ("antithetic + control variate", True, True),
)


@dataclass(frozen=True)
class Config:
    """Sizes and seeds for one run of the experiments.

    Attributes:
        seed: base seed; each experiment adds its own fixed offset.
        tolerance: the agreement target for the three European prices.
        tree_scan_max: largest step count searched for the tree to reach
            ``tolerance``.
        tree_curve_max: largest step count on the tree convergence curve.
        mc_pilot: payoffs used to measure each estimator's variance.
        mc_headline_cap: upper limit on payoffs in the headline simulation.
        mc_curve_sizes: payoff counts on the Monte Carlo convergence curve.
        mc_replications: independent repeats per point on that curve.
        greeks_payoffs: payoffs behind the pathwise greeks table.
        greeks_curve_sizes: payoff counts on the pathwise greeks curve.
        lsm_paths: paths per cell of the Longstaff-Schwartz table.
        lsm_curve_sizes: path counts on the least-squares convergence curve.
        lsm_replications: independent repeats per point on that curve.
        bermudan_steps_per_date: tree periods between exercise dates when the
            tree prices the 50-dates-a-year put.
        american_tree_steps: step counts on the American tree curve.
        american_tree_reference: step count of the reference American tree.
    """

    seed: int = 2001
    tolerance: float = 5e-5
    tree_scan_max: int = 25_000
    tree_curve_max: int = 20_000
    mc_pilot: int = 10_000_000
    mc_headline_cap: int = 2_000_000_000
    mc_curve_sizes: tuple[int, ...] = (1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000)
    mc_replications: int = 100
    greeks_payoffs: int = 10_000_000
    greeks_curve_sizes: tuple[int, ...] = (1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000)
    lsm_paths: int = 100_000
    lsm_curve_sizes: tuple[int, ...] = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000)
    lsm_replications: int = 20
    bermudan_steps_per_date: int = 200
    american_tree_steps: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000)
    american_tree_reference: int = 50_000


FULL = Config()

FAST = Config(
    tolerance=5e-4,
    tree_scan_max=3_000,
    tree_curve_max=2_000,
    mc_pilot=400_000,
    mc_headline_cap=20_000_000,
    mc_curve_sizes=(1_000, 4_000, 16_000, 64_000),
    mc_replications=30,
    greeks_payoffs=400_000,
    greeks_curve_sizes=(1_000, 4_000, 16_000, 64_000),
    lsm_paths=20_000,
    lsm_curve_sizes=(1_000, 2_000, 4_000, 8_000),
    lsm_replications=8,
    bermudan_steps_per_date=10,
    american_tree_steps=(10, 20, 50, 100, 200, 500),
    american_tree_reference=5_000,
)


def fitted_slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of log y against log x."""
    slope, _ = np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)
    return float(slope)


# --------------------------------------------------------------------------
# European: three ways
# --------------------------------------------------------------------------


def tree_steps_for_tolerance(option: Option, tolerance: float, scan_max: int) -> int:
    """Smallest n such that every tree from n to ``scan_max`` steps is within tolerance.

    Returns 0 if the tree at ``scan_max`` steps is still outside the tolerance.
    """
    exact = bs.price(option)
    last_bad = 0
    for n in range(scan_max, 0, -1):
        if abs(crr_european_sum(option, n) - exact) >= tolerance:
            last_bad = n
            break
    return 0 if last_bad == scan_max else last_bad + 1


def tree_convergence(cfg: Config) -> list[Row]:
    """Absolute error of the European tree against the closed form, by step count."""
    exact = bs.price(BASE)
    grid = np.unique(np.round(np.geomspace(10, cfg.tree_curve_max, 40)).astype(int))
    rows: list[Row] = []
    for n in grid:
        # An even and an odd step count at each scale: the error alternates in sign.
        for steps in (int(n), int(n) + 1):
            value = crr_european_sum(BASE, steps)
            rows.append({"steps": steps, "price": value, "abs_error": abs(value - exact)})
    return rows


def _estimator_runs(option: Option, n_payoffs: int, seed: int) -> dict[str, McResult]:
    return {
        name: european_mc(option, n_payoffs, seed, antithetic=anti, control_variate=control)
        for name, anti, control in ESTIMATORS
    }


def variance_reduction(cfg: Config) -> list[Row]:
    """Each estimator on the base put: error, variance-reduction factor, cost of the target.

    The factor is the variance of the plain estimate divided by the variance
    of the estimate in question, at an equal number of payoff evaluations.
    """
    exact = bs.price(BASE)
    runs = _estimator_runs(BASE, cfg.mc_pilot, cfg.seed + 1)
    rows: list[Row] = []
    for name, result in runs.items():
        rows.append(
            {
                "estimator": name,
                "payoffs": result.n_payoffs,
                "price": result.price,
                "std_error": result.std_error,
                "error": result.price - exact,
                "error_in_se": (result.price - exact) / result.std_error,
                "variance_reduction": variance_reduction_factor(runs["plain"], result),
                "payoffs_for_target": math.ceil(result.variance_per_payoff / cfg.tolerance**2),
            }
        )
    return rows


def variance_reduction_by_spot(cfg: Config) -> list[Row]:
    """Variance-reduction factors for puts in, at and out of the money."""
    rows: list[Row] = []
    for i, spot in enumerate((36.0, 40.0, 44.0)):
        runs = _estimator_runs(BASE.with_(spot=spot), cfg.mc_pilot, cfg.seed + 10 + i)
        row: Row = {"spot": spot}
        for name, result in runs.items():
            if name != "plain":
                row[name] = variance_reduction_factor(runs["plain"], result)
        rows.append(row)
    return rows


def european_three_ways(cfg: Config) -> list[Row]:
    """Build the headline table: one put by formula, tree and simulation.

    The tree uses the smallest step count that stays within the tolerance.
    The simulation uses both variance reductions, with the number of payoffs
    that a pilot run says is needed for a standard error equal to the
    tolerance, rounded up to a whole number of millions. The control
    coefficient is taken from the pilot run, which uses a different seed, so
    the headline estimate is exactly unbiased.
    """
    exact = bs.price(BASE)
    rows: list[Row] = [
        {
            "method": "Black-Scholes formula",
            "effort": "closed form",
            "price": exact,
            "error": 0.0,
            "std_error": 0.0,
        }
    ]

    steps = tree_steps_for_tolerance(BASE, cfg.tolerance, cfg.tree_scan_max)
    if steps:
        tree_value = crr_price(BASE, steps)
        rows.append(
            {
                "method": "CRR binomial tree",
                "effort": f"{steps} steps",
                "price": tree_value,
                "error": tree_value - exact,
                "std_error": 0.0,
            }
        )

    pilot = european_mc(BASE, cfg.mc_pilot, cfg.seed + 1, antithetic=True, control_variate=True)
    needed = math.ceil(pilot.variance_per_payoff / cfg.tolerance**2 / 1e6) * 1_000_000
    n_payoffs = min(needed, cfg.mc_headline_cap)
    result = european_mc(
        BASE, n_payoffs, cfg.seed + 2, antithetic=True, control_variate=True, beta=pilot.beta
    )
    rows.append(
        {
            "method": "Monte Carlo, antithetic + control variate",
            "effort": f"{n_payoffs} payoffs",
            "price": result.price,
            "error": result.price - exact,
            "std_error": result.std_error,
        }
    )
    return rows


def mc_convergence(cfg: Config) -> list[Row]:
    """Root-mean-square error of each estimator over repeated runs, by payoff count."""
    exact = bs.price(BASE)
    rows: list[Row] = []
    for name, anti, control in ESTIMATORS:
        for size_index, n in enumerate(cfg.mc_curve_sizes):
            errors = [
                european_mc(
                    BASE,
                    n,
                    cfg.seed + 1_000 + 100 * size_index + rep,
                    antithetic=anti,
                    control_variate=control,
                ).price
                - exact
                for rep in range(cfg.mc_replications)
            ]
            rows.append(
                {
                    "estimator": name,
                    "payoffs": n,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_error": float(np.mean(errors)),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Greeks: three ways
# --------------------------------------------------------------------------


def greeks_table(cfg: Config) -> list[Row]:
    """Analytic, finite-difference and Monte Carlo greeks of the base put."""
    analytic = bs.greeks(BASE).as_dict()
    finite = finite_difference_greeks(bs.price, BASE).as_dict()
    simulated = pathwise_greeks(BASE, cfg.greeks_payoffs, cfg.seed + 3)
    estimate = simulated.estimate.as_dict()
    error = simulated.std_error.as_dict()
    rows: list[Row] = []
    for name in bs.GREEK_NAMES:
        rows.append(
            {
                "greek": name,
                "analytic": analytic[name],
                "finite_difference": finite[name],
                "fd_minus_analytic": finite[name] - analytic[name],
                "monte_carlo": estimate[name],
                "mc_std_error": error[name],
                "mc_error_in_se": (estimate[name] - analytic[name]) / error[name],
            }
        )
    return rows


def fd_bump_convergence() -> list[Row]:
    """Relative error of finite-difference delta and gamma against the bump size."""
    analytic = bs.greeks(BASE)
    rows: list[Row] = []
    for exponent in np.arange(-7.0, -0.75, 0.25):
        bump = float(10.0**exponent)
        finite = finite_difference_greeks(bs.price, BASE, rel_bump=bump)
        rows.append(
            {
                "rel_bump": bump,
                "delta_rel_error": abs(finite.delta / analytic.delta - 1.0),
                "gamma_rel_error": abs(finite.gamma / analytic.gamma - 1.0),
                "vega_rel_error": abs(finite.vega / analytic.vega - 1.0),
            }
        )
    return rows


def pathwise_convergence(cfg: Config) -> list[Row]:
    """Relative root-mean-square error of the Monte Carlo greeks, by payoff count."""
    analytic = bs.greeks(BASE).as_dict()
    rows: list[Row] = []
    for size_index, n in enumerate(cfg.greeks_curve_sizes):
        squares = {name: 0.0 for name in bs.GREEK_NAMES}
        for rep in range(cfg.mc_replications):
            estimate = pathwise_greeks(
                BASE, n, cfg.seed + 5_000 + 100 * size_index + rep
            ).estimate.as_dict()
            for name in bs.GREEK_NAMES:
                squares[name] += (estimate[name] / analytic[name] - 1.0) ** 2
        row: Row = {"payoffs": n}
        for name in bs.GREEK_NAMES:
            row[f"{name}_rel_rmse"] = math.sqrt(squares[name] / cfg.mc_replications)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# American puts
# --------------------------------------------------------------------------


def bermudan_tree(option: Option, steps_per_date: int) -> float:
    """Tree value of a put exercisable 50 times a year, as in the paper's Table 1."""
    n_dates = round(LS_EXERCISE_DATES_PER_YEAR * option.maturity)
    return crr_price(option, n_dates * steps_per_date, "american", exercise_every=steps_per_date)


def longstaff_schwartz_table(cfg: Config) -> list[Row]:
    """Table 1 of Longstaff and Schwartz, with this code's values beside the paper's."""
    rows: list[Row] = []
    for index, cell in enumerate(LS_TABLE_1):
        option = Option(cell.spot, LS_STRIKE, LS_RATE, cell.vol, cell.maturity, "put")
        n_dates = round(LS_EXERCISE_DATES_PER_YEAR * cell.maturity)
        sim = lsm_price(option, n_dates, cfg.lsm_paths, cfg.seed + 100 + index)
        tree_value = bermudan_tree(option, cfg.bermudan_steps_per_date)
        tree_any_date = crr_price(option, n_dates * cfg.bermudan_steps_per_date, "american")
        rows.append(
            {
                "spot": float(cell.spot),
                "vol": cell.vol,
                "maturity": float(cell.maturity),
                "paper_fd": cell.fd_american,
                "tree_50_dates": tree_value,
                "tree_any_date": tree_any_date,
                "tree_minus_paper_fd": tree_value - cell.fd_american,
                "paper_european": cell.european,
                "european": bs.price(option),
                "paper_lsm": cell.sim_american,
                "paper_se": cell.sim_se,
                "lsm": sim.price,
                "se_pairs": sim.std_error,
                "se_naive": sim.std_error_naive,
                "lsm_minus_paper": sim.price - cell.sim_american,
                "diff_in_paper_se": (sim.price - cell.sim_american) / cell.sim_se,
                "lsm_minus_tree": sim.price - tree_value,
            }
        )
    return rows


def lsm_convergence(cfg: Config) -> list[Row]:
    """Error of least-squares Monte Carlo against the tree, by number of paths."""
    reference = bermudan_tree(AMERICAN_BASE, cfg.bermudan_steps_per_date)
    n_dates = round(LS_EXERCISE_DATES_PER_YEAR * AMERICAN_BASE.maturity)
    rows: list[Row] = []
    for size_index, n in enumerate(cfg.lsm_curve_sizes):
        errors = [
            lsm_price(AMERICAN_BASE, n_dates, n, cfg.seed + 9_000 + 100 * size_index + rep).price
            - reference
            for rep in range(cfg.lsm_replications)
        ]
        rows.append(
            {
                "paths": n,
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_error": float(np.mean(errors)),
                "std_of_estimates": float(np.std(errors, ddof=1)),
            }
        )
    return rows


def american_tree_convergence(cfg: Config) -> list[Row]:
    """Error of the American tree against a much finer tree, by step count."""
    reference = crr_price(AMERICAN_BASE, cfg.american_tree_reference, "american")
    rows: list[Row] = []
    for steps in cfg.american_tree_steps:
        value = crr_price(AMERICAN_BASE, steps, "american")
        rows.append({"steps": steps, "price": value, "abs_error": abs(value - reference)})
    rows.append({"steps": cfg.american_tree_reference, "price": reference, "abs_error": 0.0})
    return rows


# --------------------------------------------------------------------------
# Slopes
# --------------------------------------------------------------------------


def convergence_slopes(
    tree_rows: list[Row],
    mc_rows: list[Row],
    lsm_rows: list[Row],
    american_rows: list[Row],
    fd_rows: list[Row],
    pathwise_rows: list[Row],
) -> list[Row]:
    """Fitted log-log slopes of every convergence curve, beside the expected value."""

    def column(rows: list[Row], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    out: list[Row] = [
        {
            "curve": "European tree, error against steps",
            "expected": -1.0,
            "fitted": fitted_slope(column(tree_rows, "steps"), column(tree_rows, "abs_error")),
        }
    ]
    for name, _, _ in ESTIMATORS:
        sub = [r for r in mc_rows if r["estimator"] == name]
        out.append(
            {
                "curve": f"Monte Carlo ({name}), RMSE against payoffs",
                "expected": -0.5,
                "fitted": fitted_slope(column(sub, "payoffs"), column(sub, "rmse")),
            }
        )
    out.append(
        {
            "curve": "Least-squares Monte Carlo, RMSE against paths",
            "expected": -0.5,
            "fitted": fitted_slope(column(lsm_rows, "paths"), column(lsm_rows, "rmse")),
        }
    )
    finite = [r for r in american_rows if float(r["abs_error"]) > 0.0]
    out.append(
        {
            "curve": "American tree, error against steps",
            "expected": -1.0,
            "fitted": fitted_slope(column(finite, "steps"), column(finite, "abs_error")),
        }
    )
    # Fit the finite-difference slope only where truncation error dominates.
    truncation = [r for r in fd_rows if float(r["rel_bump"]) >= 1e-3]
    out.append(
        {
            "curve": "Finite-difference delta, error against bump (bump >= 1e-3)",
            "expected": 2.0,
            "fitted": fitted_slope(
                column(truncation, "rel_bump"), column(truncation, "delta_rel_error")
            ),
        }
    )
    for name in bs.GREEK_NAMES:
        out.append(
            {
                "curve": f"Monte Carlo {name}, RMSE against payoffs",
                "expected": -0.5,
                "fitted": fitted_slope(
                    column(pathwise_rows, "payoffs"), column(pathwise_rows, f"{name}_rel_rmse")
                ),
            }
        )
    return out
