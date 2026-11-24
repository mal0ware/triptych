"""The experiments behind every table and figure.

Each function takes a :class:`Config` and returns plain rows (a list of
dictionaries). Nothing here writes files or draws; see :mod:`triptych.report`.
All randomness comes from seeds held in the config, so two runs give the same
rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from triptych import black_scholes as bs
from triptych.american import (
    andersen_broadie_upper_bound,
    fit_exercise_rule,
    lsm_lower_bound,
    lsm_price,
)
from triptych.greeks import finite_difference_greeks, pathwise_greeks
from triptych.monte_carlo import McResult, european_mc, variance_reduction_factor
from triptych.option import FloatArray, Option
from triptych.published import (
    LS_EXERCISE_DATES_PER_YEAR,
    LS_RATE,
    LS_STRIKE,
    LS_TABLE_1,
)
from triptych.tree import crr_european_sum, crr_price, richardson, smoothed_european_sum

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
        richardson_scan_max: largest coarse step count searched when
            Richardson extrapolation combines trees of n and 2n steps.
        american_methods_steps: finest-tree step counts at which the plain,
            smoothed and extrapolated American trees are compared.
        american_methods_reference: coarse step count of the extrapolated
            tree used as the reference in that comparison.
        lsm_oos_paths: fresh paths on which each rule fitted for the
            least-squares convergence curve is valued out of sample.
        bound_cells: (spot, vol, maturity) of the Table 1 puts given a
            lower and an upper bound.
        bound_fit_paths: paths that fit the exercise rule for the bounds.
        bound_lower_paths: fresh paths that value the rule (lower bound).
        bound_outer: outer paths of the upper bound.
        bound_inner: inner paths per continuation value of the upper bound.
        bound_inner_sweep: inner path counts tried on the base American put.
        bound_sweep_outer: outer paths for that sweep, the same at each count.
        bootstrap_draws: resamples behind each fitted-slope interval.
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
    richardson_scan_max: int = 5_000
    american_methods_steps: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1_600, 3_200)
    american_methods_reference: int = 10_000
    lsm_oos_paths: int = 100_000
    bound_cells: tuple[tuple[float, float, float], ...] = (
        (36.0, 0.2, 1.0),
        (40.0, 0.2, 1.0),
        (44.0, 0.2, 1.0),
        (36.0, 0.4, 1.0),
        (40.0, 0.4, 1.0),
        (44.0, 0.4, 1.0),
    )
    bound_fit_paths: int = 100_000
    bound_lower_paths: int = 2_000_000
    bound_outer: int = 400
    bound_inner: int = 4_000
    bound_inner_sweep: tuple[int, ...] = (500, 1_000, 2_000, 4_000, 8_000)
    bound_sweep_outer: int = 100
    bootstrap_draws: int = 2_000


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
    richardson_scan_max=1_000,
    american_methods_steps=(25, 50, 100, 200, 400),
    american_methods_reference=1_000,
    lsm_oos_paths=20_000,
    bound_cells=((36.0, 0.2, 1.0),),
    bound_fit_paths=20_000,
    bound_lower_paths=40_000,
    bound_outer=40,
    bound_inner=400,
    bound_inner_sweep=(200, 400),
    bound_sweep_outer=20,
    bootstrap_draws=200,
)


def fitted_slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of log y against log x."""
    slope, _ = np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)
    return float(slope)


@dataclass(frozen=True)
class Curve:
    """The raw errors behind one Monte Carlo convergence curve.

    ``errors[j]`` holds one error per independent run at size ``sizes[j]``.
    The curve plots the root mean square of each.
    """

    sizes: list[float]
    errors: list[FloatArray]


def bootstrap_slope_interval(curve: Curve, draws: int, seed: int) -> tuple[float, float]:
    """95% percentile bootstrap interval for the fitted slope of an RMSE curve.

    Each draw resamples the runs at every size independently, with
    replacement, recomputes the root-mean-square errors and refits the slope.
    The runs at different sizes use different seeds, so they are independent.
    """
    rng = np.random.default_rng(seed)
    log_rmse = np.empty((len(curve.sizes), draws))
    for j, errors in enumerate(curve.errors):
        picks = rng.integers(0, errors.size, size=(draws, errors.size))
        log_rmse[j] = 0.5 * np.log(np.mean(np.square(errors[picks]), axis=1))
    slopes = np.polyfit(np.log(np.asarray(curve.sizes)), log_rmse, 1)[0]
    low, high = np.percentile(slopes, [2.5, 97.5])
    return float(low), float(high)


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


def first_count_within(error_at: Callable[[int], float], counts: range, tolerance: float) -> int:
    """Smallest count in ``counts`` from which every larger count is within tolerance.

    Scans down from the largest count and stops at the first miss. Returns 0
    if the largest count is still outside the tolerance.
    """
    first = 0
    for n in reversed(counts):
        if abs(error_at(n)) >= tolerance:
            break
        first = n
    return first


def tree_steps_by_method(cfg: Config) -> list[Row]:
    """Count the steps each European tree method needs to stay within the tolerance.

    Four methods: the plain tree, the plain tree with Richardson
    extrapolation over even step counts, the smoothed tree, and the smoothed
    tree with Richardson extrapolation. For the extrapolated methods the
    count reported is the finer of the two trees, 2n; the coarse tree of n
    steps is built as well. Puts in, at and out of the money.
    """
    rows: list[Row] = []
    top = 2 * cfg.richardson_scan_max
    for spot in (36.0, 40.0, 44.0):
        option = BASE.with_(spot=spot)
        exact = bs.price(option)

        def crr_extrapolated(fine: int, option: Option = option, exact: float = exact) -> float:
            coarse = crr_european_sum(option, fine // 2)
            return richardson(coarse, crr_european_sum(option, fine)) - exact

        def smoothed(n: int, option: Option = option, exact: float = exact) -> float:
            return smoothed_european_sum(option, n) - exact

        def smoothed_extrapolated(
            fine: int, option: Option = option, exact: float = exact
        ) -> float:
            coarse = smoothed_european_sum(option, fine // 2)
            return richardson(coarse, smoothed_european_sum(option, fine)) - exact

        for method, steps, limit in (
            (
                "plain tree",
                tree_steps_for_tolerance(option, cfg.tolerance, cfg.tree_scan_max),
                cfg.tree_scan_max,
            ),
            (
                "plain tree, Richardson, both trees even",
                first_count_within(crr_extrapolated, range(4, top + 1, 4), cfg.tolerance),
                top,
            ),
            ("smoothed tree", first_count_within(smoothed, range(1, top + 1), cfg.tolerance), top),
            (
                "smoothed tree, Richardson",
                first_count_within(smoothed_extrapolated, range(2, top + 1, 2), cfg.tolerance),
                top,
            ),
        ):
            rows.append(
                {
                    "spot": spot,
                    "method": method,
                    "steps_needed": steps,
                    "searched_up_to": limit,
                }
            )
    return rows


def american_tree_methods(cfg: Config, plain_reference: float) -> list[Row]:
    """Plain, smoothed and extrapolated American trees against a fine extrapolated tree.

    At each step count n the plain and smoothed trees have n steps and the
    extrapolated value combines smoothed trees of n/2 and n steps. The
    reference extrapolates smoothed trees of ``american_methods_reference``
    and twice that many steps. The last row is the plain tree used as the
    reference elsewhere (``plain_reference``, at ``american_tree_reference``
    steps), measured against the same reference.
    """
    option = AMERICAN_BASE

    def smoothed(n: int) -> float:
        return crr_price(option, n, "american", smooth=True)

    m = cfg.american_methods_reference
    reference = richardson(smoothed(m), smoothed(2 * m))
    rows: list[Row] = []
    for n in cfg.american_methods_steps:
        smooth = smoothed(n)
        rows.append(
            {
                "steps": n,
                "plain_error": crr_price(option, n, "american") - reference,
                "smoothed_error": smooth - reference,
                "smoothed_richardson_error": richardson(smoothed(n // 2), smooth) - reference,
                "reference": reference,
            }
        )
    rows.append(
        {
            "steps": cfg.american_tree_reference,
            "plain_error": plain_reference - reference,
            "smoothed_error": "",
            "smoothed_richardson_error": "",
            "reference": reference,
        }
    )
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

    The tree uses the smallest step count that stays within the tolerance,
    and so does the smoothed tree with Richardson extrapolation.
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

    def extrapolated_error(fine: int) -> float:
        coarse = smoothed_european_sum(BASE, fine // 2)
        return richardson(coarse, smoothed_european_sum(BASE, fine)) - exact

    top = 2 * cfg.richardson_scan_max
    fine = first_count_within(extrapolated_error, range(2, top + 1, 2), cfg.tolerance)
    if fine:
        value = richardson(
            smoothed_european_sum(BASE, fine // 2), smoothed_european_sum(BASE, fine)
        )
        rows.append(
            {
                "method": "Smoothed tree with Richardson extrapolation",
                "effort": f"{fine // 2} and {fine} steps",
                "price": value,
                "error": value - exact,
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


def mc_convergence(cfg: Config) -> tuple[list[Row], dict[str, Curve]]:
    """Root-mean-square error of each estimator over repeated runs, by payoff count.

    Returns the table and the raw errors of each estimator's curve.
    """
    exact = bs.price(BASE)
    rows: list[Row] = []
    curves: dict[str, Curve] = {}
    for name, anti, control in ESTIMATORS:
        curve = Curve(sizes=[float(n) for n in cfg.mc_curve_sizes], errors=[])
        for size_index, n in enumerate(cfg.mc_curve_sizes):
            errors = np.array(
                [
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
            )
            curve.errors.append(errors)
            rows.append(
                {
                    "estimator": name,
                    "payoffs": n,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_error": float(np.mean(errors)),
                }
            )
        curves[name] = curve
    return rows, curves


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


def pathwise_convergence(cfg: Config) -> tuple[list[Row], dict[str, Curve]]:
    """Relative root-mean-square error of the Monte Carlo greeks, by payoff count.

    Returns the table and the raw relative errors of each greek's curve.
    """
    analytic = bs.greeks(BASE).as_dict()
    rows: list[Row] = []
    curves = {
        name: Curve(sizes=[float(n) for n in cfg.greeks_curve_sizes], errors=[])
        for name in bs.GREEK_NAMES
    }
    for size_index, n in enumerate(cfg.greeks_curve_sizes):
        relative: dict[str, list[float]] = {name: [] for name in bs.GREEK_NAMES}
        for rep in range(cfg.mc_replications):
            estimate = pathwise_greeks(
                BASE, n, cfg.seed + 5_000 + 100 * size_index + rep
            ).estimate.as_dict()
            for name in bs.GREEK_NAMES:
                relative[name].append(estimate[name] / analytic[name] - 1.0)
        row: Row = {"payoffs": n}
        for name in bs.GREEK_NAMES:
            errors = np.array(relative[name])
            curves[name].errors.append(errors)
            row[f"{name}_rel_rmse"] = math.sqrt(float(np.sum(np.square(errors))) / errors.size)
        rows.append(row)
    return rows, curves


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


def lsm_convergence(cfg: Config) -> tuple[list[Row], Curve]:
    """Error of least-squares Monte Carlo against the tree, by number of paths.

    Each run fits the exercise rule on ``n`` paths and values the option two
    ways: on those same paths (in sample, the paper's estimator) and on
    ``lsm_oos_paths`` fresh paths (out of sample, a lower bound). Returns the
    table and the raw in-sample errors.
    """
    reference = bermudan_tree(AMERICAN_BASE, cfg.bermudan_steps_per_date)
    n_dates = round(LS_EXERCISE_DATES_PER_YEAR * AMERICAN_BASE.maturity)
    rows: list[Row] = []
    curve = Curve(sizes=[float(n) for n in cfg.lsm_curve_sizes], errors=[])
    for size_index, n in enumerate(cfg.lsm_curve_sizes):
        in_sample: list[float] = []
        out_of_sample: list[float] = []
        for rep in range(cfg.lsm_replications):
            seed = cfg.seed + 9_000 + 100 * size_index + rep
            rule, cash = fit_exercise_rule(AMERICAN_BASE, n_dates, n, seed)
            in_sample.append(max(float(cash.mean()), AMERICAN_BASE.intrinsic()) - reference)
            fresh = lsm_lower_bound(rule, cfg.lsm_oos_paths, seed + 500_000)
            out_of_sample.append(fresh.price - reference)
        errors = np.array(in_sample)
        curve.errors.append(errors)
        rows.append(
            {
                "paths": n,
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_error": float(np.mean(errors)),
                "std_of_estimates": float(np.std(errors, ddof=1)),
                "mean_error_out_of_sample": float(np.mean(out_of_sample)),
                "se_of_mean_out_of_sample": float(np.std(out_of_sample, ddof=1))
                / math.sqrt(len(out_of_sample)),
            }
        )
    return rows, curve


def american_bounds(cfg: Config) -> list[Row]:
    """Lower and upper bounds for some puts of Table 1, beside the tree.

    For each cell the exercise rule is fitted on ``bound_fit_paths`` paths
    set up as in the paper. The lower bound follows that rule on
    ``bound_lower_paths`` fresh paths. The upper bound is the Andersen-Broadie
    dual built from the same rule. The reference is the tree restricted to
    the same 50 exercise dates a year.
    """
    rows: list[Row] = []
    for index, (spot, vol, maturity) in enumerate(cfg.bound_cells):
        option = Option(spot, LS_STRIKE, LS_RATE, vol, maturity, "put")
        n_dates = round(LS_EXERCISE_DATES_PER_YEAR * maturity)
        rule, cash = fit_exercise_rule(
            option, n_dates, cfg.bound_fit_paths, cfg.seed + 20_000 + index
        )
        lower = lsm_lower_bound(rule, cfg.bound_lower_paths, cfg.seed + 21_000 + index)
        upper = andersen_broadie_upper_bound(
            rule, cfg.bound_outer, cfg.bound_inner, cfg.seed + 22_000 + index
        )
        tree = bermudan_tree(option, cfg.bermudan_steps_per_date)
        rows.append(
            {
                "spot": spot,
                "vol": vol,
                "maturity": maturity,
                "tree_50_dates": tree,
                "in_sample": float(cash.mean()),
                "lower": lower.price,
                "lower_se": lower.std_error,
                "upper": upper.price,
                "upper_se": upper.std_error,
                "gap": upper.price - lower.price,
                "tree_minus_lower_in_se": (tree - lower.price) / lower.std_error,
                "upper_minus_tree_in_se": (upper.price - tree) / upper.std_error,
            }
        )
    return rows


def upper_bound_by_inner_paths(cfg: Config) -> list[Row]:
    """Compute the upper bound on the base American put as the inner paths grow.

    The rule and the outer paths are the same at every inner path count, so
    the change in the bound is the effect of the nested estimates' noise.
    """
    n_dates = round(LS_EXERCISE_DATES_PER_YEAR * AMERICAN_BASE.maturity)
    rule, _ = fit_exercise_rule(AMERICAN_BASE, n_dates, cfg.bound_fit_paths, cfg.seed + 23_000)
    tree = bermudan_tree(AMERICAN_BASE, cfg.bermudan_steps_per_date)
    rows: list[Row] = []
    for n_inner in cfg.bound_inner_sweep:
        upper = andersen_broadie_upper_bound(
            rule, cfg.bound_sweep_outer, n_inner, cfg.seed + 24_000
        )
        rows.append(
            {
                "inner_paths": n_inner,
                "outer_paths": cfg.bound_sweep_outer,
                "upper": upper.price,
                "upper_se": upper.std_error,
                "upper_minus_tree": upper.price - tree,
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
    cfg: Config,
    tree_rows: list[Row],
    mc_curves: dict[str, Curve],
    lsm_curve: Curve,
    american_rows: list[Row],
    fd_rows: list[Row],
    greek_curves: dict[str, Curve],
) -> list[Row]:
    """Fitted log-log slopes of every convergence curve, beside the expected value.

    Curves built from repeated simulations get a 95% bootstrap interval for
    the slope. The tree and finite-difference curves are deterministic and
    get none.
    """

    def column(rows: list[Row], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    def deterministic(curve: str, expected: float, x: list[float], y: list[float]) -> Row:
        return {
            "curve": curve,
            "expected": expected,
            "fitted": fitted_slope(x, y),
            "ci_low": "",
            "ci_high": "",
        }

    def simulated(curve: str, data: Curve, seed: int) -> Row:
        rmse = [float(np.sqrt(np.mean(np.square(e)))) for e in data.errors]
        low, high = bootstrap_slope_interval(data, cfg.bootstrap_draws, seed)
        return {
            "curve": curve,
            "expected": -0.5,
            "fitted": fitted_slope(data.sizes, rmse),
            "ci_low": low,
            "ci_high": high,
        }

    seed = cfg.seed + 30_000
    out: list[Row] = [
        deterministic(
            "European tree, error against steps",
            -1.0,
            column(tree_rows, "steps"),
            column(tree_rows, "abs_error"),
        )
    ]
    for index, (name, _, _) in enumerate(ESTIMATORS):
        out.append(
            simulated(f"Monte Carlo ({name}), RMSE against payoffs", mc_curves[name], seed + index)
        )
    out.append(simulated("Least-squares Monte Carlo, RMSE against paths", lsm_curve, seed + 10))
    finite = [r for r in american_rows if float(r["abs_error"]) > 0.0]
    out.append(
        deterministic(
            "American tree, error against steps",
            -1.0,
            column(finite, "steps"),
            column(finite, "abs_error"),
        )
    )
    # Fit the finite-difference slope only where truncation error dominates.
    truncation = [r for r in fd_rows if float(r["rel_bump"]) >= 1e-3]
    out.append(
        deterministic(
            "Finite-difference delta, error against bump (bump >= 1e-3)",
            2.0,
            column(truncation, "rel_bump"),
            column(truncation, "delta_rel_error"),
        )
    )
    for index, name in enumerate(bs.GREEK_NAMES):
        out.append(
            simulated(
                f"Monte Carlo {name}, RMSE against payoffs", greek_curves[name], seed + 20 + index
            )
        )
    return out
