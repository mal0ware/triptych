"""American options by least-squares Monte Carlo.

Longstaff and Schwartz (2001), section 2.2. Working backwards from expiry, at
each exercise date the discounted cash flows that the paths go on to receive
are regressed on functions of the current stock price, using in-the-money
paths only. The fitted value estimates the value of continuing, equation (1).
A path is exercised when its immediate exercise value is at least the fitted
continuation value. The option value is the mean over paths of the cash flow
at the first exercise, discounted to today.

The regressors are a constant and the first three weighted Laguerre
polynomials, equations (2) to (4), evaluated at the stock price divided by the
strike (the renormalisation the paper describes in section 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from triptych.option import FloatArray, Option


def laguerre_basis(x: FloatArray) -> FloatArray:
    """Regressors of the paper's section 3: a constant and L0, L1, L2.

    L0(X) = exp(-X/2), L1(X) = exp(-X/2)(1 - X),
    L2(X) = exp(-X/2)(1 - 2X + X^2/2): equations (2), (3) and (4).
    """
    weight = np.exp(-0.5 * x)
    out: FloatArray = np.column_stack(
        [np.ones_like(x), weight, weight * (1.0 - x), weight * (1.0 - 2.0 * x + 0.5 * x * x)]
    )
    return out


def simulate_paths(
    option: Option, n_dates: int, n_paths: int, rng: np.random.Generator, *, antithetic: bool
) -> FloatArray:
    """Simulate geometric Brownian motion at ``n_dates`` equally spaced dates.

    Returns an array of shape (n_paths, n_dates); column k is the price at
    time (k + 1) * maturity / n_dates. With ``antithetic`` the second half of
    the rows mirrors the first: row i + n_paths/2 uses the negated draws of
    row i.
    """
    dt = option.maturity / n_dates
    n_draws = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((n_draws, n_dates))
    if antithetic:
        z = np.concatenate([z, -z])
    increments = (option.rate - 0.5 * option.vol**2) * dt + option.vol * math.sqrt(dt) * z
    out: FloatArray = option.spot * np.exp(np.cumsum(increments, axis=1))
    return out


@dataclass(frozen=True)
class LsmResult:
    """A least-squares Monte Carlo estimate.

    Attributes:
        price: the estimate.
        std_error: standard error. With antithetic paths it is computed from
            the averages of each mirrored pair, which are the independent
            samples.
        std_error_naive: standard deviation of the discounted cash flows
            divided by sqrt(n_paths), treating every path as independent.
        n_paths: number of simulated paths.
    """

    price: float
    std_error: float
    std_error_naive: float
    n_paths: int


def lsm_price(
    option: Option,
    n_dates: int,
    n_paths: int,
    seed: int,
    *,
    antithetic: bool = True,
    exercise_at_zero: bool = True,
) -> LsmResult:
    """Value an option exercisable at ``n_dates`` equally spaced dates.

    Args:
        option: the contract.
        n_dates: number of exercise dates, the last being expiry.
        n_paths: number of simulated paths (even if ``antithetic``).
        seed: seed for numpy's PCG64 generator.
        antithetic: generate half the paths and mirror them, as the paper does.
        exercise_at_zero: also allow exercise today, so the result is never
            below intrinsic value. The paper's exercise dates start after
            time zero.

    Returns:
        The estimate with its standard errors.
    """
    if n_dates < 1:
        raise ValueError("n_dates must be at least 1")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic paths")
    rng = np.random.default_rng(seed)
    paths = simulate_paths(option, n_dates, n_paths, rng, antithetic=antithetic)
    discount = math.exp(-option.rate * option.maturity / n_dates)

    # cash[i] is the cash flow path i will receive, discounted to the date
    # being processed.
    cash = option.payoff(paths[:, -1])
    for k in range(n_dates - 2, -1, -1):
        cash *= discount
        exercise_value = option.payoff(paths[:, k])
        in_money = np.flatnonzero(exercise_value > 0.0)
        if in_money.size <= 4:
            continue
        basis = laguerre_basis(paths[in_money, k] / option.strike)
        coefficients, *_ = np.linalg.lstsq(basis, cash[in_money], rcond=None)
        continuation = basis @ coefficients
        exercised = in_money[exercise_value[in_money] >= continuation]
        cash[exercised] = exercise_value[exercised]
    cash *= discount

    estimate = float(cash.mean())
    naive = float(cash.std(ddof=1)) / math.sqrt(n_paths)
    if antithetic:
        half = n_paths // 2
        pairs = 0.5 * (cash[:half] + cash[half:])
        std_error = float(pairs.std(ddof=1)) / math.sqrt(half)
    else:
        std_error = naive
    if exercise_at_zero:
        estimate = max(estimate, option.intrinsic())
    return LsmResult(price=estimate, std_error=std_error, std_error_naive=naive, n_paths=n_paths)
