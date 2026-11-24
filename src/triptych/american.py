"""American options by least-squares Monte Carlo, with lower and upper bounds.

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

The paper values the option on the same paths that fit the rule. Two further
estimates bracket the true value. Following the fitted rule on fresh paths
gives a lower bound, since no rule beats the optimal one. The duality of
Andersen and Broadie (2004) turns the same rule into an upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True)
class ExerciseRule:
    """An exercise rule fitted by least squares.

    ``coefficients[k]`` holds the regression coefficients at exercise date
    k + 1 (column k of a path array), or None where too few paths were in the
    money to fit, in which case the rule does not exercise there. At an early
    date the rule exercises when the option is in the money and the exercise
    value is at least the fitted continuation value. At expiry it takes the
    payoff.
    """

    option: Option
    n_dates: int
    coefficients: tuple[FloatArray | None, ...]

    @property
    def dt(self) -> float:
        """Time between exercise dates."""
        return self.option.maturity / self.n_dates

    def exercise(self, column: int, prices: FloatArray) -> NDArray[np.bool_]:
        """Where the rule exercises at the date of ``column``, given stock prices there."""
        value = self.option.payoff(prices)
        if column == self.n_dates - 1:
            at_expiry: NDArray[np.bool_] = value > 0.0
            return at_expiry
        decision = np.zeros(prices.shape, dtype=bool)
        coefficients = self.coefficients[column]
        if coefficients is None:
            return decision
        in_money = np.flatnonzero(value > 0.0)
        if in_money.size:
            basis = laguerre_basis(prices[in_money] / self.option.strike)
            decision[in_money] = value[in_money] >= basis @ coefficients
        return decision


def fit_exercise_rule(
    option: Option, n_dates: int, n_paths: int, seed: int, *, antithetic: bool = True
) -> tuple[ExerciseRule, FloatArray]:
    """Fit the Longstaff-Schwartz exercise rule on one set of simulated paths.

    Returns the rule and the discounted cash flow each fitting path receives
    under it. The mean of those cash flows is the paper's in-sample estimate.
    """
    if n_dates < 1:
        raise ValueError("n_dates must be at least 1")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic paths")
    rng = np.random.default_rng(seed)
    paths = simulate_paths(option, n_dates, n_paths, rng, antithetic=antithetic)
    discount = math.exp(-option.rate * option.maturity / n_dates)
    coefficients: list[FloatArray | None] = [None] * n_dates

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
        solution, *_ = np.linalg.lstsq(basis, cash[in_money], rcond=None)
        fitted: FloatArray = np.asarray(solution, dtype=np.float64)
        coefficients[k] = fitted
        continuation = basis @ fitted
        exercised = in_money[exercise_value[in_money] >= continuation]
        cash[exercised] = exercise_value[exercised]
    cash *= discount
    return ExerciseRule(option, n_dates, tuple(coefficients)), cash


def _standard_errors(cash: FloatArray, antithetic: bool) -> tuple[float, float]:
    """Pair-based and all-independent standard errors of the mean of ``cash``."""
    n = cash.size
    naive = float(cash.std(ddof=1)) / math.sqrt(n)
    if not antithetic:
        return naive, naive
    half = n // 2
    pairs = 0.5 * (cash[:half] + cash[half:])
    return float(pairs.std(ddof=1)) / math.sqrt(half), naive


def lsm_price(
    option: Option,
    n_dates: int,
    n_paths: int,
    seed: int,
    *,
    antithetic: bool = True,
    exercise_at_zero: bool = True,
) -> LsmResult:
    """Value an option exercisable at ``n_dates`` equally spaced dates, in sample.

    This is the paper's estimator: the same paths fit the exercise rule and
    value the option, so it is not a bound either way. See
    :func:`lsm_lower_bound` and :func:`andersen_broadie_upper_bound`.

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
    _, cash = fit_exercise_rule(option, n_dates, n_paths, seed, antithetic=antithetic)
    estimate = float(cash.mean())
    std_error, naive = _standard_errors(cash, antithetic)
    if exercise_at_zero:
        estimate = max(estimate, option.intrinsic())
    return LsmResult(price=estimate, std_error=std_error, std_error_naive=naive, n_paths=n_paths)


def follow_rule(rule: ExerciseRule, paths: FloatArray) -> FloatArray:
    """Discounted cash flow to time zero of each path when the rule is followed."""
    discount = math.exp(-rule.option.rate * rule.dt)
    cash = np.zeros(paths.shape[0])
    alive = np.arange(paths.shape[0])
    for k in range(rule.n_dates):
        stop = rule.exercise(k, paths[alive, k])
        stopped = alive[stop]
        cash[stopped] = rule.option.payoff(paths[stopped, k]) * discount ** (k + 1)
        alive = alive[~stop]
    return cash


def lsm_lower_bound(
    rule: ExerciseRule,
    n_paths: int,
    seed: int,
    *,
    antithetic: bool = True,
    chunk: int = 100_000,
) -> LsmResult:
    """Value of following a fitted rule, measured on paths it was not fitted to.

    No rule beats the optimal one, so the expectation of this estimate is at
    most the true value. The seed must differ from the one used to fit the
    rule. Exercise today is not allowed.
    """
    if antithetic and (n_paths % 2 or chunk % 2):
        raise ValueError("n_paths and chunk must be even with antithetic paths")
    rng = np.random.default_rng(seed)
    # Paths are simulated in chunks to bound memory. With antithetic paths
    # each chunk mirrors its own first half, and the pair averages are the
    # independent samples.
    cash_parts: list[FloatArray] = []
    sample_parts: list[FloatArray] = []
    for start in range(0, n_paths, chunk):
        size = min(chunk, n_paths - start)
        paths = simulate_paths(rule.option, rule.n_dates, size, rng, antithetic=antithetic)
        cash = follow_rule(rule, paths)
        cash_parts.append(cash)
        half = size // 2
        sample_parts.append(0.5 * (cash[:half] + cash[half:]) if antithetic else cash)
    cash = np.concatenate(cash_parts)
    samples = np.concatenate(sample_parts)
    return LsmResult(
        price=float(cash.mean()),
        std_error=float(samples.std(ddof=1)) / math.sqrt(samples.size),
        std_error_naive=float(cash.std(ddof=1)) / math.sqrt(n_paths),
        n_paths=n_paths,
    )


@dataclass(frozen=True)
class UpperBound:
    """An Andersen-Broadie duality estimate.

    Attributes:
        price: the upper-bound estimate.
        std_error: standard error over the outer paths.
        n_outer: outer paths.
        n_inner: inner paths per continuation value.
    """

    price: float
    std_error: float
    n_outer: int
    n_inner: int


def continuation_values(
    rule: ExerciseRule,
    start_prices: FloatArray,
    first_column: int,
    n_inner: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Value of following the rule from the next exercise date on, one per start price.

    ``first_column`` is the column of the next exercise date; the start
    prices are at the date before it (today when ``first_column`` is 0).
    Each value is the mean over ``n_inner`` paths, half of them the mirror
    images of the other half, discounted to the start date.
    """
    if n_inner % 2:
        raise ValueError("n_inner must be even")
    option = rule.option
    dt = rule.dt
    drift = (option.rate - 0.5 * option.vol**2) * dt
    shock = option.vol * math.sqrt(dt)
    discount = math.exp(-option.rate * dt)
    prices = np.repeat(start_prices, n_inner)
    cash = np.zeros(prices.size)
    alive = np.arange(prices.size)
    for k in range(first_column, rule.n_dates):
        half = rng.standard_normal((start_prices.size, n_inner // 2))
        z = np.concatenate([half, -half], axis=1).ravel()
        moved = prices[alive] * np.exp(drift + shock * z[alive])
        prices[alive] = moved
        stop = rule.exercise(k, moved)
        stopped = alive[stop]
        cash[stopped] = option.payoff(moved[stop]) * discount ** (k - first_column + 1)
        alive = alive[~stop]
    out: FloatArray = cash.reshape(start_prices.size, n_inner).mean(axis=1)
    return out


def andersen_broadie_upper_bound(
    rule: ExerciseRule,
    n_outer: int,
    n_inner: int,
    seed: int,
    *,
    exercise_at_zero: bool = True,
    chunk: int = 50,
) -> UpperBound:
    """Upper bound on the option value by the duality of Andersen and Broadie (2004).

    The martingale is built from the value of following the fitted rule.
    With dates i = 0 (today) to n, let Q_i be the value at date i of
    continuing and following the rule from date i + 1, and L_i the value of
    following the rule from date i itself: the exercise value where the rule
    stops, otherwise Q_i. With everything discounted to today,

        M_0 = 0,   M_i = M_(i-1) + L_i - Q_(i-1),

    is a martingale, because Q_(i-1) is the expected value of L_i. For any
    martingale that starts at zero the option is worth at most
    E[max_i (h_i - M_i)], where h_i is the discounted exercise value.

    Exercising out of the money pays nothing and is never better than
    holding, so the maximum may be taken over today, the in-the-money dates
    and expiry only. At an out-of-the-money date before expiry the rule
    continues, L_i = Q_i, and Q_i cancels from the next increment, so it is
    never needed. Every other Q is estimated by a nested simulation of
    ``n_inner`` paths that follows the rule. The noise in those estimates
    raises the bound in expectation.

    Args:
        rule: the fitted exercise rule.
        n_outer: outer paths.
        n_inner: inner paths for each continuation value.
        seed: seed for the outer and inner paths; use one not used to fit the rule.
        exercise_at_zero: include exercise today in the maximum, as the tree does.
        chunk: outer paths processed together, to bound memory.
    """
    option = rule.option
    n = rule.n_dates
    outer_seq, inner_seq = np.random.SeedSequence(seed).spawn(2)
    outer = simulate_paths(option, n, n_outer, np.random.default_rng(outer_seq), antithetic=False)
    inner_rng = np.random.default_rng(inner_seq)
    discount = math.exp(-option.rate * rule.dt)
    samples = np.empty(n_outer)
    for start in range(0, n_outer, chunk):
        paths = outer[start : start + chunk]
        m = paths.shape[0]
        # Running state per path: the martingale and Q at the last date where
        # they were computed. Today: M_0 = 0 and Q_0 from one nested run.
        q_last = continuation_values(rule, np.full(m, option.spot), 0, n_inner, inner_rng)
        m_last = np.zeros(m)
        best = np.full(m, option.intrinsic() if exercise_at_zero else -np.inf)
        for k in range(n):
            # Date i = k + 1, stock prices in column k.
            h = option.payoff(paths[:, k]) * discount ** (k + 1)
            if k == n - 1:
                rows = np.arange(m)
                value = h
            else:
                # Out of the money the rule continues, so L_i = Q_i and the
                # next increment subtracts Q_i again: it cancels and is never
                # estimated. Only in-the-money dates enter the maximum.
                rows = np.flatnonzero(h > 0.0)
                if rows.size == 0:
                    continue
                q_now = discount ** (k + 1) * continuation_values(
                    rule, paths[rows, k], k + 1, n_inner, inner_rng
                )
                stop = rule.exercise(k, paths[rows, k])
                value = np.where(stop, h[rows], q_now)
            m_now = m_last[rows] + value - q_last[rows]
            best[rows] = np.maximum(best[rows], h[rows] - m_now)
            if k < n - 1:
                m_last[rows] = m_now
                q_last[rows] = q_now
        samples[start : start + m] = best
    return UpperBound(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1)) / math.sqrt(n_outer),
        n_outer=n_outer,
        n_inner=n_inner,
    )
