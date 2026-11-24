"""Cox-Ross-Rubinstein binomial tree.

Cox, Ross and Rubinstein (1979): one-period value equation (3) with
p = (r - d)/(u - d); n-period formula equation (6); the parameter choice
u = exp(sigma sqrt(t/n)), d = 1/u and the early-exercise comparison for puts
are on p. 258. In the paper r is one plus the rate per period; here that is
exp(rate * dt).

Two refinements that are not in the paper. The smoothed tree replaces the
values one period before expiry by the Black-Scholes value over that last
period, which removes the kink of the payoff from the tree. Richardson
extrapolation then combines trees of n and 2n steps, 2 V(2n) - V(n), to
cancel the part of the error that falls as 1/n. Together these are the
binomial Black-Scholes method with Richardson extrapolation of Broadie and
Detemple (1996).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.special import gammaln, ndtr

from triptych.option import FloatArray, Option

Exercise = Literal["european", "american"]


def crr_parameters(option: Option, steps: int) -> tuple[float, float, float, float]:
    """Return (u, d, p, one-period discount factor) for a tree with ``steps`` periods."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = option.maturity / steps
    u = math.exp(option.vol * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(option.rate * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            "risk-neutral probability outside (0, 1): the tree needs "
            "|rate| * sqrt(dt) < vol; use more steps"
        )
    return u, d, p, 1.0 / growth


def black_scholes_values(option: Option, spots: FloatArray, tau: float) -> FloatArray:
    """European values at an array of spot prices with ``tau`` years to expiry."""
    root = option.vol * math.sqrt(tau)
    d1 = (np.log(spots / option.strike) + (option.rate + 0.5 * option.vol**2) * tau) / root
    d2 = d1 - root
    discounted_strike = option.strike * math.exp(-option.rate * tau)
    w = option.sign
    out: FloatArray = w * (spots * ndtr(w * d1) - discounted_strike * ndtr(w * d2))
    return out


def crr_price(
    option: Option,
    steps: int,
    exercise: Exercise = "european",
    exercise_every: int = 1,
    *,
    smooth: bool = False,
) -> float:
    """Value an option by backward induction on a CRR tree.

    Args:
        option: the contract.
        steps: number of periods in the tree.
        exercise: "european" for exercise at expiry only, "american" to allow
            early exercise.
        exercise_every: with "american", early exercise is allowed only at tree
            dates whose index is a multiple of this number (and at time zero).
            The default of 1 is the usual American tree. A tree with
            ``steps = m * exercise_every`` values a Bermudan option with ``m``
            equally spaced exercise dates.
        smooth: start the backward induction one period before expiry from
            the Black-Scholes value over the last period (with early exercise
            there if it is allowed), instead of from the payoff.

    Returns:
        The option value today.
    """
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    if exercise_every < 1:
        raise ValueError("exercise_every must be at least 1")
    u, _, p, discount = crr_parameters(option, steps)
    log_u = math.log(u)

    def node_prices(i: int) -> FloatArray:
        j = np.arange(i + 1, dtype=np.float64)
        out: FloatArray = option.spot * np.exp((2.0 * j - i) * log_u)
        return out

    if smooth:
        last = steps - 1
        values = black_scholes_values(option, node_prices(last), option.maturity / steps)
        if exercise == "american" and last % exercise_every == 0:
            values = np.maximum(values, option.payoff(node_prices(last)))
    else:
        last = steps
        values = option.payoff(node_prices(steps))
    for i in range(last - 1, -1, -1):
        values = discount * (p * values[1 : i + 2] + (1.0 - p) * values[: i + 1])
        if exercise == "american" and i % exercise_every == 0:
            values = np.maximum(values, option.payoff(node_prices(i)))
    return float(values[0])


def crr_european_sum(option: Option, steps: int) -> float:
    """European value from the n-period sum, CRR equation (6).

    Equal to ``crr_price(option, steps)`` up to rounding, at O(n) cost rather
    than O(n^2). Used to scan many step counts.
    """
    u, _, p, discount = crr_parameters(option, steps)
    j = np.arange(steps + 1, dtype=np.float64)
    log_pmf = (
        gammaln(steps + 1.0)
        - gammaln(j + 1.0)
        - gammaln(steps - j + 1.0)
        + j * math.log(p)
        + (steps - j) * math.log1p(-p)
    )
    terminal = option.spot * np.exp((2.0 * j - steps) * math.log(u))
    return float(discount**steps * np.sum(np.exp(log_pmf) * option.payoff(terminal)))


def smoothed_european_sum(option: Option, steps: int) -> float:
    """European value of the smoothed tree as a sum, at O(n) cost.

    Equal to ``crr_price(option, steps, smooth=True)`` up to rounding.
    """
    u, _, p, discount = crr_parameters(option, steps)
    m = steps - 1
    j = np.arange(m + 1, dtype=np.float64)
    log_pmf = (
        gammaln(m + 1.0)
        - gammaln(j + 1.0)
        - gammaln(m - j + 1.0)
        + j * math.log(p)
        + (m - j) * math.log1p(-p)
    )
    nodes = option.spot * np.exp((2.0 * j - m) * math.log(u))
    values = black_scholes_values(option, nodes, option.maturity / steps)
    return float(discount**m * np.sum(np.exp(log_pmf) * values))


def richardson(coarse: float, fine: float) -> float:
    """Extrapolate trees of n and 2n steps whose error falls as 1/n."""
    return 2.0 * fine - coarse
