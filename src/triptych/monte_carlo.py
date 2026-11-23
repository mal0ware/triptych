"""Monte Carlo for European options, with antithetic paths and a control variate.

Under the risk-neutral measure the terminal price is

    S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),

and the option value is the mean of exp(-rT) * payoff(S_T).

Antithetic paths pair each draw Z with -Z and average the two payoffs.
The control variate is the discounted terminal stock price, whose mean is
known to be S_0 exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from triptych.option import FloatArray, Option


def terminal_prices(option: Option, z: FloatArray) -> FloatArray:
    """Terminal stock prices for standard normal draws ``z``."""
    drift = (option.rate - 0.5 * option.vol**2) * option.maturity
    out: FloatArray = option.spot * np.exp(drift + option.vol * math.sqrt(option.maturity) * z)
    return out


@dataclass(frozen=True)
class McResult:
    """A Monte Carlo estimate and its sampling error.

    Attributes:
        price: the estimate.
        std_error: standard error of the estimate.
        n_payoffs: number of payoff evaluations used.
        beta: control-variate coefficient used, or 0.0 without a control.
    """

    price: float
    std_error: float
    n_payoffs: int
    beta: float = 0.0

    @property
    def variance_per_payoff(self) -> float:
        """Variance of the estimate times the number of payoff evaluations.

        This is the work-normalised variance. For plain Monte Carlo it is the
        variance of one discounted payoff.
        """
        return self.std_error**2 * self.n_payoffs


def variance_reduction_factor(plain: McResult, other: McResult) -> float:
    """How many times fewer payoff evaluations ``other`` needs for the same error."""
    return plain.variance_per_payoff / other.variance_per_payoff


def european_mc(
    option: Option,
    n_payoffs: int,
    seed: int,
    *,
    antithetic: bool = False,
    control_variate: bool = False,
    beta: float | None = None,
    chunk: int = 2_000_000,
) -> McResult:
    """Estimate a European option value by simulating the terminal price.

    Args:
        option: the contract.
        n_payoffs: number of payoff evaluations. With ``antithetic`` it must be
            even: half as many normal draws are made, each used twice.
        seed: seed for numpy's PCG64 generator.
        antithetic: pair each draw with its negative.
        control_variate: subtract ``beta * (exp(-rT) S_T - S_0)`` from each sample.
        beta: fixed control coefficient. If None it is estimated from the same
            sample by least squares, which leaves a bias of order 1/n.
        chunk: draws generated per batch, to bound memory.

    Returns:
        The estimate with its standard error.
    """
    if n_payoffs < 2:
        raise ValueError("n_payoffs must be at least 2")
    if antithetic and n_payoffs % 2:
        raise ValueError("n_payoffs must be even with antithetic paths")
    rng = np.random.default_rng(seed)
    discount = math.exp(-option.rate * option.maturity)
    n_samples = n_payoffs // 2 if antithetic else n_payoffs

    # Running sums of y, c, y^2, c^2 and y*c, where y is one independent
    # sample of the discounted payoff and c the matching mean-zero control.
    sums = np.zeros(5)
    remaining = n_samples
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        z = rng.standard_normal(m)
        s_t = terminal_prices(option, z)
        y = discount * option.payoff(s_t)
        c = discount * s_t - option.spot
        if antithetic:
            s_t_anti = terminal_prices(option, -z)
            y = 0.5 * (y + discount * option.payoff(s_t_anti))
            c = 0.5 * (c + discount * s_t_anti - option.spot)
        sums += (y.sum(), c.sum(), np.dot(y, y), np.dot(c, c), np.dot(y, c))

    mean_y, mean_c, mean_yy, mean_cc, mean_yc = (float(v) for v in sums / n_samples)
    bessel = n_samples / (n_samples - 1)
    var_y = (mean_yy - mean_y**2) * bessel
    var_c = (mean_cc - mean_c**2) * bessel
    cov_yc = (mean_yc - mean_y * mean_c) * bessel

    used_beta = 0.0
    if control_variate:
        used_beta = cov_yc / var_c if beta is None else beta
    estimate = mean_y - used_beta * mean_c
    variance = var_y - 2.0 * used_beta * cov_yc + used_beta**2 * var_c
    return McResult(
        price=estimate,
        std_error=math.sqrt(max(variance, 0.0) / n_samples),
        n_payoffs=n_payoffs,
        beta=used_beta,
    )
