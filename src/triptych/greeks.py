"""Greeks by finite difference and by pathwise Monte Carlo.

The analytic greeks live in :mod:`triptych.black_scholes`. This module gives
the two numerical routes they are checked against.

Pathwise estimators differentiate the discounted payoff along each path
(Broadie and Glasserman, 1996). With w = +1 for a call and -1 for a put, and
I the indicator that the option finishes in the money:

    delta = exp(-rT) w I S_T / S_0
    vega  = exp(-rT) w I S_T (sqrt(T) Z - sigma T)
    rho   = T exp(-rT) (w I S_T - payoff)
    theta = exp(-rT) (r payoff - w I S_T (r - sigma^2/2 + sigma Z / (2 sqrt(T))))

The payoff has a kink, so its second derivative along a path is zero almost
everywhere and the pathwise method gives no gamma. Gamma here applies the
likelihood-ratio score Z / (S_0 sigma sqrt(T)) to the pathwise delta:

    gamma = exp(-rT) w I (S_T / S_0^2) (Z / (sigma sqrt(T)) - 1)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from triptych.black_scholes import GREEK_NAMES, Greeks
from triptych.monte_carlo import terminal_prices
from triptych.option import FloatArray, Option

Pricer = Callable[[Option], float]


def finite_difference_greeks(pricer: Pricer, option: Option, rel_bump: float = 1e-3) -> Greeks:
    """Greeks of any pricing function by central differences.

    Each input is bumped up and down by ``rel_bump`` times its own value. The
    truncation error is of order ``rel_bump ** 2``.
    """
    h_s = option.spot * rel_bump
    h_v = option.vol * rel_bump
    h_t = option.maturity * rel_bump
    h_r = max(abs(option.rate), 1e-2) * rel_bump

    base = pricer(option)
    up_s = pricer(option.with_(spot=option.spot + h_s))
    dn_s = pricer(option.with_(spot=option.spot - h_s))
    up_v = pricer(option.with_(vol=option.vol + h_v))
    dn_v = pricer(option.with_(vol=option.vol - h_v))
    up_t = pricer(option.with_(maturity=option.maturity + h_t))
    dn_t = pricer(option.with_(maturity=option.maturity - h_t))
    up_r = pricer(option.with_(rate=option.rate + h_r))
    dn_r = pricer(option.with_(rate=option.rate - h_r))
    return Greeks(
        delta=(up_s - dn_s) / (2.0 * h_s),
        gamma=(up_s - 2.0 * base + dn_s) / h_s**2,
        vega=(up_v - dn_v) / (2.0 * h_v),
        theta=-(up_t - dn_t) / (2.0 * h_t),
        rho=(up_r - dn_r) / (2.0 * h_r),
    )


@dataclass(frozen=True)
class McGreeks:
    """Monte Carlo greeks with their standard errors."""

    estimate: Greeks
    std_error: Greeks
    n_payoffs: int


def _path_samples(option: Option, z: FloatArray) -> FloatArray:
    """Per-path greek estimators, one row per greek in ``GREEK_NAMES`` order."""
    t = option.maturity
    root_t = math.sqrt(t)
    discount = math.exp(-option.rate * t)
    s_t = terminal_prices(option, z)
    payoff = option.payoff(s_t)
    # w * I * S_T: the derivative of the payoff with respect to log S_T.
    slope = option.sign * np.where(payoff > 0.0, s_t, 0.0)
    drift_rate = option.rate - 0.5 * option.vol**2
    out: FloatArray = np.stack(
        [
            discount * slope / option.spot,
            discount * slope / option.spot**2 * (z / (option.vol * root_t) - 1.0),
            discount * slope * (root_t * z - option.vol * t),
            discount
            * (option.rate * payoff - slope * (drift_rate + option.vol * z / (2.0 * root_t))),
            t * discount * (slope - payoff),
        ]
    )
    return out


def pathwise_greeks(
    option: Option, n_payoffs: int, seed: int, *, antithetic: bool = True
) -> McGreeks:
    """Estimate all five greeks from one set of simulated terminal prices.

    Delta, vega, theta and rho are pathwise estimators. Gamma is the
    likelihood-ratio derivative of the pathwise delta (see the module
    docstring).
    """
    if antithetic and n_payoffs % 2:
        raise ValueError("n_payoffs must be even with antithetic paths")
    rng = np.random.default_rng(seed)
    n_samples = n_payoffs // 2 if antithetic else n_payoffs
    z = rng.standard_normal(n_samples)
    samples = _path_samples(option, z)
    if antithetic:
        samples = 0.5 * (samples + _path_samples(option, -z))
    means = samples.mean(axis=1)
    errors = samples.std(axis=1, ddof=1) / math.sqrt(n_samples)
    assert len(GREEK_NAMES) == samples.shape[0]
    return McGreeks(
        estimate=Greeks(*(float(v) for v in means)),
        std_error=Greeks(*(float(v) for v in errors)),
        n_payoffs=n_payoffs,
    )
