"""Black-Scholes closed form for European options, and its greeks.

Formulas follow Black and Scholes (1973): the call is equation (13), the put
equation (27), and the hedge ratio N(d1) equation (14). The other greeks are
derivatives of (13) and (27); see PAPER.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from triptych.option import Option

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Return the standard normal distribution function, accurate in both tails."""
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def d1_d2(option: Option) -> tuple[float, float]:
    """Return d1 and d2 of equation (13)."""
    root_t = math.sqrt(option.maturity)
    d1 = (
        math.log(option.spot / option.strike)
        + (option.rate + 0.5 * option.vol**2) * option.maturity
    ) / (option.vol * root_t)
    return d1, d1 - option.vol * root_t


def price(option: Option) -> float:
    """European option value: equation (13) for a call, (27) for a put."""
    d1, d2 = d1_d2(option)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    if option.kind == "call":
        return option.spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - option.spot * norm_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities of an option value.

    Attributes:
        delta: derivative with respect to spot.
        gamma: second derivative with respect to spot.
        vega: derivative with respect to volatility (per unit, not per point).
        theta: derivative with respect to calendar time, so minus the
            derivative with respect to time to expiry. Per year.
        rho: derivative with respect to the rate (per unit, not per point).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        """Return the greeks keyed by name, in a fixed order."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


GREEK_NAMES: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")


def greeks(option: Option) -> Greeks:
    """Analytic greeks of the Black-Scholes value."""
    d1, d2 = d1_d2(option)
    root_t = math.sqrt(option.maturity)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    sign = option.sign
    decay = -option.spot * norm_pdf(d1) * option.vol / (2.0 * root_t)
    return Greeks(
        delta=sign * norm_cdf(sign * d1),
        gamma=norm_pdf(d1) / (option.spot * option.vol * root_t),
        vega=option.spot * norm_pdf(d1) * root_t,
        theta=decay - sign * option.rate * discounted_strike * norm_cdf(sign * d2),
        rho=sign * option.maturity * discounted_strike * norm_cdf(sign * d2),
    )
