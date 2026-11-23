"""The contract being priced: a plain option on a non-dividend-paying stock."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Kind = Literal["call", "put"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Option:
    """A call or put under constant rate and volatility, with no dividends.

    Attributes:
        spot: current stock price.
        strike: exercise price.
        rate: continuously compounded risk-free rate, per year.
        vol: volatility of log returns, per square-root year.
        maturity: time to expiry in years.
        kind: "call" or "put".
    """

    spot: float
    strike: float
    rate: float
    vol: float
    maturity: float
    kind: Kind = "call"

    def __post_init__(self) -> None:
        """Reject parameters for which the model is not defined."""
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.vol <= 0 or self.maturity <= 0:
            raise ValueError("vol and maturity must be positive")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")

    @property
    def sign(self) -> float:
        """Plus one for a call, minus one for a put."""
        return 1.0 if self.kind == "call" else -1.0

    def payoff(self, prices: FloatArray) -> FloatArray:
        """Exercise value at the given stock prices."""
        out: FloatArray = np.maximum(self.sign * (prices - self.strike), 0.0)
        return out

    def intrinsic(self) -> float:
        """Exercise value today."""
        return max(self.sign * (self.spot - self.strike), 0.0)

    def with_(
        self,
        *,
        spot: float | None = None,
        strike: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
        maturity: float | None = None,
        kind: Kind | None = None,
    ) -> Option:
        """Return a copy with some fields changed."""
        return replace(
            self,
            spot=self.spot if spot is None else spot,
            strike=self.strike if strike is None else strike,
            rate=self.rate if rate is None else rate,
            vol=self.vol if vol is None else vol,
            maturity=self.maturity if maturity is None else maturity,
            kind=self.kind if kind is None else kind,
        )
