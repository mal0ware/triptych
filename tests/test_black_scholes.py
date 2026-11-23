"""Closed-form checks: parity, limits, the differential equation, published values."""

import itertools
import math
import sys

import pytest

from triptych import black_scholes as bs
from triptych.option import Kind, Option
from triptych.published import (
    CRR_MONTHS,
    CRR_ONE_PLUS_RATE,
    CRR_SPOT,
    CRR_TABLE_2,
    LS_RATE,
    LS_STRIKE,
    LS_TABLE_1,
)

KINDS: tuple[Kind, ...] = ("call", "put")

GRID = [
    Option(spot, strike, rate, vol, maturity)
    for spot, strike, rate, vol, maturity in itertools.product(
        (5.0, 36.0, 40.0, 44.0, 250.0),
        (40.0, 100.0),
        (0.0, 0.06, 0.15),
        (0.05, 0.2, 0.4, 1.0),
        (1 / 52, 0.5, 1.0, 5.0),
    )
]


@pytest.mark.parametrize("option", GRID)
def test_put_call_parity_to_machine_precision(option: Option) -> None:
    # Black and Scholes (1973), equation (25): w - u = x - c exp(r(t - t*)).
    call = bs.price(option)
    put = bs.price(option.with_(kind="put"))
    forward = option.spot - option.strike * math.exp(-option.rate * option.maturity)
    scale = option.spot + option.strike
    assert abs(call - put - forward) <= 4 * sys.float_info.epsilon * scale


@pytest.mark.parametrize("option", GRID)
def test_price_within_no_arbitrage_bounds(option: Option) -> None:
    call = bs.price(option)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    assert max(option.spot - discounted_strike, 0.0) - 1e-12 <= call <= option.spot
    put = bs.price(option.with_(kind="put"))
    assert max(discounted_strike - option.spot, 0.0) - 1e-12 <= put <= discounted_strike


def test_small_volatility_limit_is_discounted_forward_payoff() -> None:
    option = Option(40.0, 36.0, 0.06, 1e-6, 1.0)
    assert bs.price(option) == pytest.approx(40.0 - 36.0 * math.exp(-0.06), abs=1e-12)
    assert bs.price(option.with_(kind="put")) == pytest.approx(0.0, abs=1e-12)


def test_matches_european_column_of_longstaff_schwartz_table_1() -> None:
    for row in LS_TABLE_1:
        option = Option(row.spot, LS_STRIKE, LS_RATE, row.vol, row.maturity, "put")
        assert abs(bs.price(option) - row.european) <= 0.0005 + 1e-9


def test_matches_black_scholes_column_of_crr_table_2() -> None:
    rate = math.log(CRR_ONE_PLUS_RATE)
    for (vol, strike), columns in CRR_TABLE_2.items():
        for months, printed in zip(CRR_MONTHS, columns["inf"], strict=True):
            option = Option(CRR_SPOT, strike, rate, vol, months / 12)
            assert abs(bs.price(option) - printed) <= 0.005 + 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_greeks_satisfy_the_black_scholes_equation(kind: Kind) -> None:
    # Equation (7): w2 = r w - r x w1 - (1/2) v^2 x^2 w11.
    for base in GRID:
        option = base.with_(kind=kind)
        g = bs.greeks(option)
        value = bs.price(option)
        residual = (
            g.theta
            + option.rate * option.spot * g.delta
            + 0.5 * option.vol**2 * option.spot**2 * g.gamma
            - option.rate * value
        )
        assert abs(residual) <= 1e-10 * (option.spot + option.strike)


def test_call_delta_is_n_d1() -> None:
    # Equation (14).
    option = Option(40.0, 40.0, 0.06, 0.2, 1.0)
    d1, _ = bs.d1_d2(option)
    assert bs.greeks(option).delta == bs.norm_cdf(d1)


def test_call_and_put_greeks_differ_by_parity() -> None:
    option = Option(42.0, 40.0, 0.06, 0.3, 0.75)
    call, put = bs.greeks(option), bs.greeks(option.with_(kind="put"))
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-14)
    assert call.gamma == put.gamma
    assert call.vega == put.vega
    assert call.rho - put.rho == pytest.approx(option.maturity * discounted_strike, abs=1e-12)
    assert call.theta - put.theta == pytest.approx(-option.rate * discounted_strike, abs=1e-12)


def test_vega_equals_spot_squared_vol_time_gamma() -> None:
    option = Option(38.0, 40.0, 0.06, 0.4, 2.0, "put")
    g = bs.greeks(option)
    assert g.vega == pytest.approx(option.spot**2 * option.vol * option.maturity * g.gamma)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        Option(-1.0, 40.0, 0.06, 0.2, 1.0)
    with pytest.raises(ValueError):
        Option(40.0, 40.0, 0.06, 0.0, 1.0)
    with pytest.raises(ValueError):
        Option(40.0, 40.0, 0.06, 0.2, 1.0, "straddle")  # type: ignore[arg-type]
