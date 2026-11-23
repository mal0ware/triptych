"""Binomial tree checks: published values, convergence, early exercise."""

import itertools
import math

import pytest

from triptych import black_scholes as bs
from triptych.option import Kind, Option
from triptych.published import CRR_MONTHS, CRR_ONE_PLUS_RATE, CRR_SPOT, CRR_TABLE_2
from triptych.tree import crr_european_sum, crr_parameters, crr_price

PUT = Option(40.0, 40.0, 0.06, 0.2, 1.0, "put")
KINDS: tuple[Kind, ...] = ("call", "put")

PUT_GRID = [
    Option(spot, 40.0, rate, vol, maturity, "put")
    for spot, rate, vol, maturity in itertools.product(
        (20.0, 36.0, 40.0, 44.0, 60.0), (0.0, 0.06, 0.12), (0.2, 0.4), (0.25, 1.0, 2.0)
    )
]


def test_matches_crr_table_2_for_5_and_20_steps() -> None:
    rate = math.log(CRR_ONE_PLUS_RATE)
    for (vol, strike), columns in CRR_TABLE_2.items():
        for steps in (5, 20):
            for months, printed in zip(CRR_MONTHS, columns[str(steps)], strict=True):
                option = Option(CRR_SPOT, strike, rate, vol, months / 12)
                assert abs(crr_price(option, steps) - printed) <= 0.005 + 1e-9


def test_one_step_tree_is_crr_equation_3() -> None:
    u, d, p, discount = crr_parameters(PUT, 1)
    by_hand = discount * (
        p * max(PUT.strike - PUT.spot * u, 0.0) + (1 - p) * max(PUT.strike - PUT.spot * d, 0.0)
    )
    assert crr_price(PUT, 1) == pytest.approx(by_hand, rel=1e-14)
    assert p == pytest.approx((math.exp(PUT.rate * PUT.maturity) - d) / (u - d))


@pytest.mark.parametrize("steps", [1, 2, 7, 50, 501])
@pytest.mark.parametrize("kind", KINDS)
def test_backward_induction_equals_the_n_period_sum(steps: int, kind: Kind) -> None:
    option = PUT.with_(kind=kind)
    assert crr_price(option, steps) == pytest.approx(crr_european_sum(option, steps), rel=1e-11)


def test_tree_converges_to_closed_form_at_rate_one_over_n() -> None:
    exact = bs.price(PUT)
    # Even step counts only: for an at-the-money option the error alternates
    # in sign between even and odd counts, with a different constant for each.
    errors = {n: abs(crr_price(PUT, n) - exact) for n in (128, 256, 512, 1024, 2048, 4096)}
    assert errors[4096] < 2.5e-4
    for n in (128, 256, 512, 1024, 2048):
        # Doubling the steps should halve the error.
        assert errors[2 * n] / errors[n] == pytest.approx(0.5, abs=0.01)


def test_tree_error_alternates_in_sign_at_the_money() -> None:
    exact = bs.price(PUT)
    assert crr_price(PUT, 1000) < exact < crr_price(PUT, 1001)


@pytest.mark.parametrize("option", PUT_GRID)
def test_tree_is_close_to_closed_form(option: Option) -> None:
    assert crr_price(option, 2000) == pytest.approx(bs.price(option), abs=2e-3)


@pytest.mark.parametrize("option", PUT_GRID)
def test_american_put_never_below_european_or_intrinsic(option: Option) -> None:
    american = crr_price(option, 300, "american")
    assert american >= crr_price(option, 300, "european") - 1e-12
    assert american >= option.intrinsic()


@pytest.mark.parametrize("option", PUT_GRID)
def test_bermudan_lies_between_european_and_american(option: Option) -> None:
    european = crr_price(option, 300)
    bermudan = crr_price(option, 300, "american", exercise_every=30)
    american = crr_price(option, 300, "american")
    assert european - 1e-12 <= bermudan <= american + 1e-12


def test_american_call_without_dividends_equals_european_call() -> None:
    call = Option(40.0, 40.0, 0.06, 0.2, 1.0, "call")
    assert crr_price(call, 500, "american") == pytest.approx(crr_price(call, 500), rel=1e-13)


def test_zero_rate_american_put_equals_european_put() -> None:
    put = PUT.with_(rate=0.0)
    assert crr_price(put, 500, "american") == pytest.approx(crr_price(put, 500), rel=1e-13)


def test_deep_in_the_money_american_put_is_intrinsic() -> None:
    put = PUT.with_(spot=10.0)
    assert crr_price(put, 500, "american") == pytest.approx(30.0, abs=1e-12)


def test_too_few_steps_for_the_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="probability"):
        crr_price(Option(40.0, 40.0, 0.5, 0.05, 10.0), 2)
