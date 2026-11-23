"""Greeks three ways must agree."""

import pytest

from triptych import black_scholes as bs
from triptych.greeks import finite_difference_greeks, pathwise_greeks
from triptych.option import Option
from triptych.tree import crr_price

OPTIONS = [
    Option(40.0, 40.0, 0.06, 0.2, 1.0, "put"),
    Option(40.0, 40.0, 0.06, 0.2, 1.0, "call"),
    Option(36.0, 40.0, 0.06, 0.4, 2.0, "put"),
    Option(44.0, 40.0, 0.03, 0.3, 0.5, "call"),
]


@pytest.mark.parametrize("option", OPTIONS)
def test_finite_difference_matches_analytic(option: Option) -> None:
    analytic = bs.greeks(option).as_dict()
    finite = finite_difference_greeks(bs.price, option).as_dict()
    for name in bs.GREEK_NAMES:
        assert finite[name] == pytest.approx(analytic[name], rel=1e-5), name


@pytest.mark.parametrize("option", OPTIONS)
def test_monte_carlo_greeks_within_three_standard_errors(option: Option) -> None:
    analytic = bs.greeks(option).as_dict()
    simulated = pathwise_greeks(option, 400_000, seed=21)
    estimate, error = simulated.estimate.as_dict(), simulated.std_error.as_dict()
    for name in bs.GREEK_NAMES:
        assert abs(estimate[name] - analytic[name]) <= 3.0 * error[name], name


def test_monte_carlo_greeks_without_antithetic_paths() -> None:
    option = OPTIONS[0]
    analytic = bs.greeks(option).as_dict()
    simulated = pathwise_greeks(option, 400_000, seed=22, antithetic=False)
    estimate, error = simulated.estimate.as_dict(), simulated.std_error.as_dict()
    for name in bs.GREEK_NAMES:
        assert abs(estimate[name] - analytic[name]) <= 3.0 * error[name], name


def test_finite_difference_greeks_of_the_tree_are_close_to_analytic() -> None:
    option = OPTIONS[0]
    finite = finite_difference_greeks(lambda o: crr_price(o, 1000), option, rel_bump=0.01)
    analytic = bs.greeks(option)
    assert finite.delta == pytest.approx(analytic.delta, abs=2e-3)
    assert finite.vega == pytest.approx(analytic.vega, rel=2e-3)
    assert finite.rho == pytest.approx(analytic.rho, rel=2e-3)
