"""Monte Carlo checks: accuracy, unbiasedness and honest standard errors."""

import math

import numpy as np
import pytest

from triptych import black_scholes as bs
from triptych.monte_carlo import (
    european_mc,
    terminal_prices,
    variance_reduction_factor,
)
from triptych.option import Option

PUT = Option(40.0, 40.0, 0.06, 0.2, 1.0, "put")
CALL = PUT.with_(kind="call")

# (antithetic, control_variate, fixed beta or None)
Variant = tuple[bool, bool, float | None]
VARIANTS: list[Variant] = [
    (False, False, None),
    (True, False, None),
    (False, True, None),
    (True, True, None),
]


@pytest.mark.parametrize("option", [PUT, CALL, PUT.with_(spot=36.0), CALL.with_(spot=44.0)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_within_three_standard_errors_of_closed_form(option: Option, variant: Variant) -> None:
    anti, control, _ = variant
    result = european_mc(option, 400_000, seed=11, antithetic=anti, control_variate=control)
    assert abs(result.price - bs.price(option)) <= 3.0 * result.std_error
    assert result.std_error > 0.0


def _replicate(option: Option, n: int, reps: int, variant: Variant) -> tuple[float, float, float]:
    """Mean of repeated estimates, its standard error, and the mean reported SE."""
    anti, control, beta = variant
    runs = [
        european_mc(option, n, seed=500 + i, antithetic=anti, control_variate=control, beta=beta)
        for i in range(reps)
    ]
    prices = np.array([r.price for r in runs])
    mean = float(prices.mean())
    se_of_mean = float(prices.std(ddof=1)) / math.sqrt(reps)
    return mean, se_of_mean, float(np.mean([r.std_error for r in runs]))


@pytest.mark.parametrize(
    "variant",
    [
        (True, False, None),
        # A fixed beta makes the control-variate estimator exactly unbiased.
        (False, True, -0.31),
        (True, True, 1.49),
    ],
)
def test_variance_reduction_estimators_are_unbiased(variant: Variant) -> None:
    mean, se_of_mean, _ = _replicate(PUT, 2_000, 2_000, variant)
    assert abs(mean - bs.price(PUT)) <= 3.0 * se_of_mean


@pytest.mark.parametrize("variant", [(False, True, None), (True, True, None)])
def test_in_sample_beta_bias_is_small_and_shrinks(variant: Variant) -> None:
    # Fitting beta on the same sample correlates it with the control mean and
    # leaves a bias of order 1/n. It must be a small fraction of one standard
    # error, and smaller still on a larger sample.
    exact = bs.price(PUT)
    mean_small, se_small, reported_small = _replicate(PUT, 500, 4_000, variant)
    mean_large, se_large, reported_large = _replicate(PUT, 8_000, 1_000, variant)
    assert abs(mean_small - exact) <= 0.25 * reported_small + 3.0 * se_small
    assert abs(mean_large - exact) <= 0.05 * reported_large + 3.0 * se_large


@pytest.mark.parametrize("variant", VARIANTS)
def test_reported_standard_error_matches_spread_of_estimates(variant: Variant) -> None:
    reps = 400
    _, se_of_mean, mean_reported = _replicate(PUT, 4_000, reps, variant)
    observed = se_of_mean * math.sqrt(reps)
    assert mean_reported == pytest.approx(observed, rel=0.12)


def test_each_variance_reduction_reduces_variance() -> None:
    plain = european_mc(PUT, 400_000, seed=3)
    anti = european_mc(PUT, 400_000, seed=3, antithetic=True)
    control = european_mc(PUT, 400_000, seed=3, control_variate=True)
    both = european_mc(PUT, 400_000, seed=3, antithetic=True, control_variate=True)
    assert variance_reduction_factor(plain, plain) == 1.0
    assert variance_reduction_factor(plain, anti) > 1.2
    assert variance_reduction_factor(plain, control) > 1.5
    assert variance_reduction_factor(plain, both) > variance_reduction_factor(plain, anti)
    assert variance_reduction_factor(plain, both) > variance_reduction_factor(plain, control)


def test_control_with_zero_beta_is_the_plain_estimator() -> None:
    plain = european_mc(PUT, 10_000, seed=5)
    zero = european_mc(PUT, 10_000, seed=5, control_variate=True, beta=0.0)
    assert zero.price == plain.price
    assert zero.std_error == plain.std_error


def test_antithetic_estimate_matches_a_direct_computation() -> None:
    z = np.random.default_rng(9).standard_normal(5_000)
    discount = math.exp(-PUT.rate * PUT.maturity)
    pairs = (
        0.5
        * discount
        * (PUT.payoff(terminal_prices(PUT, z)) + PUT.payoff(terminal_prices(PUT, -z)))
    )
    result = european_mc(PUT, 10_000, seed=9, antithetic=True)
    assert result.price == pytest.approx(float(pairs.mean()), rel=1e-12)
    assert result.std_error == pytest.approx(float(pairs.std(ddof=1)) / math.sqrt(5_000), rel=1e-9)


def test_discounted_terminal_price_has_mean_spot() -> None:
    z = np.random.default_rng(1).standard_normal(2_000_000)
    discounted = math.exp(-PUT.rate * PUT.maturity) * terminal_prices(PUT, z)
    standard_error = float(discounted.std(ddof=1)) / math.sqrt(z.size)
    assert abs(float(discounted.mean()) - PUT.spot) <= 3.0 * standard_error


def test_same_seed_same_answer_and_chunking_does_not_change_the_draws() -> None:
    a = european_mc(PUT, 100_000, seed=42, antithetic=True, control_variate=True)
    b = european_mc(PUT, 100_000, seed=42, antithetic=True, control_variate=True)
    c = european_mc(PUT, 100_000, seed=42, antithetic=True, control_variate=True, chunk=7_777)
    assert a == b
    assert c.price == pytest.approx(a.price, rel=1e-12)


def test_odd_count_with_antithetic_is_rejected() -> None:
    with pytest.raises(ValueError):
        european_mc(PUT, 1_001, seed=1, antithetic=True)
