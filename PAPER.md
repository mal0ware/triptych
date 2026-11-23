# Sources, formulas and comparison

Every formula in the code is listed here with the place it comes from. Numbers quoted from `results/` were produced by `uv run python scripts/reproduce.py` at seed 2001.

## Sources read

| Source | Copy read | How |
| --- | --- | --- |
| Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities". Journal of Political Economy 81(3), 637-654. | Scan of the journal pages hosted by a Princeton course page (`cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf`) | Pages 643 to 647 read as images |
| Cox, J., Ross, S. and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach". Journal of Financial Economics 7(3), 229-263. | Scan of the journal pages (`static.stevereads.com/papers_to_read/option_pricing_a_simplified_approach.pdf`) | Pages 234 to 238, 249, 258 and 259 read as images |
| Longstaff, F. and Schwartz, E. (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach". Review of Financial Studies 14(1), 113-147. | The authors' copy at UCLA Anderson (`anderson.ucla.edu/documents/areas/fac/finance/least_squares.pdf`) | Text layer, pages 121 to 128 and 143 |

Two things are from secondary knowledge and were not read in a source during this build:

- The pathwise greek estimators. They are credited to Broadie and Glasserman, "Estimating Security Price Derivatives Using Simulation", Management Science 42(2), 1996. I derived the estimators below by differentiating the discounted payoff and checked them numerically against the analytic greeks. I did not open that paper.
- The analytic gamma, vega, theta and rho. Black and Scholes print only the hedge ratio, equation (14). The others are derivatives of equations (13) and (27), checked by finite differences and by substituting them into equation (7).

## Notation

`S` spot, `K` strike, `r` continuously compounded rate, `sigma` volatility, `T` time to expiry, `N` the standard normal distribution function, `n` its density. The stock pays no dividends. Black and Scholes write `x` for `S`, `c` for `K`, `v^2` for `sigma^2` and `t* - t` for `T`.

## Black and Scholes (1973)

| What | Source | Formula | Code |
| --- | --- | --- | --- |
| Valuation equation | eq. (7), p. 643 | `w2 = r w - r x w1 - (1/2) v^2 x^2 w11` | tested in `test_greeks_satisfy_the_black_scholes_equation` |
| European call | eq. (13), p. 644 | `C = S N(d1) - K exp(-rT) N(d2)`, `d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))`, `d2 = d1 - sigma sqrt(T)` | `black_scholes.price` |
| Hedge ratio | eq. (14), p. 645 | `w1 = N(d1)` | `black_scholes.greeks` (delta) |
| Put-call relation | eq. (25) and (26), p. 647 | `C - P = S - K exp(-rT)` | tested, not used to compute the put |
| European put | eq. (27), p. 647 | `P = -S N(-d1) + K exp(-rT) N(-d2)` | `black_scholes.price` |

The paper writes `d2` out in full with `r - v^2/2`; the code uses the equal form `d1 - sigma sqrt(T)`.

Greeks derived from (13) and (27), with `w = +1` for a call and `-1` for a put:

```
delta = w N(w d1)
gamma = n(d1) / (S sigma sqrt(T))
vega  = S n(d1) sqrt(T)
theta = -S n(d1) sigma / (2 sqrt(T)) - w r K exp(-rT) N(w d2)
rho   = w T K exp(-rT) N(w d2)
```

Theta is the derivative with respect to calendar time, the `w2` of equation (7). Vega and rho are per unit of volatility and rate, not per percentage point.

## Cox, Ross and Rubinstein (1979)

In the paper `r` is one plus the interest rate over one period, and `r-hat = r^(t/n)` when a year is cut into periods (p. 258). The code works with a continuously compounded rate, so its one-period growth factor is `exp(r T / n)`.

| What | Source | Formula | Code |
| --- | --- | --- | --- |
| One-period value | eq. (3), p. 234 | `C = [p Cu + (1 - p) Cd] / r`, `p = (r - d) / (u - d)` | `tree.crr_price`, each backward step |
| n-period value | eq. (6), p. 238 | `C = [sum_j (n! / (j! (n-j)!)) p^j (1-p)^(n-j) max(0, u^j d^(n-j) S - K)] / r^n` | `tree.crr_european_sum` |
| Up and down moves | p. 249 and p. 258 | `u = exp(sigma sqrt(t/n))`, `d = 1/u` | `tree.crr_parameters` |
| Early exercise of a put | p. 258 | `P = [p Pu + (1 - p) Pd] / r-hat` if this exceeds `K - S`, otherwise `P = K - S` | `tree.crr_price` with `exercise="american"` |

The code checks that `p` lies strictly between 0 and 1 and refuses to build a tree otherwise.

One addition is not in the paper. `crr_price(..., exercise_every=m)` allows early exercise only at every m-th tree date. That prices the option of Longstaff and Schwartz's Table 1, which can be exercised 50 times a year, with a tree much finer than 50 steps a year.

## Monte Carlo and variance reduction

The terminal price is sampled exactly, `S_T = S exp((r - sigma^2/2) T + sigma sqrt(T) Z)` with `Z` standard normal, which is the solution of the paper's equation (7) of Longstaff and Schwartz, `dS = r S dt + sigma S dZ`. There is no time-stepping error in the European simulations.

- Plain: the mean of `exp(-rT) payoff(S_T)`.
- Antithetic: each `Z` is used again as `-Z` and the two discounted payoffs are averaged. The standard error is computed from those pair averages, which are the independent samples.
- Control variate: the sample is `Y - beta (exp(-rT) S_T - S)`. The control has mean zero exactly because the discounted stock price is a martingale. `beta` is the least-squares slope of `Y` on the control.

The variance-reduction factor is the variance of the plain estimate divided by the variance of the other estimate when both use the same number of payoff evaluations. An antithetic pair counts as two evaluations.

Fitting `beta` on the same sample that it is applied to leaves a bias of order 1/n. A test measured it: over 2,000 repeats of 2,000 payoffs with antithetic paths and an in-sample `beta`, the mean estimate was 2.06860 against 2.06640, a bias of 0.0022 with a standard error of 0.0005. That is about a tenth of one run's standard error at that size and it shrinks faster than the standard error does. The headline simulation avoids it by taking `beta` from a separate pilot run.

## Greeks by simulation

With `w = +1` for a call and `-1` for a put, and `I` the indicator that the option finishes in the money:

```
delta = exp(-rT) w I S_T / S
vega  = exp(-rT) w I S_T (sqrt(T) Z - sigma T)
rho   = T exp(-rT) (w I S_T - payoff)
theta = exp(-rT) (r payoff - w I S_T (r - sigma^2/2 + sigma Z / (2 sqrt(T))))
gamma = exp(-rT) w I (S_T / S^2) (Z / (sigma sqrt(T)) - 1)
```

The first four differentiate the discounted payoff along the path. The payoff has a kink at the strike, so its second derivative along a path is zero almost everywhere and the same method gives no gamma. The gamma line applies the likelihood-ratio score `Z / (S sigma sqrt(T))` to the pathwise delta.

Written this way the gamma estimator is the vega estimator divided by `S^2 sigma T` on every path. So the simulated gamma and vega satisfy `vega = S^2 sigma T gamma` exactly, as the analytic ones do, and they carry the same information. Their errors in the greeks table are the same number of standard errors for that reason.

## Longstaff and Schwartz (2001)

| What | Source | In the code |
| --- | --- | --- |
| Continuation value as a conditional expectation of discounted future cash flows | eq. (1), p. 121 | `american.lsm_price`, the vector `cash` |
| Regress on in-the-money paths only | section 2.2, p. 123 | `in_money` |
| Exercise when the immediate value is positive and at least the fitted continuation value | section 2.1, p. 122 | `exercise_value >= continuation` |
| Value is the average over paths of the cash flow at the first exercise, discounted to time zero | section 2.2, p. 123 | `cash.mean()` |
| Basis: a constant and `L0(X) = exp(-X/2)`, `L1(X) = exp(-X/2)(1 - X)`, `L2(X) = exp(-X/2)(1 - 2X + X^2/2)` | eq. (2) to (4), p. 122, and section 3, p. 126 | `american.laguerre_basis` |
| Stock price divided by the strike before the basis is evaluated | section 8, p. 143 | `paths / option.strike` |
| Table 1 set-up: K = 40, r = 0.06, 50 exercise dates a year, 100,000 paths of which 50,000 are antithetic | section 3 and the note to Table 1, pp. 126 to 127 | `published.py`, `experiments.longstaff_schwartz_table` |

Differences from the paper's procedure:

1. The paper also divides the cash flows by the strike. The code regresses the cash flows in dollars. A least-squares fit is unchanged by rescaling the dependent variable, so this changes nothing but rounding.
2. `lsm_price` takes the larger of the simulated value and today's exercise value, so the result is never below intrinsic value. The paper's exercise dates start after time zero. In all 20 cells of Table 1 the simulated value is above intrinsic value, so this does not bind there.
3. The random numbers are numpy's PCG64 at fixed seeds. The paper does not say which generator it used, so its values cannot be reproduced digit for digit, only up to sampling error.
4. The paper does not say how its standard errors were computed. See the comparison below.
