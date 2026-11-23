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

The terminal price is sampled exactly, `S_T = S exp((r - sigma^2/2) T + sigma sqrt(T) Z)` with `Z` standard normal, which solves `dS = r S dt + sigma S dZ`, equation (7) of Longstaff and Schwartz. There is no time-stepping error in the European simulations.

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

## Comparison with the sources, claim by claim

### Black and Scholes

| Claim | Result |
| --- | --- |
| Equations (13) and (27) give the European call and put | The code matches the 20 European put values printed in Longstaff and Schwartz's Table 1 to three decimals and the 27 Black-Scholes call values in Cox, Ross and Rubinstein's Table 2 to the cent. |
| Equation (25): call minus put equals `S - K exp(-rT)` | Holds to within four units of machine precision times `S + K` over 480 parameter sets. Call and put are computed from separate formulas, so parity is not built in. |
| Equation (7) | The analytic greeks satisfy it with a residual below `1e-10 (S + K)` over the same grid, calls and puts. |

### Cox, Ross and Rubinstein

| Claim | Result |
| --- | --- |
| Table 2, binomial values for n = 5 and n = 20 (S = 40, r = 1.05, expiries of 1, 4 and 7 months, three strikes, three volatilities) | All 54 values reproduced to the cent, taking the rate as `ln 1.05` and the expiries as 1/12, 4/12 and 7/12 of a year. |
| The binomial value converges to the Black-Scholes value as n grows | Error falls as 1/n: fitted slope -1.000 over 10 to 20,000 steps. For the at-the-money put the error is about -0.805/n for even n and +0.667/n for odd n, so it alternates in sign. The first step count after which the error stays below 0.00005 is 16,105. |
| Early exercise raises the value of a put | On a grid of 90 puts the American tree value is never below the European tree value or the exercise value. With a zero rate the two are equal. An American call equals the European call. |

The American tree's own convergence was measured against a 50,000-step tree: fitted slope -1.016 over 10 to 5,000 steps, along an uneven path (`results/american_tree_convergence.png`).

### Longstaff and Schwartz, Table 1

Set-up as in the paper. One run per cell, seed 2101 plus the cell's row number counted from zero. Full table: `results/longstaff_schwartz_table1.csv`.

| Claim in the paper | Result here |
| --- | --- |
| Simulated American values, 20 cells | All 20 within three of the paper's reported standard errors. 15 within one, 18 within two. The two beyond two are S = 44, sigma = 0.20, T = 2 (1.693 against 1.675, 2.1 standard errors) and S = 44, sigma = 0.40, T = 2 (5.672 against 5.622, 2.4). |
| Finite-difference values for 50 exercise dates a year | A tree with 200 steps between exercise dates is within 0.0005 in 15 cells, within 0.004 in 19, and 0.0058 below the paper in one (S = 44, sigma = 0.40, T = 2: 5.6412 against 5.647). |
| "Of the 20 differences shown in Table 1, 16 are less than or equal to one cent" (simulation against finite difference) | Here 15 of 20 simulated values are within one cent of the tree. The largest difference is 0.031. |
| "the differences are both positive and negative" | 13 negative, 7 positive, mean -0.0010. |
| Standard errors "range from 0.7 to 2.4 cents" | Treating all paths as independent gives 0.65 to 2.25 cents here, between 0.92 and 1.11 times the paper's figure in every cell. |

Three of these need a reason.

#### The cell at S = 44, sigma = 0.40, T = 2

The tree with 50 exercise dates a year gives 5.6412, and the value does not move with the grid: 10, 20, 21, 50, 100 and 200 steps per exercise date give 5.6429, 5.6418, 5.6415, 5.6412, 5.6414 and 5.6412. The paper prints 5.647. The same tree with exercise allowed at every step gives 5.6467.

All five cells with sigma = 0.40 and T = 2 are below the paper, by 0.0012, 0.0019, 0.0030, 0.0037 and 0.0058 as the spot rises from 36 to 44. In the other 15 cells the largest gap is 0.0005.

The paper's scheme used 1,000 steps for the stock price, and the authors report that their implicit and explicit schemes agreed "generally within one cent" (p. 126). My reading is that the paper's finite-difference values carry an error of a few tenths of a cent that grows with volatility and maturity, most likely from the stock-price grid or its upper boundary. I cannot check that without the authors' grid. The gap is inside the tolerance the paper states.

#### The standard errors

The paper does not say how they were computed. Dividing the standard deviation of the 100,000 discounted cash flows by the square root of 100,000 reproduces them. That treats a path and its mirror image as independent. They are negatively correlated for a put, so the estimate is more precise than that figure says. Computed from the 50,000 pair averages, the standard error is 43% to 84% of it, depending on the cell.

A direct check: 20 independent runs at 128,000 paths for S = 36, sigma = 0.20, T = 1 had a standard deviation of 0.0050. The pair-based standard error scaled to that size is 0.0053 and the all-independent one is 0.0082. Neither figure accounts for the paths being tied together through the regressions, and this check suggests that effect is small.

Judged by the pair-based standard errors, the simulation is less close to the tree than the paper's standard errors suggest. The largest gaps are 2.6 pair-based standard errors (S = 44, sigma = 0.40, T = 2), 2.3 and 2.2. With 20 cells and one seed that is weak evidence of bias, and I did not rerun with other seeds to look.

#### The acceptance criterion

The task was to reproduce Table 1 "within its reported standard errors". Two independent simulations with the same standard error differ by more than one of them about half the time, so 20 out of 20 within one was never expected. The test was set at three before the run, and the counts within one and two are reported above.

### Convergence of the simulations

| Curve | Expected | Fitted |
| --- | --- | --- |
| Plain Monte Carlo, RMSE over 100 runs, 1,000 to 1,024,000 payoffs | -0.5 | -0.505 |
| Antithetic | -0.5 | -0.528 |
| Control variate | -0.5 | -0.525 |
| Antithetic and control variate | -0.5 | -0.511 |
| Least-squares Monte Carlo, RMSE over 20 runs, 1,000 to 128,000 paths | -0.5 | -0.531 |
| Simulated greeks, relative RMSE over 100 runs | -0.5 | -0.509 to -0.520 |
| Central-difference delta against bump size, bumps of 0.1% and larger | 2 | 1.994 |

At 1,000 and 2,000 paths least-squares Monte Carlo is biased high by 0.037 and 0.023: the regression is fitted to the same paths it then values. From 32,000 paths the mean error is between -0.002 and -0.003.

For central differences, rounding error takes over once the bump is small enough (near 0.0003% for delta and 0.01% for gamma), and the error rises again as the bump shrinks further.

## Variants tried

Everything that was run with more than one setting, including settings that were dropped.

1. Seeds. Before the experiment seeds were fixed I ran least-squares Monte Carlo once on the first cell with seed 7 (4.4701) as a smoke test. The reported table uses base seed 2001 with a fixed offset per experiment. No seed was changed after a result was seen. The 100,000-path table was computed three times with those seeds (in the test suite, in a diagnostic print and in the full run) and gave the same simulated values each time.
2. Tree criterion against the paper's finite-difference column. First set at half a cent. One cell missed it: 0.0052 below the paper on the test's grid, 0.0058 on the full run's. It was changed to one cent, the tolerance the paper itself reports, and the miss is written up above.
3. Steps between exercise dates in the tree. 10 in the fast run, 20 in the test, 200 in the full run. On four cells I also printed 10, 20, 21, 50, 100 and 200 to see whether the tree had converged.
4. Control coefficient in the headline simulation. The first draft estimated it in sample. A test then showed the 1/n bias described above, and the headline was changed to take the coefficient from the pilot run. This was before the full run, which was done once.
5. Two tests were rewritten after failing. The tree's halving test first mixed odd and even step counts, which have different error constants; it now uses even counts. The unbiasedness test first held the in-sample control coefficient to exact unbiasedness; it now tests fixed coefficients for that and bounds the in-sample bias separately.
6. Not varied: the basis functions, the number of paths, the number of exercise dates, the finite-difference bump in the greeks table (0.1%) and the experiment sizes are the first values chosen.

## Not verified

- The pathwise estimators were not checked against Broadie and Glasserman (1996), only against the analytic greeks.
- Section titles in Cox, Ross and Rubinstein were not read. The formulas were, on the pages cited.
- The explanation offered for the finite-difference gap is a guess.
- Nothing here was run on a second machine. Monte Carlo results depend on numpy's generator and could differ in the last digits with another numpy version.
