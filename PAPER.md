# Sources, formulas and comparison

Every formula in the code is listed here with the place it comes from. Numbers quoted from `results/` were produced by `uv run python scripts/reproduce.py` at seed 2001.

## Sources read

| Source | Copy read | How |
| --- | --- | --- |
| Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities". Journal of Political Economy 81(3), 637-654. | Scan of the journal pages hosted by a Princeton course page (`cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf`) | Pages 643 to 647 read as images |
| Cox, J., Ross, S. and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach". Journal of Financial Economics 7(3), 229-263. | Scan of the journal pages (`static.stevereads.com/papers_to_read/option_pricing_a_simplified_approach.pdf`) | Pages 234 to 238, 249, 258 and 259 read as images |
| Longstaff, F. and Schwartz, E. (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach". Review of Financial Studies 14(1), 113-147. | The authors' copy at UCLA Anderson (`anderson.ucla.edu/documents/areas/fac/finance/least_squares.pdf`) | Text layer, pages 121 to 128 and 143 |

Four things are from secondary knowledge and were not read in a source during this build:

- The pathwise greek estimators. They are credited to Broadie and Glasserman, "Estimating Security Price Derivatives Using Simulation", Management Science 42(2), 1996. I derived the estimators below by differentiating the discounted payoff and checked them numerically against the analytic greeks. I did not open that paper.
- The duality upper bound for American options, credited to Andersen and Broadie, "Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options", Management Science 50(9), 2004. I implemented it from the standard textbook statement of the dual and checked it against the tree. The argument is written out below so it can be checked without the paper.
- The smoothed binomial tree with Richardson extrapolation, credited to Broadie and Detemple, "American Option Valuation: New Bounds, Approximations, and a Comparison of Existing Methods", Review of Financial Studies 9(4), 1996. Same status: implemented from the description of the method, checked against the formula.
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

Three additions are not in the paper.

- `crr_price(..., exercise_every=m)` allows early exercise only at every m-th tree date. That prices the option of Longstaff and Schwartz's Table 1, which can be exercised 50 times a year, with a tree much finer than 50 steps a year.
- `crr_price(..., smooth=True)` starts the backward induction one step before expiry from the Black-Scholes value over the last step (`tree.black_scholes_values`), with early exercise there if allowed. `tree.smoothed_european_sum` is the same thing as an n-period sum.
- `tree.richardson(coarse, fine) = 2 fine - coarse` combines trees of n and 2n steps. If the error is `c/n + o(1/n)` the `c/n` term cancels.

Results (`results/tree_steps_by_method.csv`, tolerance 0.00005 against the formula, one-year puts with strike 40, rate 6%, volatility 20%):

| Method | Spot 36 | Spot 40 | Spot 44 |
| --- | --- | --- | --- |
| Plain tree | 18,166 | 16,105 | 12,374 |
| Plain tree, Richardson, both trees even | 9,952 | 68 | 9,544 |
| Smoothed tree | 642 | 3,621 | 3,150 |
| Smoothed tree, Richardson | 252 | 368 | 232 |

Each count is the smallest from which every larger count stays within the tolerance, searched up to 25,000 steps for the plain tree and 10,000 for the others. The extrapolated counts are the finer tree, 2n. Richardson extrapolation of the plain tree works only at the money, where the strike is on a node for every even step count. Off the money the plain tree's error oscillates with the strike's position between nodes and has no clean `c/n` term.

For the American put (spot 36, `results/american_tree_methods.csv`) the reference is the extrapolated smoothed tree of 10,000 and 20,000 steps. The plain 50,000-step tree differs from it by 0.000002. On the grid 25, 50, ..., 3,200 steps the smoothed tree stays within 0.00005 from 400 steps, the extrapolated smoothed tree from 1,600, and the plain tree only at 3,200. The extrapolated errors change sign irregularly (+0.00023 at 50 steps, -0.00032 at 100, +0.00030 at 200), as expected when early exercise adds error terms that do not fall as 1/n.

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

The task was to reproduce Table 1 "within its reported standard errors". Two independent simulations with the same standard error differ by more than one of them about half the time, so 20 out of 20 within one was never expected. The test uses three, and the counts within one and two are reported above so the reader can judge.

### Lower and upper bounds

The paper's estimator fits the exercise rule and values the option on the same paths. That estimate is not a bound in either direction: the fitted rule is worse than the optimal one, which pulls it down, and it is tuned to the paths it is valued on, which pushes it up. Two estimates that are bounds:

Lower bound (`american.lsm_lower_bound`). Fit the rule on one set of paths (`american.fit_exercise_rule`), then follow it on independent paths (`american.follow_rule`). The value of any exercise rule is at most the value of the best one, so the expected value of this estimate is at most the true price. Exercise today is not allowed in it; for every cell used here the price is above the exercise value today, so this does not bind.

Upper bound (`american.andersen_broadie_upper_bound`). Number the dates i = 0 (today) to n. Let Q_i be the value at date i of continuing and following the rule from date i + 1 on, and L_i the value of following the rule from date i: the exercise value h_i where the rule stops, otherwise Q_i. All are discounted to today. Define

```
M_0 = 0,   M_i = M_(i-1) + L_i - Q_(i-1).
```

Q_(i-1) is the expected value of L_i given what is known at i - 1, so M is a martingale that starts at zero. For any stopping time tau, E[h_tau] = E[h_tau - M_tau] <= E[max_i (h_i - M_i)], so the right-hand side is at least the price. Exercising out of the money pays nothing, so the maximum can be taken over today, the in-the-money dates and expiry only. At an out-of-the-money date before expiry the rule continues, L_i = Q_i, and Q_i cancels from the next increment, so it never has to be estimated. Every remaining Q is estimated by a nested simulation that follows the rule, with half its paths mirrored. The nested estimates are unbiased, but the maximum of noisy values is biased up, so the noise raises the bound and never lowers it in expectation.

Results (`results/american_bounds.csv`): rule fitted on 100,000 paths, lower bound on 2,000,000 fresh paths, upper bound with 400 outer paths and 4,000 inner paths per continuation value, reference the tree with 200 steps between the 50 yearly exercise dates.

| Spot | Vol | Tree | In sample | Lower (se) | Upper (se) | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 36 | 0.2 | 4.4779 | 4.4850 | 4.4752 (0.0014) | 4.4828 (0.0017) | 0.0077 |
| 40 | 0.2 | 2.3141 | 2.3139 | 2.3109 (0.0012) | 2.3167 (0.0015) | 0.0058 |
| 44 | 0.2 | 1.1099 | 1.1134 | 1.1099 (0.0012) | 1.1143 (0.0014) | 0.0045 |
| 36 | 0.4 | 7.1013 | 7.1038 | 7.0990 (0.0019) | 7.1134 (0.0025) | 0.0144 |
| 40 | 0.4 | 5.3119 | 5.3095 | 5.3088 (0.0020) | 5.3189 (0.0026) | 0.0100 |
| 44 | 0.4 | 3.9477 | 3.9502 | 3.9452 (0.0024) | 3.9560 (0.0029) | 0.0108 |

In every cell lower < tree < upper. In standard errors, the tree is 0.0 to 2.7 above the lower bound and 1.7 to 4.9 below the upper bound. Both are the expected direction: the lower bound pays for the rule's mistakes, and the upper bound pays for those and for the nested noise. In the first cell the in-sample estimate is above the upper bound.

How much of the upper bound is nested noise (`results/upper_bound_by_inner_paths.csv`, put at spot 36, volatility 0.2, a separately fitted rule and 100 outer paths held fixed):

| Inner paths | Upper bound | se | Minus tree |
| --- | --- | --- | --- |
| 500 | 4.5216 | 0.0112 | 0.0437 |
| 1,000 | 4.5002 | 0.0066 | 0.0223 |
| 2,000 | 4.4923 | 0.0049 | 0.0144 |
| 4,000 | 4.4818 | 0.0036 | 0.0040 |
| 8,000 | 4.4849 | 0.0025 | 0.0071 |

The bias falls roughly as one over the number of inner paths up to 4,000. From 4,000 to 8,000 it does not fall further, within the standard errors. I did not separate the share of the final gap that is rule error from the share that is nested noise.

### Convergence of the simulations

Root-mean-square error over 100 runs per point (20 for least-squares Monte Carlo). Interval: 95% percentile bootstrap, resampling the runs at each point independently, 2,000 draws (`experiments.bootstrap_slope_interval`).

| Curve | Expected | Fitted | 95% interval |
| --- | --- | --- | --- |
| Plain Monte Carlo, 1,000 to 1,024,000 payoffs | -0.5 | -0.505 | -0.528 to -0.480 |
| Antithetic | -0.5 | -0.528 | -0.554 to -0.502 |
| Control variate | -0.5 | -0.525 | -0.548 to -0.501 |
| Antithetic and control variate | -0.5 | -0.511 | -0.538 to -0.484 |
| Least-squares Monte Carlo, 1,000 to 128,000 paths | -0.5 | -0.531 | -0.608 to -0.456 |
| Simulated greeks, relative RMSE | -0.5 | -0.509 to -0.520 | each of the five contains -0.5 |
| Central-difference delta against bump size, bumps of 0.1% and larger | 2 | 1.994 | deterministic |

Two of the ten simulated intervals, antithetic and control variate, end just short of -0.5. About half a miss would be expected by chance among ten 95% intervals. The finite-sample curves can also bend slightly at the smallest sizes. I did not run more replications to tell these apart.

Least-squares Monte Carlo against the tree (`results/convergence_lsm.csv`). Each run fits the rule on n paths and values it twice: on the same paths and on 100,000 fresh ones.

| Paths | In-sample mean error | Out-of-sample mean error (se) |
| --- | --- | --- |
| 1,000 | +0.037 | -0.036 (0.003) |
| 2,000 | +0.023 | -0.021 (0.003) |
| 8,000 | +0.011 | -0.011 (0.001) |
| 32,000 | -0.003 | -0.004 (0.001) |
| 128,000 | -0.002 | -0.003 (0.002) |

The out-of-sample value is below the tree at every size, as a lower bound should be, and the gap closes as the rule improves with more fitting paths. The in-sample value is above the tree at small sizes, where the regression overfits the paths it then values.

For central differences, rounding error takes over once the bump is small enough (near 0.0003% for delta and 0.01% for gamma), and the error rises again as the bump shrinks further.

## Variants tried

Everything that was run with more than one setting, including settings that were dropped.

1. Seeds. Least-squares Monte Carlo was run once on the first cell with seed 7 (4.4701) as a smoke test. The reported tables use base seed 2001 with a fixed offset per experiment. The 100,000-path table was computed three times with those seeds (in the test suite, in a diagnostic print and in the full run) and gave the same simulated values each time.
2. Tree criterion against the paper's finite-difference column. First set at half a cent. One cell missed it: 0.0052 below the paper on the test's grid, 0.0058 on the full run's. It was widened to one cent, the tolerance the paper itself reports. This change is post hoc and the miss is written up above.
3. Steps between exercise dates in the tree. 10 in the fast run, 20 in the test, 200 in the full run. On four cells I also printed 10, 20, 21, 50, 100 and 200 to see whether the tree had converged.
4. Control coefficient in the headline simulation. The first draft estimated it in sample. A test then showed the 1/n bias described above, and the headline was changed to take the coefficient from the pilot run.
5. Two tests were rewritten after failing. The tree's halving test first mixed odd and even step counts, which have different error constants; it now uses even counts. The unbiasedness test first held the in-sample control coefficient to exact unbiasedness; it now tests fixed coefficients for that and bounds the in-sample bias separately.
6. The upper bound, tried on the put at spot 36 with one exploratory rule and outer paths equal in number to inner paths. The first version estimated a continuation value at every date, in and out of the money, with plain inner paths: 4.610 and 4.525 with 200 and 500 inner paths. Skipping out-of-the-money dates changed little: 4.631, 4.526 and 4.507 with 200, 500 and 1,000. Mirroring half the inner paths as well gave 4.514 and 4.497 at 500 and 1,000. All versions are valid upper bounds; the last has the least nested noise and is the one in the code.
7. Bound sizes. A first full-size run of the bounds used 400,000 paths for the lower bound and 250 outer paths for the upper bound. The lower bound's standard error (0.003 to 0.005) was then as large as the gap between the bounds, and in two cells the lower bound came out above the tree by 1.6 and 1.7 standard errors. The sizes were raised to 2,000,000 and 400 after seeing that, which is post hoc. At the first sizes the tree was also inside the bracket in every cell, within two standard errors.
8. A test of Richardson extrapolation on the smoothed tree first required its error to be below 5% of the finer tree's, then 25%. Both failed at spot 36, where the ratio is 0.16 at 200 steps and 0.29 at 800. The test now requires below 50%, plus an error below 0.00005 at 400 and 800 steps. The thresholds were set after seeing the numbers.
9. Not varied: the basis functions, the number of paths in Table 1, the number of exercise dates, the finite-difference bump in the greeks table (0.1%), the 4,000 inner paths of the bounds, and the 2,000 bootstrap draws.

## Not verified

- The pathwise estimators were not checked against Broadie and Glasserman (1996), only against the analytic greeks.
- Section titles in Cox, Ross and Rubinstein were not read. The formulas were, on the pages cited.
- The explanation offered for the finite-difference gap is a guess.
- The upper bound and the smoothed tree were not checked against the numbers in Andersen and Broadie (2004) or Broadie and Detemple (1996), only against the tree and the formula.
- Nothing here was run on a second machine. Monte Carlo results depend on numpy's generator and could differ in the last digits with another numpy version.
