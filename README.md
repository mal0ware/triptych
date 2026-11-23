# Triptych

**One option, priced three ways, until all three agree.**

## Result

One European put (spot 40, strike 40, rate 6%, volatility 20%, one year) priced by formula, by tree and by simulation:

| Method | Effort | Price | Error against the formula |
| --- | --- | --- | --- |
| Black-Scholes formula | closed form | 2.066401 | |
| Cox-Ross-Rubinstein tree | 16,105 steps | 2.066442 | +0.000041 |
| Monte Carlo, antithetic paths and a control variate | 442 million payoffs | 2.066470 ± 0.000050 | +0.000069 |

The formula and the tree agree to four decimals. The simulation was sized for a standard error of 0.00005, landed 1.4 standard errors high, and rounds to 2.0665, one unit off in the fourth decimal. Plain Monte Carlo would have needed 4.4 billion payoffs for the same standard error; the two variance reductions together cut that tenfold.

For American puts, least-squares Monte Carlo reproduces all 20 values in Table 1 of Longstaff and Schwartz (2001) within three of the paper's standard errors: 15 within one, 18 within two, the largest gap 2.4.

![Error of the tree, of four Monte Carlo estimators and of least-squares Monte Carlo, each against the expected slope](results/triptych.png)

Left: the tree's error falls as one over the number of steps. Middle: every Monte Carlo estimator's error falls as one over the square root of the number of payoffs, and variance reduction moves the line down without changing its slope. Right: the same square-root rate for the American put, measured against a tree.

## Reproduce

```
uv sync
uv run pytest
uv run python scripts/reproduce.py
```

The last command rebuilds every table and figure in `results/`. It took 114 seconds here. `--fast` runs small sizes in a few seconds and writes to `results-fast/`, which is not committed; CI uses it.

## The question

A formula, a tree and a simulation should give the same price. How fast does each one get there, and what happens when the option can be exercised early?

## What was built

- European options by the Black-Scholes formula, a Cox-Ross-Rubinstein tree, and Monte Carlo
- Variance reduction: antithetic paths and a control variate (the discounted stock price), with the factor each one buys
- Greeks three ways: by formula, by finite difference, and by simulation
- American puts by tree and by least-squares Monte Carlo
- A convergence plot for every method, with the expected slope drawn and the fitted slope tabulated

### What variance reduction buys

Measured on 10 million payoffs for the put above. The factor is how many times fewer payoffs the estimator needs than plain Monte Carlo for the same error.

| Estimator | Standard error | Factor | Payoffs for a standard error of 0.00005 |
| --- | --- | --- | --- |
| Plain | 0.001052 | 1.0 | 4.43 billion |
| Antithetic | 0.000824 | 1.6 | 2.72 billion |
| Control variate | 0.000696 | 2.3 | 1.94 billion |
| Both | 0.000332 | 10.0 | 441 million |

The factor depends on moneyness. With spot at 36 the three factors are 3.9, 3.8 and 29. With spot at 44 they are 1.2, 1.6 and 19. Used together the two tricks beat the product of their separate factors in all three cases.

### Greeks three ways

Same put. Finite differences are central differences of the formula with a 0.1% bump. The simulation uses 10 million payoffs.

| Greek | Formula | Finite difference | Simulation | Simulation error, in standard errors |
| --- | --- | --- | --- | --- |
| Delta | -0.344578 | -0.344579 | -0.344673 | -1.4 |
| Gamma | 0.046034 | 0.046034 | 0.046046 | 1.0 |
| Vega | 14.730806 | 14.730805 | 14.734573 | 1.0 |
| Theta | -0.522109 | -0.522109 | -0.522223 | -0.4 |
| Rho | -15.849531 | -15.849531 | -15.853905 | -1.4 |

Finite differences match the formula to at least five significant figures. The simulated greeks are all within 1.5 standard errors. Simulated gamma and vega are the same estimator up to a constant, which is why their errors are equal (see `PAPER.md`).

### Convergence rates

| Curve | Expected slope | Fitted slope |
| --- | --- | --- |
| European tree, error against steps | -1 | -1.000 |
| Monte Carlo, four estimators, error against payoffs | -0.5 | -0.505 to -0.528 |
| Least-squares Monte Carlo, error against paths | -0.5 | -0.531 |
| American tree, error against steps | -1 | -1.016 |
| Finite-difference delta, error against bump size | 2 | 1.994 |
| Simulated greeks, error against payoffs | -0.5 | -0.509 to -0.520 |

The other figures are `results/american_tree_convergence.png` and `results/greeks_convergence.png`. All tables are in `results/tables.md` and as CSV files beside it.

## Sources

- Black and Scholes, "The Pricing of Options and Corporate Liabilities", Journal of Political Economy 81(3), 1973
- Cox, Ross and Rubinstein, "Option Pricing: A Simplified Approach", Journal of Financial Economics 7(3), 1979
- Longstaff and Schwartz, "Valuing American Options by Simulation: A Simple Least-Squares Approach", Review of Financial Studies 14(1), 2001

`PAPER.md` lists each formula with its equation number and the page it was read from.

## Comparison with the sources

What matches:

- The formula matches all 20 European values in Longstaff and Schwartz's Table 1 to three decimals, and all 27 Black-Scholes values in Cox, Ross and Rubinstein's Table 2 to the cent.
- The tree matches all 54 values for 5 and 20 steps in Cox, Ross and Rubinstein's Table 2 to the cent.
- Least-squares Monte Carlo, set up as in the paper (100,000 paths, half of them antithetic, 50 exercise dates a year, a constant and three Laguerre polynomials), is within three reported standard errors of the paper's value in all 20 cells.

What does not match:

- Monte Carlo and the formula differ in the fourth decimal, as described at the top.
- The tree, restricted to the paper's 50 exercise dates a year, is within 0.0005 of the paper's finite-difference value in 15 of 20 cells and within 0.004 in 19. In the last cell (spot 44, volatility 40%, two years) the tree gives 5.6412 and the paper 5.647. The tree is converged there: 50, 100 and 200 steps per exercise date give 5.6412, 5.6414 and 5.6412. The paper says its two finite-difference schemes agreed "generally within one cent", so the gap is inside the paper's own tolerance. All five cells with volatility 40% and two years sit 0.001 to 0.006 below the paper.
- The paper's standard errors are larger than the sampling error. They are reproduced, to within 11%, by treating all 100,000 paths as independent. Half the paths mirror the other half, and the standard error computed from the mirrored pairs is 43% to 84% of that figure. Twenty repeated runs at 128,000 paths gave a spread of 0.0050 where the pair-based figure predicts 0.0053 and the all-independent figure 0.0082.

Full detail is in `PAPER.md`.

## Data

None. Every number comes from a formula or a seeded simulation. `scripts/fetch_data.py` exists only because the shared layout names it.

## How it is tested

`uv run pytest` runs 1,316 tests in about 15 seconds.

- Put-call parity to within four units of machine precision, over 480 parameter sets
- The Black-Scholes differential equation holds for the analytic greeks
- The tree matches the published Table 2 of Cox, Ross and Rubinstein, equals their n-period sum, and halves its error when the steps double
- Monte Carlo is within three standard errors of the formula for every estimator
- The variance-reduction estimators are unbiased over 2,000 repeated runs, and the reported standard errors match the spread of those runs
- An American put is never worth less than the European put or its exercise value
- Greeks agree three ways
- Least-squares Monte Carlo reproduces Table 1 of Longstaff and Schwartz
- Two runs of the reproduce pipeline write identical files

## Layout

```
src/triptych/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  nothing to download for this project
results/           committed figures and tables
PAPER.md           citations, formulas, comparison with the sources
```

## Roadmap

- [x] Project skeleton, CI, and the reproduce command
- [x] European options by the Black-Scholes formula, a Cox-Ross-Rubinstein tree, and Monte Carlo
- [x] Variance reduction: antithetic paths and control variates, with the factor each one buys
- [x] Greeks three ways: by formula, by finite difference, and pathwise
- [x] American puts by tree and by least-squares Monte Carlo
- [x] Convergence plots for every method
- [x] `PAPER.md` comparison with the sources
- [x] Results and limits written up at the top of this file

## Limits

- Constant volatility and constant rates throughout, and no dividends. Meniscus is where volatility stops being constant.
- The headline is one at-the-money put. The step and path counts are specific to it. The variance-reduction factors alone range from 10 to 29 across the three spots tried.
- The tree reaches four decimals only by brute force. No smoothing or extrapolation was tried.
- The control variate is the stock price alone. A better control would cut the 442 million payoffs further; none was tried.
- Least-squares Monte Carlo is run in sample, as in the paper: the same paths fit the exercise rule and value the option. At 128,000 paths its mean error against the tree over 20 runs was -0.002 with a standard error of 0.001. That hints at the low bias an imperfect exercise rule should give, without establishing it.
- Gamma by simulation is a likelihood-ratio estimator applied to the pathwise delta. A purely pathwise gamma does not exist for a payoff with a kink.
- The paper's random numbers are unknown, so its Table 1 can be matched only up to sampling error.
- The CI workflow has not run. GitHub refused to start the job on the first push because of the account's Actions billing, so the checks have only been run locally, on Windows.
- This is textbook material. The point is that it is exact, tested, and that every number can be reproduced.
- This is research code. It is not investment advice and it does not trade.

## The name

A triptych is one picture in three panels. Here the picture is the price of an option.

## Licence

MIT. See `LICENSE`.
