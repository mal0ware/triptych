# Triptych

**One option, priced three ways, until all three agree.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

A formula, a tree and a simulation should give the same price. How fast does each one get there, and what happens when the option can be exercised early?

## What this will contain

- European options by the Black-Scholes formula, a Cox-Ross-Rubinstein tree, and Monte Carlo
- Variance reduction: antithetic paths and control variates, with the factor each one buys
- Greeks three ways: by formula, by finite difference, and pathwise
- American puts by tree and by least-squares Monte Carlo
- Convergence plots for every method

## The result it is built to show

Three methods agreeing to four decimal places, and a table of how many steps or paths each needed to get there.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Black and Scholes, "The Pricing of Options and Corporate Liabilities", Journal of Political Economy 81(3), 1973
- Cox, Ross and Rubinstein, "Option Pricing: A Simplified Approach", Journal of Financial Economics 7(3), 1979
- Longstaff and Schwartz, "Valuing American Options by Simulation: A Simple Least-Squares Approach", Review of Financial Studies 14(1), 2001

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

None.

## How it is tested

The maths is tested before any result is trusted.

- Put-call parity to machine precision
- The tree converges to the closed form
- Least-squares Monte Carlo reproduces Table 1 of the Longstaff-Schwartz paper within its standard errors

## Layout

```
src/triptych/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] European options by the Black-Scholes formula, a Cox-Ross-Rubinstein tree, and Monte Carlo
- [ ] Variance reduction: antithetic paths and control variates, with the factor each one buys
- [ ] Greeks three ways: by formula, by finite difference, and pathwise
- [ ] American puts by tree and by least-squares Monte Carlo
- [ ] Convergence plots for every method
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Constant volatility and constant rates throughout. Meniscus is where volatility stops being constant.
- This is textbook material. The point is that it is exact, tested, and that every number can be reproduced.
- This is research code. It is not investment advice and it does not trade.

## The name

A triptych is one picture in three panels. Here the picture is the price of an option.

## Licence

MIT. See `LICENSE`.
