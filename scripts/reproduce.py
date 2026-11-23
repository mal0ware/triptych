"""Rebuild every table and figure.

    uv run python scripts/reproduce.py          full run, writes results/
    uv run python scripts/reproduce.py --fast   small sizes, writes results-fast/

The fast run exists for CI. Its numbers are noisier and are not the ones
quoted in the README.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from triptych.experiments import FAST, FULL
from triptych.report import run_all

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Parse arguments and run the experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="small sizes, under two minutes")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    out: Path = args.out or ROOT / ("results-fast" if args.fast else "results")
    start = time.perf_counter()
    tables = run_all(FAST if args.fast else FULL, out)
    elapsed = time.perf_counter() - start
    print(f"wrote {len(tables)} tables and 3 figures to {out} in {elapsed:.0f} s")


if __name__ == "__main__":
    main()
