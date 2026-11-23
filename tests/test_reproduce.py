"""The reproduce pipeline runs and is deterministic."""

import dataclasses
from pathlib import Path

from triptych.experiments import FAST, fitted_slope, tree_steps_for_tolerance
from triptych.option import Option
from triptych.report import format_value, markdown_table, run_all

TINY = dataclasses.replace(
    FAST,
    tolerance=5e-3,
    tree_scan_max=400,
    tree_curve_max=200,
    mc_pilot=20_000,
    mc_headline_cap=100_000,
    mc_curve_sizes=(1_000, 4_000),
    mc_replications=4,
    greeks_payoffs=20_000,
    greeks_curve_sizes=(1_000, 4_000),
    lsm_paths=2_000,
    lsm_curve_sizes=(1_000, 2_000),
    lsm_replications=2,
    bermudan_steps_per_date=2,
    american_tree_steps=(10, 20, 50),
    american_tree_reference=500,
)


def test_two_runs_write_identical_tables(tmp_path: Path) -> None:
    first = run_all(TINY, tmp_path / "a", figures=False)
    second = run_all(TINY, tmp_path / "b", figures=False)
    assert first == second
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert "tables.md" in names
    assert "longstaff_schwartz_table1.csv" in names
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_figures_are_written(tmp_path: Path) -> None:
    run_all(TINY, tmp_path)
    for name in ("triptych.png", "american_tree_convergence.png", "greeks_convergence.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_tree_steps_for_tolerance_is_past_the_last_miss() -> None:
    option = Option(40.0, 40.0, 0.06, 0.2, 1.0, "put")
    steps = tree_steps_for_tolerance(option, 5e-3, 400)
    assert 100 < steps < 400
    assert tree_steps_for_tolerance(option, 1e-9, 50) == 0


def test_fitted_slope_recovers_a_power_law() -> None:
    x = [1.0, 10.0, 100.0]
    assert abs(fitted_slope(x, [3.0 * v**-0.5 for v in x]) + 0.5) < 1e-12


def test_table_formatting() -> None:
    assert format_value(3) == "3"
    assert format_value(0.0) == "0"
    assert format_value(2.0664010044) == "2.066401"
    assert format_value(4.2e-5) == "4.200e-05"
    assert markdown_table([{"a": 1, "b": "x"}]).splitlines()[2] == "| 1 | x |"
