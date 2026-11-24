"""Run every experiment and write the tables and figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from triptych import experiments as ex
from triptych.black_scholes import GREEK_NAMES
from triptych.experiments import Config, Row

COLOURS = ("#1f4e79", "#c0504d", "#4f8a3c", "#7f5aa2", "#b8860b")


def format_value(value: float | int | str) -> str:
    """Format one table cell. Floats get a fixed number of significant digits."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e7:
        return f"{value:.3e}"
    return f"{value:.6f}"


def write_csv(rows: list[Row], path: Path) -> None:
    """Write rows to a CSV file with a header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow([format_value(v) for v in row.values()])


def markdown_table(rows: list[Row]) -> str:
    """Render rows as a GitHub-flavoured markdown table."""
    header = list(rows[0].keys())
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(v) for v in row.values()) + " |")
    return "\n".join(lines)


def _reference_line(ax: Axes, x: list[float], y_anchor: float, slope: float, label: str) -> None:
    """Draw a dashed line of the given log-log slope through (x[0], y_anchor)."""
    xs = np.array([min(x), max(x)], dtype=float)
    ax.plot(xs, y_anchor * (xs / xs[0]) ** slope, "k--", linewidth=1.0, label=label)


def _loglog(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=7, frameon=False)


def _col(rows: list[Row], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def plot_tree_panel(ax: Axes, rows: list[Row]) -> None:
    """European tree error against the number of steps."""
    steps, errors = _col(rows, "steps"), _col(rows, "abs_error")
    ax.plot(steps, errors, "o", markersize=3, color=COLOURS[0], label="CRR tree")
    _reference_line(ax, steps, errors[0], -1.0, "slope -1")
    _loglog(ax, "tree steps", "absolute error in price", "Tree against the formula")


def plot_mc_panel(ax: Axes, rows: list[Row]) -> None:
    """Monte Carlo root-mean-square error against the number of payoffs."""
    for (name, _, _), colour in zip(ex.ESTIMATORS, COLOURS, strict=False):
        sub = [r for r in rows if r["estimator"] == name]
        ax.plot(
            _col(sub, "payoffs"), _col(sub, "rmse"), "o-", markersize=3, color=colour, label=name
        )
    plain = [r for r in rows if r["estimator"] == "plain"]
    _reference_line(ax, _col(plain, "payoffs"), 1.6 * _col(plain, "rmse")[0], -0.5, "slope -1/2")
    _loglog(ax, "payoff evaluations", "RMSE of price", "Monte Carlo against the formula")


def plot_lsm_panel(ax: Axes, rows: list[Row]) -> None:
    """Least-squares Monte Carlo error against the number of paths."""
    paths = _col(rows, "paths")
    ax.plot(paths, _col(rows, "rmse"), "o-", markersize=3, color=COLOURS[0], label="RMSE")
    ax.plot(
        paths,
        [abs(v) for v in _col(rows, "mean_error")],
        "s:",
        markersize=3,
        color=COLOURS[1],
        label="|mean error|, in sample",
    )
    ax.plot(
        paths,
        [abs(v) for v in _col(rows, "mean_error_out_of_sample")],
        "^:",
        markersize=3,
        color=COLOURS[2],
        label="|mean error|, out of sample",
    )
    _reference_line(ax, paths, 1.6 * _col(rows, "rmse")[0], -0.5, "slope -1/2")
    _loglog(ax, "simulated paths", "error in price", "American put: simulation against the tree")


def figure_triptych(
    tree_rows: list[Row], mc_rows: list[Row], lsm_rows: list[Row], path: Path
) -> None:
    """Draw the headline figure: the three methods converging, one panel each."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    plot_tree_panel(axes[0], tree_rows)
    plot_mc_panel(axes[1], mc_rows)
    plot_lsm_panel(axes[2], lsm_rows)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_american_tree(rows: list[Row], reference_steps: int, path: Path) -> None:
    """American tree error against the number of steps."""
    finite = [r for r in rows if float(r["abs_error"]) > 0.0]
    steps, errors = _col(finite, "steps"), _col(finite, "abs_error")
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.plot(steps, errors, "o-", markersize=3, color=COLOURS[0], label="CRR tree, early exercise")
    _reference_line(ax, steps, errors[0], -1.0, "slope -1")
    _loglog(
        ax,
        "tree steps",
        f"absolute difference from {reference_steps}-step tree",
        "American put by tree",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_greeks(fd_rows: list[Row], pathwise_rows: list[Row], path: Path) -> None:
    """Greek errors: finite difference against bump size, Monte Carlo against payoffs."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    bumps = _col(fd_rows, "rel_bump")
    for name, colour in zip(("delta", "gamma", "vega"), COLOURS, strict=False):
        ax_errors = [max(v, 1e-17) for v in _col(fd_rows, f"{name}_rel_error")]
        axes[0].plot(bumps, ax_errors, "o-", markersize=3, color=colour, label=name)
    large = [b for b in bumps if b >= 1e-3]
    anchor = next(float(r["delta_rel_error"]) for r in fd_rows if float(r["rel_bump"]) >= 1e-3)
    _reference_line(axes[0], large, 3.0 * anchor, 2.0, "slope 2")
    _loglog(
        axes[0],
        "relative bump size",
        "relative error",
        "Central differences of the formula",
    )

    payoffs = _col(pathwise_rows, "payoffs")
    for name, colour in zip(GREEK_NAMES, COLOURS, strict=True):
        axes[1].plot(
            payoffs,
            _col(pathwise_rows, f"{name}_rel_rmse"),
            "o-",
            markersize=3,
            color=colour,
            label=name,
        )
    _reference_line(
        axes[1], payoffs, 1.6 * _col(pathwise_rows, "gamma_rel_rmse")[0], -0.5, "slope -1/2"
    )
    _loglog(axes[1], "payoff evaluations", "relative RMSE", "Monte Carlo greeks")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_all(cfg: Config, out: Path, *, figures: bool = True) -> dict[str, list[Row]]:
    """Run every experiment, write every table (and figure) into ``out``.

    Returns the tables keyed by file stem.
    """
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[Row]] = {}

    tables["european_three_ways"] = ex.european_three_ways(cfg)
    tables["variance_reduction"] = ex.variance_reduction(cfg)
    tables["variance_reduction_by_spot"] = ex.variance_reduction_by_spot(cfg)
    tables["greeks"] = ex.greeks_table(cfg)
    tables["longstaff_schwartz_table1"] = ex.longstaff_schwartz_table(cfg)
    tables["american_bounds"] = ex.american_bounds(cfg)
    tables["upper_bound_by_inner_paths"] = ex.upper_bound_by_inner_paths(cfg)
    tables["tree_steps_by_method"] = ex.tree_steps_by_method(cfg)
    tables["convergence_tree"] = ex.tree_convergence(cfg)
    tables["convergence_mc"], mc_curves = ex.mc_convergence(cfg)
    tables["convergence_lsm"], lsm_curve = ex.lsm_convergence(cfg)
    tables["convergence_american_tree"] = ex.american_tree_convergence(cfg)
    plain_reference = float(tables["convergence_american_tree"][-1]["price"])
    tables["american_tree_methods"] = ex.american_tree_methods(cfg, plain_reference)
    tables["convergence_fd_bump"] = ex.fd_bump_convergence()
    tables["convergence_mc_greeks"], greek_curves = ex.pathwise_convergence(cfg)
    tables["convergence_slopes"] = ex.convergence_slopes(
        cfg,
        tables["convergence_tree"],
        mc_curves,
        lsm_curve,
        tables["convergence_american_tree"],
        tables["convergence_fd_bump"],
        greek_curves,
    )

    for stem, rows in tables.items():
        write_csv(rows, out / f"{stem}.csv")

    summary = [
        "european_three_ways",
        "variance_reduction",
        "variance_reduction_by_spot",
        "greeks",
        "longstaff_schwartz_table1",
        "american_bounds",
        "upper_bound_by_inner_paths",
        "tree_steps_by_method",
        "american_tree_methods",
        "convergence_lsm",
        "convergence_slopes",
    ]
    text = "\n\n".join(f"## {stem}\n\n{markdown_table(tables[stem])}" for stem in summary)
    (out / "tables.md").write_text(text + "\n", encoding="utf-8", newline="\n")

    if figures:
        figure_triptych(
            tables["convergence_tree"],
            tables["convergence_mc"],
            tables["convergence_lsm"],
            out / "triptych.png",
        )
        figure_american_tree(
            tables["convergence_american_tree"],
            cfg.american_tree_reference,
            out / "american_tree_convergence.png",
        )
        figure_greeks(
            tables["convergence_fd_bump"],
            tables["convergence_mc_greeks"],
            out / "greeks_convergence.png",
        )
    return tables
