"""Plotting helpers for naked-singularity results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from numerical_solutions.calculations.naked_singularity.solver import SimulationResult


def plot_results(result: SimulationResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Save metric and scalar-field plots and return their paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metric_path = destination / "metric_function.png"
    scalar_path = destination / "scalar_field.png"

    metric_figure, metric_axes = plt.subplots(figsize=(8, 5))
    metric_axes.plot(result.x, result.state[0], color="purple", label=r"$f(x)$")
    metric_axes.plot(
        result.x,
        1.0 - 2.0 / result.x,
        color="black",
        linestyle="--",
        label=r"Schwarzschild $1-2/x$",
    )
    metric_axes.set(xlabel="x", ylabel="f(x)", title="Metric function")
    metric_axes.grid(True, linestyle="--", alpha=0.5)
    metric_axes.legend()
    metric_figure.tight_layout()
    metric_figure.savefig(metric_path, dpi=160)
    plt.close(metric_figure)

    scalar_figure, scalar_axes = plt.subplots(figsize=(8, 5))
    scalar_axes.semilogy(result.x, np.abs(result.state[1]), color="purple")
    scalar_axes.set(xlabel="x", ylabel=r"$|\Phi(x)|$", title="Scalar field")
    scalar_axes.grid(True, which="both", linestyle="--", alpha=0.5)
    scalar_figure.tight_layout()
    scalar_figure.savefig(scalar_path, dpi=160)
    plt.close(scalar_figure)

    return metric_path, scalar_path
