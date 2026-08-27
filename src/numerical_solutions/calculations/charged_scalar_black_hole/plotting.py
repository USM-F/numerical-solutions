"""Plots for a charged scalar black-hole configuration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BlackHoleSolution,
)


def plot_solution(
    solution: BlackHoleSolution, output_path: str | Path
) -> Path:
    """Save all five fields and the reconstructed metric function in one figure."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    labels = (r"$f$", r"$\psi$", r"$\psi'$", r"$\tau$", r"$\mu$")
    for axis, values, label in zip(axes.flat[:5], solution.state, labels, strict=True):
        axis.plot(solution.radius, values, color="purple")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.4)
    axes.flat[5].plot(solution.radius, solution.metric_F, color="black")
    axes.flat[5].set_ylabel(r"$F$")
    axes.flat[5].grid(True, linestyle="--", alpha=0.4)
    for axis in axes[-1]:
        axis.set_xlabel(r"$r/h$")
    figure.suptitle(solution.classification.label)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
