"""Command-line interface for the numerical-solutions project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from numerical_solutions.calculations.naked_singularity.plotting import plot_results
from numerical_solutions.calculations.naked_singularity.solver import (
    IntegrationError,
    SimulationConfig,
    solve_backward_forward,
)


def _add_naked_singularity_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "naked-singularity",
        help="solve the dimensionless scalar-field naked-singularity problem",
    )
    parser.add_argument("--mass-parameter", type=float, default=1.0, metavar="M_PRIME")
    parser.add_argument("--amplitude", type=float, default=1.0, metavar="C")
    parser.add_argument("--x-min", type=float, default=1.8)
    parser.add_argument("--x-max", type=float, default=7.0)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--accuracy-threshold", type=float, default=1e-6)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.set_defaults(command_handler=_run_naked_singularity)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="numerical-solutions",
        description="Numerical solutions for general-relativity models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_naked_singularity_parser(subparsers)
    return parser


def _run_naked_singularity(args: argparse.Namespace) -> int:
    config = SimulationConfig(
        mass_parameter=args.mass_parameter,
        amplitude=args.amplitude,
        x_min=args.x_min,
        x_max=args.x_max,
        rtol=args.rtol,
        atol=args.atol,
        accuracy_threshold=args.accuracy_threshold,
        samples=args.samples,
    )
    result = solve_backward_forward(config)
    metric_path, scalar_path = plot_results(result, args.output_dir)

    labels = ("f", "Phi", "Phi'")
    print("Naked singularity calculation")
    print(f"m'={config.mass_parameter:.8g}, C={config.amplitude:.8g}")
    print(f"interval=[{config.x_min:.8g}, {config.x_max:.8g}]")
    print(f"f_min={result.f_min:.9g} at x={result.x_at_f_min:.9g}")
    print(f"Phi(x_min)={result.inner_state[1]:.9g}")
    horizon_status = "detected" if result.horizon_detected else "not detected"
    print(f"horizon on interval: {horizon_status}")
    for index, label in enumerate(labels):
        print(
            f"round-trip {label}: "
            f"abs={result.errors.absolute[index]:.6e}, "
            f"rel={result.errors.relative[index]:.6e}"
        )
    accuracy_status = "met" if result.errors.threshold_met else "NOT MET"
    print(
        f"accuracy threshold {config.accuracy_threshold:.6e}: "
        f"{accuracy_status} (max relative={result.errors.max_relative:.6e})"
    )
    print(f"metric plot: {metric_path}")
    print(f"scalar-field plot: {scalar_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.command_handler(args))
    except (IntegrationError, OSError, ValueError, FloatingPointError) as error:
        parser.exit(1, f"error: {error}\n")

    return 1  # pragma: no cover - argparse handlers always return or exit
