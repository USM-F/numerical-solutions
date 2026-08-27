"""Command-line interface for the numerical-solutions project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from numerical_solutions.calculations.charged_scalar_black_hole.atlas import (
    AtlasConfig,
    search_atlas,
)
from numerical_solutions.calculations.charged_scalar_black_hole.plotting import (
    plot_solution as plot_black_hole_solution,
)
from numerical_solutions.calculations.charged_scalar_black_hole.results import (
    InvalidRegion,
    ResultStore,
)
from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BoundarySeed,
    BoundaryValueError,
    SolverConfig as BlackHoleSolverConfig,
    solve_black_hole,
)
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


def _add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon-epsilon", type=float, default=1e-3)
    parser.add_argument("--r-max", type=float, default=20.0)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--bc-tol", type=float, default=1e-9)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--initial-nodes", type=int, default=240)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--minimum-homotopy-step", type=float, default=1e-3)
    parser.add_argument("--disable-homotopy", action="store_true")


def _black_hole_solver_config(args: argparse.Namespace) -> BlackHoleSolverConfig:
    return BlackHoleSolverConfig(
        horizon_epsilon=args.horizon_epsilon,
        r_max=args.r_max,
        tolerance=args.tol,
        boundary_tolerance=args.bc_tol,
        max_nodes=args.max_nodes,
        initial_nodes=args.initial_nodes,
        samples=args.samples,
        minimum_homotopy_step=args.minimum_homotopy_step,
        use_homotopy=not args.disable_homotopy,
    )


def _add_charged_black_hole_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "charged-black-hole",
        help="solve the corrected charged scalar black-hole problem from section 2.6",
    )
    actions = parser.add_subparsers(dest="black_hole_action", required=True)

    solve_parser = actions.add_parser("solve", help="solve one boundary-value seed")
    solve_parser.add_argument("--charge", type=float, default=1.0, metavar="E")
    solve_parser.add_argument("--scalar-mass", type=float, default=1.0, metavar="M")
    solve_parser.add_argument("--beta", type=float, default=0.0)
    solve_parser.add_argument("--psi-h", type=float, default=0.3)
    solve_parser.add_argument("--seed-mu-h", type=float, default=0.4)
    solve_parser.add_argument("--seed-alpha", type=float, default=0.2)
    solve_parser.add_argument("--seed-adm-mass", type=float, default=0.5)
    solve_parser.add_argument("--seed-p", type=float, default=1.0)
    solve_parser.add_argument("--seed-q", type=float, default=0.2)
    solve_parser.add_argument("--psi-nodes", type=int, default=0)
    solve_parser.add_argument("--tau-nodes", type=int, default=0)
    solve_parser.add_argument(
        "--output-dir", type=Path, default=Path("results/charged-black-hole")
    )
    _add_solver_arguments(solve_parser)
    solve_parser.set_defaults(command_handler=_run_charged_black_hole_solve)

    atlas_parser = actions.add_parser(
        "atlas", help="run the finite deterministic demonstration atlas"
    )
    atlas_parser.add_argument(
        "--masses", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    atlas_parser.add_argument(
        "--charge-mass-ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    atlas_parser.add_argument("--beta-m2-max", type=float, default=1.0)
    atlas_parser.add_argument("--beta-m2-step", type=float, default=0.05)
    atlas_parser.add_argument("--psi-h-mass-step", type=float, default=0.05)
    atlas_parser.add_argument("--minimum-step", type=float, default=1e-4)
    atlas_parser.add_argument("--starts", type=int, default=8)
    atlas_parser.add_argument("--random-seed", type=int, default=20230612)
    atlas_parser.add_argument("--jobs", type=int, default=1)
    atlas_parser.add_argument("--retry-invalid", action="store_true")
    atlas_parser.add_argument(
        "--output-dir", type=Path, default=Path("results/charged-black-hole")
    )
    _add_solver_arguments(atlas_parser)
    atlas_parser.set_defaults(command_handler=_run_charged_black_hole_atlas)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="numerical-solutions",
        description="Numerical solutions for general-relativity models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_naked_singularity_parser(subparsers)
    _add_charged_black_hole_parser(subparsers)
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


def _run_charged_black_hole_solve(args: argparse.Namespace) -> int:
    solver_config = _black_hole_solver_config(args)
    seed = BoundarySeed(
        mu_h=args.seed_mu_h,
        alpha=args.seed_alpha,
        adm_mass=args.seed_adm_mass,
        scalar_amplitude=args.seed_p,
        tau_amplitude=args.seed_q,
        psi_nodes=args.psi_nodes,
        tau_nodes=args.tau_nodes,
    )
    solution = solve_black_hole(
        charge=args.charge,
        scalar_mass=args.scalar_mass,
        beta=args.beta,
        psi_h=args.psi_h,
        seed=seed,
        config=solver_config,
    )
    store = ResultStore(args.output_dir)
    if not solution.classification.valid:
        store.append_invalid_region(
            InvalidRegion(
                status="numerically_unresolved",
                bounds={
                    "m": [args.scalar_mass, args.scalar_mass],
                    "e": [args.charge, args.charge],
                    "beta": [args.beta, args.beta],
                    "psi_h": [args.psi_h, args.psi_h],
                },
                node_templates=((args.psi_nodes, args.tau_nodes),),
                attempted_starts=1,
                best_residual=solution.max_equation_residual,
                failure_kind="physical-or-refinement-validation-failed",
                message="; ".join(solution.classification.reasons),
                solver_settings=asdict(solver_config),
            )
        )
        print("Charged scalar black-hole calculation")
        print("valid=False")
        print("diagnostics: " + "; ".join(solution.classification.reasons))
        print(f"unresolved-region JSONL: {store.invalid_regions_path}")
        return 2
    branch_id = (
        f"single-n{solution.classification.psi_nodes}-"
        f"{solution.classification.tau_nodes}"
    )
    record = store.append_solution(
        solution, branch_id=branch_id, termination_reason="single-solve"
    )
    plot_path = plot_black_hole_solution(
        solution, store.plot_dir / f"{record['configuration_id']}.png"
    )
    print("Charged scalar black-hole calculation")
    print(f"configuration: {solution.classification.label}")
    print(
        f"alpha={solution.alpha:.9g}, beta={solution.beta:.9g}, "
        f"e={solution.charge:.9g}, m={solution.scalar_mass:.9g}"
    )
    print(
        f"psi_h={solution.psi_h:.9g}, mu_h={solution.mu_h:.9g}, "
        f"f'(1)={solution.f_prime_h:.9g}"
    )
    print(
        f"M={solution.adm_mass:.9g}, p={solution.scalar_amplitude:.9g}, "
        f"q={solution.tau_amplitude:.9g}, g={solution.decay_rate:.9g}"
    )
    print(f"max equation residual={solution.max_equation_residual:.6e}")
    print(f"valid={solution.classification.valid}")
    print(f"JSONL: {store.configurations_path}")
    print(f"plot: {plot_path}")
    return 0


def _run_charged_black_hole_atlas(args: argparse.Namespace) -> int:
    atlas = AtlasConfig(
        masses=tuple(args.masses),
        charge_mass_ratios=tuple(args.charge_mass_ratios),
        beta_mass_squared_max=args.beta_m2_max,
        beta_mass_squared_step=args.beta_m2_step,
        psi_h_mass_step=args.psi_h_mass_step,
        minimum_continuation_step=args.minimum_step,
        starts_per_node_template=args.starts,
        random_seed=args.random_seed,
        parallel_workers=args.jobs,
    )
    summary = search_atlas(
        args.output_dir,
        atlas_config=atlas,
        solver_config=_black_hole_solver_config(args),
        retry_invalid=args.retry_invalid,
    )
    print("Charged scalar black-hole atlas")
    print(f"attempted starts: {summary.attempted_starts}")
    print(f"base configurations: {summary.base_solutions}")
    print(f"continued configurations: {summary.continued_solutions}")
    print(f"invalid/unresolved regions: {summary.invalid_regions}")
    print(f"output: {summary.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.command_handler(args))
    except (
        BoundaryValueError,
        IntegrationError,
        OSError,
        ValueError,
        FloatingPointError,
    ) as error:
        parser.exit(1, f"error: {error}\n")

    return 1  # pragma: no cover - argparse handlers always return or exit
