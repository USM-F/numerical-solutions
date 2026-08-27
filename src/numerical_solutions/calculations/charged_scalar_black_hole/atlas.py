"""Deterministic multistart and continuation over the demonstration atlas."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import qmc

from numerical_solutions.calculations.charged_scalar_black_hole.plotting import (
    plot_solution,
)
from numerical_solutions.calculations.charged_scalar_black_hole.results import (
    InvalidRegion,
    ResultStore,
    deduplicate_solutions,
)
from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BlackHoleSolution,
    BoundarySeed,
    BoundaryValueError,
    SolverConfig,
    solve_black_hole,
    solve_black_hole_pseudo_arclength,
)
from numerical_solutions.calculations.charged_scalar_black_hole.series import SeriesError


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    """Finite, explicitly non-exhaustive search domain from the project plan."""

    masses: tuple[float, ...] = (0.5, 1.0, 2.0)
    charge_mass_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    beta_mass_squared_max: float = 1.0
    beta_mass_squared_step: float = 0.05
    psi_h_mass_step: float = 0.05
    minimum_continuation_step: float = 1e-4
    starts_per_node_template: int = 8
    node_templates: tuple[tuple[int, int], ...] = (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 2),
    )
    random_seed: int = 20230612
    parallel_workers: int = 1

    def __post_init__(self) -> None:
        if not self.masses or not self.charge_mass_ratios:
            raise ValueError("the atlas axes must not be empty")
        if any(value <= 0.0 for value in (*self.masses, *self.charge_mass_ratios)):
            raise ValueError("masses and charge/mass ratios must be positive")
        if self.beta_mass_squared_max < 0.0:
            raise ValueError("beta_mass_squared_max must be non-negative")
        if (
            self.beta_mass_squared_step <= 0.0
            or self.psi_h_mass_step <= 0.0
            or self.minimum_continuation_step <= 0.0
        ):
            raise ValueError("continuation steps must be positive")
        if self.starts_per_node_template < 1:
            raise ValueError("starts_per_node_template must be positive")
        if isinstance(self.parallel_workers, bool) or self.parallel_workers < 1:
            raise ValueError("parallel_workers must be positive")


@dataclass(frozen=True, slots=True)
class AtlasSummary:
    attempted_starts: int
    base_solutions: int
    continued_solutions: int
    invalid_regions: int
    output_dir: Path


def _multistart_seeds(
    mass: float,
    charge: float,
    node_template: tuple[int, int],
    count: int,
    random_seed: int,
) -> Iterable[tuple[float, BoundarySeed]]:
    sampler = qmc.Sobol(d=6, scramble=True, seed=random_seed)
    exponent = int(math.ceil(math.log2(count)))
    samples = sampler.random_base2(exponent)[:count]
    psi_nodes, tau_nodes = node_template
    for sample in samples:
        compactness = 0.05 + 0.90 * sample[0]
        mu_ratio = 10.0 ** (-1.0 + 2.0 * sample[1])
        mu_h = mass * mu_ratio
        psi_h = compactness / math.sqrt(mass * mass + mu_h * mu_h)
        chi = 10.0 ** (-2.0 + 4.0 * sample[2])
        alpha = charge * (chi - 1.0) / (2.0 * mass * mass)
        adm_mass = 0.5 * 10.0 ** sample[3]
        scalar_amplitude = (-1.0 if psi_nodes % 2 else 1.0) * 10.0 ** (
            -1.0 + 2.0 * sample[4]
        )
        tau_amplitude = (-1.0 if tau_nodes % 2 else 1.0) * 10.0 ** (
            -1.0 + 2.0 * sample[5]
        )
        yield psi_h, BoundarySeed(
            mu_h=mu_h,
            alpha=alpha,
            adm_mass=adm_mass,
            scalar_amplitude=scalar_amplitude,
            tau_amplitude=tau_amplitude,
            psi_nodes=psi_nodes,
            tau_nodes=tau_nodes,
        )


def _seed_from_solution(solution: BlackHoleSolution) -> BoundarySeed:
    return BoundarySeed(
        mu_h=solution.mu_h,
        alpha=solution.alpha,
        adm_mass=solution.adm_mass,
        scalar_amplitude=solution.scalar_amplitude,
        tau_amplitude=solution.tau_amplitude,
        psi_nodes=solution.classification.psi_nodes,
        tau_nodes=solution.classification.tau_nodes,
    )


def _attempt_start(
    task: tuple[float, float, float, float, BoundarySeed, SolverConfig],
) -> tuple[
    BlackHoleSolution | None,
    str | None,
    float | None,
    dict[str, object] | None,
]:
    mass, charge, beta, psi_h, boundary_seed, solver = task
    try:
        root = solve_black_hole(
            charge=charge,
            scalar_mass=mass,
            beta=beta,
            psi_h=psi_h,
            seed=boundary_seed,
            config=solver,
        )
    except (BoundaryValueError, SeriesError, ValueError) as error:
        return (
            None,
            str(error),
            getattr(error, "best_residual", None),
            getattr(error, "candidate", None),
        )
    rejected_candidate = (
        _rejected_solution_snapshot(root, boundary_seed)
        if not root.classification.valid
        else None
    )
    residual = (
        rejected_candidate["diagnostics"]["ranking_residual"]
        if rejected_candidate is not None
        else root.max_equation_residual
    )
    return root, None, float(residual), rejected_candidate


def _rejected_solution_snapshot(
    solution: BlackHoleSolution, source_seed: BoundarySeed
) -> dict[str, object]:
    """Convert a fully evaluated but rejected solution to a diagnostic candidate."""

    ranking_residual = max(
        solution.max_equation_residual, solution.boundary_residual
    )
    return {
        "problem": {
            "e": solution.charge,
            "m": solution.scalar_mass,
            "beta": solution.beta,
            "psi_h": solution.psi_h,
        },
        "source_seed": asdict(source_seed),
        "fitted_parameters": {
            "mu_h": solution.mu_h,
            "alpha": solution.alpha,
            "M": solution.adm_mass,
            "p": solution.scalar_amplitude,
            "q": solution.tau_amplitude,
        },
        "diagnostics": {
            "residual_kind": "max-of-normalized-equation-and-boundary-residual",
            "ranking_residual": ranking_residual,
            "max_equation_residual": solution.max_equation_residual,
            "max_raw_equation_residual": solution.max_raw_equation_residual,
            "boundary_residual": solution.boundary_residual,
            "solver_status": solution.solver_status,
            "solver_message": solution.solver_message,
            "mesh_nodes": solution.mesh_nodes,
            "stage": "physical-validation-or-refinement",
        },
        "solution_summary": solution.summary(),
        "radius": solution.radius,
        "state": solution.state,
        "metric_F": solution.metric_F,
    }


def _branch_id(solution: BlackHoleSolution) -> str:
    return (
        f"m{solution.scalar_mass:g}-e{solution.charge:g}-"
        f"n{solution.classification.psi_nodes}-{solution.classification.tau_nodes}-"
        f"a{solution.alpha:.6g}"
    )


def _failure_category(message: str) -> str:
    lowered = message.lower()
    if "singular jacobian" in lowered:
        return "singular-collocation-jacobian"
    if "maximum number of mesh nodes" in lowered:
        return "maximum-mesh-nodes"
    if "horizon coefficient recurrence is singular" in lowered:
        return "singular-horizon-recurrence"
    if "refinement" in lowered:
        return "refinement-failed"
    if "residual" in lowered:
        return "residual-or-physical-validation"
    return "other-numerical-failure"


def _continue_beta(
    base: BlackHoleSolution,
    atlas: AtlasConfig,
    solver: SolverConfig,
) -> tuple[list[BlackHoleSolution], str]:
    results: list[BlackHoleSolution] = []
    scaled_beta = base.beta * base.scalar_mass * base.scalar_mass
    step = atlas.beta_mass_squared_step
    previous = base
    reason = "beta-range-complete"
    while scaled_beta + step <= atlas.beta_mass_squared_max + 1e-14:
        target_scaled = min(scaled_beta + step, atlas.beta_mass_squared_max)
        target_beta = target_scaled / (base.scalar_mass * base.scalar_mass)
        try:
            candidate = solve_black_hole(
                charge=base.charge,
                scalar_mass=base.scalar_mass,
                beta=target_beta,
                psi_h=previous.psi_h,
                seed=_seed_from_solution(previous),
                config=solver,
            )
        except BoundaryValueError:
            step *= 0.5
            if step < atlas.minimum_continuation_step:
                reason = "continuation-step-below-minimum"
                break
            continue
        if not candidate.classification.valid:
            reason = "physical-validation-failed"
            break
        results.append(candidate)
        previous = candidate
        scaled_beta = target_scaled
        step = min(step * 1.25, atlas.beta_mass_squared_step)
    return results, reason


def _continue_psi_direction(
    base: BlackHoleSolution,
    direction: int,
    atlas: AtlasConfig,
    solver: SolverConfig,
) -> tuple[list[BlackHoleSolution], str]:
    """Continue one beta=0 branch, switching to pseudo-arclength at a fold."""

    points = [base]
    step = atlas.psi_h_mass_step
    reason = "psi-h-domain-complete"
    for _ in range(200):
        current = points[-1]
        target = current.psi_h + direction * step / base.scalar_mass
        if target <= 0.0 or target * base.scalar_mass >= 1.0:
            break
        try:
            candidate = solve_black_hole(
                charge=base.charge,
                scalar_mass=base.scalar_mass,
                beta=base.beta,
                psi_h=target,
                seed=_seed_from_solution(current),
                config=solver,
                warm_start=current,
            )
        except BoundaryValueError:
            candidate = None

        if (candidate is None or not candidate.classification.valid) and len(points) >= 2:
            try:
                candidate = solve_black_hole_pseudo_arclength(
                    points[-2], current, step, config=solver
                )
            except (BoundaryValueError, ValueError):
                candidate = None

        if candidate is None or not candidate.classification.valid:
            step *= 0.5
            if step < atlas.minimum_continuation_step:
                reason = "psi-h-continuation-step-below-minimum"
                break
            continue

        compactness = candidate.psi_h * math.sqrt(
            candidate.scalar_mass**2 + candidate.mu_h**2
        )
        if not 0.05 <= compactness <= 0.95:
            reason = "psi-h-search-domain-boundary"
            break
        coordinate_change = np.max(
            np.abs(_solution_coordinates(candidate) - _solution_coordinates(current))
            / np.maximum(np.abs(_solution_coordinates(current)), 1.0)
        )
        if coordinate_change < 1e-8:
            step *= 0.5
            if step < atlas.minimum_continuation_step:
                reason = "pseudo-arclength-returned-duplicate"
                break
            continue
        points.append(candidate)
        step = min(step * 1.25, atlas.psi_h_mass_step)
    else:
        reason = "psi-h-maximum-steps"
    return points[1:], reason


def _solution_coordinates(solution: BlackHoleSolution) -> np.ndarray:
    return np.array(
        [
            solution.psi_h,
            solution.mu_h,
            solution.alpha,
            solution.adm_mass,
            solution.scalar_amplitude,
            solution.tau_amplitude,
        ],
        dtype=float,
    )


def _continue_psi(
    base: BlackHoleSolution,
    atlas: AtlasConfig,
    solver: SolverConfig,
) -> tuple[list[BlackHoleSolution], str]:
    lower, lower_reason = _continue_psi_direction(base, -1, atlas, solver)
    upper, upper_reason = _continue_psi_direction(base, 1, atlas, solver)
    points = list(reversed(lower)) + [base] + upper
    return deduplicate_solutions(points), f"lower:{lower_reason};upper:{upper_reason}"


def search_atlas(
    output_dir: str | Path,
    *,
    atlas_config: AtlasConfig | None = None,
    solver_config: SolverConfig | None = None,
    retry_invalid: bool = False,
) -> AtlasSummary:
    """Search the declared finite atlas and persist every result and failed cell."""

    atlas = atlas_config or AtlasConfig()
    solver = solver_config or SolverConfig()
    store = ResultStore(output_dir)
    store.prepare()
    known_invalid = set() if retry_invalid else store.invalid_fingerprints()
    attempted = 0
    skipped_attempted = 0
    base_count = 0
    continuation_count = 0
    invalid_count = 0
    all_base: list[BlackHoleSolution] = []
    aggregate_failures: Counter[str] = Counter()
    expected_invalid_fingerprints: set[str] = set()
    executor = (
        ProcessPoolExecutor(max_workers=atlas.parallel_workers)
        if atlas.parallel_workers > 1
        else None
    )

    for mass_index, mass in enumerate(atlas.masses):
        for ratio_index, ratio in enumerate(atlas.charge_mass_ratios):
            charge = mass * ratio
            for template_index, node_template in enumerate(atlas.node_templates):
                seed_number = (
                    atlas.random_seed
                    + 10_000 * mass_index
                    + 1_000 * ratio_index
                    + 100 * template_index
                )
                settings = {
                    **asdict(solver),
                    "starts_per_node_template": atlas.starts_per_node_template,
                    "sobol_seed": atlas.random_seed,
                    "cell_sobol_seed": seed_number,
                    "psi_h_mass_step": atlas.psi_h_mass_step,
                    "beta_mass_squared_step": atlas.beta_mass_squared_step,
                    "minimum_continuation_step": atlas.minimum_continuation_step,
                }
                bounds = {
                    "m": [mass, mass],
                    "e_over_m": [ratio, ratio],
                    "beta_m_squared": [0.0, 0.0],
                    "compactness": [0.05, 0.95],
                    "mu_h_over_m": [0.1, 10.0],
                    "chi": [1e-2, 1e2],
                }
                pending_region = InvalidRegion(
                    status="numerically_unresolved",
                    bounds=bounds,
                    node_templates=(node_template,),
                    attempted_starts=atlas.starts_per_node_template,
                    best_residual=None,
                    failure_kind="no-converged-root",
                    message="all deterministic starts failed",
                    solver_settings=settings,
                )
                expected_invalid_fingerprints.add(pending_region.fingerprint)
                if pending_region.fingerprint in known_invalid:
                    skipped_attempted += pending_region.attempted_starts
                    continue

                roots: list[BlackHoleSolution] = []
                failures: list[str] = []
                best_residual: float | None = None
                best_residual_kind: str | None = None
                best_candidate: dict[str, object] | None = None
                tasks = [
                    (mass, charge, 0.0, psi_h, boundary_seed, solver)
                    for psi_h, boundary_seed in _multistart_seeds(
                        mass,
                        charge,
                        node_template,
                        atlas.starts_per_node_template,
                        seed_number,
                    )
                ]
                outcomes = (
                    executor.map(_attempt_start, tasks)
                    if executor is not None
                    else map(_attempt_start, tasks)
                )
                for root, failure, error_residual, candidate in outcomes:
                    attempted += 1
                    if failure is not None:
                        failures.append(failure)
                        if error_residual is not None and math.isfinite(error_residual):
                            if best_residual is None or error_residual < best_residual:
                                best_residual = error_residual
                                best_candidate = candidate
                                if candidate is not None:
                                    best_residual_kind = str(
                                        candidate["diagnostics"]["residual_kind"]
                                    )
                        continue
                    assert root is not None
                    if root.classification.valid:
                        roots.append(root)
                    else:
                        assert candidate is not None
                        assert error_residual is not None
                        if best_residual is None or error_residual < best_residual:
                            best_residual = error_residual
                            best_candidate = candidate
                            best_residual_kind = str(
                                candidate["diagnostics"]["residual_kind"]
                            )
                        failures.append("; ".join(root.classification.reasons))

                roots = deduplicate_solutions(roots)
                if not roots:
                    invalid_count += 1
                    failure_counts = Counter(
                        _failure_category(message) for message in failures
                    )
                    aggregate_failures.update(failure_counts)
                    failure_kind = (
                        failure_counts.most_common(1)[0][0]
                        if failure_counts
                        else pending_region.failure_kind
                    )
                    message = (
                        "; ".join(
                            f"{kind}={count}"
                            for kind, count in sorted(failure_counts.items())
                        )
                        if failure_counts
                        else pending_region.message
                    )
                    best_candidate_id = None
                    if best_candidate is not None:
                        best_candidate_id = store.append_best_candidate(
                            best_candidate,
                            cell_fingerprint=pending_region.fingerprint,
                            failure_kind=failure_kind,
                            failure_message=message,
                        )
                    store.append_invalid_region(
                        InvalidRegion(
                            status="numerically_unresolved",
                            bounds=bounds,
                            node_templates=(node_template,),
                            attempted_starts=atlas.starts_per_node_template,
                            best_residual=best_residual,
                            best_residual_kind=best_residual_kind,
                            best_candidate_id=best_candidate_id,
                            failure_kind=failure_kind,
                            message=message,
                            solver_settings=settings,
                            failure_counts=dict(sorted(failure_counts.items())),
                        )
                    )
                    continue
                all_base.extend(roots)

    if executor is not None:
        executor.shutdown(wait=True)
    all_base = deduplicate_solutions(all_base)
    for base in all_base:
        branch_id = _branch_id(base)
        base_count += 1
        psi_branch, psi_reason = _continue_psi(base, atlas, solver)
        for psi_index, psi_solution in enumerate(psi_branch):
            if psi_solution is not base:
                continuation_count += 1
            psi_termination = (
                psi_reason
                if psi_index == len(psi_branch) - 1
                else "psi-h-continued"
            )
            record = store.append_solution(
                psi_solution,
                branch_id=branch_id,
                termination_reason=psi_termination,
            )
            plot_solution(
                psi_solution, store.plot_dir / f"{record['configuration_id']}.png"
            )
            beta_branch, beta_reason = _continue_beta(psi_solution, atlas, solver)
            for beta_index, solution in enumerate(beta_branch):
                continuation_count += 1
                termination = (
                    beta_reason
                    if beta_index == len(beta_branch) - 1
                    else "beta-continued"
                )
                record = store.append_solution(
                    solution, branch_id=branch_id, termination_reason=termination
                )
                plot_solution(
                    solution, store.plot_dir / f"{record['configuration_id']}.png"
                )

    matching_invalid = {
        record["fingerprint"]: record
        for record in store.invalid_records()
        if record["fingerprint"] in expected_invalid_fingerprints
    }
    total_attempted = attempted + skipped_attempted
    total_invalid = len(matching_invalid)
    matching_candidates = {
        record["cell_fingerprint"]: record
        for record in store.best_candidate_records()
        if record["cell_fingerprint"] in expected_invalid_fingerprints
    }
    aggregate_failures = Counter()
    for record in matching_invalid.values():
        aggregate_failures.update(record.get("failure_counts", {}))

    report_lines = [
        "# Отчёт поиска конфигураций",
        "",
        f"- Проверено стартов: {total_attempted}",
        f"- Найдено базовых конфигураций: {base_count}",
        f"- Получено точек продолжения: {continuation_count}",
        f"- Неразрешённых областей: {total_invalid}",
        f"- Сохранено лучших отклонённых кандидатов: "
        f"{len(matching_candidates)}",
        "",
        "## Точная область поиска",
        "",
        f"- $m$: `{list(atlas.masses)}`",
        f"- $e/m$: `{list(atlas.charge_mass_ratios)}`",
        f"- $0\\leq\\beta m^2\\leq$: `{atlas.beta_mass_squared_max}`",
        "- $\\psi_h\\sqrt{m^2+\\mu_h^2}$: `[0.05, 0.95]`",
        "- $\\mu_h/m$: `[0.1, 10]`",
        "- $(e^2+2\\alpha e m^2)/e^2$: `[1e-2, 1e2]`",
        f"- Стартов на узловую маску: `{atlas.starts_per_node_template}`",
        f"- Узловые маски: `{list(atlas.node_templates)}`",
        f"- Настройки решателя: `{asdict(solver)}`",
        "",
        "## Диагностика отказов",
        "",
        *(
            [f"- `{kind}`: {count}" for kind, count in sorted(aggregate_failures.items())]
            or ["- Новых запусков не выполнялось или отказов не было."]
        ),
        "",
        "Отсутствие корней означает только отсутствие найденных решений в "
        "записанной конечной области поиска.",
    ]
    (store.root / "search-report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return AtlasSummary(
        attempted_starts=total_attempted,
        base_solutions=base_count,
        continued_solutions=continuation_count,
        invalid_regions=total_invalid,
        output_dir=store.root,
    )
