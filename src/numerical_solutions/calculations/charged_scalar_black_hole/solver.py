"""Collocation solver and physical diagnostics for section 2.6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import CubicSpline

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    DegenerateSystemError,
    ModelParameters,
    equation_residuals,
    field_derivatives,
    metric_F_derivative,
    normalized_equation_residuals,
)
from numerical_solutions.calculations.charged_scalar_black_hole.series import (
    HorizonSeries,
    InfinitySeries,
    SeriesError,
    build_horizon_series,
    build_infinity_series,
)


class BoundaryValueError(RuntimeError):
    """Raised when the black-hole boundary-value problem cannot be validated."""

    def __init__(
        self,
        message: str,
        *,
        best_residual: float | None = None,
        candidate: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.best_residual = best_residual
        self.candidate = candidate


_CHI_BOUNDS = (1e-3, 1e3)
_ADM_MASS_BOUNDS = (0.1, 20.0)


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Numerical settings for one collocation solve."""

    horizon_epsilon: float = 1e-3
    r_max: float = 20.0
    tolerance: float = 1e-7
    boundary_tolerance: float = 1e-9
    max_nodes: int = 50_000
    initial_nodes: int = 240
    samples: int = 1200
    residual_threshold: float = 1e-6
    stability_tolerance: float = 1e-5
    max_refinements: int = 3
    minimum_homotopy_step: float = 1e-3
    use_homotopy: bool = True

    def __post_init__(self) -> None:
        finite = {
            "horizon_epsilon": self.horizon_epsilon,
            "r_max": self.r_max,
            "tolerance": self.tolerance,
            "boundary_tolerance": self.boundary_tolerance,
            "residual_threshold": self.residual_threshold,
            "stability_tolerance": self.stability_tolerance,
            "minimum_homotopy_step": self.minimum_homotopy_step,
        }
        for name, value in finite.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.horizon_epsilon >= 0.1:
            raise ValueError("horizon_epsilon must be smaller than 0.1")
        if self.r_max <= 1.0 + self.horizon_epsilon:
            raise ValueError("r_max must lie outside the horizon expansion point")
        for name in ("max_nodes", "initial_nodes", "samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 20:
                raise ValueError(f"{name} must be an integer of at least 20")
        if (
            isinstance(self.max_refinements, bool)
            or not isinstance(self.max_refinements, int)
            or self.max_refinements < 1
        ):
            raise ValueError("max_refinements must be a positive integer")
        if not isinstance(self.use_homotopy, bool):
            raise ValueError("use_homotopy must be boolean")


@dataclass(frozen=True, slots=True)
class BoundarySeed:
    """Physical initial guess for the five free matching parameters."""

    mu_h: float = 0.4
    alpha: float = 0.2
    adm_mass: float = 0.5
    scalar_amplitude: float = 1.0
    tau_amplitude: float = 0.2
    psi_nodes: int = 0
    tau_nodes: int = 0

    def __post_init__(self) -> None:
        for name in ("mu_h", "adm_mass"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("alpha", "scalar_amplitude", "tau_amplitude"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("psi_nodes", "tau_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ConfigurationClassification:
    """Physical and nodal classification of a computed exterior."""

    geometry: str
    psi_nodes: int
    tau_nodes: int
    non_extremal: bool
    asymptotically_flat: bool
    exterior_regular: bool
    valid: bool
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"BH(n_psi={self.psi_nodes},n_tau={self.tau_nodes})"


@dataclass(frozen=True, slots=True)
class BlackHoleSolution:
    """One sampled branch point and all diagnostics needed for publication."""

    charge: float
    scalar_mass: float
    beta: float
    psi_h: float
    mu_h: float
    alpha: float
    adm_mass: float
    scalar_amplitude: float
    tau_amplitude: float
    decay_rate: float
    f_prime_h: float
    F_h: float
    surface_gravity: float
    horizon_series_order: int
    horizon_series_error: float
    infinity_series_error: float
    max_equation_residual: float
    boundary_residual: float
    solver_status: int
    solver_message: str
    mesh_nodes: int
    horizon_epsilon: float
    r_max: float
    classification: ConfigurationClassification
    radius: NDArray[np.float64]
    state: NDArray[np.float64]
    metric_F: NDArray[np.float64]
    refinement_count: int = 0
    refinement_verified: bool = False
    max_observable_change: float = math.inf
    max_raw_equation_residual: float = math.inf
    scalar_tail_power: float = math.nan
    tau_tail_power: float = math.nan
    solver_tolerance: float = math.nan
    solver_boundary_tolerance: float = math.nan
    solver_max_nodes: int = 0
    solver_stability_tolerance: float = math.nan
    solver_used_homotopy: bool = False

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable record without the sampled profiles."""

        return {
            "schema_version": 1,
            "equation_revision": "CSF2023-06-12-corrected-v6",
            "classification": {
                "geometry": self.classification.geometry,
                "label": self.classification.label,
                "n_psi": self.classification.psi_nodes,
                "n_tau": self.classification.tau_nodes,
                "non_extremal": self.classification.non_extremal,
                "asymptotically_flat": self.classification.asymptotically_flat,
                "exterior_regular": self.classification.exterior_regular,
                "valid": self.classification.valid,
                "reasons": list(self.classification.reasons),
            },
            "parameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "e": self.charge,
                "m": self.scalar_mass,
                "psi_h": self.psi_h,
                "mu_h": self.mu_h,
            },
            "horizon": {
                "f_prime": self.f_prime_h,
                "F_h": self.F_h,
                "surface_gravity": self.surface_gravity,
            },
            "asymptotics": {
                "M": self.adm_mass,
                "p": self.scalar_amplitude,
                "q": self.tau_amplitude,
                "g": self.decay_rate,
                "s_psi": (
                    self.scalar_tail_power
                    if math.isfinite(self.scalar_tail_power)
                    else None
                ),
                "s_tau": (
                    self.tau_tail_power
                    if math.isfinite(self.tau_tail_power)
                    else None
                ),
            },
            "diagnostics": {
                "residual_kind": "normalized-backward-error",
                "horizon_series_order": self.horizon_series_order,
                "horizon_control_order": 8,
                "horizon_series_error": self.horizon_series_error,
                "infinity_used_order": 2,
                "infinity_control_order": 3,
                "infinity_series_error": self.infinity_series_error,
                "max_equation_residual": self.max_equation_residual,
                "max_raw_equation_residual": (
                    self.max_raw_equation_residual
                    if math.isfinite(self.max_raw_equation_residual)
                    else None
                ),
                "boundary_residual": self.boundary_residual,
                "solver_status": self.solver_status,
                "solver_message": self.solver_message,
                "mesh_nodes": self.mesh_nodes,
                "horizon_epsilon": self.horizon_epsilon,
                "r_max": self.r_max,
                "refinement_count": self.refinement_count,
                "refinement_verified": self.refinement_verified,
                "max_observable_change": (
                    self.max_observable_change
                    if math.isfinite(self.max_observable_change)
                    else None
                ),
                "solver_settings": {
                    "tol": (
                        self.solver_tolerance
                        if math.isfinite(self.solver_tolerance)
                        else None
                    ),
                    "bc_tol": (
                        self.solver_boundary_tolerance
                        if math.isfinite(self.solver_boundary_tolerance)
                        else None
                    ),
                    "max_nodes": self.solver_max_nodes or None,
                    "stability_tolerance": (
                        self.solver_stability_tolerance
                        if math.isfinite(self.solver_stability_tolerance)
                        else None
                    ),
                    "use_homotopy": self.solver_used_homotopy,
                },
            },
        }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _encode_bounded(value: float, bounds: tuple[float, float], name: str) -> float:
    lower, upper = bounds
    if not lower < value < upper:
        raise ValueError(f"{name} seed must lie strictly between {lower} and {upper}")
    return _logit((value - lower) / (upper - lower))


def _decode_bounded(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return lower + (upper - lower) * _sigmoid(value)


def _mu_upper_bound(
    psi_h: float, scalar_mass: float, beta: float
) -> float:
    radicand = 1.0 / (psi_h * psi_h) - scalar_mass * scalar_mass
    if radicand <= 0.0:
        raise ValueError("psi_h must satisfy m*psi_h<1 for a non-extremal horizon")
    upper = math.sqrt(radicand) * (1.0 - 1e-8)
    if beta > 0.0:
        upper = min(upper, (1.0 - 1e-8) / math.sqrt(beta))
    return upper


def _encode_parameters(
    seed: BoundarySeed, psi_h: float, charge: float, scalar_mass: float, beta: float
) -> NDArray[np.float64]:
    upper = _mu_upper_bound(psi_h, scalar_mass, beta)
    fraction = min(max(seed.mu_h / upper, 1e-8), 1.0 - 1e-8)
    chi = 1.0 + 2.0 * seed.alpha * scalar_mass * scalar_mass / charge
    if chi <= 0.0:
        raise ValueError("the alpha seed does not admit a decaying electromagnetic tail")
    return np.array(
        [
            _logit(fraction),
            _encode_bounded(chi, _CHI_BOUNDS, "chi"),
            _encode_bounded(seed.adm_mass, _ADM_MASS_BOUNDS, "adm_mass"),
        ],
        dtype=float,
    )


def _decode_parameters(
    encoded: NDArray[np.float64],
    psi_h: float,
    charge: float,
    scalar_mass: float,
    beta: float,
) -> tuple[float, float, float]:
    upper = _mu_upper_bound(psi_h, scalar_mass, beta)
    mu_h = upper * _sigmoid(float(encoded[0]))
    chi = _decode_bounded(float(encoded[1]), _CHI_BOUNDS)
    alpha = charge * (chi - 1.0) / (2.0 * scalar_mass * scalar_mass)
    adm_mass = _decode_bounded(float(encoded[2]), _ADM_MASS_BOUNDS)
    return mu_h, alpha, adm_mass


def _initial_mesh(config: SolverConfig) -> NDArray[np.float64]:
    inner = config.horizon_epsilon
    outer = config.r_max - 1.0
    offsets = np.geomspace(inner, outer, config.initial_nodes)
    return 1.0 + offsets


def _count_nodes(values: NDArray[np.float64]) -> int:
    threshold = max(float(np.max(np.abs(values))) * 1e-7, 1e-14)
    significant = values[np.abs(values) > threshold]
    if significant.size < 2:
        return 0
    return int(np.count_nonzero(significant[1:] * significant[:-1] < 0.0))


def _initial_profile(
    radius: NDArray[np.float64],
    parameters: ModelParameters,
    psi_h: float,
    mu_h: float,
    seed: BoundarySeed,
    left: HorizonSeries,
    left_order: int,
    right: InfinitySeries,
) -> NDArray[np.float64]:
    offset = radius - 1.0
    span = radius[-1] - radius[0]
    phase = (radius - radius[0]) / span
    blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
    right_states = np.column_stack(
        [right.state(float(point), order=2) for point in radius]
    )

    horizon_slope = float(left.metric[1])
    metric_inner = 1.0 - np.exp(-horizon_slope * offset)
    metric = (1.0 - blend) * metric_inner + blend * right_states[0]

    psi_shape = np.cos(seed.psi_nodes * math.pi * phase)
    horizon_log_slope = float(left.field[1]) / psi_h
    field_inner = (
        psi_h
        * np.exp(np.clip(horizon_log_slope * offset, -700.0, 700.0))
        * psi_shape
    )
    field = (1.0 - blend) * field_inner + blend * right_states[1]
    tau_shape = np.cos(seed.tau_nodes * math.pi * phase)
    tau_inner = (
        mu_h
        * offset
        * np.exp(-parameters.decay_rate * offset)
        * tau_shape
    )
    tau = (1.0 - blend) * tau_inner + blend * right_states[3]
    field_prime = np.gradient(field, radius, edge_order=2)
    mu = np.gradient(tau, radius, edge_order=2)
    state = np.vstack((metric, field, field_prime, tau, mu))
    state[:, 0] = left.state(radius[0] - 1.0, left_order)
    state[:, -1] = right.state(radius[-1], order=2)
    return state


def _solver_derivatives(
    r: float,
    state: NDArray[np.float64],
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    # The undivided row-scaled block remains regular farther into an
    # exponentially small tail.  The independently expanded beta=0 system is
    # retained as a mandatory regression oracle rather than the primary RHS.
    return field_derivatives(r, state, parameters)


def _classify(
    radius: NDArray[np.float64],
    state: NDArray[np.float64],
    f_prime_h: float,
    max_residual: float,
    residual_threshold: float,
) -> ConfigurationClassification:
    reasons: list[str] = []
    non_extremal = bool(f_prime_h > 0.0)
    exterior_regular = bool(np.all(np.isfinite(state)) and np.all(state[0] > 0.0))
    asymptotically_flat = bool(
        abs(state[0, -1] - 1.0) < max(0.2, 4.0 / radius[-1])
        and abs(state[1, -1]) < max(abs(state[1, 0]) * 1e-3, 1e-8)
        and abs(state[3, -1]) < max(abs(state[3, 0]) * 1e-3, 1e-8)
    )
    if not non_extremal:
        reasons.append("f'(1)<=0")
    if not exterior_regular:
        reasons.append("f<=0 or a non-finite field occurs outside the horizon")
    if not asymptotically_flat:
        reasons.append("the sampled exterior has not reached its decaying asymptotics")
    if max_residual > residual_threshold:
        reasons.append("the equation residual exceeds the publication threshold")
    valid = not reasons
    return ConfigurationClassification(
        geometry="static-spherical-asymptotically-flat-nonextremal-black-hole",
        psi_nodes=_count_nodes(state[1]),
        tau_nodes=_count_nodes(state[3]),
        non_extremal=non_extremal,
        asymptotically_flat=asymptotically_flat,
        exterior_regular=exterior_regular,
        valid=valid,
        reasons=tuple(reasons),
    )


def _solve_once(
    *,
    charge: float,
    scalar_mass: float,
    beta: float,
    psi_h: float,
    seed: BoundarySeed | None = None,
    config: SolverConfig | None = None,
    initial_solution: BlackHoleSolution | None = None,
) -> BlackHoleSolution:
    """Solve one collocation problem without publication refinements."""

    chosen_seed = seed or BoundarySeed()
    chosen_config = config or SolverConfig()
    encoded_seed = _encode_parameters(
        chosen_seed, psi_h, charge, scalar_mass, beta
    )
    initial_mu, initial_alpha, initial_mass = _decode_parameters(
        encoded_seed, psi_h, charge, scalar_mass, beta
    )
    initial_parameters = ModelParameters(initial_alpha, beta, charge, scalar_mass)
    initial_left = build_horizon_series(initial_parameters, psi_h, initial_mu)
    effective_epsilon = chosen_config.horizon_epsilon
    while True:
        try:
            initial_order = initial_left.select_order(
                effective_epsilon, chosen_config.boundary_tolerance
            )
            break
        except SeriesError as error:
            effective_epsilon *= 0.5
            if effective_epsilon < 1e-8:
                raise BoundaryValueError(
                    "the horizon expansion did not reach the boundary tolerance "
                    "before epsilon=1e-8"
                ) from error
    if effective_epsilon != chosen_config.horizon_epsilon:
        chosen_config = replace(chosen_config, horizon_epsilon=effective_epsilon)
    radius = _initial_mesh(chosen_config)
    initial_right = build_infinity_series(
        initial_parameters,
        initial_mass,
        chosen_seed.scalar_amplitude,
        chosen_seed.tau_amplitude,
    )
    initial_state = _initial_profile(
        radius,
        initial_parameters,
        psi_h,
        initial_mu,
        chosen_seed,
        initial_left,
        initial_order,
        initial_right,
    )
    if initial_solution is not None:
        shared = (radius >= initial_solution.radius[0]) & (
            radius <= initial_solution.radius[-1]
        )
        for component in range(initial_state.shape[0]):
            initial_state[component, shared] = np.interp(
                radius[shared],
                initial_solution.radius,
                initial_solution.state[component],
            )
        initial_state[:, 0] = initial_left.state(
            chosen_config.horizon_epsilon, initial_order
        )
        initial_state[:, -1] = initial_right.state(chosen_config.r_max, order=2)

    def rhs(
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        encoded: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _, alpha, _ = _decode_parameters(
            encoded, psi_h, charge, scalar_mass, beta
        )
        physical = ModelParameters(alpha, beta, charge, scalar_mass)
        result = np.empty_like(values)
        for index, point in enumerate(points):
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    result[:, index] = _solver_derivatives(
                        float(point), values[:, index], physical
                    )
                if not np.all(np.isfinite(result[:, index])):
                    raise FloatingPointError("non-finite collocation derivative")
            except (ValueError, DegenerateSystemError, FloatingPointError):
                result[:, index] = np.full(5, 1e12)
        return result

    def boundary(
        left_value: NDArray[np.float64],
        right_value: NDArray[np.float64],
        encoded: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        try:
            mu_h, alpha, adm_mass = _decode_parameters(
                encoded, psi_h, charge, scalar_mass, beta
            )
            physical = ModelParameters(alpha, beta, charge, scalar_mass)
            left_series = build_horizon_series(physical, psi_h, mu_h)
            left_order = left_series.select_order(
                chosen_config.horizon_epsilon, chosen_config.boundary_tolerance
            )
            right_series = build_infinity_series(
                physical, adm_mass, 1.0, 1.0
            )
            expected_left = left_series.state(
                chosen_config.horizon_epsilon, left_order
            )
            scalar_log_derivative, tau_log_derivative = (
                right_series.log_derivatives(chosen_config.r_max, order=2)
            )
            left_scale = np.maximum(
                np.abs(expected_left),
                np.array([1e-4, 1e-3, 1e-3, 1e-4, 1e-3]),
            )
            metric_expected = 1.0 - 2.0 * adm_mass / chosen_config.r_max
            scalar_robin = right_value[2] - scalar_log_derivative * right_value[1]
            tau_robin = right_value[4] - tau_log_derivative * right_value[3]
            scalar_scale = max(
                abs(right_value[2]),
                abs(scalar_log_derivative * right_value[1]),
                1e-300,
            )
            tau_scale = max(
                abs(right_value[4]),
                abs(tau_log_derivative * right_value[3]),
                1e-300,
            )
            return np.concatenate(
                (
                    (left_value - expected_left) / left_scale,
                    np.array(
                        [
                            (right_value[0] - metric_expected)
                            / max(abs(metric_expected), 1e-2),
                            scalar_robin / scalar_scale,
                            tau_robin / tau_scale,
                        ]
                    ),
                )
            )
        except (ValueError, OverflowError, SeriesError, FloatingPointError):
            return np.full(8, 1e8, dtype=float)

    def failed_candidate_snapshot(candidate, *, stage: str) -> dict[str, Any]:
        """Preserve a failed iterate so atlas diagnostics remain inspectable."""

        rms = np.asarray(candidate.rms_residuals, dtype=float)
        finite_rms = np.abs(rms[np.isfinite(rms)])
        maximum_rms = float(np.max(finite_rms)) if finite_rms.size else None
        minimum_rms = float(np.min(finite_rms)) if finite_rms.size else None
        try:
            boundary_values = boundary(
                candidate.y[:, 0], candidate.y[:, -1], candidate.p
            )
            finite_boundary = np.abs(
                boundary_values[np.isfinite(boundary_values)]
            )
            boundary_error = (
                float(np.max(finite_boundary)) if finite_boundary.size else None
            )
        except Exception:  # A failed iterate may not have decodable parameters.
            boundary_error = None
        quality_terms = [
            value
            for value in (maximum_rms, boundary_error)
            if value is not None and math.isfinite(value)
        ]
        ranking_residual = max(quality_terms) if quality_terms else None
        try:
            mu_h, alpha, adm_mass = _decode_parameters(
                candidate.p, psi_h, charge, scalar_mass, beta
            )
            fitted_parameters: dict[str, float] | None = {
                "mu_h": mu_h,
                "alpha": alpha,
                "M": adm_mass,
            }
        except (ValueError, OverflowError):
            fitted_parameters = None
        return {
            "problem": {
                "e": charge,
                "m": scalar_mass,
                "beta": beta,
                "psi_h": psi_h,
            },
            "source_seed": asdict(chosen_seed),
            "fitted_parameters": fitted_parameters,
            "diagnostics": {
                "residual_kind": (
                    "max-of-collocation-rms-relative-and-normalized-boundary"
                ),
                "ranking_residual": ranking_residual,
                "max_interval_rms_residual": maximum_rms,
                "min_interval_rms_residual": minimum_rms,
                "boundary_residual": boundary_error,
                "solver_status": int(candidate.status),
                "solver_message": str(candidate.message),
                "mesh_nodes": int(candidate.x.size),
                "stage": stage,
            },
            "radius": np.asarray(candidate.x, dtype=float),
            "state": np.asarray(candidate.y, dtype=float),
            "encoded_parameters": np.asarray(candidate.p, dtype=float),
        }

    if not chosen_config.use_homotopy:
        try:
            solution = solve_bvp(
                rhs,
                boundary,
                radius,
                initial_state,
                p=encoded_seed,
                tol=chosen_config.tolerance,
                bc_tol=chosen_config.boundary_tolerance,
                max_nodes=chosen_config.max_nodes,
                verbose=0,
            )
        except Exception as error:
            raise BoundaryValueError(f"collocation failed: {error}") from error
        if not solution.success:
            failed_candidate = failed_candidate_snapshot(solution, stage="direct")
            raise BoundaryValueError(
                f"collocation failed: {solution.message}",
                best_residual=failed_candidate["diagnostics"]["ranking_residual"],
                candidate=failed_candidate,
            )

    baseline = tuple(
        CubicSpline(radius, initial_state[component])
        for component in range(initial_state.shape[0])
    )

    def homotopy_rhs(
        fraction: float,
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        encoded: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        physical = rhs(points, values, encoded)
        if fraction == 1.0:
            return physical
        baseline_derivative = np.vstack(
            [spline(points, 1) for spline in baseline]
        )
        return (1.0 - fraction) * baseline_derivative + fraction * physical

    fraction = 0.0
    homotopy_step = 0.2
    mesh = radius
    guess = initial_state
    parameters_guess = encoded_seed
    if chosen_config.use_homotopy:
        solution = None
    while chosen_config.use_homotopy and fraction < 1.0 - 1e-14:
        target = min(1.0, fraction + homotopy_step)

        def staged_rhs(points, values, encoded, target=target):
            return homotopy_rhs(target, points, values, encoded)

        def staged_boundary(left_value, right_value, encoded, target=target):
            physical_boundary = boundary(left_value, right_value, encoded)
            parameter_anchor = encoded - encoded_seed
            return np.concatenate(
                (
                    physical_boundary[:5],
                    (1.0 - target) * parameter_anchor
                    + target * physical_boundary[5:],
                )
            )

        try:
            candidate = solve_bvp(
                staged_rhs,
                staged_boundary,
                mesh,
                guess,
                p=parameters_guess,
                tol=(
                    chosen_config.tolerance
                    if target == 1.0
                    else max(chosen_config.tolerance, 1e-4)
                ),
                bc_tol=(
                    chosen_config.boundary_tolerance
                    if target == 1.0
                    else max(chosen_config.boundary_tolerance, 1e-6)
                ),
                max_nodes=chosen_config.max_nodes,
                verbose=0,
            )
        except Exception as error:  # SciPy may wrap singular failures poorly.
            candidate = None
            failure_message = str(error)
        else:
            failure_message = str(candidate.message)
        if candidate is None or not candidate.success:
            homotopy_step *= 0.5
            if homotopy_step < chosen_config.minimum_homotopy_step:
                failed_candidate = (
                    failed_candidate_snapshot(
                        candidate, stage=f"homotopy-lambda-{target:.12g}"
                    )
                    if candidate is not None
                    else None
                )
                raise BoundaryValueError(
                    f"collocation homotopy failed near lambda={target:.6g}: "
                    f"{failure_message}",
                    best_residual=(
                        failed_candidate["diagnostics"]["ranking_residual"]
                        if failed_candidate is not None
                        else None
                    ),
                    candidate=failed_candidate,
                )
            continue
        solution = candidate
        fraction = target
        mesh = solution.x
        guess = solution.y
        parameters_guess = solution.p
        homotopy_step = min(homotopy_step * 1.5, 1.0 - fraction)

    if solution is None:  # pragma: no cover - protected by the branches above.
        raise BoundaryValueError("collocation homotopy produced no solution")

    mu_h, alpha, adm_mass = _decode_parameters(
        solution.p, psi_h, charge, scalar_mass, beta
    )
    physical = ModelParameters(alpha, beta, charge, scalar_mass)
    left_series = build_horizon_series(physical, psi_h, mu_h)
    left_order = left_series.select_order(
        chosen_config.horizon_epsilon, chosen_config.boundary_tolerance
    )
    amplitude_radius = min(
        chosen_config.r_max,
        max(
            5.0,
            5.0 / physical.scalar_mass,
            5.0 / physical.decay_rate,
        ),
    )
    amplitude_state = np.asarray(solution.sol(amplitude_radius), dtype=float)
    unit_series = build_infinity_series(physical, adm_mass, 1.0, 1.0)
    unit_state = unit_series.state(amplitude_radius, order=2)

    def fit_amplitude(observed: NDArray[np.float64], basis: NDArray[np.float64]) -> float:
        scale = float(np.max(np.abs(basis)))
        if not math.isfinite(scale) or scale <= 1e-300:
            raise BoundaryValueError("an asymptotic amplitude underflowed")
        normalized = basis / scale
        return float(np.dot(normalized, observed) / (scale * np.dot(normalized, normalized)))

    scalar_amplitude = fit_amplitude(amplitude_state[1:3], unit_state[1:3])
    tau_amplitude = fit_amplitude(amplitude_state[3:5], unit_state[3:5])
    right_series = build_infinity_series(
        physical, adm_mass, scalar_amplitude, tau_amplitude
    )

    sample_radius = np.geomspace(
        chosen_config.horizon_epsilon,
        chosen_config.r_max - 1.0,
        chosen_config.samples,
    ) + 1.0
    sample_state = np.asarray(solution.sol(sample_radius), dtype=float)
    sample_derivative = np.asarray(solution.sol(sample_radius, 1), dtype=float)
    F_prime = np.empty(chosen_config.samples, dtype=float)
    residuals = np.empty((4, chosen_config.samples), dtype=float)
    normalized_residuals = np.empty((4, chosen_config.samples), dtype=float)
    for index, point in enumerate(sample_radius):
        F_prime[index] = metric_F_derivative(
            point, sample_state[:, index], physical
        )
        derivative_values = np.array(
            [
                sample_derivative[0, index],
                sample_derivative[2, index],
                sample_derivative[3, index],
                sample_derivative[4, index],
            ]
        )
        residuals[:, index] = equation_residuals(
            point,
            sample_state[:, index],
            derivative_values,
            physical,
        )
        normalized_residuals[:, index] = normalized_equation_residuals(
            point,
            sample_state[:, index],
            derivative_values,
            physical,
        )
    metric_F = cumulative_trapezoid(
        F_prime[::-1], sample_radius[::-1], initial=0.0
    )[::-1]
    f_prime_h = float(left_series.metric[1])
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(16)
    horizon_offsets = (
        0.5
        * chosen_config.horizon_epsilon
        * (quadrature_nodes + 1.0)
    )
    horizon_F_prime = np.array(
        [
            metric_F_derivative(
                1.0 + offset,
                left_series.state(float(offset), left_order),
                physical,
            )
            for offset in horizon_offsets
        ]
    )
    horizon_integral = (
        0.5
        * chosen_config.horizon_epsilon
        * float(np.dot(quadrature_weights, horizon_F_prime))
    )
    F_h = float(metric_F[0] - horizon_integral)
    surface_gravity = 0.5 * math.exp(F_h) * f_prime_h
    max_raw_residual = float(np.max(np.abs(residuals)))
    max_residual = float(np.max(np.abs(normalized_residuals)))
    boundary_residual = float(
        np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1], solution.p)))
    )
    classification = _classify(
        sample_radius,
        sample_state,
        f_prime_h,
        max_residual,
        chosen_config.residual_threshold,
    )
    return BlackHoleSolution(
        charge=charge,
        scalar_mass=scalar_mass,
        beta=beta,
        psi_h=psi_h,
        mu_h=mu_h,
        alpha=alpha,
        adm_mass=adm_mass,
        scalar_amplitude=scalar_amplitude,
        tau_amplitude=tau_amplitude,
        decay_rate=physical.decay_rate,
        f_prime_h=f_prime_h,
        F_h=F_h,
        surface_gravity=surface_gravity,
        horizon_series_order=left_order,
        horizon_series_error=left_series.truncation_error(
            chosen_config.horizon_epsilon, left_order
        ),
        infinity_series_error=right_series.truncation_error(
            chosen_config.r_max, 2
        ),
        max_equation_residual=max_residual,
        boundary_residual=boundary_residual,
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        mesh_nodes=int(solution.x.size),
        horizon_epsilon=chosen_config.horizon_epsilon,
        r_max=chosen_config.r_max,
        classification=classification,
        radius=sample_radius,
        state=sample_state,
        metric_F=metric_F,
        max_raw_equation_residual=max_raw_residual,
        scalar_tail_power=right_series.scalar_power,
        tau_tail_power=right_series.tau_power,
        solver_tolerance=chosen_config.tolerance,
        solver_boundary_tolerance=chosen_config.boundary_tolerance,
        solver_max_nodes=chosen_config.max_nodes,
        solver_stability_tolerance=chosen_config.stability_tolerance,
        solver_used_homotopy=chosen_config.use_homotopy,
    )


def _observable_change(
    previous: BlackHoleSolution, current: BlackHoleSolution
) -> float:
    """Return the normalized maximum change of publication observables."""

    left = np.array(
        [
            previous.alpha,
            previous.mu_h,
            previous.f_prime_h,
            previous.F_h,
            previous.adm_mass,
            previous.scalar_amplitude,
            previous.tau_amplitude,
            previous.decay_rate,
            previous.surface_gravity,
        ],
        dtype=float,
    )
    right = np.array(
        [
            current.alpha,
            current.mu_h,
            current.f_prime_h,
            current.F_h,
            current.adm_mass,
            current.scalar_amplitude,
            current.tau_amplitude,
            current.decay_rate,
            current.surface_gravity,
        ],
        dtype=float,
    )
    scale = np.maximum.reduce((np.abs(left), np.abs(right), np.ones_like(left)))
    return float(np.max(np.abs(right - left) / scale))


def _set_refinement_status(
    solution: BlackHoleSolution,
    *,
    count: int,
    change: float,
    verified: bool,
    failure_reason: str | None = None,
) -> BlackHoleSolution:
    reasons = list(solution.classification.reasons)
    if not verified:
        reasons.append(
            failure_reason
            or "observables did not stabilize under boundary and tolerance refinement"
        )
    classification = replace(
        solution.classification,
        valid=solution.classification.valid and verified,
        reasons=tuple(dict.fromkeys(reasons)),
    )
    return replace(
        solution,
        classification=classification,
        refinement_count=count,
        refinement_verified=verified,
        max_observable_change=change,
    )


def solve_black_hole(
    *,
    charge: float,
    scalar_mass: float,
    beta: float,
    psi_h: float,
    seed: BoundarySeed | None = None,
    config: SolverConfig | None = None,
    warm_start: BlackHoleSolution | None = None,
) -> BlackHoleSolution:
    """Solve and independently refine one corrected section-2.6 BVP.

    A configuration is publishable only after at least one solve with a
    smaller horizon cutoff, larger outer boundary and tighter collocation
    tolerances.  Further refinements are attempted until both the third
    asymptotic term and the principal observables meet ``stability_tolerance``.
    """

    chosen_config = config or SolverConfig()
    current = _solve_once(
        charge=charge,
        scalar_mass=scalar_mass,
        beta=beta,
        psi_h=psi_h,
        seed=seed,
        config=chosen_config,
        initial_solution=warm_start,
    )
    last_change = math.inf
    for refinement in range(1, chosen_config.max_refinements + 1):
        refined_config = replace(
            chosen_config,
            horizon_epsilon=max(current.horizon_epsilon * 0.5, 1e-8),
            r_max=max(current.r_max * 1.25, current.r_max + 2.0),
            tolerance=max(chosen_config.tolerance * 0.3**refinement, 1e-10),
            boundary_tolerance=max(
                chosen_config.boundary_tolerance * 0.3**refinement, 1e-12
            ),
            initial_nodes=min(
                chosen_config.max_nodes,
                max(
                    chosen_config.initial_nodes,
                    int(chosen_config.initial_nodes * 1.25**refinement),
                ),
            ),
        )
        refined_seed = BoundarySeed(
            mu_h=current.mu_h,
            alpha=current.alpha,
            adm_mass=current.adm_mass,
            scalar_amplitude=current.scalar_amplitude,
            tau_amplitude=current.tau_amplitude,
            psi_nodes=current.classification.psi_nodes,
            tau_nodes=current.classification.tau_nodes,
        )
        try:
            candidate = _solve_once(
                charge=charge,
                scalar_mass=scalar_mass,
                beta=beta,
                psi_h=psi_h,
                seed=refined_seed,
                config=refined_config,
                initial_solution=current,
            )
        except BoundaryValueError as error:
            return _set_refinement_status(
                current,
                count=refinement - 1,
                change=last_change,
                verified=False,
                failure_reason=f"refinement failed: {error}",
            )
        last_change = _observable_change(current, candidate)
        current = candidate
        if (
            current.classification.valid
            and current.infinity_series_error <= chosen_config.stability_tolerance
            and last_change <= chosen_config.stability_tolerance
        ):
            return _set_refinement_status(
                current,
                count=refinement,
                change=last_change,
                verified=True,
            )
    return _set_refinement_status(
        current,
        count=chosen_config.max_refinements,
        change=last_change,
        verified=False,
    )


def _branch_coordinates(solution: BlackHoleSolution) -> NDArray[np.float64]:
    return np.array(
        [
            solution.psi_h,
            solution.mu_h,
            solution.alpha,
            solution.adm_mass,
        ],
        dtype=float,
    )


def solve_black_hole_pseudo_arclength(
    previous: BlackHoleSolution,
    current: BlackHoleSolution,
    step: float,
    *,
    config: SolverConfig | None = None,
) -> BlackHoleSolution:
    """Take a secant pseudo-arclength step with ``psi_h`` free.

    The ordinary boundary-value problem has three free matching parameters and
    eight boundary conditions.  Here ``psi_h`` is promoted to a fourth unknown;
    the ninth condition is the hyperplane normal to the scaled secant in
    ``(psi_h, mu_h, alpha, M)``.  The resulting point is subsequently
    passed through the same fixed-``psi_h`` publication refinements as an
    ordinary solve.
    """

    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("pseudo-arclength step must be positive and finite")
    fixed_previous = (previous.charge, previous.scalar_mass, previous.beta)
    fixed_current = (current.charge, current.scalar_mass, current.beta)
    if fixed_previous != fixed_current:
        raise ValueError("pseudo-arclength endpoints must have the same e, m and beta")

    charge, scalar_mass, beta = fixed_current
    chosen = config or SolverConfig()
    local_config = replace(
        chosen,
        horizon_epsilon=current.horizon_epsilon,
        r_max=current.r_max,
    )
    previous_coordinates = _branch_coordinates(previous)
    current_coordinates = _branch_coordinates(current)
    scales = np.maximum.reduce(
        (
            np.abs(previous_coordinates),
            np.abs(current_coordinates),
            np.ones(4, dtype=float),
        )
    )
    tangent = (current_coordinates - previous_coordinates) / scales
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise BoundaryValueError("the secant is too short for pseudo-arclength")
    tangent /= tangent_norm
    predictor_scaled = current_coordinates / scales + step * tangent
    predictor = predictor_scaled * scales
    psi_upper = (1.0 - 1e-8) / scalar_mass
    predictor[0] = float(np.clip(predictor[0], 1e-8, psi_upper))
    predicted_seed = BoundarySeed(
        mu_h=max(float(predictor[1]), 1e-8),
        alpha=float(predictor[2]),
        adm_mass=max(float(predictor[3]), 1e-8),
        scalar_amplitude=current.scalar_amplitude,
        tau_amplitude=current.tau_amplitude,
        psi_nodes=current.classification.psi_nodes,
        tau_nodes=current.classification.tau_nodes,
    )

    psi_encoded = _logit(float(np.clip(predictor[0] / psi_upper, 1e-8, 1 - 1e-8)))
    encoded_seed = np.concatenate(
        (
            np.array([psi_encoded]),
            _encode_parameters(
                predicted_seed, predictor[0], charge, scalar_mass, beta
            ),
        )
    )

    def decode(
        encoded: NDArray[np.float64],
    ) -> tuple[float, float, float, float]:
        psi_h = psi_upper * _sigmoid(float(encoded[0]))
        mu_h, alpha, adm_mass = _decode_parameters(
            encoded[1:], psi_h, charge, scalar_mass, beta
        )
        return psi_h, mu_h, alpha, adm_mass

    initial_psi, initial_mu, initial_alpha, initial_mass = decode(encoded_seed)
    initial_parameters = ModelParameters(initial_alpha, beta, charge, scalar_mass)
    initial_left = build_horizon_series(
        initial_parameters, initial_psi, initial_mu
    )
    effective_epsilon = local_config.horizon_epsilon
    while True:
        try:
            initial_order = initial_left.select_order(
                effective_epsilon, local_config.boundary_tolerance
            )
            break
        except SeriesError as error:
            effective_epsilon *= 0.5
            if effective_epsilon < 1e-8:
                raise BoundaryValueError(
                    "pseudo-arclength horizon expansion did not converge"
                ) from error
    local_config = replace(local_config, horizon_epsilon=effective_epsilon)
    radius = _initial_mesh(local_config)
    initial_right = build_infinity_series(
        initial_parameters,
        initial_mass,
        current.scalar_amplitude,
        current.tau_amplitude,
    )
    initial_state = _initial_profile(
        radius,
        initial_parameters,
        initial_psi,
        initial_mu,
        predicted_seed,
        initial_left,
        initial_order,
        initial_right,
    )
    shared = (radius >= current.radius[0]) & (radius <= current.radius[-1])
    for component in range(5):
        initial_state[component, shared] = np.interp(
            radius[shared], current.radius, current.state[component]
        )
    initial_state[:, 0] = initial_left.state(
        local_config.horizon_epsilon, initial_order
    )
    initial_state[:, -1] = initial_right.state(local_config.r_max, order=2)

    def rhs(
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        encoded: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _, _, alpha, _ = decode(encoded)
        physical = ModelParameters(alpha, beta, charge, scalar_mass)
        result = np.empty_like(values)
        for index, point in enumerate(points):
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    result[:, index] = _solver_derivatives(
                        float(point), values[:, index], physical
                    )
                if not np.all(np.isfinite(result[:, index])):
                    raise FloatingPointError("non-finite collocation derivative")
            except (ValueError, DegenerateSystemError, FloatingPointError):
                result[:, index] = np.full(5, 1e12)
        return result

    def boundary(
        left_value: NDArray[np.float64],
        right_value: NDArray[np.float64],
        encoded: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        try:
            psi_h, mu_h, alpha, adm_mass = decode(encoded)
            physical = ModelParameters(alpha, beta, charge, scalar_mass)
            left_series = build_horizon_series(physical, psi_h, mu_h)
            left_order = left_series.select_order(
                local_config.horizon_epsilon, local_config.boundary_tolerance
            )
            right_series = build_infinity_series(
                physical, adm_mass, 1.0, 1.0
            )
            expected_left = left_series.state(
                local_config.horizon_epsilon, left_order
            )
            scalar_log_derivative, tau_log_derivative = (
                right_series.log_derivatives(local_config.r_max, order=2)
            )
            left_scale = np.maximum(
                np.abs(expected_left), np.array([1e-4, 1e-3, 1e-3, 1e-4, 1e-3])
            )
            metric_expected = 1.0 - 2.0 * adm_mass / local_config.r_max
            scalar_robin = right_value[2] - scalar_log_derivative * right_value[1]
            tau_robin = right_value[4] - tau_log_derivative * right_value[3]
            scalar_scale = max(
                abs(right_value[2]),
                abs(scalar_log_derivative * right_value[1]),
                1e-300,
            )
            tau_scale = max(
                abs(right_value[4]),
                abs(tau_log_derivative * right_value[3]),
                1e-300,
            )
            coordinates = np.array([psi_h, mu_h, alpha, adm_mass])
            phase = float(np.dot(coordinates / scales - predictor_scaled, tangent))
            return np.concatenate(
                (
                    (left_value - expected_left) / left_scale,
                    np.array(
                        [
                            (right_value[0] - metric_expected)
                            / max(abs(metric_expected), 1e-2),
                            scalar_robin / scalar_scale,
                            tau_robin / tau_scale,
                        ]
                    ),
                    np.array([phase]),
                )
            )
        except (ValueError, OverflowError, SeriesError, FloatingPointError):
            return np.full(9, 1e8, dtype=float)

    try:
        pseudo_solution = solve_bvp(
            rhs,
            boundary,
            radius,
            initial_state,
            p=encoded_seed,
            tol=local_config.tolerance,
            bc_tol=local_config.boundary_tolerance,
            max_nodes=local_config.max_nodes,
            verbose=0,
        )
    except Exception as error:
        raise BoundaryValueError(
            f"pseudo-arclength collocation failed: {error}"
        ) from error
    if not pseudo_solution.success:
        raise BoundaryValueError(
            f"pseudo-arclength collocation failed: {pseudo_solution.message}"
        )

    psi_h, mu_h, alpha, adm_mass = decode(pseudo_solution.p)
    return solve_black_hole(
        charge=charge,
        scalar_mass=scalar_mass,
        beta=beta,
        psi_h=psi_h,
        seed=BoundarySeed(
            mu_h=mu_h,
            alpha=alpha,
            adm_mass=adm_mass,
            scalar_amplitude=current.scalar_amplitude,
            tau_amplitude=current.tau_amplitude,
            psi_nodes=current.classification.psi_nodes,
            tau_nodes=current.classification.tau_nodes,
        ),
        config=chosen,
        warm_start=current,
    )
