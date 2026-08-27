"""Backward-forward integration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from numerical_solutions.calculations.naked_singularity.model import (
    asymptotic_state,
    system_equations,
)


class IntegrationError(RuntimeError):
    """Raised when the numerical integrator cannot complete a requested leg."""


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Inputs controlling one dimensionless naked-singularity calculation."""

    mass_parameter: float = 1.0
    amplitude: float = 1.0
    x_min: float = 1.8
    x_max: float = 7.0
    rtol: float = 1e-9
    atol: float = 1e-12
    accuracy_threshold: float = 1e-6
    samples: int = 1000

    def __post_init__(self) -> None:
        finite_values = {
            "mass_parameter": self.mass_parameter,
            "amplitude": self.amplitude,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "rtol": self.rtol,
            "atol": self.atol,
            "accuracy_threshold": self.accuracy_threshold,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("mass_parameter", "x_min", "x_max", "rtol", "atol"):
            if finite_values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.accuracy_threshold <= 0.0:
            raise ValueError("accuracy_threshold must be positive")
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise ValueError("samples must be an integer")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Round-trip error diagnostics."""

    absolute: NDArray[np.float64]
    relative: NDArray[np.float64]
    max_relative: float
    threshold_met: bool


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Solution samples and diagnostics from a backward-forward integration."""

    config: SimulationConfig
    x: NDArray[np.float64]
    state: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    inner_state: NDArray[np.float64]
    roundtrip_state: NDArray[np.float64]
    errors: ErrorMetrics
    f_min: float
    x_at_f_min: float
    horizon_detected: bool
    backward_evaluations: int
    forward_evaluations: int


def calculate_errors(
    reference: ArrayLike,
    actual: ArrayLike,
    threshold: float,
) -> ErrorMetrics:
    """Calculate absolute and component-wise relative round-trip errors."""

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a positive finite number")
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    if reference_array.shape != actual_array.shape:
        raise ValueError("reference and actual states must have the same shape")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(actual_array)):
        raise ValueError("states must contain only finite numbers")

    absolute = np.abs(actual_array - reference_array)
    denominator = np.maximum(np.abs(reference_array), np.finfo(float).eps)
    relative = absolute / denominator
    max_relative = float(np.max(relative))
    return ErrorMetrics(
        absolute=absolute,
        relative=relative,
        max_relative=max_relative,
        threshold_met=max_relative < threshold,
    )


def _check_integration(solution: object, endpoint: float, leg: str) -> None:
    if not solution.success:
        raise IntegrationError(f"{leg} integration failed: {solution.message}")
    if not np.isclose(solution.t[-1], endpoint, rtol=0.0, atol=1e-12):
        raise IntegrationError(f"{leg} integration stopped before x={endpoint}")
    if not np.all(np.isfinite(solution.y)):
        raise IntegrationError(f"{leg} integration produced non-finite values")


def solve_backward_forward(config: SimulationConfig) -> SimulationResult:
    """Integrate from infinity inward and then return to the starting point."""

    initial_state = asymptotic_state(
        config.x_max,
        config.mass_parameter,
        config.amplitude,
    )

    def right_hand_side(x: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return system_equations(x, state, config.mass_parameter)

    backward = solve_ivp(
        right_hand_side,
        (config.x_max, config.x_min),
        initial_state,
        method="Radau",
        rtol=config.rtol,
        atol=config.atol,
    )
    _check_integration(backward, config.x_min, "backward")
    inner_state = backward.y[:, -1].copy()

    sample_points = np.linspace(config.x_min, config.x_max, config.samples)
    forward = solve_ivp(
        right_hand_side,
        (config.x_min, config.x_max),
        inner_state,
        method="Radau",
        t_eval=sample_points,
        rtol=config.rtol,
        atol=config.atol,
    )
    _check_integration(forward, config.x_max, "forward")

    roundtrip_state = forward.y[:, -1].copy()
    errors = calculate_errors(
        initial_state,
        roundtrip_state,
        config.accuracy_threshold,
    )
    minimum_index = int(np.argmin(forward.y[0]))
    metric_samples = np.concatenate((backward.y[0], forward.y[0]))

    return SimulationResult(
        config=config,
        x=forward.t.copy(),
        state=forward.y.copy(),
        initial_state=initial_state,
        inner_state=inner_state,
        roundtrip_state=roundtrip_state,
        errors=errors,
        f_min=float(forward.y[0, minimum_index]),
        x_at_f_min=float(forward.t[minimum_index]),
        horizon_detected=bool(np.any(metric_samples <= 0.0)),
        backward_evaluations=int(backward.nfev),
        forward_evaluations=int(forward.nfev),
    )
