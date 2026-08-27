"""Equations and asymptotic data for the naked-singularity calculation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def system_equations(
    x: float,
    state: ArrayLike,
    mass_parameter: float,
) -> NDArray[np.float64]:
    """Return derivatives for the state ``[f, Phi, Phi']``.

    The equations are the first-order form of equations (2) and (3) in the
    source presentation after introducing the dimensionless radius ``x`` and
    mass parameter ``m' = M m``.
    """

    metric, field, field_derivative = np.asarray(state, dtype=float)
    if x == 0.0:
        raise FloatingPointError("the differential system is singular at x=0")
    if metric == 0.0:
        raise FloatingPointError("the differential system is singular at f=0")

    x_squared = x * x
    mass_squared = mass_parameter * mass_parameter
    metric_derivative = (
        1.0
        - metric * (1.0 + x_squared * field_derivative * field_derivative)
        - mass_squared * x_squared * field * field
    ) / x
    field_second_derivative = (
        mass_squared * field
        - (field_derivative / x)
        * (
            x * metric_derivative
            + metric * x_squared * field_derivative * field_derivative
            + 2.0 * metric
        )
    ) / metric

    return np.array(
        [metric_derivative, field_derivative, field_second_derivative],
        dtype=float,
    )


def asymptotic_phi(x: float, mass_parameter: float, amplitude: float) -> float:
    """Evaluate the scalar-field asymptotic approximation at large ``x``."""

    if not math.isfinite(x) or x <= 0.0:
        raise ValueError("x must be a positive finite number")
    if not math.isfinite(mass_parameter) or mass_parameter <= 0.0:
        raise ValueError("mass_parameter must be a positive finite number")
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    return amplitude * math.exp(-mass_parameter * x) / x ** (mass_parameter + 1.0)


def asymptotic_state(
    x: float,
    mass_parameter: float,
    amplitude: float,
) -> NDArray[np.float64]:
    """Return asymptotic initial data ``[f, Phi, Phi']`` at ``x``."""

    field = asymptotic_phi(x, mass_parameter, amplitude)
    field_derivative = -field * (
        mass_parameter + (mass_parameter + 1.0) / x
    )
    metric = 1.0 - 2.0 / x
    return np.array([metric, field, field_derivative], dtype=float)
