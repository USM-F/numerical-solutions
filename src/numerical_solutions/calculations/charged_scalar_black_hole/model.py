"""Corrected equations from section 2.6 of ``CSF2023-06-12.pdf``.

The state is ``[f, psi, psi', tau, mu]``.  The implementation retains the
implicit two-by-two derivative block from equations (2.30)--(2.31), rather
than expanding it into a large and difficult-to-audit rational expression.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DegenerateSystemError(RuntimeError):
    """Raised when the differential system loses its regular derivative block."""


@dataclass(frozen=True, slots=True)
class ModelParameters:
    """Dimensionless physical parameters after setting the horizon radius to one."""

    alpha: float
    beta: float
    charge: float
    scalar_mass: float

    def __post_init__(self) -> None:
        values = {
            "alpha": self.alpha,
            "beta": self.beta,
            "charge": self.charge,
            "scalar_mass": self.scalar_mass,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.charge <= 0.0:
            raise ValueError("charge must be positive")
        if self.scalar_mass <= 0.0:
            raise ValueError("scalar_mass must be positive")

    @property
    def decay_rate(self) -> float:
        """Return the corrected positive exponent ``g`` of the electromagnetic tail."""

        m = self.scalar_mass
        e = self.charge
        numerator = e * e + 2.0 * self.alpha * e * m * m
        denominator = 1.0 + self.beta * m * m
        radicand = m * m + numerator / denominator
        if numerator <= 0.0 or radicand <= m * m:
            raise ValueError(
                "the parameters do not admit the required decaying electromagnetic mode"
            )
        return math.sqrt(radicand) - m


def _validate_point(r: float, state: ArrayLike) -> NDArray[np.float64]:
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("r must be a positive finite number")
    values = np.asarray(state, dtype=float)
    if values.shape != (5,):
        raise ValueError("state must have shape (5,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain finite values")
    return values


def metric_f_derivative(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
) -> float:
    """Solve equation (2.29) for ``f'``."""

    metric, field, field_prime, tau, mu = _validate_point(r, state)
    if metric == 0.0:
        raise DegenerateSystemError("equation (2.29) is singular at f=0")
    e = parameters.charge
    m = parameters.scalar_mass
    alpha = parameters.alpha
    beta = parameters.beta
    matter = (
        metric * field_prime * field_prime
        + e * e * tau * tau * field * field / metric
        + m * m * field * field
        + mu * mu * field * field
        - 2.0 * alpha * e * field * field_prime * tau * mu
        + beta
        * (
            metric * field_prime * field_prime * mu * mu
            - 3.0 * e * e * tau * tau * mu * mu * field * field / metric
        )
    )
    return (1.0 - metric) / r - r * matter


def metric_F_derivative(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
) -> float:
    """Evaluate the quadrature (2.25) for the second metric function ``F``."""

    metric, field, field_prime, tau, mu = _validate_point(r, state)
    if metric == 0.0:
        raise DegenerateSystemError("equation (2.25) is singular at f=0")
    e = parameters.charge
    alpha = parameters.alpha
    beta = parameters.beta
    kinetic_factor = 1.0 - beta * mu * mu
    return (
        r * field_prime * field_prime * kinetic_factor
        + e * e * r * tau * tau * field * field / (metric * metric) * kinetic_factor
        - 2.0
        * alpha
        * e
        * r
        * field
        * field_prime
        * tau
        * mu
        / metric
    )


def tau_derivative(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
) -> float:
    """Solve equation (2.32) for ``tau'``."""

    metric, field, field_prime, tau, mu = _validate_point(r, state)
    if metric == 0.0:
        raise DegenerateSystemError("equation (2.32) is singular at f=0")
    e = parameters.charge
    alpha = parameters.alpha
    beta = parameters.beta
    kinetic_factor = 1.0 - beta * mu * mu
    return (
        mu
        + 2.0
        * alpha
        * e
        * r
        * field
        * field_prime
        * mu
        * tau
        * tau
        / metric
        - (
            r * field_prime * field_prime * tau
            + e * e * r * tau**3 * field * field / (metric * metric)
        )
        * kinetic_factor
    )


def derivative_block(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
    *,
    determinant_tolerance: float = 1e-12,
) -> tuple[float, float, float, float, float]:
    """Return ``f'``, ``psi''``, ``tau'``, ``mu'`` and ``F'``.

    Equations (2.30) and (2.31) are linear in ``psi''`` and ``mu'`` after
    ``f'``, ``tau'`` and ``F'`` have been evaluated.  The determinant is
    scaled before it is compared with ``determinant_tolerance``.
    """

    metric, field, field_prime, tau, mu = _validate_point(r, state)
    if metric == 0.0:
        raise DegenerateSystemError("the derivative block is singular at f=0")

    e = parameters.charge
    m = parameters.scalar_mass
    alpha_e = parameters.alpha * e
    beta = parameters.beta
    kinetic_factor = 1.0 - beta * mu * mu

    f_prime = metric_f_derivative(r, state, parameters)
    tau_prime = tau_derivative(r, state, parameters)
    F_prime = metric_F_derivative(r, state, parameters)

    a11 = metric * kinetic_factor
    a12 = -2.0 * beta * metric * mu * field_prime + alpha_e * field * tau
    c1 = (
        field_prime
        * (f_prime + metric * (F_prime + 2.0 / r))
        * kinetic_factor
        + e * e * field * tau * tau / metric * kinetic_factor
        + 2.0 * alpha_e * field * tau * mu / r
        + alpha_e * field * mu * mu
        + mu * mu * field
        - m * m * field
    )

    maxwell_factor = field * field + beta * (
        metric * field_prime * field_prime
        - e * e * field * field * tau * tau / metric
    )
    a21 = 2.0 * beta * mu * metric * field_prime - alpha_e * tau * field
    a22 = maxwell_factor
    d0 = (
        2.0 * field * field_prime
        + 2.0 * maxwell_factor / r
        + beta
        * (
            f_prime * field_prime * field_prime
            + f_prime * e * e * field * field * tau * tau / (metric * metric)
            - 2.0 * e * e * field * field_prime * tau * tau / metric
            - 2.0 * e * e * field * field * tau * tau_prime / metric
        )
    )
    r0 = (
        e * e * field * field * tau / metric * kinetic_factor
        - F_prime * alpha_e * field * field_prime * tau
        + alpha_e
        * tau
        * (2.0 * field * field_prime / r + field_prime * field_prime)
    )

    matrix = np.array([[a11, a12], [a21, a22]], dtype=float)
    right_hand_side = np.array([-c1, r0 - mu * d0], dtype=float)
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= np.finfo(float).tiny):
        raise DegenerateSystemError(
            "equations (2.30)--(2.31) have a vanishing derivative row"
        )
    scaled_matrix = matrix / row_scale[:, None]
    scaled_right_hand_side = right_hand_side / row_scale
    determinant = float(np.linalg.det(scaled_matrix))
    condition = float(np.linalg.cond(scaled_matrix))
    if (
        not math.isfinite(determinant)
        or not math.isfinite(condition)
        or abs(determinant) <= determinant_tolerance
        or condition >= 1.0 / determinant_tolerance
    ):
        raise DegenerateSystemError(
            "equations (2.30)--(2.31) have a singular derivative block"
        )
    field_second, mu_prime = np.linalg.solve(
        scaled_matrix, scaled_right_hand_side
    )
    if not np.all(np.isfinite((field_second, mu_prime))):
        raise DegenerateSystemError("the derivative block produced non-finite values")
    return (
        f_prime,
        float(field_second),
        tau_prime,
        float(mu_prime),
        F_prime,
    )


def field_derivatives(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    """Return the explicit first-order derivative of ``[f, psi, psi', tau, mu]``."""

    values = _validate_point(r, state)
    f_prime, field_second, tau_prime, mu_prime, _ = derivative_block(
        r, values, parameters
    )
    return np.array(
        [f_prime, values[2], field_second, tau_prime, mu_prime], dtype=float
    )


def field_derivatives_beta_zero(
    r: float,
    state: ArrayLike,
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    """Independent specialization of equations (2.33)--(2.36)."""

    if parameters.beta != 0.0:
        raise ValueError("field_derivatives_beta_zero requires beta=0")
    metric, field, field_prime, tau, mu = _validate_point(r, state)
    if metric == 0.0 or field == 0.0:
        raise DegenerateSystemError("the beta=0 equations require non-zero f and psi")
    e = parameters.charge
    m = parameters.scalar_mass
    alpha_e = parameters.alpha * e

    f_prime = metric_f_derivative(r, state, parameters)
    tau_prime = tau_derivative(r, state, parameters)
    F_prime = metric_F_derivative(r, state, parameters)

    a11 = metric
    a12 = alpha_e * field * tau
    rhs1 = -(
        field_prime * (f_prime + metric * F_prime + 2.0 * metric / r)
        + field
        * (
            e * e * tau * tau / metric
            + 2.0 * alpha_e * tau * mu / r
            + alpha_e * mu * mu
            + mu * mu
            - m * m
        )
    )
    a21 = -alpha_e * tau / field
    a22 = 1.0
    rhs2 = (
        e * e * tau / metric
        - mu * (2.0 * field_prime / field + 2.0 / r)
        + alpha_e
        * tau
        * (
            field_prime * field_prime / (field * field)
            + 2.0 * field_prime / (r * field)
            - F_prime * field_prime / field
        )
    )
    matrix = np.array([[a11, a12], [a21, a22]], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise DegenerateSystemError("the beta=0 derivative block is singular")
    field_second, mu_prime = np.linalg.solve(
        matrix, np.array([rhs1, rhs2], dtype=float)
    )
    return np.array(
        [f_prime, field_prime, field_second, tau_prime, mu_prime], dtype=float
    )


def equation_residuals(
    r: float,
    state: ArrayLike,
    derivatives: ArrayLike,
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    """Evaluate the unexpanded residuals of equations (2.29)--(2.32).

    ``derivatives`` is ``[f', psi'', tau', mu']``.  This function is kept
    algebraically separate from :func:`field_derivatives` and is used for
    validation and series-substitution tests.
    """

    metric, field, field_prime, tau, mu = _validate_point(r, state)
    derivative_values = np.asarray(derivatives, dtype=float)
    if derivative_values.shape != (4,) or not np.all(np.isfinite(derivative_values)):
        raise ValueError("derivatives must be a finite array with shape (4,)")
    f_prime, field_second, tau_prime, mu_prime = derivative_values
    if metric == 0.0:
        raise DegenerateSystemError("the original residuals are singular at f=0")

    e = parameters.charge
    m = parameters.scalar_mass
    alpha_e = parameters.alpha * e
    beta = parameters.beta
    kinetic_factor = 1.0 - beta * mu * mu

    F_prime = metric_F_derivative(r, state, parameters)
    matter = (
        metric * field_prime * field_prime
        + e * e * tau * tau * field * field / metric
        + m * m * field * field
        + mu * mu * field * field
        - 2.0 * alpha_e * field * field_prime * tau * mu
        + beta
        * (
            metric * field_prime * field_prime * mu * mu
            - 3.0 * e * e * tau * tau * mu * mu * field * field / metric
        )
    )
    residual_29 = -f_prime / r - (metric - 1.0) / (r * r) - matter

    residual_30 = (
        metric * field_second * kinetic_factor
        + field_prime
        * (
            f_prime * kinetic_factor
            - 2.0 * beta * metric * mu * mu_prime
            + metric * (F_prime + 2.0 / r) * kinetic_factor
        )
        + e * e * field * tau * tau / metric * kinetic_factor
        + 2.0 * alpha_e * field * tau * mu / r
        + alpha_e * field * tau * mu_prime
        + alpha_e * field * mu * mu
        + mu * mu * field
        - m * m * field
    )

    maxwell_factor = field * field + beta * (
        metric * field_prime * field_prime
        - e * e * field * field * tau * tau / metric
    )
    left_31 = mu_prime * maxwell_factor + mu * (
        2.0 * field * field_prime
        + 2.0 * maxwell_factor / r
        + beta
        * (
            f_prime * field_prime * field_prime
            + 2.0 * metric * field_prime * field_second
            + f_prime * e * e * field * field * tau * tau / (metric * metric)
            - 2.0 * e * e * field * field_prime * tau * tau / metric
            - 2.0 * e * e * field * field * tau * tau_prime / metric
        )
    )
    right_31 = (
        e * e * field * field * tau / metric * kinetic_factor
        - F_prime * alpha_e * field * field_prime * tau
        + alpha_e
        * tau
        * (
            2.0 * field * field_prime / r
            + field_prime * field_prime
            + field * field_second
        )
    )
    residual_31 = left_31 - right_31

    residual_32 = (
        tau_prime
        - 2.0
        * alpha_e
        * r
        * field
        * field_prime
        * mu
        * tau
        * tau
        / metric
        + (
            r * field_prime * field_prime * tau
            + e * e * r * tau**3 * field * field / (metric * metric)
        )
        * kinetic_factor
        - mu
    )
    return np.array(
        [residual_29, residual_30, residual_31, residual_32], dtype=float
    )


def normalized_equation_residuals(
    r: float,
    state: ArrayLike,
    derivatives: ArrayLike,
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    """Return dimensionless backward residuals of (2.29)--(2.32).

    Each equation is divided by one plus a local magnitude assembled from the
    derivatives and fields that occur in it.  The additive one avoids turning
    a correctly decaying tail into a spurious large relative error when every
    term is below machine scale.
    """

    values = _validate_point(r, state)
    derivative_values = np.asarray(derivatives, dtype=float)
    raw = equation_residuals(r, values, derivative_values, parameters)
    f_prime, field_second, tau_prime, mu_prime = derivative_values
    metric, field, field_prime, tau, mu = values
    F_prime = metric_F_derivative(r, values, parameters)
    common = (
        abs(f_prime)
        + abs(field_prime)
        + abs(field_second)
        + abs(tau_prime)
        + abs(mu_prime)
        + abs(F_prime)
    )
    scales = np.array(
        [
            1.0 + abs(f_prime / r) + abs((metric - 1.0) / (r * r)) + common,
            1.0 + abs(metric * field_second) + abs(field_prime) * common + common,
            1.0
            + (field * field + abs(parameters.beta) * field_prime * field_prime)
            * (abs(mu_prime) + abs(field_second) + common),
            1.0 + abs(tau_prime) + abs(mu) + abs(tau) * common,
        ],
        dtype=float,
    )
    return raw / scales
