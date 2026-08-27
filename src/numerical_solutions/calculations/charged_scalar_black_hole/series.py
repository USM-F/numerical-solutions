"""Horizon and infinity expansions for the charged scalar black-hole model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    ModelParameters,
)


class SeriesError(RuntimeError):
    """Raised when a formal expansion is singular or insufficiently accurate."""


def _constant(value: float, size: int) -> NDArray[np.float64]:
    result = np.zeros(size, dtype=float)
    result[0] = value
    return result


def _multiply(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.convolve(left, right)[: left.size]


def _inverse(series: NDArray[np.float64]) -> NDArray[np.float64]:
    if not math.isfinite(float(series[0])) or abs(series[0]) <= 1e-14:
        raise SeriesError("cannot invert a series with zero constant coefficient")
    result = np.zeros_like(series)
    result[0] = 1.0 / series[0]
    for order in range(1, series.size):
        result[order] = -np.dot(series[1 : order + 1], result[order - 1 :: -1]) / series[0]
    return result


def _divide(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    return _multiply(numerator, _inverse(denominator))


def _derivative(series: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.zeros_like(series)
    indices = np.arange(1, series.size, dtype=float)
    result[:-1] = indices * series[1:]
    return result


def _shift(series: NDArray[np.float64], count: int = 1) -> NDArray[np.float64]:
    result = np.zeros_like(series)
    if count < series.size:
        result[count:] = series[:-count]
    return result


def _evaluate(series: NDArray[np.float64], value: float, order: int) -> float:
    return float(np.polynomial.polynomial.polyval(value, series[: order + 1]))


def _horizon_residual_series(
    metric: NDArray[np.float64],
    field: NDArray[np.float64],
    tau: NDArray[np.float64],
    mu: NDArray[np.float64],
    parameters: ModelParameters,
) -> tuple[NDArray[np.float64], ...]:
    """Return series of the original residuals after cancelling explicit ``x`` factors."""

    size = metric.size
    one = _constant(1.0, size)
    x = _shift(one)
    radius = one + x
    inverse_radius = _inverse(radius)
    inverse_radius_squared = _multiply(inverse_radius, inverse_radius)

    metric_hat = np.zeros(size, dtype=float)
    metric_hat[:-1] = metric[1:]
    tau_hat = np.zeros(size, dtype=float)
    tau_hat[:-1] = tau[1:]
    inverse_metric_hat = _inverse(metric_hat)
    inverse_metric_hat_squared = _multiply(inverse_metric_hat, inverse_metric_hat)

    metric_prime = _derivative(metric)
    field_prime = _derivative(field)
    field_second = _derivative(field_prime)
    tau_prime = _derivative(tau)
    mu_prime = _derivative(mu)

    e = parameters.charge
    m = parameters.scalar_mass
    alpha_e = parameters.alpha * e
    beta = parameters.beta

    field_squared = _multiply(field, field)
    field_prime_squared = _multiply(field_prime, field_prime)
    tau_hat_squared = _multiply(tau_hat, tau_hat)
    mu_squared = _multiply(mu, mu)
    kinetic_factor = one - beta * mu_squared

    tau_squared_over_metric = _shift(
        _multiply(tau_hat_squared, inverse_metric_hat)
    )
    tau_squared_over_metric_squared = _multiply(
        tau_hat_squared, inverse_metric_hat_squared
    )
    tau_over_metric = _multiply(tau_hat, inverse_metric_hat)

    F_prime = (
        _multiply(radius, _multiply(field_prime_squared, kinetic_factor))
        + e
        * e
        * _multiply(
            radius,
            _multiply(
                _multiply(tau_squared_over_metric_squared, field_squared),
                kinetic_factor,
            ),
        )
        - 2.0
        * alpha_e
        * _multiply(
            radius,
            _multiply(
                _multiply(_multiply(field, field_prime), tau_over_metric), mu
            ),
        )
    )

    matter = (
        _multiply(metric, field_prime_squared)
        + e * e * _multiply(tau_squared_over_metric, field_squared)
        + m * m * field_squared
        + _multiply(mu_squared, field_squared)
        - 2.0
        * alpha_e
        * _multiply(_multiply(_multiply(field, field_prime), tau), mu)
        + beta
        * (
            _multiply(_multiply(metric, field_prime_squared), mu_squared)
            - 3.0
            * e
            * e
            * _multiply(
                _multiply(tau_squared_over_metric, mu_squared), field_squared
            )
        )
    )
    residual_29 = (
        -_multiply(metric_prime, inverse_radius)
        - _multiply(metric - one, inverse_radius_squared)
        - matter
    )

    residual_30 = (
        _multiply(_multiply(metric, field_second), kinetic_factor)
        + _multiply(
            field_prime,
            _multiply(metric_prime, kinetic_factor)
            - 2.0 * beta * _multiply(_multiply(metric, mu), mu_prime)
            + _multiply(
                metric,
                _multiply(F_prime + 2.0 * inverse_radius, kinetic_factor),
            ),
        )
        + e
        * e
        * _multiply(
            _multiply(field, tau_squared_over_metric), kinetic_factor
        )
        + 2.0
        * alpha_e
        * _multiply(_multiply(_multiply(field, tau), mu), inverse_radius)
        + alpha_e * _multiply(_multiply(field, tau), mu_prime)
        + alpha_e * _multiply(field, mu_squared)
        + _multiply(mu_squared, field)
        - m * m * field
    )

    maxwell_factor = field_squared + beta * (
        _multiply(metric, field_prime_squared)
        - e * e * _multiply(field_squared, tau_squared_over_metric)
    )
    maxwell_brace = (
        2.0 * _multiply(field, field_prime)
        + 2.0 * _multiply(maxwell_factor, inverse_radius)
        + beta
        * (
            _multiply(metric_prime, field_prime_squared)
            + 2.0 * _multiply(_multiply(metric, field_prime), field_second)
            + e
            * e
            * _multiply(
                _multiply(metric_prime, field_squared),
                tau_squared_over_metric_squared,
            )
            - 2.0
            * e
            * e
            * _multiply(
                _multiply(field, field_prime), tau_squared_over_metric
            )
            - 2.0
            * e
            * e
            * _multiply(
                _multiply(field_squared, tau_over_metric), tau_prime
            )
        )
    )
    left_31 = _multiply(mu_prime, maxwell_factor) + _multiply(mu, maxwell_brace)
    right_31 = (
        e
        * e
        * _multiply(_multiply(field_squared, tau_over_metric), kinetic_factor)
        - alpha_e
        * _multiply(F_prime, _multiply(_multiply(field, field_prime), tau))
        + alpha_e
        * _multiply(
            tau,
            2.0
            * _multiply(_multiply(field, field_prime), inverse_radius)
            + field_prime_squared
            + _multiply(field, field_second),
        )
    )
    residual_31 = left_31 - right_31

    tau_cubic_over_metric_squared = _shift(
        _multiply(_multiply(tau_hat_squared, tau_hat), inverse_metric_hat_squared)
    )
    residual_32 = (
        tau_prime
        - 2.0
        * alpha_e
        * _multiply(
            radius,
            _shift(
                _multiply(
                    _multiply(_multiply(field, field_prime), mu),
                    _multiply(tau_hat_squared, inverse_metric_hat),
                )
            ),
        )
        + _multiply(
            (
                _multiply(radius, _multiply(field_prime_squared, tau))
                + e
                * e
                * _multiply(
                    radius,
                    _multiply(tau_cubic_over_metric_squared, field_squared),
                )
            ),
            kinetic_factor,
        )
        - mu
    )
    return residual_29, residual_30, residual_31, residual_32


@dataclass(frozen=True, slots=True)
class HorizonSeries:
    """Numerical Frobenius coefficients at the non-extremal horizon."""

    metric: NDArray[np.float64]
    field: NDArray[np.float64]
    tau: NDArray[np.float64]
    mu: NDArray[np.float64]
    psi_h: float
    mu_h: float
    max_order: int

    @property
    def field_prime(self) -> NDArray[np.float64]:
        return _derivative(self.field)

    def state(self, epsilon: float, order: int) -> NDArray[np.float64]:
        """Evaluate ``[f, psi, psi', tau, mu]`` at ``r=1+epsilon``."""

        if not 1 <= order <= self.max_order:
            raise ValueError("order is outside the available horizon series")
        truncated_field = self.field.copy()
        truncated_field[order + 1 :] = 0.0
        field_prime = _derivative(truncated_field)
        return np.array(
            [
                _evaluate(self.metric, epsilon, order),
                _evaluate(self.field, epsilon, order),
                _evaluate(field_prime, epsilon, min(order - 1, field_prime.size - 1)),
                _evaluate(self.tau, epsilon, order),
                _evaluate(self.mu, epsilon, order),
            ],
            dtype=float,
        )

    def truncation_error(self, epsilon: float, order: int) -> float:
        """Estimate normalized value/derivative error from the first omitted term."""

        if order >= self.max_order:
            raise ValueError("an omitted coefficient is required for error estimation")
        index = order + 1
        state = self.state(epsilon, order)
        value_terms = np.array(
            [
                self.metric[index] * epsilon**index,
                self.field[index] * epsilon**index,
                index * self.field[index] * epsilon ** (index - 1),
                self.tau[index] * epsilon**index,
                self.mu[index] * epsilon**index,
            ],
            dtype=float,
        )
        derivative_terms = np.array(
            [
                index * self.metric[index] * epsilon ** (index - 1),
                index
                * (index - 1)
                * self.field[index]
                * epsilon ** max(index - 2, 0),
                index * self.tau[index] * epsilon ** (index - 1),
                index * self.mu[index] * epsilon ** (index - 1),
            ],
            dtype=float,
        )
        normalized_values = np.abs(value_terms) / np.maximum(np.abs(state), 1.0)
        return float(max(np.max(normalized_values), np.max(np.abs(derivative_terms))))

    def select_order(self, epsilon: float, bc_tolerance: float) -> int:
        """Choose the first acceptable runtime order among five, six and seven."""

        if epsilon <= 0.0 or bc_tolerance <= 0.0:
            raise ValueError("epsilon and bc_tolerance must be positive")
        for order in (5, 6, 7):
            if self.truncation_error(epsilon, order) < 0.1 * bc_tolerance:
                return order
        raise SeriesError(
            "the eighth-order horizon series requires a smaller horizon epsilon"
        )


def build_horizon_series(
    parameters: ModelParameters,
    psi_h: float,
    mu_h: float,
    *,
    order: int = 8,
) -> HorizonSeries:
    """Construct the regular horizon expansion recursively through ``order``."""

    if not math.isfinite(psi_h) or psi_h <= 0.0:
        raise ValueError("psi_h must be positive and finite")
    if not math.isfinite(mu_h) or mu_h <= 0.0:
        raise ValueError("mu_h must be positive and finite")
    if order < 2:
        raise ValueError("order must be at least two")

    m = parameters.scalar_mass
    e = parameters.charge
    beta = parameters.beta
    metric_1 = 1.0 - (m * m + mu_h * mu_h) * psi_h * psi_h
    kinetic_0 = 1.0 - beta * mu_h * mu_h
    if metric_1 <= 0.0:
        raise SeriesError("the horizon is extremal or has f'(1)<0")
    if abs(kinetic_0) <= 1e-12:
        raise SeriesError("the horizon lies on 1-beta*mu_h^2=0")

    size = order + 3
    metric = np.zeros(size, dtype=float)
    field = np.zeros(size, dtype=float)
    tau = np.zeros(size, dtype=float)
    mu = np.zeros(size, dtype=float)
    metric[1] = metric_1
    field[0] = psi_h
    field[1] = (
        psi_h
        * (m * m - (1.0 + parameters.alpha * e) * mu_h * mu_h)
        / (metric_1 * kinetic_0)
    )
    tau[1] = mu_h
    mu[0] = mu_h

    # The x^0 Maxwell equation fixes mu_1.  The remaining four coefficients
    # at each order enter linearly, so a small numerical coefficient matrix is
    # more stable and auditable than enormous closed expressions.
    residual = _horizon_residual_series(metric, field, tau, mu, parameters)[2]
    base = residual[0]
    mu[1] = 1.0
    slope = _horizon_residual_series(metric, field, tau, mu, parameters)[2][0] - base
    if abs(slope) <= 1e-14:
        raise SeriesError("the horizon Maxwell recurrence is singular at order zero")
    mu[1] = -base / slope

    arrays = (metric, field, tau, mu)
    for residual_order in range(1, order):
        coefficient_index = residual_order + 1
        base_residual = np.array(
            [
                residual_series[residual_order]
                for residual_series in _horizon_residual_series(
                    metric, field, tau, mu, parameters
                )
            ],
            dtype=float,
        )
        matrix = np.empty((4, 4), dtype=float)
        for column, coefficients in enumerate(arrays):
            coefficients[coefficient_index] = 1.0
            changed = _horizon_residual_series(metric, field, tau, mu, parameters)
            matrix[:, column] = np.array(
                [series[residual_order] for series in changed], dtype=float
            ) - base_residual
            coefficients[coefficient_index] = 0.0
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition) or condition >= 1e13:
            raise SeriesError(
                "the horizon coefficient recurrence is singular at order "
                f"{residual_order} (condition={condition:.3e})"
            )
        solution = np.linalg.solve(matrix, -base_residual)
        for coefficients, value in zip(arrays, solution, strict=True):
            coefficients[coefficient_index] = value

    return HorizonSeries(
        metric=metric[: order + 1].copy(),
        field=field[: order + 1].copy(),
        tau=tau[: order + 1].copy(),
        mu=mu[: order + 1].copy(),
        psi_h=psi_h,
        mu_h=mu_h,
        max_order=order,
    )


def _factored_derivative(
    coefficients: NDArray[np.float64], exponent: float, power: float
) -> NDArray[np.float64]:
    """Differentiate ``exp(-exponent*r) r**power A(1/r)`` modulo its factor."""

    result = -exponent * coefficients
    for index in range(1, coefficients.size):
        result[index] += (power - index + 1.0) * coefficients[index - 1]
    return result


def _scalar_tail_residual(
    coefficients: NDArray[np.float64], mass: float, adm_mass: float, power: float
) -> NDArray[np.float64]:
    size = coefficients.size
    f = _constant(1.0, size)
    f[1] = -2.0 * adm_mass
    transport = np.zeros(size, dtype=float)
    if size > 1:
        transport[1] = 2.0
    if size > 2:
        transport[2] = -2.0 * adm_mass
    first = _factored_derivative(coefficients, mass, power)
    second = _factored_derivative(first, mass, power)
    return _multiply(f, second) + _multiply(transport, first) - mass * mass * coefficients


def _build_scalar_tail(
    mass: float, adm_mass: float, order: int, size: int
) -> tuple[float, NDArray[np.float64]]:
    power = -1.0 - adm_mass * mass
    coefficients = np.zeros(size, dtype=float)
    coefficients[0] = 1.0
    for index in range(1, order + 1):
        residual_index = index + 1
        base = _scalar_tail_residual(coefficients, mass, adm_mass, power)[
            residual_index
        ]
        coefficients[index] = 1.0
        slope = (
            _scalar_tail_residual(coefficients, mass, adm_mass, power)[
                residual_index
            ]
            - base
        )
        if abs(slope) <= 1e-14:
            raise SeriesError("the scalar asymptotic recurrence is singular")
        coefficients[index] = -base / slope
    return power, coefficients


def _tau_tail_residual(
    coefficients: NDArray[np.float64],
    exponent: float,
    power: float,
    scalar_coefficients: NDArray[np.float64],
    scalar_power: float,
    adm_mass: float,
    parameters: ModelParameters,
) -> NDArray[np.float64]:
    size = coefficients.size
    one = _constant(1.0, size)
    f = one.copy()
    f[1] = -2.0 * adm_mass
    f_prime = np.zeros(size, dtype=float)
    if size > 2:
        f_prime[2] = 2.0 * adm_mass
    inverse_f = _inverse(f)

    scalar_first = _factored_derivative(
        scalar_coefficients, parameters.scalar_mass, scalar_power
    )
    scalar_second = _factored_derivative(
        scalar_first, parameters.scalar_mass, scalar_power
    )
    log_scalar = _divide(scalar_first, scalar_coefficients)
    scalar_ratio_second = _divide(scalar_second, scalar_coefficients)
    log_scalar_squared = _multiply(log_scalar, log_scalar)

    beta = parameters.beta
    alpha_e = parameters.alpha * parameters.charge
    derivative_factor = one + beta * _multiply(f, log_scalar_squared)
    transport = (
        2.0 * log_scalar
        + 2.0 * _shift(derivative_factor)
        + beta
        * (
            _multiply(f_prime, log_scalar_squared)
            + 2.0
            * _multiply(_multiply(f, log_scalar), scalar_ratio_second)
        )
    )
    potential = (
        parameters.charge * parameters.charge * inverse_f
        + alpha_e
        * (
            2.0 * _shift(log_scalar)
            + log_scalar_squared
            + scalar_ratio_second
        )
    )
    first = _factored_derivative(coefficients, exponent, power)
    second = _factored_derivative(first, exponent, power)
    return _multiply(derivative_factor, second) + _multiply(transport, first) - _multiply(
        potential, coefficients
    )


@dataclass(frozen=True, slots=True)
class InfinitySeries:
    """Polynomial sectors of the decaying large-radius expansion."""

    scalar_coefficients: NDArray[np.float64]
    tau_coefficients: NDArray[np.float64]
    scalar_power: float
    tau_power: float
    decay_rate: float
    scalar_mass: float
    adm_mass: float
    scalar_amplitude: float
    tau_amplitude: float
    order: int

    def log_derivatives(self, radius: float, order: int = 2) -> tuple[float, float]:
        """Return ``psi'/psi`` and ``tau'/tau`` without exponential underflow."""

        if radius <= 1.0:
            raise ValueError("the infinity expansion requires r>1")
        if not 0 <= order <= self.order:
            raise ValueError("order is outside the available infinity series")
        z = 1.0 / radius
        scalar_coefficients = self.scalar_coefficients.copy()
        scalar_coefficients[order + 1 :] = 0.0
        tau_coefficients = self.tau_coefficients.copy()
        tau_coefficients[order + 1 :] = 0.0
        scalar_polynomial = float(
            np.polynomial.polynomial.polyval(z, scalar_coefficients)
        )
        tau_polynomial = float(
            np.polynomial.polynomial.polyval(z, tau_coefficients)
        )
        if abs(scalar_polynomial) <= 1e-14 or abs(tau_polynomial) <= 1e-14:
            raise SeriesError("an asymptotic polynomial crosses zero at the boundary")
        scalar_first = _factored_derivative(
            scalar_coefficients, self.scalar_mass, self.scalar_power
        )
        tau_first = _factored_derivative(
            tau_coefficients, self.decay_rate, self.tau_power
        )
        return (
            float(np.polynomial.polynomial.polyval(z, scalar_first))
            / scalar_polynomial,
            float(np.polynomial.polynomial.polyval(z, tau_first))
            / tau_polynomial,
        )

    def state(self, radius: float, order: int = 2) -> NDArray[np.float64]:
        if radius <= 1.0:
            raise ValueError("the infinity expansion requires r>1")
        if not 0 <= order <= self.order:
            raise ValueError("order is outside the available infinity series")
        z = 1.0 / radius
        scalar_coefficients = self.scalar_coefficients.copy()
        scalar_coefficients[order + 1 :] = 0.0
        tau_coefficients = self.tau_coefficients.copy()
        tau_coefficients[order + 1 :] = 0.0
        scalar_polynomial = np.polynomial.polynomial.polyval(z, scalar_coefficients)
        scalar_first = _factored_derivative(
            scalar_coefficients, self.scalar_mass, self.scalar_power
        )
        tau_polynomial = np.polynomial.polynomial.polyval(z, tau_coefficients)
        tau_first = _factored_derivative(
            tau_coefficients, self.decay_rate, self.tau_power
        )
        scalar_factor = math.exp(-self.scalar_mass * radius) * radius**self.scalar_power
        tau_factor = math.exp(-self.decay_rate * radius) * radius**self.tau_power
        return np.array(
            [
                1.0 - 2.0 * self.adm_mass / radius,
                self.scalar_amplitude * scalar_factor * scalar_polynomial,
                self.scalar_amplitude
                * scalar_factor
                * np.polynomial.polynomial.polyval(z, scalar_first),
                self.tau_amplitude * tau_factor * tau_polynomial,
                self.tau_amplitude
                * tau_factor
                * np.polynomial.polynomial.polyval(z, tau_first),
            ],
            dtype=float,
        )

    def truncation_error(self, radius: float, used_order: int = 2) -> float:
        if used_order >= self.order:
            raise ValueError("an omitted infinity coefficient is required")
        index = used_order + 1
        z = 1.0 / radius
        scalar_term = abs(self.scalar_coefficients[index] * z**index)
        tau_term = abs(self.tau_coefficients[index] * z**index)
        return float(max(scalar_term, tau_term))


def build_infinity_series(
    parameters: ModelParameters,
    adm_mass: float,
    scalar_amplitude: float,
    tau_amplitude: float,
    *,
    order: int = 3,
) -> InfinitySeries:
    """Construct the algebraic sectors multiplying the two decaying exponentials."""

    if not math.isfinite(adm_mass) or adm_mass <= 0.0:
        raise ValueError("adm_mass must be positive and finite")
    if not math.isfinite(scalar_amplitude) or not math.isfinite(tau_amplitude):
        raise ValueError("tail amplitudes must be finite")
    if order < 1:
        raise ValueError("order must be positive")
    size = order + 4
    scalar_power, scalar_coefficients = _build_scalar_tail(
        parameters.scalar_mass, adm_mass, order, size
    )
    decay_rate = parameters.decay_rate

    tau_coefficients = np.zeros(size, dtype=float)
    tau_coefficients[0] = 1.0
    residual_at_zero = _tau_tail_residual(
        tau_coefficients,
        decay_rate,
        0.0,
        scalar_coefficients,
        scalar_power,
        adm_mass,
        parameters,
    )[1]
    residual_at_one = _tau_tail_residual(
        tau_coefficients,
        decay_rate,
        1.0,
        scalar_coefficients,
        scalar_power,
        adm_mass,
        parameters,
    )[1]
    slope = residual_at_one - residual_at_zero
    if abs(slope) <= 1e-14:
        raise SeriesError("the electromagnetic asymptotic power is indeterminate")
    tau_power = -residual_at_zero / slope

    for index in range(1, order + 1):
        residual_index = index + 1
        base = _tau_tail_residual(
            tau_coefficients,
            decay_rate,
            tau_power,
            scalar_coefficients,
            scalar_power,
            adm_mass,
            parameters,
        )[residual_index]
        tau_coefficients[index] = 1.0
        changed = _tau_tail_residual(
            tau_coefficients,
            decay_rate,
            tau_power,
            scalar_coefficients,
            scalar_power,
            adm_mass,
            parameters,
        )[residual_index]
        coefficient_slope = changed - base
        if abs(coefficient_slope) <= 1e-14:
            raise SeriesError("the electromagnetic asymptotic recurrence is singular")
        tau_coefficients[index] = -base / coefficient_slope

    return InfinitySeries(
        scalar_coefficients=scalar_coefficients[: order + 1].copy(),
        tau_coefficients=tau_coefficients[: order + 1].copy(),
        scalar_power=scalar_power,
        tau_power=float(tau_power),
        decay_rate=decay_rate,
        scalar_mass=parameters.scalar_mass,
        adm_mass=adm_mass,
        scalar_amplitude=scalar_amplitude,
        tau_amplitude=tau_amplitude,
        order=order,
    )
