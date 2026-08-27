import numpy as np

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    ModelParameters,
    equation_residuals,
)
from numerical_solutions.calculations.charged_scalar_black_hole.series import (
    build_horizon_series,
    build_infinity_series,
)


def _series_derivatives(series, epsilon: float) -> np.ndarray:
    f_prime = np.polynomial.polynomial.polyval(
        epsilon, np.arange(1, series.metric.size) * series.metric[1:]
    )
    psi_second = np.polynomial.polynomial.polyval(
        epsilon,
        np.arange(1, series.field.size - 1)
        * np.arange(2, series.field.size)
        * series.field[2:],
    )
    tau_prime = np.polynomial.polynomial.polyval(
        epsilon, np.arange(1, series.tau.size) * series.tau[1:]
    )
    mu_prime = np.polynomial.polynomial.polyval(
        epsilon, np.arange(1, series.mu.size) * series.mu[1:]
    )
    return np.array([f_prime, psi_second, tau_prime, mu_prime])


def test_horizon_series_has_required_leading_coefficients() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    series = build_horizon_series(parameters, psi_h=0.3, mu_h=0.4)

    expected_f1 = 1.0 - (1.0 + 0.4**2) * 0.3**2
    expected_psi1 = 0.3 * (1.0 - 1.2 * 0.4**2) / (
        expected_f1 * (1.0 - 0.1 * 0.4**2)
    )

    assert series.metric[1] == expected_f1
    assert series.tau[1] == 0.4
    assert series.field[1] == expected_psi1
    assert series.max_order == 8


def test_eighth_order_horizon_series_satisfies_original_system() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    series = build_horizon_series(parameters, psi_h=0.3, mu_h=0.4)
    epsilon = 1e-2

    residual = equation_residuals(
        1.0 + epsilon,
        series.state(epsilon, 8),
        _series_derivatives(series, epsilon),
        parameters,
    )

    assert np.max(np.abs(residual)) < 1e-11


def test_horizon_error_decreases_with_order_and_epsilon() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.0, charge=1.0, scalar_mass=1.0)
    series = build_horizon_series(parameters, psi_h=0.3, mu_h=0.4)

    assert series.truncation_error(1e-3, 6) < series.truncation_error(1e-3, 5)
    assert series.truncation_error(5e-4, 5) < series.truncation_error(1e-3, 5)
    assert series.select_order(1e-3, 1e-9) in (5, 6, 7)

    for order in (5, 6, 7):
        ratio = series.truncation_error(
            5e-4, order
        ) / series.truncation_error(1e-3, order)
        assert ratio == 2.0 ** (-(order - 1))


def test_infinity_series_contains_corrected_scalar_power_and_third_order_control() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    series = build_infinity_series(
        parameters,
        adm_mass=0.5,
        scalar_amplitude=1.0,
        tau_amplitude=0.2,
    )

    assert series.scalar_power == -1.5
    assert series.order == 3
    assert np.all(np.isfinite(series.state(20.0, order=2)))
    state = series.state(20.0, order=2)
    scalar_log_derivative, tau_log_derivative = series.log_derivatives(
        20.0, order=2
    )
    np.testing.assert_allclose(scalar_log_derivative, state[2] / state[1])
    np.testing.assert_allclose(tau_log_derivative, state[4] / state[3])
    assert series.truncation_error(40.0, 2) < series.truncation_error(20.0, 2)
