"""Independent SymPy substitutions used by the mandatory CI verification job."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    ModelParameters,
)
from numerical_solutions.calculations.charged_scalar_black_hole.series import (
    build_horizon_series,
    build_infinity_series,
)


pytestmark = pytest.mark.symbolic

_MIN_POWER = -3
_MAX_POWER = 9


class _ExactLaurent:
    """Small exact Laurent algebra independent of the runtime NumPy series."""

    def __init__(self, coefficients=None):
        self.coefficients = {
            int(power): sp.cancel(value)
            for power, value in (coefficients or {}).items()
            if value != 0 and _MIN_POWER <= int(power) <= _MAX_POWER
        }

    @classmethod
    def from_list(cls, values):
        return cls({power: value for power, value in enumerate(values)})

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls({0: sp.sympify(value)})

    def coefficient(self, power):
        return self.coefficients.get(power, sp.S.Zero)

    def __add__(self, other):
        other = self.coerce(other)
        return self.__class__(
            {
                power: self.coefficient(power) + other.coefficient(power)
                for power in range(_MIN_POWER, _MAX_POWER + 1)
            }
        )

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(
            {power: -value for power, value in self.coefficients.items()}
        )

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        coefficients = {}
        for left_power, left_value in self.coefficients.items():
            for right_power, right_value in other.coefficients.items():
                power = left_power + right_power
                if _MIN_POWER <= power <= _MAX_POWER:
                    coefficients[power] = coefficients.get(power, 0) + left_value * right_value
        return self.__class__(coefficients)

    __rmul__ = __mul__

    def inverse(self):
        valuation = min(
            power for power, value in self.coefficients.items() if value != 0
        )
        leading = self.coefficient(valuation)
        regular_inverse = [sp.cancel(1 / leading)]
        largest_index = _MAX_POWER + valuation
        for index in range(1, largest_index + 1):
            value = -sum(
                self.coefficient(valuation + offset)
                * regular_inverse[index - offset]
                for offset in range(1, index + 1)
            ) / leading
            regular_inverse.append(sp.cancel(value))
        return self.__class__(
            {
                index - valuation: value
                for index, value in enumerate(regular_inverse)
            }
        )

    def __truediv__(self, other):
        return self * self.coerce(other).inverse()

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("only non-negative integral powers are supported")
        result = self.coerce(1)
        for _ in range(power):
            result = result * self
        return result

    def derivative(self):
        return self.__class__(
            {
                power - 1: power * value
                for power, value in self.coefficients.items()
                if power != 0
            }
        )


def _exact_original_residuals(f, psi, tau, mu, *, alpha, beta, e, m):
    x = _ExactLaurent({1: 1})
    r = 1 + x
    fp = f.derivative()
    vp = psi.derivative()
    vpp = vp.derivative()
    tp = tau.derivative()
    mup = mu.derivative()
    kinetic = 1 - beta * mu**2
    Fp = (
        r * vp**2 * kinetic
        + e**2 * r * tau**2 * psi**2 / f**2 * kinetic
        - 2 * alpha * e * r * psi * vp * tau * mu / f
    )
    residual_29 = (
        -fp / r
        - (f - 1) / r**2
        - f * vp**2
        - e**2 * tau**2 * psi**2 / f
        - m**2 * psi**2
        - mu**2 * psi**2
        + 2 * alpha * e * psi * vp * tau * mu
        - beta * (f * vp**2 * mu**2 - 3 * e**2 * tau**2 * mu**2 * psi**2 / f)
    )
    residual_30 = (
        f * vpp * kinetic
        + vp * (fp * kinetic - 2 * beta * f * mu * mup + f * (Fp + 2 / r) * kinetic)
        + e**2 * psi * tau**2 / f * kinetic
        + 2 * alpha * e * psi * tau * mu / r
        + alpha * e * psi * tau * mup
        + alpha * e * psi * mu**2
        + mu**2 * psi
        - m**2 * psi
    )
    factor = psi**2 + beta * (f * vp**2 - e**2 * psi**2 * tau**2 / f)
    residual_31 = (
        mup * factor
        + mu
        * (
            2 * psi * vp
            + 2 * factor / r
            + beta
            * (
                fp * vp**2
                + 2 * f * vp * vpp
                + fp * e**2 * psi**2 * tau**2 / f**2
                - 2 * e**2 * psi * vp * tau**2 / f
                - 2 * e**2 * psi**2 * tau * tp / f
            )
        )
        - e**2 * psi**2 * tau / f * kinetic
        + Fp * alpha * e * psi * vp * tau
        - alpha * e * tau * (2 * psi * vp / r + vp**2 + psi * vpp)
    )
    residual_32 = (
        tp
        - 2 * alpha * e * r * psi * vp * mu * tau**2 / f
        + (r * vp**2 * tau + e**2 * r * tau**3 * psi**2 / f**2) * kinetic
        - mu
    )
    return residual_29, residual_30, residual_31, residual_32


def _independent_exact_horizon_series():
    alpha = sp.Rational(1, 5)
    beta = sp.Rational(1, 10)
    e = m = sp.Integer(1)
    psi_h = sp.Rational(3, 10)
    mu_h = sp.Rational(2, 5)
    f = [sp.S.Zero] * 9
    psi = [sp.S.Zero] * 9
    tau = [sp.S.Zero] * 9
    mu = [sp.S.Zero] * 9
    f[1] = 1 - (m**2 + mu_h**2) * psi_h**2
    psi[0] = psi_h
    psi[1] = psi_h * (m**2 - (1 + alpha * e) * mu_h**2) / (
        f[1] * (1 - beta * mu_h**2)
    )
    tau[1] = mu_h
    mu[0] = mu_h

    mu_one = sp.symbols("mu_1")
    mu[1] = mu_one
    residual = _exact_original_residuals(
        *map(_ExactLaurent.from_list, (f, psi, tau, mu)),
        alpha=alpha,
        beta=beta,
        e=e,
        m=m,
    )[2].coefficient(0)
    mu[1] = sp.solve(residual, mu_one)[0]

    arrays = (f, psi, tau, mu)
    for residual_order in range(1, 8):
        symbols = sp.symbols(f"f_{residual_order + 1} psi_{residual_order + 1} tau_{residual_order + 1} mu_{residual_order + 1}")
        for coefficients, symbol in zip(arrays, symbols, strict=True):
            coefficients[residual_order + 1] = symbol
        residuals = _exact_original_residuals(
            *map(_ExactLaurent.from_list, arrays),
            alpha=alpha,
            beta=beta,
            e=e,
            m=m,
        )
        equations = [item.coefficient(residual_order) for item in residuals]
        matrix, right = sp.linear_eq_to_matrix(equations, symbols)
        solution = matrix.inv() * right
        for coefficients, value in zip(arrays, solution, strict=True):
            coefficients[residual_order + 1] = sp.cancel(value)
    return arrays, (alpha, beta, e, m)


@pytest.fixture(scope="module")
def exact_horizon_series():
    return _independent_exact_horizon_series()


def test_horizon_series_is_built_independently_and_cancels_every_equation(
    exact_horizon_series,
) -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    runtime = build_horizon_series(parameters, psi_h=0.3, mu_h=0.4)
    exact_arrays, constants = exact_horizon_series
    for exact, numerical in zip(
        exact_arrays,
        (runtime.metric, runtime.field, runtime.tau, runtime.mu),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.array([float(value) for value in exact]),
            numerical,
            atol=2e-11,
            rtol=2e-11,
        )

    residuals = _exact_original_residuals(
        *map(_ExactLaurent.from_list, exact_arrays),
        alpha=constants[0],
        beta=constants[1],
        e=constants[2],
        m=constants[3],
    )
    for residual in residuals:
        assert all(residual.coefficient(power) == 0 for power in range(-2, 8))
    assert any(residual.coefficient(8) != 0 for residual in residuals)


@pytest.mark.parametrize("changed_array", range(4))
def test_horizon_series_negative_control_detects_each_changed_coefficient(
    changed_array: int,
    exact_horizon_series,
) -> None:
    exact_arrays, constants = exact_horizon_series
    changed = [list(values) for values in exact_arrays]
    changed[changed_array][8] += sp.Rational(1, 1000)
    residuals = _exact_original_residuals(
        *map(_ExactLaurent.from_list, changed),
        alpha=constants[0],
        beta=constants[1],
        e=constants[2],
        m=constants[3],
    )

    assert any(
        residual.coefficient(power) != 0
        for residual in residuals
        for power in range(8)
    )


def _factored_derivative(expression: sp.Expr, z: sp.Symbol, exponent: float, power: float) -> sp.Expr:
    return -exponent * expression + power * z * expression - z**2 * sp.diff(expression, z)


def test_right_series_cancels_scalar_and_electromagnetic_polynomial_sectors() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    runtime = build_infinity_series(parameters, 0.5, 1.0, 0.2)
    z = sp.symbols("z")
    f = 1 - z
    fp = z**2
    a1, a2, a3 = sp.symbols("a1 a2 a3")
    scalar_power = sp.Rational(-3, 2)
    scalar = 1 + a1 * z + a2 * z**2 + a3 * z**3
    scalar_first = _factored_derivative(
        scalar, z, sp.Integer(1), scalar_power
    )
    scalar_second = _factored_derivative(
        scalar_first, z, sp.Integer(1), scalar_power
    )
    raw_scalar_residual = sp.series(
        f * scalar_second + (2 * z - z**2) * scalar_first - scalar,
        z,
        0,
        6,
    ).removeO().expand()
    scalar_solution = sp.solve(
        [raw_scalar_residual.coeff(z, power) for power in (2, 3, 4)],
        (a1, a2, a3),
        dict=True,
    )[0]
    scalar = sp.expand(scalar.subs(scalar_solution))
    scalar_first = _factored_derivative(scalar, z, 1, scalar_power)
    scalar_second = _factored_derivative(scalar_first, z, 1, scalar_power)
    scalar_residual = sp.series(
        f * scalar_second + (2 * z - z**2) * scalar_first - scalar,
        z,
        0,
        5,
    ).removeO().expand()
    for power in range(5):
        assert scalar_residual.coeff(z, power) == 0
    np.testing.assert_allclose(
        [1.0, *(float(scalar_solution[item]) for item in (a1, a2, a3))],
        runtime.scalar_coefficients,
        atol=1e-12,
        rtol=1e-12,
    )

    log_scalar = sp.series(scalar_first / scalar, z, 0, 6).removeO()
    scalar_ratio_second = sp.series(scalar_second / scalar, z, 0, 6).removeO()
    derivative_factor = 1 + sp.Rational(1, 10) * f * log_scalar**2
    transport = (
        2 * log_scalar
        + 2 * z * derivative_factor
        + sp.Rational(1, 10)
        * (fp * log_scalar**2 + 2 * f * log_scalar * scalar_ratio_second)
    )
    potential = 1 / f + sp.Rational(1, 5) * (
        2 * z * log_scalar + log_scalar**2 + scalar_ratio_second
    )
    exact_decay = sp.sqrt(sp.Rational(25, 11)) - 1
    tau_power, b1, b2, b3 = sp.symbols("s_tau b1 b2 b3")
    tau = 1 + b1 * z + b2 * z**2 + b3 * z**3
    tau_first = _factored_derivative(tau, z, exact_decay, tau_power)
    tau_second = _factored_derivative(tau_first, z, exact_decay, tau_power)
    raw_tau_residual = sp.series(
        derivative_factor * tau_second + transport * tau_first - potential * tau,
        z,
        0,
        5,
    ).removeO().expand()
    tau_solution = sp.solve(
        [raw_tau_residual.coeff(z, power) for power in (1, 2, 3, 4)],
        (tau_power, b1, b2, b3),
        dict=True,
        simplify=False,
    )[0]
    tau = sp.expand(tau.subs(tau_solution))
    tau_first = _factored_derivative(
        tau, z, exact_decay, tau_solution[tau_power]
    )
    tau_second = _factored_derivative(
        tau_first, z, exact_decay, tau_solution[tau_power]
    )
    tau_residual = sp.series(
        derivative_factor * tau_second + transport * tau_first - potential * tau,
        z,
        0,
        5,
    ).removeO().expand()
    for power in range(5):
        assert sp.simplify(tau_residual.coeff(z, power)) == 0
    assert abs(float(exact_decay) - runtime.decay_rate) < 1e-14
    assert abs(float(tau_solution[tau_power]) - runtime.tau_power) < 1e-11
    np.testing.assert_allclose(
        [1.0, *(float(tau_solution[item]) for item in (b1, b2, b3))],
        runtime.tau_coefficients,
        atol=1e-10,
        rtol=1e-10,
    )


def test_general_derivative_equations_have_the_exact_beta_zero_limit() -> None:
    r, f, psi, v, tau, mu = sp.symbols(
        "r f psi v tau mu", nonzero=True
    )
    f_prime, v_prime, tau_prime, mu_prime, F_prime = sp.symbols(
        "f_prime v_prime tau_prime mu_prime F_prime"
    )
    alpha_e, e, mass, beta = sp.symbols("alpha_e e m beta")
    kinetic = 1 - beta * mu**2
    general_scalar = (
        f * v_prime * kinetic
        + v * (f_prime * kinetic - 2 * beta * f * mu * mu_prime + f * (F_prime + 2 / r) * kinetic)
        + e**2 * psi * tau**2 / f * kinetic
        + 2 * alpha_e * psi * tau * mu / r
        + alpha_e * psi * tau * mu_prime
        + alpha_e * psi * mu**2
        + mu**2 * psi
        - mass**2 * psi
    )
    beta_zero_scalar = (
        f * v_prime
        + alpha_e * psi * tau * mu_prime
        + v * (f_prime + f * F_prime + 2 * f / r)
        + psi
        * (
            e**2 * tau**2 / f
            + 2 * alpha_e * tau * mu / r
            + alpha_e * mu**2
            + mu**2
            - mass**2
        )
    )
    assert sp.expand(general_scalar.subs(beta, 0) - beta_zero_scalar) == 0

    factor = psi**2 + beta * (f * v**2 - e**2 * psi**2 * tau**2 / f)
    general_maxwell = (
        mu_prime * factor
        + mu
        * (
            2 * psi * v
            + 2 * factor / r
            + beta
            * (
                f_prime * v**2
                + 2 * f * v * v_prime
                + f_prime * e**2 * psi**2 * tau**2 / f**2
                - 2 * e**2 * psi * v * tau**2 / f
                - 2 * e**2 * psi**2 * tau * tau_prime / f
            )
        )
        - e**2 * psi**2 * tau / f * kinetic
        + F_prime * alpha_e * psi * v * tau
        - alpha_e * tau * (2 * psi * v / r + v**2 + psi * v_prime)
    )
    specialized_maxwell = (
        -alpha_e * tau / psi * v_prime
        + mu_prime
        - (
            e**2 * tau / f
            - mu * (2 * v / psi + 2 / r)
            + alpha_e
            * tau
            * (v**2 / psi**2 + 2 * v / (r * psi) - F_prime * v / psi)
        )
    )
    assert sp.cancel(general_maxwell.subs(beta, 0) / psi**2 - specialized_maxwell) == 0

    corrected_decay = sp.sqrt(
        mass**2 + (e**2 + 2 * alpha_e * mass**2) / (1 + beta * mass**2)
    ) - mass
    assert sp.limit(corrected_decay, beta, 0) == sp.sqrt(
        mass**2 + e**2 + 2 * alpha_e * mass**2
    ) - mass


def test_static_spherical_geometry_and_radial_reduction() -> None:
    r, theta = sp.symbols("r theta", positive=True)
    radial_metric = sp.Function("B")(r)
    time_metric = sp.Function("A")(r)
    field = sp.Function("psi")(r)
    metric = sp.diag(
        time_metric**2,
        -radial_metric**2,
        -r**2,
        -r**2 * sp.sin(theta) ** 2,
    )
    assert sp.factor(metric.det()) == (
        -time_metric**2 * radial_metric**2 * r**4 * sp.sin(theta) ** 2
    )

    radial_dalembertian = sp.simplify(
        sp.diff(
            time_metric * radial_metric * r**2 * (-1 / radial_metric**2) * sp.diff(field, r),
            r,
        )
        / (time_metric * radial_metric * r**2)
    )
    reduced = -(
        sp.diff(field, r, 2)
        + (
            sp.diff(time_metric, r) / time_metric
            - sp.diff(radial_metric, r) / radial_metric
            + 2 / r
        )
        * sp.diff(field, r)
    ) / radial_metric**2
    assert sp.simplify(radial_dalembertian - reduced) == 0


def test_schwarzschild_limit_has_zero_ricci_tensor() -> None:
    coordinates = t, r, theta, phi = sp.symbols("t r theta phi", positive=True)
    mass = sp.symbols("M", positive=True)
    f = 1 - 2 * mass / r
    metric = sp.diag(f, -1 / f, -r**2, -r**2 * sp.sin(theta) ** 2)
    inverse = sp.simplify(metric.inv())
    dimension = 4
    christoffel = [[[
        sp.simplify(
            sum(
                inverse[i, l]
                * (
                    sp.diff(metric[l, k], coordinates[j])
                    + sp.diff(metric[l, j], coordinates[k])
                    - sp.diff(metric[j, k], coordinates[l])
                )
                / 2
                for l in range(dimension)
            )
        )
        for k in range(dimension)] for j in range(dimension)] for i in range(dimension)]
    ricci = [[
        sp.simplify(
            sum(
                sp.diff(christoffel[k][i][j], coordinates[k])
                - sp.diff(christoffel[k][i][k], coordinates[j])
                + sum(
                    christoffel[k][k][l] * christoffel[l][i][j]
                    - christoffel[k][j][l] * christoffel[l][i][k]
                    for l in range(dimension)
                )
                for k in range(dimension)
            )
        )
        for j in range(dimension)] for i in range(dimension)]

    assert all(component == 0 for row in ricci for component in row)
