import math

import numpy as np
import pytest

from numerical_solutions.calculations.naked_singularity.model import (
    asymptotic_phi,
    asymptotic_state,
    system_equations,
)


def test_system_equations_matches_first_order_form() -> None:
    derivative = system_equations(2.0, [0.5, 0.1, -0.2], 1.0)

    np.testing.assert_allclose(derivative, [0.19, -0.2, 0.492])


def test_system_equations_rejects_singular_points() -> None:
    with pytest.raises(FloatingPointError, match="x=0"):
        system_equations(0.0, [1.0, 0.1, 0.1], 1.0)
    with pytest.raises(FloatingPointError, match="f=0"):
        system_equations(1.0, [0.0, 0.1, 0.1], 1.0)


def test_asymptotic_phi_uses_presentation_formula() -> None:
    expected = math.exp(-7.0) / 7.0**2

    assert asymptotic_phi(7.0, 1.0, 1.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "mass_parameter", "amplitude"),
    [
        (0.0, 1.0, 1.0),
        (7.0, 0.0, 1.0),
        (7.0, 1.0, math.inf),
    ],
)
def test_asymptotic_phi_validates_inputs(
    x: float,
    mass_parameter: float,
    amplitude: float,
) -> None:
    with pytest.raises(ValueError):
        asymptotic_phi(x, mass_parameter, amplitude)


def test_asymptotic_state_includes_metric_and_derivative() -> None:
    state = asymptotic_state(7.0, 1.0, 1.0)
    field = math.exp(-7.0) / 7.0**2

    np.testing.assert_allclose(
        state,
        [5.0 / 7.0, field, -field * (1.0 + 2.0 / 7.0)],
    )
