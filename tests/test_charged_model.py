import math

import numpy as np

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    DegenerateSystemError,
    ModelParameters,
    equation_residuals,
    field_derivatives,
    field_derivatives_beta_zero,
    normalized_equation_residuals,
)


def test_corrected_decay_rate_uses_decaying_characteristic_root() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)

    expected = math.sqrt(1.0 + 1.4 / 1.1) - 1.0

    assert parameters.decay_rate == expected
    assert parameters.decay_rate > 0.0


def test_explicit_derivatives_satisfy_original_equations() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    state = np.array([0.4, 0.2, -0.1, 0.03, 0.2])

    derivatives = field_derivatives(2.0, state, parameters)
    residuals = equation_residuals(
        2.0,
        state,
        derivatives[[0, 2, 3, 4]],
        parameters,
    )

    np.testing.assert_allclose(residuals, 0.0, atol=1e-13, rtol=0.0)


def test_general_system_reduces_to_independent_beta_zero_system() -> None:
    parameters = ModelParameters(alpha=-0.1, beta=0.0, charge=0.8, scalar_mass=1.2)
    state = np.array([0.6, 0.25, -0.08, 0.04, 0.3])

    general = field_derivatives(2.5, state, parameters)
    specialized = field_derivatives_beta_zero(2.5, state, parameters)

    np.testing.assert_allclose(general, specialized, atol=1e-13, rtol=1e-13)


def test_decay_condition_is_enforced() -> None:
    parameters = ModelParameters(alpha=-1.0, beta=0.0, charge=1.0, scalar_mass=1.0)

    try:
        parameters.decay_rate
    except ValueError as error:
        assert "decaying electromagnetic mode" in str(error)
    else:  # pragma: no cover - protects the physical admissibility condition
        raise AssertionError("non-decaying parameters were accepted")


def test_metric_zero_is_a_degenerate_surface() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)

    try:
        field_derivatives(2.0, np.array([0.0, 0.2, -0.1, 0.03, 0.2]), parameters)
    except DegenerateSystemError:
        pass
    else:
        raise AssertionError("f=0 outside the regularized horizon was accepted")


def test_exponentially_small_tail_is_row_scaled_not_declared_degenerate() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    amplitude = math.exp(-20.0)
    state = np.array(
        [0.95, amplitude, -amplitude, 0.2 * amplitude, -0.1 * amplitude]
    )

    derivatives = field_derivatives(20.0, state, parameters)

    assert np.all(np.isfinite(derivatives))


def test_normalized_residual_is_bounded_by_raw_backward_error() -> None:
    parameters = ModelParameters(alpha=0.2, beta=0.1, charge=1.0, scalar_mass=1.0)
    state = np.array([0.4, 0.2, -0.1, 0.03, 0.2])
    derivatives = field_derivatives(2.0, state, parameters)[[0, 2, 3, 4]]
    derivatives[1] += 1e-5

    raw = equation_residuals(2.0, state, derivatives, parameters)
    normalized = normalized_equation_residuals(
        2.0, state, derivatives, parameters
    )

    assert np.all(np.abs(normalized) <= np.abs(raw) + 1e-30)
    assert np.max(np.abs(normalized)) > 0.0
