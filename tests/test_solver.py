import math

import numpy as np
import pytest

from numerical_solutions.calculations.naked_singularity.solver import (
    SimulationConfig,
    calculate_errors,
    solve_backward_forward,
)


def test_simulation_config_defaults_match_presentation() -> None:
    config = SimulationConfig()

    assert config.mass_parameter == 1.0
    assert config.amplitude == 1.0
    assert config.x_min == 1.8
    assert config.x_max == 7.0
    assert config.rtol == 1e-9
    assert config.atol == 1e-12
    assert config.accuracy_threshold == 1e-6


@pytest.mark.parametrize(
    "changes",
    [
        {"mass_parameter": 0.0},
        {"amplitude": math.inf},
        {"x_min": 0.0},
        {"x_min": 7.0},
        {"rtol": 0.0},
        {"atol": -1.0},
        {"accuracy_threshold": 0.0},
        {"samples": 1},
        {"samples": 2.5},
    ],
)
def test_simulation_config_rejects_invalid_values(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**changes)


def test_calculate_errors_uses_component_relative_errors() -> None:
    errors = calculate_errors([2.0, 4.0], [2.2, 3.0], threshold=0.3)

    np.testing.assert_allclose(errors.absolute, [0.2, 1.0])
    np.testing.assert_allclose(errors.relative, [0.1, 0.25])
    assert errors.max_relative == pytest.approx(0.25)
    assert errors.threshold_met


def test_calculate_errors_validates_shapes_and_threshold() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_errors([1.0], [1.0, 2.0], threshold=1e-6)
    with pytest.raises(ValueError, match="threshold"):
        calculate_errors([1.0], [1.0], threshold=0.0)


def test_default_solution_reproduces_presentation_numerics() -> None:
    result = solve_backward_forward(SimulationConfig())

    assert result.x.shape == (1000,)
    assert result.state.shape == (3, 1000)
    assert result.f_min == pytest.approx(0.078956, rel=2e-3)
    assert result.x_at_f_min == pytest.approx(2.068, abs=0.01)
    assert result.inner_state[1] == pytest.approx(0.886288, rel=2e-4)
    assert not result.horizon_detected
    assert result.backward_evaluations > 0
    assert result.forward_evaluations > 0

    assert result.errors.absolute[0] < 5e-12
    assert result.errors.absolute[1] == pytest.approx(1.64e-10, rel=0.5)
    assert result.errors.absolute[2] == pytest.approx(1.69e-10, rel=0.5)
    assert result.errors.max_relative == pytest.approx(8.82e-6, rel=0.5)
    assert not result.errors.threshold_met
