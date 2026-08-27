from pathlib import Path

import numpy as np

from numerical_solutions.calculations.naked_singularity.plotting import plot_results
from numerical_solutions.calculations.naked_singularity.solver import (
    ErrorMetrics,
    SimulationConfig,
    SimulationResult,
)


def _synthetic_result() -> SimulationResult:
    x = np.linspace(1.8, 7.0, 10)
    state = np.vstack(
        (
            0.1 + (x - 2.0) ** 2,
            np.exp(-x),
            -np.exp(-x),
        )
    )
    zero_error = np.zeros(3)
    return SimulationResult(
        config=SimulationConfig(samples=10),
        x=x,
        state=state,
        initial_state=state[:, -1],
        inner_state=state[:, 0],
        roundtrip_state=state[:, -1],
        errors=ErrorMetrics(zero_error, zero_error, 0.0, True),
        f_min=float(np.min(state[0])),
        x_at_f_min=float(x[np.argmin(state[0])]),
        horizon_detected=False,
        backward_evaluations=1,
        forward_evaluations=1,
    )


def test_plot_results_writes_both_png_files(tmp_path: Path) -> None:
    metric_path, scalar_path = plot_results(_synthetic_result(), tmp_path)

    assert metric_path == tmp_path / "metric_function.png"
    assert scalar_path == tmp_path / "scalar_field.png"
    assert metric_path.stat().st_size > 0
    assert scalar_path.stat().st_size > 0
