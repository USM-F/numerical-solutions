import numpy as np

from numerical_solutions.calculations.charged_scalar_black_hole import solver


def _solution(*, alpha: float = 0.2, infinity_error: float = 1e-7):
    radius = np.linspace(1.001, 20.0, 40)
    state = np.vstack(
        (
            1.0 - 1.0 / radius,
            np.exp(-radius),
            -np.exp(-radius),
            (radius - 1.0) * np.exp(-radius),
            (2.0 - radius) * np.exp(-radius),
        )
    )
    classification = solver.ConfigurationClassification(
        geometry="static-spherical-asymptotically-flat-nonextremal-black-hole",
        psi_nodes=0,
        tau_nodes=0,
        non_extremal=True,
        asymptotically_flat=True,
        exterior_regular=True,
        valid=True,
        reasons=(),
    )
    return solver.BlackHoleSolution(
        charge=1.0,
        scalar_mass=1.0,
        beta=0.0,
        psi_h=0.3,
        mu_h=0.4,
        alpha=alpha,
        adm_mass=0.5,
        scalar_amplitude=1.0,
        tau_amplitude=0.2,
        decay_rate=0.5,
        f_prime_h=0.9,
        F_h=-0.1,
        surface_gravity=0.4,
        horizon_series_order=6,
        horizon_series_error=1e-12,
        infinity_series_error=infinity_error,
        max_equation_residual=1e-9,
        boundary_residual=1e-10,
        solver_status=0,
        solver_message="ok",
        mesh_nodes=100,
        horizon_epsilon=1e-3,
        r_max=20.0,
        classification=classification,
        radius=radius,
        state=state,
        metric_F=np.zeros_like(radius),
        max_raw_equation_residual=2e-9,
    )


def test_publication_solve_refines_both_boundaries_and_tolerances(monkeypatch) -> None:
    calls = []
    initial = _solution(alpha=0.2)
    refined = _solution(alpha=0.2 + 1e-6)

    def fake_solve_once(**kwargs):
        calls.append(kwargs)
        return initial if len(calls) == 1 else refined

    monkeypatch.setattr(solver, "_solve_once", fake_solve_once)
    config = solver.SolverConfig(max_refinements=1)

    result = solver.solve_black_hole(
        charge=1.0, scalar_mass=1.0, beta=0.0, psi_h=0.3, config=config
    )

    assert result.classification.valid
    assert result.refinement_verified
    assert result.refinement_count == 1
    assert calls[1]["config"].horizon_epsilon < calls[0]["config"].horizon_epsilon
    assert calls[1]["config"].r_max > calls[0]["config"].r_max
    assert calls[1]["config"].tolerance < calls[0]["config"].tolerance
    assert calls[1]["initial_solution"] is initial


def test_unstable_observables_are_not_publishable(monkeypatch) -> None:
    calls = 0

    def fake_solve_once(**kwargs):
        nonlocal calls
        calls += 1
        return _solution(alpha=0.2 if calls == 1 else 0.4)

    monkeypatch.setattr(solver, "_solve_once", fake_solve_once)

    result = solver.solve_black_hole(
        charge=1.0,
        scalar_mass=1.0,
        beta=0.0,
        psi_h=0.3,
        config=solver.SolverConfig(max_refinements=1),
    )

    assert not result.classification.valid
    assert not result.refinement_verified
    assert result.max_observable_change > 1e-5


def test_pseudo_arclength_adds_a_ninth_phase_condition(monkeypatch) -> None:
    previous = _solution(alpha=0.19)
    current = _solution(alpha=0.2)
    current = solver.replace(current, psi_h=0.31)
    observed = {}

    class FakeCollocation:
        success = True
        message = "ok"

        def __init__(self, parameters):
            self.p = parameters

    def fake_solve_bvp(rhs, boundary, radius, state, *, p, **kwargs):
        observed["conditions"] = boundary(state[:, 0], state[:, -1], p).size
        observed["unknowns"] = p.size
        return FakeCollocation(p)

    expected = _solution(alpha=0.21)
    monkeypatch.setattr(solver, "solve_bvp", fake_solve_bvp)
    monkeypatch.setattr(solver, "solve_black_hole", lambda **kwargs: expected)

    result = solver.solve_black_hole_pseudo_arclength(
        previous,
        current,
        0.01,
        config=solver.SolverConfig(
            horizon_epsilon=1e-2,
            boundary_tolerance=1e-3,
            initial_nodes=20,
            samples=20,
        ),
    )

    assert result is expected
    assert observed == {"conditions": 9, "unknowns": 4}
