import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from numerical_solutions.calculations.charged_scalar_black_hole.results import (
    InvalidRegion,
    ResultStore,
    deduplicate_solutions,
)
from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BlackHoleSolution,
    ConfigurationClassification,
)


def _solution(alpha: float = 0.2) -> BlackHoleSolution:
    radius = np.linspace(1.001, 10.0, 20)
    state = np.vstack(
        (
            1.0 - 1.0 / radius,
            np.exp(-radius),
            -np.exp(-radius),
            (radius - 1.0) * np.exp(-radius),
            (2.0 - radius) * np.exp(-radius),
        )
    )
    classification = ConfigurationClassification(
        geometry="static-spherical-asymptotically-flat-nonextremal-black-hole",
        psi_nodes=0,
        tau_nodes=0,
        non_extremal=True,
        asymptotically_flat=True,
        exterior_regular=True,
        valid=True,
        reasons=(),
    )
    return BlackHoleSolution(
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
        infinity_series_error=1e-8,
        max_equation_residual=1e-9,
        boundary_residual=1e-10,
        solver_status=0,
        solver_message="ok",
        mesh_nodes=100,
        horizon_epsilon=1e-3,
        r_max=10.0,
        classification=classification,
        radius=radius,
        state=state,
        metric_F=np.zeros_like(radius),
        refinement_count=1,
        refinement_verified=True,
        max_observable_change=1e-7,
        max_raw_equation_residual=2e-9,
    )


def test_result_store_writes_jsonl_profile_csv_and_markdown(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)

    record = store.append_solution(
        _solution(), branch_id="branch-0", termination_reason="test"
    )

    saved = json.loads(store.configurations_path.read_text(encoding="utf-8"))
    assert saved["configuration_id"] == record["configuration_id"]
    assert saved["classification"]["label"] == "BH(n_psi=0,n_tau=0)"
    assert saved["diagnostics"]["residual_kind"] == "normalized-backward-error"
    assert saved["diagnostics"]["max_raw_equation_residual"] == 2e-9
    assert (tmp_path / saved["profile"]).stat().st_size > 0
    assert store.csv_path.stat().st_size > 0
    assert store.markdown_path.stat().st_size > 0


def test_invalid_region_fingerprint_is_reusable(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    region = InvalidRegion(
        status="numerically_unresolved",
        bounds={"m": [1.0, 1.0]},
        node_templates=((0, 0),),
        attempted_starts=8,
        best_residual=None,
        failure_kind="no-root",
        message="test",
        solver_settings={"tol": 1e-7},
    )

    store.append_invalid_region(region)

    assert store.invalid_fingerprints() == {region.fingerprint}


def test_best_rejected_candidate_preserves_parameters_and_profile(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path)
    candidate = {
        "problem": {"e": 1.0, "m": 1.0, "beta": 0.0, "psi_h": 0.2},
        "source_seed": {"mu_h": 0.3, "alpha": 0.1},
        "fitted_parameters": {"mu_h": 0.31, "alpha": 0.12, "M": 0.55},
        "diagnostics": {
            "residual_kind": (
                "max-of-collocation-rms-relative-and-normalized-boundary"
            ),
            "ranking_residual": 2e-4,
        },
        "radius": np.array([1.001, 2.0]),
        "state": np.ones((5, 2)),
        "encoded_parameters": np.zeros(3),
    }

    candidate_id = store.append_best_candidate(
        candidate,
        cell_fingerprint="cell-1",
        failure_kind="singular-collocation-jacobian",
        failure_message="test",
    )

    record = store.best_candidate_records()[0]
    assert record["candidate_id"] == candidate_id
    assert record["fitted_parameters"]["M"] == 0.55
    assert record["diagnostics"]["ranking_residual"] == 2e-4
    profile = np.load(tmp_path / record["profile"])
    assert profile["state"].shape == (5, 2)


def test_deduplication_keeps_distinct_parameter_roots() -> None:
    first = _solution(alpha=0.2)
    duplicate = _solution(alpha=0.2 + 1e-8)
    distinct = _solution(alpha=0.4)

    assert deduplicate_solutions([first, duplicate, distinct]) == [first, distinct]


def test_result_store_rejects_unrefined_configuration(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    unrefined = replace(
        _solution(),
        refinement_verified=False,
        max_observable_change=float("inf"),
    )

    with pytest.raises(ValueError, match="refinement-verified"):
        store.append_solution(
            unrefined, branch_id="invalid", termination_reason="test"
        )
