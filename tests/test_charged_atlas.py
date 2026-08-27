from pathlib import Path

import numpy as np

from numerical_solutions.calculations.charged_scalar_black_hole import atlas
from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    SolverConfig,
)


def test_atlas_resume_reports_total_without_repeating_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    attempts = 0

    def fail_start(task):
        nonlocal attempts
        attempts += 1
        candidate = {
            "problem": {"e": 1.0, "m": 1.0, "beta": 0.0, "psi_h": 0.2},
            "source_seed": {"mu_h": 0.3},
            "fitted_parameters": {"mu_h": 0.31, "alpha": 0.2, "M": 0.5},
            "diagnostics": {
                "residual_kind": (
                    "max-of-collocation-rms-relative-and-normalized-boundary"
                ),
                "ranking_residual": 0.25,
            },
            "radius": np.array([1.001, 2.0]),
            "state": np.zeros((5, 2)),
            "encoded_parameters": np.zeros(3),
        }
        return None, "collocation failed: test", 0.25, candidate

    monkeypatch.setattr(atlas, "_attempt_start", fail_start)
    atlas_config = atlas.AtlasConfig(
        masses=(1.0,),
        charge_mass_ratios=(1.0,),
        beta_mass_squared_max=0.0,
        starts_per_node_template=1,
        node_templates=((0, 0),),
    )
    solver_config = SolverConfig(
        initial_nodes=20,
        samples=20,
        max_nodes=20,
        use_homotopy=False,
    )

    first = atlas.search_atlas(
        tmp_path, atlas_config=atlas_config, solver_config=solver_config
    )
    second = atlas.search_atlas(
        tmp_path, atlas_config=atlas_config, solver_config=solver_config
    )

    assert attempts == 1
    assert first.attempted_starts == second.attempted_starts == 1
    assert first.invalid_regions == second.invalid_regions == 1
    assert len((tmp_path / "invalid-regions.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "best-candidates.jsonl").read_text().splitlines()) == 1
    candidate = store_record = atlas.ResultStore(tmp_path).best_candidate_records()[0]
    assert candidate["diagnostics"]["ranking_residual"] == 0.25
    assert (tmp_path / store_record["profile"]).is_file()
