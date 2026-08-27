import os
from pathlib import Path
import subprocess
import sys

import pytest

from numerical_solutions.cli import build_parser, main


def test_parser_accepts_naked_singularity_parameters() -> None:
    args = build_parser().parse_args(
        [
            "naked-singularity",
            "--mass-parameter",
            "2.0",
            "--amplitude",
            "0.5",
            "--samples",
            "25",
        ]
    )

    assert args.command == "naked-singularity"
    assert args.mass_parameter == 2.0
    assert args.amplitude == 0.5
    assert args.samples == 25


def test_parser_accepts_charged_black_hole_parameters() -> None:
    args = build_parser().parse_args(
        [
            "charged-black-hole",
            "solve",
            "--charge",
            "0.8",
            "--scalar-mass",
            "1.2",
            "--beta",
            "0.1",
            "--psi-h",
            "0.25",
        ]
    )

    assert args.command == "charged-black-hole"
    assert args.black_hole_action == "solve"
    assert args.charge == 0.8
    assert args.scalar_mass == 1.2
    assert args.beta == 0.1
    assert args.psi_h == 0.25


def test_main_reports_invalid_configuration() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["naked-singularity", "--x-min", "7", "--x-max", "1.8"])

    assert exit_info.value.code == 1


def test_module_cli_end_to_end(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(repository_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_solutions",
            "naked-singularity",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "f_min=" in completed.stdout
    assert "accuracy threshold 1.000000e-06: NOT MET" in completed.stdout
    assert (tmp_path / "metric_function.png").stat().st_size > 0
    assert (tmp_path / "scalar_field.png").stat().st_size > 0


def test_charged_atlas_cli_end_to_end_records_unresolved_search(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_solutions",
            "charged-black-hole",
            "atlas",
            "--masses",
            "1",
            "--charge-mass-ratios",
            "1",
            "--starts",
            "1",
            "--disable-homotopy",
            "--beta-m2-max",
            "0",
            "--horizon-epsilon",
            "0.01",
            "--r-max",
            "5",
            "--tol",
            "0.01",
            "--bc-tol",
            "0.001",
            "--max-nodes",
            "100",
            "--initial-nodes",
            "20",
            "--samples",
            "20",
            "--output-dir",
            str(tmp_path / "results"),
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "attempted starts: 7" in completed.stdout
    assert (tmp_path / "results" / "invalid-regions.jsonl").stat().st_size > 0
    assert (tmp_path / "results" / "search-report.md").stat().st_size > 0
