"""Persistent, versioned output for black-hole configurations and failed searches."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BlackHoleSolution,
)


EQUATION_REVISION = "CSF2023-06-12-corrected-v6"


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class InvalidRegion:
    """A bounded search cell that was invalid or numerically unresolved."""

    status: str
    bounds: dict[str, Any]
    node_templates: tuple[tuple[int, int], ...]
    attempted_starts: int
    best_residual: float | None
    failure_kind: str
    message: str
    solver_settings: dict[str, Any]
    failure_counts: dict[str, int] = field(default_factory=dict)
    last_valid_configuration: str | None = None
    best_residual_kind: str | None = None
    best_candidate_id: str | None = None
    equation_revision: str = EQUATION_REVISION
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.status not in {"invalid", "numerically_unresolved"}:
            raise ValueError("status must be invalid or numerically_unresolved")
        if self.attempted_starts < 0:
            raise ValueError("attempted_starts must be non-negative")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "equation_revision": self.equation_revision,
                "bounds": self.bounds,
                "node_templates": self.node_templates,
                "solver_settings": self.solver_settings,
            }
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "equation_revision": self.equation_revision,
            "fingerprint": self.fingerprint,
            "recorded_at": self.recorded_at,
            "status": self.status,
            "bounds": self.bounds,
            "node_templates": [list(item) for item in self.node_templates],
            "attempted_starts": self.attempted_starts,
            "best_residual": self.best_residual,
            "best_residual_kind": self.best_residual_kind,
            "best_candidate_id": self.best_candidate_id,
            "failure_kind": self.failure_kind,
            "message": self.message,
            "failure_counts": self.failure_counts,
            "last_valid_configuration": self.last_valid_configuration,
            "solver_settings": self.solver_settings,
        }


class ResultStore:
    """Append-only JSONL store with derived CSV and Markdown summaries."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.profile_dir = self.root / "profiles"
        self.candidate_profile_dir = self.root / "candidate-profiles"
        self.plot_dir = self.root / "plots"
        self.configurations_path = self.root / "configurations.jsonl"
        self.invalid_regions_path = self.root / "invalid-regions.jsonl"
        self.best_candidates_path = self.root / "best-candidates.jsonl"
        self.csv_path = self.root / "configurations.csv"
        self.markdown_path = self.root / "configurations.md"

    def prepare(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_profile_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def append_solution(
        self,
        solution: BlackHoleSolution,
        *,
        branch_id: str,
        termination_reason: str,
    ) -> dict[str, Any]:
        if not solution.classification.valid or not solution.refinement_verified:
            raise ValueError(
                "only refinement-verified physical configurations may be published"
            )
        self.prepare()
        summary = solution.summary()
        configuration_id = _canonical_hash(
            {
                "branch_id": branch_id,
                "parameters": summary["parameters"],
                "asymptotics": summary["asymptotics"],
            }
        )
        profile_relative = Path("profiles") / f"{configuration_id}.npz"
        np.savez_compressed(
            self.root / profile_relative,
            radius=solution.radius,
            f=solution.state[0],
            psi=solution.state[1],
            psi_prime=solution.state[2],
            tau=solution.state[3],
            mu=solution.state[4],
            F=solution.metric_F,
        )
        record = {
            **summary,
            "configuration_id": configuration_id,
            "branch_id": branch_id,
            "termination_reason": termination_reason,
            "profile": str(profile_relative),
        }
        with self.configurations_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.rebuild_summaries()
        return record

    def append_invalid_region(self, region: InvalidRegion) -> None:
        self.prepare()
        with self.invalid_regions_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(region.as_record(), ensure_ascii=False, sort_keys=True) + "\n"
            )

    def append_best_candidate(
        self,
        candidate: dict[str, Any],
        *,
        cell_fingerprint: str,
        failure_kind: str,
        failure_message: str,
    ) -> str:
        """Persist the closest rejected iterate and its raw sampled profile."""

        self.prepare()
        diagnostics = candidate.get("diagnostics", {})
        residual = diagnostics.get("ranking_residual")
        candidate_id = _canonical_hash(
            {
                "equation_revision": EQUATION_REVISION,
                "cell_fingerprint": cell_fingerprint,
                "problem": candidate.get("problem"),
                "source_seed": candidate.get("source_seed"),
                "fitted_parameters": candidate.get("fitted_parameters"),
                "ranking_residual": residual,
            }
        )
        profile_relative: Path | None = None
        radius = candidate.get("radius")
        state = candidate.get("state")
        if radius is not None and state is not None:
            profile_relative = Path("candidate-profiles") / f"{candidate_id}.npz"
            arrays: dict[str, Any] = {
                "radius": np.asarray(radius, dtype=float),
                "state": np.asarray(state, dtype=float),
            }
            if candidate.get("encoded_parameters") is not None:
                arrays["encoded_parameters"] = np.asarray(
                    candidate["encoded_parameters"], dtype=float
                )
            if candidate.get("metric_F") is not None:
                arrays["F"] = np.asarray(candidate["metric_F"], dtype=float)
            np.savez_compressed(self.root / profile_relative, **arrays)
        record = {
            "schema_version": 1,
            "equation_revision": EQUATION_REVISION,
            "candidate_id": candidate_id,
            "cell_fingerprint": cell_fingerprint,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "failure_kind": failure_kind,
            "failure_message": failure_message,
            "problem": candidate.get("problem"),
            "source_seed": candidate.get("source_seed"),
            "fitted_parameters": candidate.get("fitted_parameters"),
            "diagnostics": diagnostics,
            "solution_summary": candidate.get("solution_summary"),
            "profile": str(profile_relative) if profile_relative is not None else None,
        }
        with self.best_candidates_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return candidate_id

    def best_candidate_records(self) -> list[dict[str, Any]]:
        """Return the latest best-candidate record for every atlas cell."""

        if not self.best_candidates_path.exists():
            return []
        latest: dict[str, dict[str, Any]] = {}
        with self.best_candidates_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("equation_revision") == EQUATION_REVISION:
                    latest[str(record["cell_fingerprint"])] = record
        return list(latest.values())

    def invalid_fingerprints(self) -> set[str]:
        return {
            str(record["fingerprint"])
            for record in self.invalid_records()
        }

    def invalid_records(self) -> list[dict[str, Any]]:
        if not self.invalid_regions_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.invalid_regions_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if record.get("equation_revision") == EQUATION_REVISION:
                        records.append(record)
        return records

    def records(self) -> list[dict[str, Any]]:
        if not self.configurations_path.exists():
            return []
        with self.configurations_path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def rebuild_summaries(self) -> None:
        records = self.records()
        columns = (
            "configuration_id",
            "branch_id",
            "label",
            "n_psi",
            "n_tau",
            "valid",
            "alpha",
            "beta",
            "e",
            "m",
            "psi_h",
            "mu_h",
            "f_prime_h",
            "F_h",
            "M",
            "p",
            "q",
            "g",
            "surface_gravity",
            "max_equation_residual",
            "max_raw_equation_residual",
            "boundary_residual",
            "termination_reason",
        )
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow(_flatten_record(record))

        lines = [
            "# Конфигурации заряженной скалярной чёрной дыры",
            "",
            "Этот файл автоматически построен из `configurations.jsonl`.",
            "",
            "| ID | Ветвь | Конфигурация | α | β | e | m | M | max residual |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for record in records:
            flat = _flatten_record(record)
            lines.append(
                "| {configuration_id} | {branch_id} | {label} | {alpha:.6g} | "
                "{beta:.6g} | {e:.6g} | {m:.6g} | {M:.6g} | "
                "{max_equation_residual:.3e} |".format(**flat)
            )
        self.markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    classification = record["classification"]
    parameters = record["parameters"]
    horizon = record["horizon"]
    asymptotics = record["asymptotics"]
    diagnostics = record["diagnostics"]
    return {
        "configuration_id": record["configuration_id"],
        "branch_id": record["branch_id"],
        "label": classification["label"],
        "n_psi": classification["n_psi"],
        "n_tau": classification["n_tau"],
        "valid": classification["valid"],
        "alpha": parameters["alpha"],
        "beta": parameters["beta"],
        "e": parameters["e"],
        "m": parameters["m"],
        "psi_h": parameters["psi_h"],
        "mu_h": parameters["mu_h"],
        "f_prime_h": horizon["f_prime"],
        "F_h": horizon["F_h"],
        "M": asymptotics["M"],
        "p": asymptotics["p"],
        "q": asymptotics["q"],
        "g": asymptotics["g"],
        "surface_gravity": horizon["surface_gravity"],
        "max_equation_residual": diagnostics["max_equation_residual"],
        "max_raw_equation_residual": diagnostics["max_raw_equation_residual"],
        "boundary_residual": diagnostics["boundary_residual"],
        "termination_reason": record["termination_reason"],
    }


def deduplicate_solutions(
    solutions: Iterable[BlackHoleSolution], *, relative_tolerance: float = 1e-5
) -> list[BlackHoleSolution]:
    """Deduplicate roots using nodal labels and principal scalar parameters."""

    unique: list[BlackHoleSolution] = []
    for candidate in solutions:
        candidate_vector = np.array(
            [
                candidate.psi_h,
                candidate.alpha,
                candidate.mu_h,
                candidate.adm_mass,
            ],
            dtype=float,
        )
        duplicate = False
        for existing in unique:
            if (
                candidate.classification.psi_nodes
                != existing.classification.psi_nodes
                or candidate.classification.tau_nodes
                != existing.classification.tau_nodes
            ):
                continue
            existing_vector = np.array(
                [
                    existing.psi_h,
                    existing.alpha,
                    existing.mu_h,
                    existing.adm_mass,
                ],
                dtype=float,
            )
            scale = np.maximum(np.abs(existing_vector), 1.0)
            if np.max(np.abs(candidate_vector - existing_vector) / scale) < relative_tolerance:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique
