"""Charged scalar black-hole boundary-value calculation."""

from numerical_solutions.calculations.charged_scalar_black_hole.model import (
    DegenerateSystemError,
    ModelParameters,
    equation_residuals,
    field_derivatives,
    normalized_equation_residuals,
)
from numerical_solutions.calculations.charged_scalar_black_hole.solver import (
    BlackHoleSolution,
    BoundarySeed,
    BoundaryValueError,
    SolverConfig,
    solve_black_hole,
)

__all__ = [
    "DegenerateSystemError",
    "ModelParameters",
    "BlackHoleSolution",
    "BoundarySeed",
    "BoundaryValueError",
    "SolverConfig",
    "equation_residuals",
    "field_derivatives",
    "normalized_equation_residuals",
    "solve_black_hole",
]
