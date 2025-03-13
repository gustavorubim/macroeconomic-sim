"""
Solution module for the DSGE model.

This module provides methods for solving the DSGE model,
including perturbation and projection methods.
"""

from dsge.solution.perturbation import PerturbationSolver
from dsge.solution.projection import ProjectionSolver

__all__ = [
    "PerturbationSolver",
    "ProjectionSolver",
]