"""
Core module for the DSGE model.

This module contains the core components of the DSGE model,
including the base Smets and Wouters model and the steady state solver.
"""

from dsge.core.base_model import SmetsWoutersModel, ModelVariables
from dsge.core.steady_state import (
    compute_steady_state,
    compute_base_steady_state,
    compute_financial_extension_steady_state,
    compute_open_economy_extension_steady_state,
    compute_fiscal_extension_steady_state,
    check_steady_state,
)

__all__ = [
    "SmetsWoutersModel",
    "ModelVariables",
    "compute_steady_state",
    "compute_base_steady_state",
    "compute_financial_extension_steady_state",
    "compute_open_economy_extension_steady_state",
    "compute_fiscal_extension_steady_state",
    "check_steady_state",
]