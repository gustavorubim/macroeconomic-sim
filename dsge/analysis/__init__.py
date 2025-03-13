"""
Analysis module for the DSGE model.

This module provides functionality for analyzing the DSGE model,
including impulse response functions, shock decomposition, and diagnostics.
"""

from dsge.analysis.impulse_response import ImpulseResponseFunctions
from dsge.analysis.decomposition import ShockDecomposition
from dsge.analysis.diagnostics import ModelDiagnostics

__all__ = [
    "ImpulseResponseFunctions",
    "ShockDecomposition",
    "ModelDiagnostics",
]