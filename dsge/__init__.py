"""
DSGE model package.

This package provides a comprehensive implementation of the Smets and Wouters (2007)
DSGE model with various extensions, including financial frictions, open economy features,
and fiscal policy extensions.

The package includes modules for:
- Core model components
- Data fetching and processing
- Solution methods
- Estimation
- Analysis
- Forecasting

References:
    Smets, F., & Wouters, R. (2007). Shocks and frictions in US business cycles: 
    A Bayesian DSGE approach. American Economic Review, 97(3), 586-606.
"""

from dsge.core import SmetsWoutersModel, ModelVariables
from dsge.core import compute_steady_state, check_steady_state
from dsge.solution import PerturbationSolver, ProjectionSolver
from dsge.data import DataFetcher, DataProcessor
from dsge.estimation import BayesianEstimator, PriorDistribution, PriorSet
from dsge.analysis import ImpulseResponseFunctions, ShockDecomposition, ModelDiagnostics
from dsge.forecasting import BaselineForecaster, ScenarioForecaster, UncertaintyQuantifier

__version__ = "0.1.0"

__all__ = [
    # Core
    "SmetsWoutersModel",
    "ModelVariables",
    "compute_steady_state",
    "check_steady_state",
    
    # Solution
    "PerturbationSolver",
    "ProjectionSolver",
    
    # Data
    "DataFetcher",
    "DataProcessor",
    
    # Estimation
    "BayesianEstimator",
    "PriorDistribution",
    "PriorSet",
    
    # Analysis
    "ImpulseResponseFunctions",
    "ShockDecomposition",
    "ModelDiagnostics",
    
    # Forecasting
    "BaselineForecaster",
    "ScenarioForecaster",
    "UncertaintyQuantifier",
]