"""
Initialization module for DSGE model forecasting.

This module provides classes and utilities for generating forecasts
from estimated DSGE models.
"""

"""
DSGE model forecasting functionality.
"""

from dsge.forecasting.baseline import BaselineForecaster
from dsge.forecasting.scenario import ScenarioForecaster
from dsge.forecasting.uncertainty import UncertaintyQuantifier, UncertaintyBands

__all__ = [
    'BaselineForecaster',
    'ScenarioForecaster',
    'UncertaintyQuantifier',
    'UncertaintyBands'
]

class ScenarioForecaster(BaselineForecaster):
    """
    Extension of BaselineForecaster for scenario analysis.
    Placeholder for future implementation.
    """
    pass

class UncertaintyQuantifier(BaselineForecaster):
    """
    Extension of BaselineForecaster for uncertainty quantification.
    Placeholder for future implementation.
    """
    pass

__all__ = [
    'BaselineForecaster',
    'ScenarioForecaster',
    'UncertaintyQuantifier',
]