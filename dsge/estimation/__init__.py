"""
Estimation module for the DSGE model.

This module provides functionality for estimating the DSGE model
using Bayesian methods.
"""

from dsge.estimation.bayesian import BayesianEstimator, Prior
from dsge.estimation.priors import PriorDistribution, PriorSet, create_default_priors
from dsge.estimation.posteriors import PosteriorAnalysis

__all__ = [
    "BayesianEstimator",
    "Prior",
    "PriorDistribution",
    "PriorSet",
    "create_default_priors",
    "PosteriorAnalysis",
]