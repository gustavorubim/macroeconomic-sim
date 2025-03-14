"""
Initialization module for DSGE model estimation.

This module provides classes and utilities for estimating DSGE models
using various methods, with a focus on Bayesian estimation.
"""

from dsge.estimation.bayesian import BayesianEstimator, Prior, PriorSet

class PriorDistribution(Prior):
    """Alias for Prior class for backward compatibility."""
    pass

def create_default_priors() -> PriorSet:
    """
    Create a set of default priors for the Smets-Wouters model.
    
    Returns:
        PriorSet: Default prior specifications
    """
    priors = {
        # Steady state related parameters
        "alpha": Prior("alpha", "beta", {"mean": 0.3, "std": 0.05}, bounds=(0.2, 0.4)),
        "beta": Prior("beta", "beta", {"loc": 0.99, "scale": 0.002}, bounds=(0.95, 0.999)),
        "delta": Prior("delta", "beta", {"loc": 0.025, "scale": 0.005}, bounds=(0.01, 0.05)),
        
        # Household preferences
        "sigma_c": Prior("sigma_c", "gamma", {"loc": 2.0, "scale": 0.1}, bounds=(1.0, 3.0)),
        "h": Prior("h", "beta", {"loc": 0.7, "scale": 0.1}, bounds=(0.5, 0.95)),
        "sigma_l": Prior("sigma_l", "gamma", {"loc": 2.0, "scale": 0.75}, bounds=(0.5, 5.0)),
        
        # Price and wage rigidities
        "xi_p": Prior("xi_p", "beta", {"loc": 0.75, "scale": 0.05}, bounds=(0.5, 0.9)),
        "xi_w": Prior("xi_w", "beta", {"loc": 0.75, "scale": 0.05}, bounds=(0.5, 0.9)),
        "iota_p": Prior("iota_p", "beta", {"loc": 0.5, "scale": 0.15}, bounds=(0.0, 1.0)),
        "iota_w": Prior("iota_w", "beta", {"loc": 0.5, "scale": 0.15}, bounds=(0.0, 1.0)),
        # Monetary policy
        "rho_r": Prior("rho_r", "beta", {"loc": 0.8, "scale": 0.1}, bounds=(0.5, 0.97)),
        "phi_pi": Prior("phi_pi", "normal", {"loc": 1.5, "scale": 0.25}, bounds=(1.0, 3.0)),
        "phi_y": Prior("phi_y", "gamma", {"loc": 0.125, "scale": 0.05}, bounds=(0.0, 0.5)),
        "phi_dy": Prior("phi_dy", "gamma", {"loc": 0.125, "scale": 0.05}, bounds=(0.0, 0.5)),
        
        # Steady state rates
        "pi_bar": Prior("pi_bar", "gamma", {"loc": 1.005, "scale": 0.0025}, bounds=(1.0, 1.02)),
        "r_bar": Prior("r_bar", "gamma", {"loc": 1.01, "scale": 0.0025}, bounds=(1.0, 1.02)),
        
        # Shock persistence
        "technology_rho": Prior("technology_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        "preference_rho": Prior("preference_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        "investment_rho": Prior("investment_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        "government_rho": Prior("government_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        "monetary_rho": Prior("monetary_rho", "beta", {"loc": 0.5, "scale": 0.2}, bounds=(0.0, 0.8)),
        "price_markup_rho": Prior("price_markup_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        "wage_markup_rho": Prior("wage_markup_rho", "beta", {"loc": 0.9, "scale": 0.05}, bounds=(0.5, 0.999)),
        
        # Shock standard deviations
        "technology_sigma": Prior("technology_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "preference_sigma": Prior("preference_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "investment_sigma": Prior("investment_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "government_sigma": Prior("government_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "monetary_sigma": Prior("monetary_sigma", "gamma", {"loc": 0.005, "scale": 0.0025}, bounds=(0.0, 0.05)),
        "price_markup_sigma": Prior("price_markup_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "wage_markup_sigma": Prior("wage_markup_sigma", "gamma", {"loc": 0.01, "scale": 0.005}, bounds=(0.0, 0.05)),
        "wage_markup_sigma": Prior("wage_markup_sigma", "invgamma", {"a": 3, "b": 0.01}, bounds=(0.0, 0.05)),
    }
    
    return PriorSet(priors)

__all__ = [
    'BayesianEstimator',
    'Prior',
    'PriorSet',
    'PriorDistribution',
    'create_default_priors',
]