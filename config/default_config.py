"""
Default configuration for the DSGE model.
This file contains the default parameters for the Smets and Wouters model
and its extensions.
"""

from typing import Dict, Any, List, Optional

# Base Smets and Wouters model parameters
BASE_MODEL_PARAMS = {
    # Household parameters
    "beta": 0.99,  # Discount factor
    "sigma_c": 1.5,  # Intertemporal elasticity of substitution
    "h": 0.7,  # Habit persistence
    "sigma_l": 2.0,  # Labor supply elasticity
    
    # Production parameters
    "alpha": 0.3,  # Capital share
    "delta": 0.025,  # Depreciation rate
    "epsilon_p": 10.0,  # Price elasticity of demand
    "xi_p": 0.75,  # Calvo price stickiness
    "iota_p": 0.5,  # Price indexation
    
    # Wage setting parameters
    "epsilon_w": 10.0,  # Wage elasticity of labor demand
    "xi_w": 0.75,  # Calvo wage stickiness
    "iota_w": 0.5,  # Wage indexation
    
    # Monetary policy parameters
    "rho_r": 0.8,  # Interest rate smoothing
    "phi_pi": 1.5,  # Response to inflation
    "phi_y": 0.125,  # Response to output gap
    "phi_dy": 0.125,  # Response to output growth
    
    # Steady state parameters
    "pi_bar": 1.005,  # Steady state inflation (quarterly)
    "r_bar": 1.01,  # Steady state real interest rate (quarterly)
    "g_y": 1.005,  # Steady state output growth (quarterly)
}

# Financial frictions extension parameters
FINANCIAL_EXTENSION_PARAMS = {
    "enabled": False,  # Whether to enable financial frictions
    "mu": 0.12,  # Monitoring cost
    "sigma_omega": 0.3,  # Standard deviation of idiosyncratic shock
    "gamma": 0.975,  # Survival rate of entrepreneurs
    "lambda_b": 0.05,  # Bank leverage constraint
}

# Open economy extension parameters
OPEN_ECONOMY_EXTENSION_PARAMS = {
    "enabled": False,  # Whether to enable open economy features
    "alpha_h": 0.7,  # Home bias in consumption
    "eta": 1.5,  # Elasticity of substitution between home and foreign goods
    "phi_b": 0.01,  # Debt elasticity of interest rate
}

# Fiscal policy extension parameters
FISCAL_EXTENSION_PARAMS = {
    "enabled": False,  # Whether to enable fiscal policy extension
    "g_y_ratio": 0.2,  # Government spending to GDP ratio
    "b_y_ratio": 0.6,  # Government debt to GDP ratio
    "tau_c": 0.2,  # Consumption tax rate
    "tau_k": 0.35,  # Capital tax rate
    "tau_l": 0.28,  # Labor tax rate
    "rho_g": 0.9,  # Government spending persistence
    "phi_b": 0.1,  # Fiscal rule response to debt
}

# Shock processes parameters
SHOCK_PARAMS = {
    "technology": {
        "rho": 0.95,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
    "preference": {
        "rho": 0.9,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
    "investment": {
        "rho": 0.9,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
    "government": {
        "rho": 0.9,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
    "monetary": {
        "rho": 0.5,  # Persistence
        "sigma": 0.003,  # Standard deviation
    },
    "price_markup": {
        "rho": 0.9,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
    "wage_markup": {
        "rho": 0.9,  # Persistence
        "sigma": 0.01,  # Standard deviation
    },
}

# Solution method parameters
SOLUTION_PARAMS = {
    "method": "perturbation",  # "perturbation" or "projection"
    "perturbation_order": 2,  # 1, 2, or 3
    "projection_method": "chebyshev",  # "chebyshev" or "finite_elements"
    "projection_nodes": 5,  # Number of nodes per dimension
}

# Estimation parameters
ESTIMATION_PARAMS = {
    "method": "bayesian",  # "bayesian" or "maximum_likelihood"
    "mcmc_algorithm": "metropolis_hastings",  # MCMC algorithm
    "num_chains": 4,  # Number of MCMC chains
    "num_draws": 10000,  # Number of draws per chain
    "burn_in": 5000,  # Number of burn-in draws
    "tune": 1000,  # Number of tuning iterations
    "target_acceptance": 0.234,  # Target acceptance rate
}

# Data parameters
DATA_PARAMS = {
    "start_date": "1984-01-01",
    "end_date": "2019-12-31",
    "frequency": "quarterly",
    "series": {
        "gdp": {
            "source": "FRED",
            "series_id": "GDPC1",
            "transformation": "log_diff",
        },
        "consumption": {
            "source": "FRED",
            "series_id": "PCECC96",
            "transformation": "log_diff",
        },
        "investment": {
            "source": "FRED",
            "series_id": "GPDIC1",
            "transformation": "log_diff",
        },
        "hours": {
            "source": "FRED",
            "series_id": "HOANBS",
            "transformation": "log_diff",
        },
        "wages": {
            "source": "FRED",
            "series_id": "COMPRNFB",
            "transformation": "log_diff",
        },
        "inflation": {
            "source": "FRED",
            "series_id": "GDPDEF",
            "transformation": "log_diff",
        },
        "interest_rate": {
            "source": "FRED",
            "series_id": "FEDFUNDS",
            "transformation": "level",
        },
    },
    "detrending": {
        "method": "hp_filter",  # "hp_filter", "linear", "bandpass", etc.
        "lambda": 1600,  # HP filter smoothing parameter
    },
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    "style": "seaborn-whitegrid",
    "context": "paper",
    "dpi": 300,
    "figsize": (10, 6),
    "save_format": "pdf",
    "save_path": "figures/",
}

# Default configuration dictionary
DEFAULT_CONFIG = {
    "base_model": BASE_MODEL_PARAMS,
    "financial_extension": FINANCIAL_EXTENSION_PARAMS,
    "open_economy_extension": OPEN_ECONOMY_EXTENSION_PARAMS,
    "fiscal_extension": FISCAL_EXTENSION_PARAMS,
    "shocks": SHOCK_PARAMS,
    "solution": SOLUTION_PARAMS,
    "estimation": ESTIMATION_PARAMS,
    "data": DATA_PARAMS,
    "visualization": VISUALIZATION_PARAMS,
}


def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration dictionary.
    
    Returns:
        Dict[str, Any]: The default configuration dictionary.
    """
    return DEFAULT_CONFIG.copy()


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the configuration dictionary with the provided updates.
    
    Args:
        config (Dict[str, Any]): The original configuration dictionary.
        updates (Dict[str, Any]): The updates to apply.
        
    Returns:
        Dict[str, Any]: The updated configuration dictionary.
    """
    # Create a deep copy of the config
    result = config.copy()
    
    # Apply updates recursively
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = update_config(result[key], value)
        else:
            result[key] = value
            
    return result