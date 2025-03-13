"""
Steady state solver for the DSGE model.

This module provides functions for computing the steady state of the
Smets and Wouters (2007) DSGE model and its extensions.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from scipy.optimize import root

from config.config_manager import ConfigManager


def compute_base_steady_state(config: ConfigManager) -> Dict[str, float]:
    """
    Compute the steady state of the base Smets and Wouters model.
    
    Args:
        config (ConfigManager): Model configuration.
        
    Returns:
        Dict[str, float]: Dictionary of steady state values.
    """
    # Extract parameters
    params = config.get("base_model")
    beta = params["beta"]
    delta = params["delta"]
    alpha = params["alpha"]
    g_y = params["g_y"]
    pi_bar = params["pi_bar"]
    
    # Compute steady state values
    r_k_ss = 1/beta - (1 - delta)  # Rental rate of capital
    w_ss = 1.0  # Normalized real wage
    l_ss = 1.0  # Normalized labor
    k_l_ratio = alpha / (1 - alpha) * w_ss / r_k_ss  # Capital-labor ratio
    k_ss = k_l_ratio * l_ss  # Capital stock
    y_ss = k_ss**alpha * l_ss**(1-alpha)  # Output
    i_ss = delta * k_ss  # Investment
    c_ss = y_ss - i_ss  # Consumption
    r_ss = 1/beta * pi_bar  # Nominal interest rate
    
    # Return steady state values
    return {
        "output": y_ss,
        "consumption": c_ss,
        "investment": i_ss,
        "capital": k_ss,
        "capital_utilization": 1.0,
        "labor": l_ss,
        "real_wage": w_ss,
        "rental_rate": r_k_ss,
        "inflation": pi_bar,
        "price_markup": 1.0,
        "wage_markup": 1.0,
        "nominal_interest": r_ss,
        "real_interest": r_ss / pi_bar,
    }


def compute_financial_extension_steady_state(
    base_ss: Dict[str, float], 
    config: ConfigManager
) -> Dict[str, float]:
    """
    Compute the steady state with financial frictions extension.
    
    Args:
        base_ss (Dict[str, float]): Base model steady state values.
        config (ConfigManager): Model configuration.
        
    Returns:
        Dict[str, float]: Dictionary of steady state values with financial extension.
    """
    # Check if financial extension is enabled
    if not config.get("financial_extension.enabled"):
        return base_ss
    
    # Extract parameters
    params = config.get("financial_extension")
    mu = params["mu"]  # Monitoring cost
    sigma_omega = params["sigma_omega"]  # Standard deviation of idiosyncratic shock
    gamma = params["gamma"]  # Survival rate of entrepreneurs
    lambda_b = params["lambda_b"]  # Bank leverage constraint
    
    # Extract base steady state values
    y_ss = base_ss["output"]
    k_ss = base_ss["capital"]
    r_k_ss = base_ss["rental_rate"]
    r_ss = base_ss["nominal_interest"]
    pi_bar = base_ss["inflation"]
    
    # Compute financial variables
    # External finance premium
    s_ss = 1.05  # Steady state external finance premium (5% above risk-free rate)
    
    # Entrepreneur's net worth
    n_ss = k_ss / (1 + (s_ss - 1) / lambda_b)
    
    # Bank assets
    b_ss = k_ss - n_ss
    
    # Loan rate
    r_l_ss = r_ss * s_ss
    
    # Default threshold
    omega_bar_ss = compute_default_threshold(r_k_ss, r_l_ss, sigma_omega)
    
    # Default rate
    default_rate_ss = compute_default_rate(omega_bar_ss, sigma_omega)
    
    # Update steady state values
    ss = base_ss.copy()
    ss.update({
        "external_finance_premium": s_ss,
        "entrepreneur_net_worth": n_ss,
        "bank_assets": b_ss,
        "loan_rate": r_l_ss,
        "default_threshold": omega_bar_ss,
        "default_rate": default_rate_ss,
    })
    
    return ss


def compute_open_economy_extension_steady_state(
    base_ss: Dict[str, float], 
    config: ConfigManager
) -> Dict[str, float]:
    """
    Compute the steady state with open economy extension.
    
    Args:
        base_ss (Dict[str, float]): Base model steady state values.
        config (ConfigManager): Model configuration.
        
    Returns:
        Dict[str, float]: Dictionary of steady state values with open economy extension.
    """
    # Check if open economy extension is enabled
    if not config.get("open_economy_extension.enabled"):
        return base_ss
    
    # Extract parameters
    params = config.get("open_economy_extension")
    alpha_h = params["alpha_h"]  # Home bias in consumption
    eta = params["eta"]  # Elasticity of substitution between home and foreign goods
    phi_b = params["phi_b"]  # Debt elasticity of interest rate
    
    # Extract base steady state values
    y_ss = base_ss["output"]
    c_ss = base_ss["consumption"]
    i_ss = base_ss["investment"]
    r_ss = base_ss["nominal_interest"]
    
    # Compute open economy variables
    # Real exchange rate (normalized to 1 in steady state)
    q_ss = 1.0
    
    # Net foreign assets (balanced trade in steady state)
    nfa_ss = 0.0
    
    # Exports and imports (balanced trade in steady state)
    x_ss = (1 - alpha_h) * y_ss
    m_ss = (1 - alpha_h) * (c_ss + i_ss)
    
    # Terms of trade
    tot_ss = 1.0
    
    # Update steady state values
    ss = base_ss.copy()
    ss.update({
        "real_exchange_rate": q_ss,
        "net_foreign_assets": nfa_ss,
        "exports": x_ss,
        "imports": m_ss,
        "terms_of_trade": tot_ss,
    })
    
    return ss


def compute_fiscal_extension_steady_state(
    base_ss: Dict[str, float], 
    config: ConfigManager
) -> Dict[str, float]:
    """
    Compute the steady state with fiscal policy extension.
    
    Args:
        base_ss (Dict[str, float]): Base model steady state values.
        config (ConfigManager): Model configuration.
        
    Returns:
        Dict[str, float]: Dictionary of steady state values with fiscal extension.
    """
    # Check if fiscal extension is enabled
    if not config.get("fiscal_extension.enabled"):
        return base_ss
    
    # Extract parameters
    params = config.get("fiscal_extension")
    g_y_ratio = params["g_y_ratio"]  # Government spending to GDP ratio
    b_y_ratio = params["b_y_ratio"]  # Government debt to GDP ratio
    tau_c = params["tau_c"]  # Consumption tax rate
    tau_k = params["tau_k"]  # Capital tax rate
    tau_l = params["tau_l"]  # Labor tax rate
    
    # Extract base steady state values
    y_ss = base_ss["output"]
    c_ss = base_ss["consumption"]
    i_ss = base_ss["investment"]
    k_ss = base_ss["capital"]
    l_ss = base_ss["labor"]
    w_ss = base_ss["real_wage"]
    r_k_ss = base_ss["rental_rate"]
    r_ss = base_ss["nominal_interest"]
    pi_bar = base_ss["inflation"]
    
    # Compute fiscal variables
    # Government spending
    g_ss = g_y_ratio * y_ss
    
    # Government debt
    b_g_ss = b_y_ratio * y_ss
    
    # Tax revenues
    tax_c_ss = tau_c * c_ss  # Consumption tax revenue
    tax_k_ss = tau_k * r_k_ss * k_ss  # Capital tax revenue
    tax_l_ss = tau_l * w_ss * l_ss  # Labor tax revenue
    tax_ss = tax_c_ss + tax_k_ss + tax_l_ss  # Total tax revenue
    
    # Government budget constraint
    # In steady state: g + r*b = tax + (b - b(-1))
    # With constant debt: g + r*b = tax
    # So: tax = g + (r-1)*b
    transfer_ss = tax_ss - g_ss - (r_ss/pi_bar - 1) * b_g_ss
    
    # Update steady state values
    ss = base_ss.copy()
    ss.update({
        "government_spending": g_ss,
        "government_debt": b_g_ss,
        "consumption_tax": tax_c_ss,
        "capital_tax": tax_k_ss,
        "labor_tax": tax_l_ss,
        "total_tax": tax_ss,
        "transfers": transfer_ss,
    })
    
    # Adjust consumption for taxes
    ss["consumption"] = c_ss / (1 + tau_c)
    
    # Adjust output for government spending
    ss["output"] = y_ss + g_ss
    
    return ss


def compute_default_threshold(r_k: float, r_l: float, sigma_omega: float) -> float:
    """
    Compute the default threshold for the financial frictions extension.
    
    Args:
        r_k (float): Rental rate of capital.
        r_l (float): Loan rate.
        sigma_omega (float): Standard deviation of idiosyncratic shock.
        
    Returns:
        float: Default threshold.
    """
    # This is a simplified version. In a full implementation, this would
    # solve for the default threshold that satisfies the lender's zero-profit condition.
    return r_l / r_k


def compute_default_rate(omega_bar: float, sigma_omega: float) -> float:
    """
    Compute the default rate for the financial frictions extension.
    
    Args:
        omega_bar (float): Default threshold.
        sigma_omega (float): Standard deviation of idiosyncratic shock.
        
    Returns:
        float: Default rate.
    """
    # This is a simplified version. In a full implementation, this would
    # compute the cumulative distribution function of the log-normal distribution
    # at the default threshold.
    from scipy.stats import norm
    return norm.cdf((np.log(omega_bar) + 0.5 * sigma_omega**2) / sigma_omega)


def compute_steady_state(config: ConfigManager) -> Dict[str, float]:
    """
    Compute the steady state of the DSGE model with all enabled extensions.
    
    Args:
        config (ConfigManager): Model configuration.
        
    Returns:
        Dict[str, float]: Dictionary of steady state values.
    """
    # Compute base steady state
    ss = compute_base_steady_state(config)
    
    # Apply extensions if enabled
    if config.get("financial_extension.enabled"):
        ss = compute_financial_extension_steady_state(ss, config)
    
    if config.get("open_economy_extension.enabled"):
        ss = compute_open_economy_extension_steady_state(ss, config)
    
    if config.get("fiscal_extension.enabled"):
        ss = compute_fiscal_extension_steady_state(ss, config)
    
    return ss


def check_steady_state(ss: Dict[str, float], config: ConfigManager) -> bool:
    """
    Check if the steady state values satisfy the model equations.
    
    Args:
        ss (Dict[str, float]): Steady state values.
        config (ConfigManager): Model configuration.
        
    Returns:
        bool: True if the steady state is valid, False otherwise.
    """
    # Extract parameters
    params = config.get("base_model")
    beta = params["beta"]
    delta = params["delta"]
    alpha = params["alpha"]
    
    # Check Euler equation
    euler_eq = abs(1 - beta * (ss["rental_rate"] + 1 - delta))
    
    # Check production function
    prod_fn = abs(ss["output"] - ss["capital"]**alpha * ss["labor"]**(1-alpha))
    
    # Check capital accumulation
    cap_acc = abs(ss["investment"] - delta * ss["capital"])
    
    # Check resource constraint
    res_con = abs(ss["output"] - ss["consumption"] - ss["investment"])
    
    # Check if all equations are satisfied (with some tolerance)
    tol = 1e-6
    return (euler_eq < tol and prod_fn < tol and cap_acc < tol and res_con < tol)