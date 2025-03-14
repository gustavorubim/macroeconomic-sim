"""
Debug script to identify which equation is causing complex number issues
in the steady state calculations.
"""

import sympy as sp
import numpy as np
import sys

# Add the project root to Python path to import modules
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dsge.core import SmetsWoutersModel
from config.config_manager import ConfigManager


def debug_equations():
    """Debug steady state equations to find which one produces complex numbers."""
    print("Initializing model...")
    config = ConfigManager()
    model = SmetsWoutersModel(config)
    
    print("Creating symbolic variables...")
    # Define symbolic variables (copied from PerturbationSolver.create_symbolic_model)
    # State variables (predetermined)
    k = sp.Symbol('k')  # Capital
    a = sp.Symbol('a')  # Technology shock
    b = sp.Symbol('b')  # Preference shock
    i_shock = sp.Symbol('i_shock')  # Investment shock
    g = sp.Symbol('g')  # Government spending shock
    r_shock = sp.Symbol('r_shock')  # Monetary policy shock
    p_shock = sp.Symbol('p_shock')  # Price markup shock
    w_shock = sp.Symbol('w_shock')  # Wage markup shock
    
    # Control variables (non-predetermined)
    y = sp.Symbol('y')  # Output
    c = sp.Symbol('c')  # Consumption
    i = sp.Symbol('i')  # Investment
    l = sp.Symbol('l')  # Labor
    w = sp.Symbol('w')  # Real wage
    r_k = sp.Symbol('r_k')  # Rental rate of capital
    pi = sp.Symbol('pi')  # Inflation
    r = sp.Symbol('r')  # Nominal interest rate
    
    print("Extracting parameters...")
    # Define parameters (copied from PerturbationSolver init)
    params = model.params
    alpha = params["alpha"]
    beta = params["beta"]
    delta = params["delta"]
    sigma_c = params["sigma_c"]
    h = params["h"]
    sigma_l = params["sigma_l"]
    xi_p = params["xi_p"]
    xi_w = params["xi_w"]
    iota_p = params["iota_p"]
    iota_w = params["iota_w"]
    rho_r = params["rho_r"]
    phi_pi = params["phi_pi"]
    phi_y = params["phi_y"]
    phi_dy = params["phi_dy"]
    pi_bar = params["pi_bar"]
    r_bar = params["r_bar"]
    
    # Shock persistence parameters
    rho_a = params["technology_rho"]
    rho_b = params["preference_rho"]
    rho_i = params["investment_rho"]
    rho_g = params["government_rho"]
    rho_r = params["monetary_rho"]
    rho_p = params["price_markup_rho"]
    rho_w = params["wage_markup_rho"]
    
    print("Setting up equations...")
    # Define model equations (copied from PerturbationSolver.create_symbolic_model)
    # Production function
    eq1 = y - a * k**alpha * l**(1-alpha)
    
    # Capital accumulation
    eq2 = k - (1 - delta) * k - i
    
    # Resource constraint
    eq3 = y - c - i - g
    
    # Consumption Euler equation
    eq4 = c**(-sigma_c) - beta * (1 + r) / pi * c**(-sigma_c)
    
    # Labor supply
    eq5 = w - l**sigma_l * c**sigma_c
    
    # Capital demand
    eq6 = r_k - alpha * a * k**(alpha-1) * l**(1-alpha)
    
    # Labor demand
    eq7 = w - (1 - alpha) * a * k**alpha * l**(-alpha)
    
    # Phillips curve
    eq8 = pi - (1 - xi_p) * (w / ((1 - alpha) * a * k**alpha * l**(-alpha))) - xi_p * pi
    
    # Monetary policy rule
    eq9 = r - rho_r * r - (1 - rho_r) * (r_bar + phi_pi * (pi - pi_bar) + phi_y * (y - y) + phi_dy * (y - y)) - r_shock
    
    # Shock processes
    eq10 = a - rho_a * a
    eq11 = b - rho_b * b
    eq12 = i_shock - rho_i * i_shock
    eq13 = g - rho_g * g
    eq14 = r_shock - rho_r * r_shock
    eq15 = p_shock - rho_p * p_shock
    eq16 = w_shock - rho_w * w_shock
    
    # Collect equations
    equations = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16]
    equation_names = ["Production function", "Capital accumulation", "Resource constraint", 
                    "Consumption Euler", "Labor supply", "Capital demand", "Labor demand", 
                    "Phillips curve", "Monetary policy", "Technology shock", "Preference shock", 
                    "Investment shock", "Government shock", "Monetary shock", 
                    "Price markup shock", "Wage markup shock"]
    
    # Initialize steady state values from model
    print("Computing model steady state...")
    model.compute_steady_state()
    ss = model.steady_state
    
    print("Setting up steady state values...")
    # Set up steady state values as dictionary for substitution
    ss_values = {
        k: ss["capital"],
        y: ss["output"],
        c: ss["consumption"],
        i: ss["investment"],
        l: ss["labor"],
        w: ss["real_wage"],
        r_k: ss["rental_rate"],
        pi: ss["inflation"],
        r: ss["nominal_interest"]
    }
    
    # Critical fix: Set technology shock to 1.0 instead of 0
    # This prevents division by zero in equations
    ss_values[a] = 1.0  # Set technology to 1 in steady state
    
    # Set other shocks to zero
    ss_values[b] = 0.0
    ss_values[i_shock] = 0.0
    ss_values[g] = 0.0
    ss_values[r_shock] = 0.0
    ss_values[p_shock] = 0.0
    ss_values[w_shock] = 0.0
    
    print("\nEvaluating each equation at steady state...")
    print("-" * 50)
    
    # Evaluate each equation
    for i, (eq, name) in enumerate(zip(equations, equation_names)):
        print(f"Equation {i+1}: {name}")
        try:
            result = eq.subs(ss_values)
            float_result = float(result)
            print(f"  Result: {float_result}")
            if abs(float_result) > 1e-6:
                print(f"  WARNING: Equation not satisfied at steady state. Residual = {float_result}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Equation form: {eq}")
            print(f"  Substituted form: {eq.subs(ss_values)}")
        print()
    
    print("Debugging complete.")


if __name__ == "__main__":
    debug_equations()