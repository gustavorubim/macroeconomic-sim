#!/usr/bin/env python
"""
Simple example of using the DSGE model with passing the fixed solver.

This script demonstrates how to create a model, solve it, and generate impulse responses
using the fixed perturbation solver and correctly handling the state dimensions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel
# Import the fixed perturbation solver instead of the original
from dsge.solution.perturbation_fixed import PerturbationSolver
from dsge.analysis import ImpulseResponseFunctions


def main():
    """Main function."""
    # Create configuration
    config = ConfigManager()
    
    # Create model
    model = SmetsWoutersModel(config)
    
    # Solve model using perturbation method with our fixed solver
    # Use first-order perturbation for simplicity
    solver = PerturbationSolver(model, order=1)
    print("Solving model with fixed perturbation solver...")
    solver.solve()
    print("Model solved successfully!")
    
    # Create impulse response functions and pass our already solved model and solver
    irf = ImpulseResponseFunctions(model, config)
    
    # CRITICAL FIX: Directly assign our fixed solver instance to irf.solver
    # This prevents ImpulseResponseFunctions from creating a new instance
    # of the original unfixed PerturbationSolver
    irf.solver = solver
    print("Using fixed solver for impulse response calculations...")
    
    # MANUALLY compute impulse responses for each shock
    print("Computing impulse responses manually...")
    
    # Define the shock names we want to analyze
    shock_names = ["technology", "monetary", "government"]
    
    # Create storage for IRFs
    irfs = {}
    
    # Period length
    periods = 40
    
    # For each shock, do a separate simulation
    for shock_name in shock_names:
        print(f"Computing IRF for {shock_name} shock...")
        
        # Get shock standard deviation
        shock_std = model.params.get(f"{shock_name}_sigma", 0.01)
        
        # Create shock vector with the correct dimensions (8 states)
        shocks = np.zeros((periods, 8))  # 8 is the correct number of states in our model
        
        # Set the appropriate shock index based on our solver's state order
        # In the solver, states are ordered as: [k, a, b, i_shock, g, r_shock, p_shock, w_shock]
        shock_indices = {
            "technology": 1,      # a
            "preference": 2,      # b
            "investment": 3,      # i_shock
            "government": 4,      # g
            "monetary": 5,        # r_shock
            "price_markup": 6,    # p_shock
            "wage_markup": 7      # w_shock
        }
        
        # Apply the shock at time 0
        shock_idx = shock_indices[shock_name]
        shocks[0, shock_idx] = 1.0 * shock_std  # shock_size = 1.0
        
        # Simulate model with zero initial states and the shock
        states_sim, controls_sim = solver.simulate(
            periods=periods,
            initial_states=np.zeros(8),  # Correct dimension: 8 states
            shocks=shocks
        )
        
        # Store IRFs in the same format ImpulseResponseFunctions expects
        irfs[shock_name] = {
            # States
            "capital": states_sim[:, 0],
            
            # Shocks
            "technology_shock": states_sim[:, 1],
            "preference_shock": states_sim[:, 2],
            "investment_shock": states_sim[:, 3],
            "government_shock": states_sim[:, 4],
            "monetary_shock": states_sim[:, 5],
            "price_markup_shock": states_sim[:, 6],
            "wage_markup_shock": states_sim[:, 7],
            
            # Controls
            "output": controls_sim[:, 0],
            "consumption": controls_sim[:, 1],
            "investment": controls_sim[:, 2],
            "labor": controls_sim[:, 3],
            "real_wage": controls_sim[:, 4],
            "rental_rate": controls_sim[:, 5],
            "inflation": controls_sim[:, 6],
            "nominal_interest": controls_sim[:, 7],
        }
    
    print("Impulse responses computed successfully!")
    
    # Store the IRFs in the irf object for plotting
    irf.irfs = irfs
    
    # Plot impulse responses
    print("Generating impulse response plots...")
    fig = irf.plot_irfs(
        shock_names=shock_names,
        variable_names=["output", "consumption", "investment", "inflation", "nominal_interest"],
        figsize=(12, 10),
        title="Impulse Response Functions"
    )
    
    # Create output directory
    os.makedirs("results/example", exist_ok=True)
    
    # Save figure
    fig.savefig("results/example/impulse_responses.png")
    plt.close(fig)
    
    print("Example completed. Results saved to results/example/")


if __name__ == "__main__":
    main()