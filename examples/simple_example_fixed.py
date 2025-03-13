#!/usr/bin/env python
"""
Simple example of using the DSGE model.

This script demonstrates how to create a model, solve it, and generate impulse responses.
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
    
    # Create impulse response functions
    irf = ImpulseResponseFunctions(model, config)
    
    # Compute impulse responses
    print("Computing impulse responses...")
    irfs = irf.compute_irfs(
        shock_names=["technology", "monetary", "government"],
        periods=40,
        shock_size=1.0
    )
    print("Impulse responses computed successfully!")
    
    # Plot impulse responses
    print("Generating impulse response plots...")
    fig = irf.plot_irfs(
        shock_names=["technology", "monetary", "government"],
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