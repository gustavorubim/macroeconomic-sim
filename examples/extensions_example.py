#!/usr/bin/env python
"""
Example of using the DSGE model with extensions.

This script demonstrates how to create a model with extensions,
solve it, and compare the results with the base model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver
from dsge.analysis import ImpulseResponseFunctions


def main():
    """Main function."""
    # Create configuration
    config = ConfigManager()
    
    # Create base model
    base_model = SmetsWoutersModel(config)
    
    # Create model with financial frictions
    financial_config = ConfigManager()
    financial_config.enable_extension("financial_extension")
    financial_model = SmetsWoutersModel(financial_config)
    
    # Create model with open economy features
    open_economy_config = ConfigManager()
    open_economy_config.enable_extension("open_economy_extension")
    open_economy_model = SmetsWoutersModel(open_economy_config)
    
    # Create model with fiscal policy extension
    fiscal_config = ConfigManager()
    fiscal_config.enable_extension("fiscal_extension")
    fiscal_model = SmetsWoutersModel(fiscal_config)
    
    # Create model with all extensions
    all_extensions_config = ConfigManager()
    all_extensions_config.enable_extension("financial_extension")
    all_extensions_config.enable_extension("open_economy_extension")
    all_extensions_config.enable_extension("fiscal_extension")
    all_extensions_model = SmetsWoutersModel(all_extensions_config)
    
    # Solve models
    models = {
        "Base": base_model,
        "Financial": financial_model,
        "Open Economy": open_economy_model,
        "Fiscal": fiscal_model,
        "All Extensions": all_extensions_model
    }
    
    solvers = {}
    for name, model in models.items():
        solver = PerturbationSolver(model, order=1)
        solver.solve()
        solvers[name] = solver
    
    # Create output directory
    os.makedirs("results/extensions", exist_ok=True)
    
    # Compare impulse responses to monetary policy shock
    plt.figure(figsize=(12, 10))
    
    variables = ["output", "consumption", "investment", "inflation", "nominal_interest"]
    periods = 40
    
    for i, var_name in enumerate(variables):
        plt.subplot(3, 2, i + 1)
        
        for name, model in models.items():
            # Create impulse response functions
            irf = ImpulseResponseFunctions(model, config)
            
            # Compute impulse responses
            irfs = irf.compute_irfs(
                shock_names=["monetary"],
                periods=periods,
                shock_size=1.0
            )
            
            # Plot impulse response
            plt.plot(irfs["monetary"][var_name], label=name)
        
        plt.title(var_name)
        plt.xlabel("Periods")
        plt.ylabel("Deviation from SS")
        plt.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/extensions/monetary_shock_comparison.png")
    plt.close()
    
    # Compare impulse responses to technology shock
    plt.figure(figsize=(12, 10))
    
    for i, var_name in enumerate(variables):
        plt.subplot(3, 2, i + 1)
        
        for name, model in models.items():
            # Create impulse response functions
            irf = ImpulseResponseFunctions(model, config)
            
            # Compute impulse responses
            irfs = irf.compute_irfs(
                shock_names=["technology"],
                periods=periods,
                shock_size=1.0
            )
            
            # Plot impulse response
            plt.plot(irfs["technology"][var_name], label=name)
        
        plt.title(var_name)
        plt.xlabel("Periods")
        plt.ylabel("Deviation from SS")
        plt.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/extensions/technology_shock_comparison.png")
    plt.close()
    
    # Compare impulse responses to fiscal shock
    plt.figure(figsize=(12, 10))
    
    for i, var_name in enumerate(variables):
        plt.subplot(3, 2, i + 1)
        
        for name, model in models.items():
            # Create impulse response functions
            irf = ImpulseResponseFunctions(model, config)
            
            # Compute impulse responses
            irfs = irf.compute_irfs(
                shock_names=["government"],
                periods=periods,
                shock_size=1.0
            )
            
            # Plot impulse response
            plt.plot(irfs["government"][var_name], label=name)
        
        plt.title(var_name)
        plt.xlabel("Periods")
        plt.ylabel("Deviation from SS")
        plt.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/extensions/fiscal_shock_comparison.png")
    plt.close()
    
    print("Example completed. Results saved to results/extensions/")


if __name__ == "__main__":
    main()