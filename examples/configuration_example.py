#!/usr/bin/env python
"""
Configuration Management Example

This script demonstrates various ways to work with the DSGE model configuration system:
1. Creating and modifying configurations
2. Loading and saving configuration files
3. Parameter sensitivity analysis
4. Configuration version control
5. Programmatically generating configurations

The example shows best practices for managing model configurations for reproducible research.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

# In a real implementation, these would be imported from the package
# For demonstration purposes, we'll create a simplified version
class ConfigManager:
    """Class for managing DSGE model configurations."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to a JSON configuration file.
                If provided, the configuration will be loaded from this file.
                If not provided, the default configuration will be used.
        """
        # Start with the default configuration
        self.config = self.get_default_config()
        
        # If a config path is provided, load and apply it
        if config_path is not None:
            self.load_config(config_path)
        
        print(f"Initialized configuration manager{' with ' + config_path if config_path else ' with default config'}")
    
    def get_default_config(self):
        """
        Get the default configuration.
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            "base_model": {
                "beta": 0.99,
                "alpha": 0.33,
                "delta": 0.025,
                "sigma_c": 1.5,
                "h": 0.7,
                "sigma_l": 2.0,
                "xi_p": 0.75,
                "xi_w": 0.75,
                "iota_p": 0.5,
                "iota_w": 0.5,
                "rho_r": 0.8,
                "phi_pi": 1.5,
                "phi_y": 0.125,
                "phi_dy": 0.125,
                "pi_bar": 1.005,
                "r_bar": 1.0101
            },
            "financial_extension": {
                "enabled": False,
                "mu": 0.12,
                "kappa": 0.05,
                "theta": 0.972
            },
            "open_economy_extension": {
                "enabled": False,
                "alpha_h": 0.7,
                "eta": 1.5,
                "omega": 0.5
            },
            "fiscal_extension": {
                "enabled": False,
                "g_y": 0.2,
                "b_y": 0.6,
                "rho_g": 0.9,
                "tax_rule": "debt_stabilizing"
            },
            "shocks": {
                "technology": {
                    "rho": 0.95,
                    "sigma": 0.01
                },
                "preference": {
                    "rho": 0.9,
                    "sigma": 0.02
                },
                "investment": {
                    "rho": 0.85,
                    "sigma": 0.01
                },
                "government": {
                    "rho": 0.8,
                    "sigma": 0.01
                },
                "monetary": {
                    "rho": 0.5,
                    "sigma": 0.005
                },
                "price_markup": {
                    "rho": 0.7,
                    "sigma": 0.01
                },
                "wage_markup": {
                    "rho": 0.7,
                    "sigma": 0.01
                }
            },
            "solution": {
                "method": "perturbation",
                "perturbation_order": 1,
                "projection_method": "chebyshev",
                "projection_nodes": 5
            },
            "data": {
                "start_date": "1966-01-01",
                "end_date": "2019-12-31",
                "variables": {
                    "gdp": {
                        "source": "FRED",
                        "series_id": "GDPC1",
                        "transformation": "log_difference"
                    },
                    "inflation": {
                        "source": "FRED",
                        "series_id": "PCEPILFE",
                        "transformation": "log_difference"
                    },
                    "interest_rate": {
                        "source": "FRED",
                        "series_id": "FEDFUNDS",
                        "transformation": "divide_by_4"
                    }
                }
            },
            "estimation": {
                "method": "bayesian",
                "mcmc_algorithm": "metropolis_hastings",
                "num_chains": 4,
                "num_draws": 10000,
                "burn_in": 5000,
                "tune": 2000,
                "target_acceptance": 0.25
            }
        }
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to a JSON configuration file.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Update the current configuration with the loaded one
        self.config = self.update_config(self.config, loaded_config)
        print(f"Loaded configuration from {config_path}")
    
    def save_config(self, config_path):
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path (str): Path where to save the configuration.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Saved configuration to {config_path}")
    
    def update_config(self, base_config, updates):
        """
        Update a configuration dictionary with another one.
        
        Args:
            base_config (dict): Base configuration to update
            updates (dict): Updates to apply
            
        Returns:
            dict: Updated configuration
        """
        result = base_config.copy()
        
        for key, value in updates.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively update nested dictionaries
                result[key] = self.update_config(result[key], value)
            else:
                # Replace or add values
                result[key] = value
        
        return result
    
    def update(self, updates):
        """
        Update the configuration with the provided updates.
        
        Args:
            updates (dict): The updates to apply to the configuration.
        """
        self.config = self.update_config(self.config, updates)
        print("Updated configuration")
    
    def get(self, key=None, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str, optional): The key to get. If None, the entire configuration is returned.
            default (any, optional): The default value to return if the key is not found.
        
        Returns:
            any: The configuration value.
        """
        if key is None:
            return self.config
        
        # Handle nested keys with dot notation (e.g., "base_model.beta")
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): The key to set.
            value (any): The value to set.
        """
        # Handle nested keys with dot notation (e.g., "base_model.beta")
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
        
        print(f"Set {key} = {value}")
    
    def enable_extension(self, extension_name):
        """
        Enable a model extension.
        
        Args:
            extension_name (str): The name of the extension to enable.
                Must be one of: "financial_extension", "open_economy_extension", "fiscal_extension".
        
        Raises:
            ValueError: If the extension name is not valid.
        """
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}. "
                            f"Must be one of: {', '.join(valid_extensions)}")
        
        self.set(f"{extension_name}.enabled", True)
        print(f"Enabled {extension_name}")
    
    def disable_extension(self, extension_name):
        """
        Disable a model extension.
        
        Args:
            extension_name (str): The name of the extension to disable.
                Must be one of: "financial_extension", "open_economy_extension", "fiscal_extension".
        
        Raises:
            ValueError: If the extension name is not valid.
        """
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}. "
                            f"Must be one of: {', '.join(valid_extensions)}")
        
        self.set(f"{extension_name}.enabled", False)
        print(f"Disabled {extension_name}")
    
    def set_solution_method(self, method, **kwargs):
        """
        Set the solution method.
        
        Args:
            method (str): The solution method to use.
                Must be one of: "perturbation", "projection".
            **kwargs: Additional parameters for the solution method.
                For perturbation: perturbation_order (int): The order of perturbation (1, 2, or 3).
                For projection: projection_method (str): The projection method ("chebyshev" or "finite_elements").
                               projection_nodes (int): The number of nodes per dimension.
        
        Raises:
            ValueError: If the method is not valid.
        """
        valid_methods = ["perturbation", "projection"]
        if method not in valid_methods:
            raise ValueError(f"Invalid solution method: {method}. "
                            f"Must be one of: {', '.join(valid_methods)}")
        
        updates = {"solution": {"method": method}}
        
        if method == "perturbation" and "perturbation_order" in kwargs:
            order = kwargs["perturbation_order"]
            if order not in [1, 2, 3]:
                raise ValueError(f"Invalid perturbation order: {order}. Must be 1, 2, or 3.")
            updates["solution"]["perturbation_order"] = order
        
        elif method == "projection":
            if "projection_method" in kwargs:
                proj_method = kwargs["projection_method"]
                valid_proj_methods = ["chebyshev", "finite_elements"]
                if proj_method not in valid_proj_methods:
                    raise ValueError(f"Invalid projection method: {proj_method}. "
                                    f"Must be one of: {', '.join(valid_proj_methods)}")
                updates["solution"]["projection_method"] = proj_method
            
            if "projection_nodes" in kwargs:
                nodes = kwargs["projection_nodes"]
                if not isinstance(nodes, int) or nodes < 2:
                    raise ValueError(f"Invalid number of nodes: {nodes}. Must be an integer >= 2.")
                updates["solution"]["projection_nodes"] = nodes
        
        self.update(updates)
        print(f"Set solution method to {method} with parameters {kwargs}")
    
    def set_data_range(self, start_date, end_date):
        """
        Set the date range for data.
        
        Args:
            start_date (str): The start date in ISO format (YYYY-MM-DD).
            end_date (str): The end date in ISO format (YYYY-MM-DD).
        """
        self.update({
            "data": {
                "start_date": start_date,
                "end_date": end_date
            }
        })
        print(f"Set data range to {start_date} - {end_date}")
    
    def set_estimation_params(self, **kwargs):
        """
        Set estimation parameters.
        
        Args:
            **kwargs: Estimation parameters to set.
                method (str): The estimation method ("bayesian" or "maximum_likelihood").
                mcmc_algorithm (str): The MCMC algorithm for Bayesian estimation.
                num_chains (int): The number of MCMC chains.
                num_draws (int): The number of draws per chain.
                burn_in (int): The number of burn-in draws.
                tune (int): The number of tuning iterations.
                target_acceptance (float): The target acceptance rate.
        """
        updates = {"estimation": {}}
        
        for key, value in kwargs.items():
            if key == "method":
                valid_methods = ["bayesian", "maximum_likelihood"]
                if value not in valid_methods:
                    raise ValueError(f"Invalid estimation method: {value}. "
                                    f"Must be one of: {', '.join(valid_methods)}")
            
            updates["estimation"][key] = value
        
        self.update(updates)
        print(f"Set estimation parameters: {kwargs}")


def create_parameter_sweep(base_config, parameter, values):
    """
    Create a set of configurations by sweeping a parameter through a range of values.
    
    Args:
        base_config (ConfigManager): Base configuration
        parameter (str): Parameter to sweep (dot notation)
        values (list): List of values to use
        
    Returns:
        list: List of (value, config) tuples
    """
    configs = []
    
    for value in values:
        # Create a new config based on the base config
        config = ConfigManager()
        config.update(base_config.get())
        
        # Set the parameter value
        config.set(parameter, value)
        
        configs.append((value, config))
    
    return configs


def create_grid_search(base_config, parameters):
    """
    Create a grid search of configurations by varying multiple parameters.
    
    Args:
        base_config (ConfigManager): Base configuration
        parameters (dict): Dictionary mapping parameters to lists of values
        
    Returns:
        list: List of (param_values, config) tuples
    """
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    configs = []
    
    # Generate all combinations of parameter values
    for values in product(*param_values):
        # Create a new config based on the base config
        config = ConfigManager()
        config.update(base_config.get())
        
        # Set parameter values
        param_dict = {}
        for name, value in zip(param_names, values):
            config.set(name, value)
            param_dict[name] = value
        
        configs.append((param_dict, config))
    
    return configs


def main():
    """Main function demonstrating configuration management."""
    # Create output directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("results/config", exist_ok=True)
    
    print("=== DSGE Model Configuration Management Example ===")
    
    print("\n1. Basic Configuration Operations")
    # Create default configuration
    config = ConfigManager()
    
    # Save default configuration
    config.save_config("config/default_config.json")
    
    # Modify some parameters
    config.set("base_model.beta", 0.98)
    config.set("base_model.sigma_c", 2.0)
    config.set("shocks.technology.sigma", 0.02)
    
    # Enable an extension
    config.enable_extension("financial_extension")
    
    # Set solution method
    config.set_solution_method("perturbation", perturbation_order=2)
    
    # Save modified configuration
    config.save_config("config/modified_config.json")
    
    # Load a configuration
    loaded_config = ConfigManager("config/default_config.json")
    
    print("\n2. Configuration Comparison")
    # Compare configurations
    def compare_configs(config1, config2, name1="Config 1", name2="Config 2"):
        """Compare two configurations and print differences."""
        print(f"\nComparing {name1} vs {name2}:")
        
        def find_differences(dict1, dict2, path=""):
            differences = []
            
            # Check keys in dict1
            for key in dict1:
                if key not in dict2:
                    differences.append((f"{path}.{key}" if path else key, dict1[key], "Not present"))
                elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    differences.extend(find_differences(dict1[key], dict2[key], f"{path}.{key}" if path else key))
                elif dict1[key] != dict2[key]:
                    differences.append((f"{path}.{key}" if path else key, dict1[key], dict2[key]))
            
            # Check keys in dict2 that are not in dict1
            for key in dict2:
                if key not in dict1:
                    differences.append((f"{path}.{key}" if path else key, "Not present", dict2[key]))
            
            return differences
        
        differences = find_differences(config1.get(), config2.get())
        
        if differences:
            print(f"Found {len(differences)} differences:")
            for key, val1, val2 in differences:
                print(f"  {key}: {val1} vs {val2}")
        else:
            print("No differences found.")
    
    compare_configs(config, loaded_config, "Modified Config", "Default Config")
    
    print("\n3. Parameter Sensitivity Analysis")
    # Create base configuration for sensitivity analysis
    base_config = ConfigManager()
    
    # Parameter sweep for beta (discount factor)
    beta_values = [0.95, 0.96, 0.97, 0.98, 0.99]
    beta_configs = create_parameter_sweep(base_config, "base_model.beta", beta_values)
    
    # Parameter sweep for phi_pi (Taylor rule inflation response)
    phi_pi_values = [1.1, 1.3, 1.5, 1.7, 1.9]
    phi_pi_configs = create_parameter_sweep(base_config, "base_model.phi_pi", phi_pi_values)
    
    # Save all configurations
    for i, (value, config) in enumerate(beta_configs):
        config.save_config(f"config/beta_{value:.2f}.json")
    
    for i, (value, config) in enumerate(phi_pi_configs):
        config.save_config(f"config/phi_pi_{value:.1f}.json")
    
    # Visualize parameter sensitivity
    plt.figure(figsize=(12, 5))
    
    # Plot beta sensitivity (hypothetical IRF values)
    plt.subplot(1, 2, 1)
    for value, _ in beta_configs:
        # In a real implementation, this would compute IRFs using the config
        # For demonstration, we'll use synthetic data
        periods = 20
        irf_values = np.exp(-np.arange(periods) / (10 * value))
        plt.plot(irf_values, label=f"β = {value:.2f}")
    
    plt.title("Output Response to Monetary Shock\nVarying Discount Factor (β)")
    plt.xlabel("Periods")
    plt.ylabel("Deviation from Steady State")
    plt.legend()
    plt.grid(True)
    
    # Plot phi_pi sensitivity (hypothetical IRF values)
    plt.subplot(1, 2, 2)
    for value, _ in phi_pi_configs:
        # In a real implementation, this would compute IRFs using the config
        # For demonstration, we'll use synthetic data
        periods = 20
        irf_values = np.exp(-np.arange(periods) * value / 15)
        plt.plot(irf_values, label=f"φ_π = {value:.1f}")
    
    plt.title("Output Response to Monetary Shock\nVarying Inflation Response (φ_π)")
    plt.xlabel("Periods")
    plt.ylabel("Deviation from Steady State")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/config/parameter_sensitivity.png")
    print(f"Saved parameter sensitivity plot to results/config/parameter_sensitivity.png")
    
    print("\n4. Grid Search for Optimal Parameters")
    # Define parameter grid
    param_grid = {
        "base_model.phi_pi": [1.3, 1.5, 1.7],
        "base_model.phi_y": [0.1, 0.125, 0.15],
        "base_model.rho_r": [0.7, 0.8, 0.9]
    }
    
    # Create grid search
    grid_configs = create_grid_search(base_config, param_grid)
    print(f"Created {len(grid_configs)} configurations for grid search")
    
    # In a real implementation, this would evaluate each configuration
    # For demonstration, we'll use synthetic performance metrics
    results = []
    for params, config in grid_configs:
        # Save configuration
        param_str = "_".join([f"{k.split('.')[-1]}_{v}" for k, v in params.items()])
        config.save_config(f"config/grid_{param_str}.json")
        
        # Compute synthetic performance metric
        # In a real implementation, this would be a model fit measure
        phi_pi = params["base_model.phi_pi"]
        phi_y = params["base_model.phi_y"]
        rho_r = params["base_model.rho_r"]
        
        # Synthetic metric: higher is better
        metric = (phi_pi - 1.5)**2 + (phi_y - 0.125)**2 + (rho_r - 0.8)**2
        metric = 1.0 / (1.0 + metric)
        
        results.append((params, metric))
    
    # Sort results by metric (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 5 configurations
    print("\nTop 5 configurations:")
    for i, (params, metric) in enumerate(results[:5]):
        print(f"{i+1}. Metric: {metric:.4f}, Parameters: {params}")
    
    # Create optimal configuration
    optimal_params = results[0][0]
    optimal_config = ConfigManager()
    for key, value in optimal_params.items():
        optimal_config.set(key, value)
    
    # Save optimal configuration
    optimal_config.save_config("config/optimal_config.json")
    print(f"Saved optimal configuration to config/optimal_config.json")
    
    print("\n5. Configuration Version Control")
    # Create a configuration history
    history = []
    
    # Initial configuration
    initial_config = ConfigManager()
    history.append({
        "version": "1.0.0",
        "date": "2025-01-01",
        "description": "Initial configuration",
        "config": initial_config.get()
    })
    
    # First update: change discount factor
    update1_config = ConfigManager()
    update1_config.update(initial_config.get())
    update1_config.set("base_model.beta", 0.98)
    history.append({
        "version": "1.1.0",
        "date": "2025-02-15",
        "description": "Updated discount factor",
        "config": update1_config.get()
    })
    
    # Second update: enable financial extension
    update2_config = ConfigManager()
    update2_config.update(update1_config.get())
    update2_config.enable_extension("financial_extension")
    history.append({
        "version": "2.0.0",
        "date": "2025-03-10",
        "description": "Enabled financial extension",
        "config": update2_config.get()
    })
    
    # Third update: change solution method
    update3_config = ConfigManager()
    update3_config.update(update2_config.get())
    update3_config.set_solution_method("perturbation", perturbation_order=2)
    history.append({
        "version": "2.1.0",
        "date": "2025-04-05",
        "description": "Changed to second-order perturbation",
        "config": update3_config.get()
    })
    
    # Save configuration history
    with open("config/config_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved configuration history to config/config_history.json")
    print(f"Configuration history contains {len(history)} versions")
    
    # Print configuration history
    print("\nConfiguration version history:")
    for entry in history:
        print(f"Version {entry['version']} ({entry['date']}): {entry['description']}")
    
    print("\nConfiguration management example completed successfully.")


if __name__ == "__main__":
    main()