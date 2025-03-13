#!/usr/bin/env python
"""
Sensitivity Analysis Example

This script demonstrates techniques for parameter sensitivity analysis:
1. One-at-a-time parameter sensitivity
2. Global sensitivity analysis
3. Morris method for screening
4. Sobol' indices for importance ranking
5. Visualizing parameter interactions

The example shows how to assess model robustness and parameter importance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import itertools
from scipy import stats
from SALib.sample import saltelli, morris
from SALib.analyze import sobol, morris as morris_analyze

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sensitivity.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dsge_sensitivity")


# Mock DSGE model classes for demonstration
class ConfigManager:
    """Mock configuration manager for DSGE model."""
    
    def __init__(self, config_path=None):
        """Initialize the configuration manager."""
        # Default configuration
        self.config = {
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
            "solution": {
                "method": "perturbation",
                "perturbation_order": 1
            }
        }
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration
            self._update_config(self.config, loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path):
        """Save configuration to a JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def _update_config(self, base_config, updates):
        """Update a configuration dictionary with another one."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                # Recursively update nested dictionaries
                self._update_config(base_config[key], value)
            else:
                # Replace or add values
                base_config[key] = value
    
    def get(self, key=None, default=None):
        """Get a configuration value."""
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
        """Set a configuration value."""
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


class SmetsWoutersModel:
    """Mock Smets-Wouters model for demonstration."""
    
    def __init__(self, config):
        """Initialize the model."""
        self.config = config
        self.params = {
            "beta": config.get("base_model.beta", 0.99),
            "alpha": config.get("base_model.alpha", 0.33),
            "delta": config.get("base_model.delta", 0.025),
            "sigma_c": config.get("base_model.sigma_c", 1.5),
            "h": config.get("base_model.h", 0.7),
            "sigma_l": config.get("base_model.sigma_l", 2.0),
            "xi_p": config.get("base_model.xi_p", 0.75),
            "xi_w": config.get("base_model.xi_w", 0.75),
            "iota_p": config.get("base_model.iota_p", 0.5),
            "iota_w": config.get("base_model.iota_w", 0.5),
            "rho_r": config.get("base_model.rho_r", 0.8),
            "phi_pi": config.get("base_model.phi_pi", 1.5),
            "phi_y": config.get("base_model.phi_y", 0.125),
            "phi_dy": config.get("base_model.phi_dy", 0.125),
            "pi_bar": config.get("base_model.pi_bar", 1.005),
            "r_bar": config.get("base_model.r_bar", 1.0101)
        }
        
        logger.info(f"Initialized model with parameters: {self.params}")
    
    def compute_impulse_response(self, shock_name, periods=40, shock_size=1.0):
        """Compute impulse response function."""
        logger.info(f"Computing impulse response for {shock_name} shock")
        
        # Different decay rates for different variables and shocks
        if shock_name == "technology":
            decay_rates = {
                "output": 20,
                "consumption": 25,
                "investment": 15,
                "inflation": 10,
                "nominal_interest": 8
            }
            signs = {
                "output": 1,
                "consumption": 1,
                "investment": 1,
                "inflation": -1,
                "nominal_interest": -0.5
            }
        elif shock_name == "monetary":
            decay_rates = {
                "output": 10,
                "consumption": 12,
                "investment": 8,
                "inflation": 15,
                "nominal_interest": 5
            }
            signs = {
                "output": -1,
                "consumption": -1,
                "investment": -1,
                "inflation": -1,
                "nominal_interest": 1
            }
        else:  # Generic shock
            decay_rates = {
                "output": 15,
                "consumption": 18,
                "investment": 12,
                "inflation": 10,
                "nominal_interest": 8
            }
            signs = {
                "output": 1,
                "consumption": 1,
                "investment": 1,
                "inflation": 1,
                "nominal_interest": 1
            }
        
        # Generate IRFs
        np.random.seed(42)  # For reproducibility
        irf = {}
        for var, decay in decay_rates.items():
            t = np.arange(periods)
            
            # Modify decay rates based on parameters to simulate parameter sensitivity
            # This is a simplified model of how parameters affect impulse responses
            modified_decay = decay
            
            # Discount factor (beta) affects persistence
            modified_decay *= (self.params["beta"] / 0.99) ** 2
            
            # Price stickiness (xi_p) affects inflation dynamics
            if var == "inflation" or var == "nominal_interest":
                modified_decay *= (self.params["xi_p"] / 0.75) ** 0.5
            
            # Habit formation (h) affects consumption dynamics
            if var == "consumption":
                modified_decay *= (self.params["h"] / 0.7) ** 0.5
            
            # Taylor rule parameters affect interest rate and inflation dynamics
            if var == "nominal_interest" or var == "inflation":
                modified_decay *= (self.params["phi_pi"] / 1.5) ** 0.3
            
            # Generate response
            response = signs[var] * shock_size * np.exp(-t / modified_decay)
            
            # Add some noise
            response += 0.02 * np.random.randn(periods)
            
            irf[var] = response
        
        logger.info(f"Impulse response computed for {shock_name} shock")
        
        return irf
    
    def compute_variance_decomposition(self):
        """Compute variance decomposition."""
        logger.info("Computing variance decomposition")
        
        # In a real implementation, this would compute the contribution of each shock
        # to the variance of each variable
        # For demonstration, we'll generate synthetic data
        
        # Shocks
        shocks = ["technology", "preference", "investment", "government", "monetary", "price_markup", "wage_markup"]
        
        # Variables
        variables = ["output", "consumption", "investment", "inflation", "nominal_interest"]
        
        # Generate random variance decomposition
        np.random.seed(42)
        decomposition = {}
        
        for var in variables:
            # Generate random contributions that sum to 1
            contributions = np.random.rand(len(shocks))
            contributions = contributions / np.sum(contributions)
            
            # Store as dictionary
            decomposition[var] = {shock: contrib for shock, contrib in zip(shocks, contributions)}
        
        logger.info("Variance decomposition computed")
        
        return decomposition
    
    def compute_model_moments(self):
        """Compute model moments."""
        logger.info("Computing model moments")
        
        # In a real implementation, this would compute the moments of the model variables
        # For demonstration, we'll generate synthetic data
        
        # Variables
        variables = ["output", "consumption", "investment", "inflation", "nominal_interest"]
        
        # Generate random moments
        np.random.seed(42)
        moments = {}
        
        # Standard deviations
        moments["std"] = {var: 0.5 + 0.5 * np.random.rand() for var in variables}
        
        # Autocorrelations
        moments["autocorr"] = {var: 0.7 + 0.2 * np.random.rand() for var in variables}
        
        # Cross-correlations
        moments["corr"] = {}
        for var1, var2 in itertools.combinations(variables, 2):
            moments["corr"][(var1, var2)] = -0.5 + np.random.rand()
        
        logger.info("Model moments computed")
        
        return moments


# 1. One-at-a-time Sensitivity Analysis
class OATSensitivityAnalysis:
    """Class for one-at-a-time sensitivity analysis."""
    
    def __init__(self, base_config):
        """Initialize the sensitivity analysis."""
        self.base_config = base_config
        
        logger.info("Initialized one-at-a-time sensitivity analysis")
    
    def analyze_parameter(self, parameter, values, shock_name, periods=40):
        """Analyze sensitivity to a parameter."""
        logger.info(f"Analyzing sensitivity to parameter {parameter} with values {values}")
        
        # Results for each parameter value
        results = []
        
        # Generate impulse responses for each parameter value
        for value in values:
            # Create a new configuration
            config = ConfigManager()
            config.update(self.base_config.get())
            
            # Set parameter value
            config.set(parameter, value)
            
            # Create model
            model = SmetsWoutersModel(config)
            
            # Compute impulse response
            irf = model.compute_impulse_response(shock_name, periods=periods)
            
            # Store result
            results.append(irf)
        
        # Compute sensitivity metrics
        sensitivity = {}
        variables = results[0].keys()
        
        for var in variables:
            # Compute standard deviation across parameter values
            responses = np.array([result[var] for result in results])
            std_dev = np.std(responses, axis=0)
            
            # Compute mean response
            mean_response = np.mean(responses, axis=0)
            
            # Compute coefficient of variation (normalized sensitivity)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = np.mean(np.abs(std_dev / (mean_response + 1e-10)))
            
            sensitivity[var] = cv
        
        logger.info(f"Sensitivity analysis completed for parameter {parameter}")
        
        return {
            "parameter": parameter,
            "values": values,
            "results": results,
            "sensitivity": sensitivity
        }
    
    def plot_sensitivity(self, results, output_dir="results/sensitivity"):
        """Plot sensitivity analysis results."""
        logger.info("Plotting sensitivity analysis results")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        parameter = results["parameter"]
        values = results["values"]
        irfs = results["results"]
        
        # Variables
        variables = list(irfs[0].keys())
        
        # Create plots for each variable
        for var in variables:
            plt.figure(figsize=(10, 6))
            
            # Plot IRFs for each parameter value
            for i, value in enumerate(values):
                plt.plot(irfs[i][var], label=f"{parameter}={value}")
            
            plt.title(f"Sensitivity of {var} to {parameter}")
            plt.xlabel("Periods")
            plt.ylabel("Response")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"oat_{parameter}_{var}.png"))
            plt.close()
        
        # Create sensitivity bar chart
        plt.figure(figsize=(10, 6))
        
        # Sort variables by sensitivity
        sorted_vars = sorted(results["sensitivity"].items(), key=lambda x: x[1], reverse=True)
        var_names = [var for var, _ in sorted_vars]
        sensitivities = [sens for _, sens in sorted_vars]
        
        # Create bar chart
        plt.bar(var_names, sensitivities)
        plt.title(f"Sensitivity to {parameter}")
        plt.xlabel("Variable")
        plt.ylabel("Sensitivity (CV)")
        plt.grid(True, axis='y')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"oat_{parameter}_sensitivity.png"))
        plt.close()
        
        logger.info(f"Sensitivity plots saved to {output_dir}")


# 2. Morris Method for Screening
class MorrisSensitivityAnalysis:
    """Class for Morris method sensitivity analysis."""
    
    def __init__(self, base_config):
        """Initialize the sensitivity analysis."""
        self.base_config = base_config
        
        logger.info("Initialized Morris method sensitivity analysis")
    
    def analyze(self, problem, num_trajectories=10, shock_name="technology", periods=40):
        """
        Perform Morris method sensitivity analysis.
        
        Args:
            problem (dict): Problem definition for SALib
            num_trajectories (int): Number of trajectories
            shock_name (str): Name of the shock
            periods (int): Number of periods
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Performing Morris method analysis with {num_trajectories} trajectories")
        
        # Generate samples
        param_values = morris.sample(problem, N=num_trajectories, num_levels=4)
        
        # Number of samples
        num_samples = param_values.shape[0]
        logger.info(f"Generated {num_samples} parameter samples")
        
        # Evaluate model for each sample
        model_outputs = []
        
        for i, X in enumerate(param_values):
            logger.info(f"Evaluating sample {i+1}/{num_samples}")
            
            # Create configuration
            config = ConfigManager()
            config.update(self.base_config.get())
            
            # Set parameter values
            for j, name in enumerate(problem["names"]):
                config.set(name, X[j])
            
            # Create model
            model = SmetsWoutersModel(config)
            
            # Compute impulse response
            irf = model.compute_impulse_response(shock_name, periods=periods)
            
            # Extract peak responses
            peak_responses = {var: np.max(np.abs(response)) for var, response in irf.items()}
            
            # Store results
            model_outputs.append(peak_responses)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(model_outputs)
        
        # Analyze results for each variable
        analysis_results = {}
        
        for var in results_df.columns:
            # Extract outputs for this variable
            Y = results_df[var].values
            
            # Perform Morris analysis
            morris_result = morris_analyze.analyze(
                problem, param_values, Y, 
                print_to_console=False, 
                num_levels=4
            )
            
            # Store results
            analysis_results[var] = {
                "mu": dict(zip(problem["names"], morris_result["mu"])),
                "mu_star": dict(zip(problem["names"], morris_result["mu_star"])),
                "sigma": dict(zip(problem["names"], morris_result["sigma"])),
                "mu_star_conf": dict(zip(problem["names"], morris_result["mu_star_conf"]))
            }
        
        logger.info("Morris method analysis completed")
        
        return {
            "problem": problem,
            "param_values": param_values,
            "results_df": results_df,
            "analysis": analysis_results
        }
    
    def plot_results(self, results, output_dir="results/sensitivity"):
        """Plot Morris method results."""
        logger.info("Plotting Morris method results")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        problem = results["problem"]
        analysis = results["analysis"]
        
        # Variables
        variables = list(analysis.keys())
        
        # Create plots for each variable
        for var in variables:
            plt.figure(figsize=(10, 6))
            
            # Extract results
            mu_star = [analysis[var]["mu_star"][name] for name in problem["names"]]
            sigma = [analysis[var]["sigma"][name] for name in problem["names"]]
            
            # Create scatter plot
            plt.scatter(mu_star, sigma, s=80, alpha=0.6)
            
            # Add parameter names as labels
            for i, name in enumerate(problem["names"]):
                plt.annotate(name.split(".")[-1], (mu_star[i], sigma[i]), 
                            xytext=(5, 5), textcoords="offset points")
            
            plt.title(f"Morris Method: {var}")
            plt.xlabel("μ* (Mean of Absolute Elementary Effects)")
            plt.ylabel("σ (Standard Deviation of Elementary Effects)")
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"morris_{var}.png"))
            plt.close()
        
        # Create bar chart of overall importance
        plt.figure(figsize=(12, 8))
        
        # Compute average mu_star across all variables
        avg_mu_star = {}
        for name in problem["names"]:
            avg_mu_star[name] = np.mean([analysis[var]["mu_star"][name] for var in variables])
        
        # Sort by importance
        sorted_params = sorted(avg_mu_star.items(), key=lambda x: x[1], reverse=True)
        param_names = [name.split(".")[-1] for name, _ in sorted_params]
        importances = [imp for _, imp in sorted_params]
        
        # Create bar chart
        plt.bar(param_names, importances)
        plt.title("Parameter Importance (Morris Method)")
        plt.xlabel("Parameter")
        plt.ylabel("Average μ*")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "morris_importance.png"))
        plt.close()
        
        logger.info(f"Morris method plots saved to {output_dir}")


# 3. Sobol' Method for Global Sensitivity Analysis
class SobolSensitivityAnalysis:
    """Class for Sobol' method sensitivity analysis."""
    
    def __init__(self, base_config):
        """Initialize the sensitivity analysis."""
        self.base_config = base_config
        
        logger.info("Initialized Sobol' method sensitivity analysis")
    
    def analyze(self, problem, n_samples=512, shock_name="technology", periods=40):
        """
        Perform Sobol' method sensitivity analysis.
        
        Args:
            problem (dict): Problem definition for SALib
            n_samples (int): Number of samples
            shock_name (str): Name of the shock
            periods (int): Number of periods
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Performing Sobol' method analysis with {n_samples} samples")
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
        
        # Number of samples
        num_samples = param_values.shape[0]
        logger.info(f"Generated {num_samples} parameter samples")
        
        # Evaluate model for each sample
        model_outputs = []
        
        for i, X in enumerate(param_values):
            if i % 100 == 0:
                logger.info(f"Evaluating sample {i+1}/{num_samples}")
            
            # Create configuration
            config = ConfigManager()
            config.update(self.base_config.get())
            
            # Set parameter values
            for j, name in enumerate(problem["names"]):
                config.set(name, X[j])
            
            # Create model
            model = SmetsWoutersModel(config)
            
            # Compute impulse response
            irf = model.compute_impulse_response(shock_name, periods=periods)
            
            # Extract peak responses
            peak_responses = {var: np.max(np.abs(response)) for var, response in irf.items()}
            
            # Store results
            model_outputs.append(peak_responses)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(model_outputs)
        
        # Analyze results for each variable
        analysis_results = {}
        
        for var in results_df.columns:
            # Extract outputs for this variable
            Y = results_df[var].values
            
            # Perform Sobol' analysis
            sobol_result = sobol.analyze(
                problem, Y, 
                calc_second_order=True, 
                print_to_console=False
            )
            
            # Store results
            analysis_results[var] = {
                "S1": dict(zip(problem["names"], sobol_result["S1"])),
                "S2": {(problem["names"][i], problem["names"][j]): sobol_result["S2"][i, j]
                      for i in range(len(problem["names"]))
                      for j in range(i+1, len(problem["names"]))},
                "ST": dict(zip(problem["names"], sobol_result["ST"]))
            }
        
        logger.info("Sobol' method analysis completed")
        
        return {
            "problem": problem,
            "param_values": param_values,
            "results_df": results_df,
            "analysis": analysis_results
        }
    
    def plot_results(self, results, output_dir="results/sensitivity"):
        """Plot Sobol' method results."""
        logger.info("Plotting Sobol' method results")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        problem = results["problem"]
        analysis = results["analysis"]
        
        # Variables
        variables = list(analysis.keys())
        
        # Create plots for each variable
        for var in variables:
            # First-order indices
            plt.figure(figsize=(10, 6))
            
            # Extract results
            S1 = [analysis[var]["S1"][name] for name in problem["names"]]
            
            # Sort by importance
            sorted_indices = np.argsort(S1)[::-1]
            sorted_names = [problem["names"][i].split(".")[-1] for i in sorted_indices]
            sorted_S1 = [S1[i] for i in sorted_indices]
            
            # Create bar chart
            plt.bar(sorted_names, sorted_S1)
            plt.title(f"First-order Sobol' Indices: {var}")
            plt.xlabel("Parameter")
            plt.ylabel("S1")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"sobol_S1_{var}.png"))
            plt.close()
            
            # Total-order indices
            plt.figure(figsize=(10, 6))
            
            # Extract results
            ST = [analysis[var]["ST"][name] for name in problem["names"]]
            
            # Sort by importance
            sorted_indices = np.argsort(ST)[::-1]
            sorted_names = [problem["names"][i].split(".")[-1] for i in sorted_indices]
            sorted_ST = [ST[i] for i in sorted_indices]
            
            # Create bar chart
            plt.bar(sorted_names, sorted_ST)
            plt.title(f"Total-order Sobol' Indices: {var}")
            plt.xlabel("Parameter")
            plt.ylabel("ST")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"sobol_ST_{var}.png"))
            plt.close()
        
        # Create heatmap of second-order indices for the first variable
        var = variables[0]
        plt.figure(figsize=(10, 8))
        
        # Extract parameter names (short form)
        short_names = [name.split(".")[-1] for name in problem["names"]]
        
        # Create matrix of second-order indices
        n_params = len(problem["names"])
        S2_matrix = np.zeros((n_params, n_params))
        
        for i in range(n_params):
            for j in range(i+1, n_params):
                key = (problem["names"][i], problem["names"][j])
                S2_matrix[i, j] = analysis[var]["S2"].get(key, 0)
                S2_matrix[j, i] = S2_matrix[i, j]  # Symmetric
        
        # Create heatmap
        plt.imshow(S2_matrix, cmap='viridis')
        plt.colorbar(label="S2")
        plt.title(f"Second-order Sobol' Indices: {var}")
        plt.xticks(range(n_params), short_names, rotation=45)
        plt.yticks(range(n_params), short_names)
        
        # Add text annotations
        for i in range(n_params):
            for j in range(n_params):
                if i != j:
                    plt.text(j, i, f"{S2_matrix[i, j]:.2f}", 
                            ha="center", va="center", 
                            color="white" if S2_matrix[i, j] > 0.3 else "black")
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"sobol_S2_{var}.png"))
        plt.close()
        
        # Create bar chart of overall importance
        plt.figure(figsize=(12, 8))
        
        # Compute average total-order indices across all variables
        avg_ST = {}
        for name in problem["names"]:
            avg_ST[name] = np.mean([analysis[var]["ST"][name] for var in variables])
        
        # Sort by importance
        sorted_params = sorted(avg_ST.items(), key=lambda x: x[1], reverse=True)
        param_names = [name.split(".")[-1] for name, _ in sorted_params]
        importances = [imp for _, imp in sorted_params]
        
        # Create bar chart
        plt.bar(param_names, importances)
        plt.title("Parameter Importance (Sobol' Method)")
        plt.xlabel("Parameter")
        plt.ylabel("Average Total-order Index")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "sobol_importance.png"))
        plt.close()
        
        logger.info(f"Sobol' method plots saved to {output_dir}")


def main():
    """Main function demonstrating sensitivity analysis techniques."""
    print("=== DSGE Model Sensitivity Analysis Example ===")
    
    # Create output directories
    os.makedirs("results/sensitivity", exist_ok=True)
    
    # 1. One-at-a-time Sensitivity Analysis
    print("\n1. One-at-a-time Sensitivity Analysis")
    
    # Create base configuration
    base_config = ConfigManager()
    
    # Create OAT analyzer
    oat_analyzer = OATSensitivityAnalysis(base_config)
    
    # Analyze sensitivity to beta (discount factor)
    beta_results = oat_analyzer.analyze_parameter(
        "base_model.beta", [0.97, 0.98, 0.99, 0.995], "technology"
    )
    
    # Plot results
    oat_analyzer.plot_sensitivity(beta_results)
    
    # Analyze sensitivity to phi_pi (Taylor rule inflation response)
    phi_pi_results = oat_analyzer.analyze_parameter(
        "base_model.phi_pi", [1.3, 1.5, 1.7, 2.0], "monetary"
    )
    
    # Plot results
    oat_analyzer.plot_sensitivity(phi_pi_results)
    
    # Print sensitivity results
    print("\nParameter Sensitivity Results:")
    print("\nBeta (discount factor):")
    for var, sensitivity in beta_results["sensitivity"].items():
        print(f"  {var}: {sensitivity:.4f}")
    
    print("\nPhi_pi (Taylor rule inflation response):")
    for var, sensitivity in phi_pi_results["sensitivity"].items():
        print(f"  {var}: {sensitivity:.4f}")
    
    # 2. Morris Method for Screening
    print("\n2. Morris Method for Screening")
    
    # Define problem for Morris method
    problem = {
        "num_vars": 6,
        "names": ["base_model.beta", "base_model.sigma_c", "base_model.h", 
                 "base_model.xi_p", "base_model.xi_w", "base_model.phi_pi"],
        "bounds": [[0.97, 0.999], [1.0, 2.0], [0.5, 0.9], 
                  [0.6, 0.9], [0.6, 0.9], [1.2, 2.0]]
    }
    
    # Create Morris analyzer
    morris_analyzer = MorrisSensitivityAnalysis(base_config)
    
    # Perform Morris analysis
    morris_results = morris_analyzer.analyze(problem, num_trajectories=10)
    
    # Plot results
    morris_analyzer.plot_results(morris_results)
    
    # Print Morris results for output variable
    print("\nMorris Method Results for Output:")
    var = "output"
    for name in problem["names"]:
        short_name = name.split(".")[-1]
        mu_star = morris_results["analysis"][var]["mu_star"][name]
        sigma = morris_results["analysis"][var]["sigma"][name]
        print(f"  {short_name}: μ*={mu_star:.4f}, σ={sigma:.4f}")
    
    # 3. Sobol' Method for Global Sensitivity Analysis
    print("\n3. Sobol' Method for Global Sensitivity Analysis")
    
    # Define problem for Sobol' method (use fewer parameters for demonstration)
    problem = {
        "num_vars": 4,
        "names": ["base_model.beta", "base_model.h", "base_model.xi_p", "base_model.phi_pi"],
        "bounds": [[0.97, 0.999], [0.5, 0.9], [0.6, 0.9], [1.2, 2.0]]
    }
    
    # Create Sobol' analyzer
    sobol_analyzer = SobolSensitivityAnalysis(base_config)
    
    # Perform Sobol' analysis with a small sample size for demonstration
    sobol_results = sobol_analyzer.analyze(problem, n_samples=64)
    
    # Plot results
    sobol_analyzer.plot_results(sobol_results)
    
    # Print Sobol' results for output variable
    print("\nSobol' Method Results for Output:")
    var = "output"
    for name in problem["names"]:
        short_name = name.split(".")[-1]
        S1 = sobol_results["analysis"][var]["S1"][name]
        ST = sobol_results["analysis"][var]["ST"][name]
        print(f"  {short_name}: S1={S1:.4f}, ST={ST:.4f}")
    
    print("\nSensitivity analysis example completed successfully.")


if __name__ == "__main__":
    main()