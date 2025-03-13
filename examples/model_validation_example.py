#!/usr/bin/env python
"""
Model Validation Example

This script demonstrates techniques for validating DSGE model outputs:
1. Replicating published results
2. Comparing with alternative model implementations
3. Sensitivity testing for validation
4. Validation metrics and reporting

The example shows how to ensure model correctness and reliability.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import re
import tempfile
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dsge_validation")


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
    
    def compute_steady_state(self):
        """Compute the steady state of the model."""
        logger.info("Computing steady state")
        
        # Extract parameters
        beta = self.params["beta"]
        delta = self.params["delta"]
        alpha = self.params["alpha"]
        pi_bar = self.params["pi_bar"]
        
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
        
        # Store steady state values
        steady_state = {
            "output": y_ss,
            "consumption": c_ss,
            "investment": i_ss,
            "capital": k_ss,
            "labor": l_ss,
            "real_wage": w_ss,
            "rental_rate": r_k_ss,
            "inflation": pi_bar,
            "nominal_interest": r_ss,
            "real_interest": r_ss / pi_bar,
        }
        
        logger.info(f"Steady state computed: {steady_state}")
        
        return steady_state
    
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
            response = signs[var] * shock_size * np.exp(-t / decay)
            
            # Add some noise
            response += 0.05 * np.random.randn(periods)
            
            irf[var] = response
        
        logger.info(f"Impulse response computed for {shock_name} shock")
        
        return irf


# 1. Benchmark Data Loader
class BenchmarkDataLoader:
    """Class for loading benchmark data for model validation."""
    
    @staticmethod
    def load_published_results(source, shock_name=None):
        """Load published results for validation."""
        logger.info(f"Loading published results from {source}")
        
        # In a real implementation, this would load data from files or databases
        # For demonstration, we'll generate synthetic benchmark data
        
        # Smets and Wouters (2007) benchmark
        if source == "smets_wouters_2007":
            # Different responses for different shocks
            if shock_name == "technology":
                benchmark = {
                    "output": np.array([0.45, 0.42, 0.39, 0.36, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23,
                                       0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]),
                    "consumption": np.array([0.25, 0.27, 0.28, 0.29, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24,
                                           0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14]),
                    "investment": np.array([0.10, 0.15, 0.18, 0.20, 0.21, 0.21, 0.20, 0.19, 0.18, 0.17,
                                          0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07]),
                    "inflation": np.array([-0.15, -0.12, -0.10, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.03,
                                         -0.02, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]),
                    "nominal_interest": np.array([-0.10, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.03, -0.02, -0.02,
                                               -0.02, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
                }
            elif shock_name == "monetary":
                benchmark = {
                    "output": np.array([-0.30, -0.25, -0.20, -0.16, -0.13, -0.10, -0.08, -0.06, -0.05, -0.04,
                                      -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00]),
                    "consumption": np.array([-0.20, -0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.07, -0.06, -0.05,
                                          -0.04, -0.03, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, 0.00]),
                    "investment": np.array([-0.40, -0.35, -0.30, -0.25, -0.21, -0.18, -0.15, -0.12, -0.10, -0.08,
                                         -0.07, -0.06, -0.05, -0.04, -0.03, -0.03, -0.02, -0.02, -0.01, -0.01]),
                    "inflation": np.array([-0.20, -0.15, -0.12, -0.09, -0.07, -0.05, -0.04, -0.03, -0.02, -0.02,
                                        -0.01, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
                    "nominal_interest": np.array([0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
                                              0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00])
                }
            else:
                benchmark = {}
        
        # Christiano, Eichenbaum, and Evans (2005) benchmark
        elif source == "christiano_et_al_2005":
            # Different responses for different shocks
            if shock_name == "monetary":
                benchmark = {
                    "output": np.array([-0.35, -0.30, -0.25, -0.20, -0.15, -0.12, -0.09, -0.07, -0.05, -0.04,
                                      -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00, 0.00]),
                    "consumption": np.array([-0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.07, -0.06, -0.05, -0.04,
                                          -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00, 0.00]),
                    "investment": np.array([-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.16, -0.13, -0.10, -0.08,
                                         -0.06, -0.05, -0.04, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, 0.00]),
                    "inflation": np.array([-0.15, -0.12, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.02,
                                        -0.01, -0.01, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
                    "nominal_interest": np.array([0.30, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04,
                                              0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00])
                }
            else:
                benchmark = {}
        else:
            logger.warning(f"Unknown source: {source}")
            benchmark = {}
        
        # Extend benchmark data to match the requested periods if needed
        for var in benchmark:
            if len(benchmark[var]) < 40:
                # Extend with zeros
                extension = np.zeros(40 - len(benchmark[var]))
                benchmark[var] = np.concatenate([benchmark[var], extension])
        
        logger.info(f"Loaded benchmark data from {source} for {shock_name} shock")
        
        return benchmark
    
    @staticmethod
    def load_alternative_implementation(implementation, shock_name=None):
        """Load results from an alternative implementation."""
        logger.info(f"Loading results from alternative implementation: {implementation}")
        
        # In a real implementation, this would load data from files or databases
        # For demonstration, we'll generate synthetic data
        
        # Dynare implementation
        if implementation == "dynare":
            # Different responses for different shocks
            if shock_name == "technology":
                alt_results = {
                    "output": np.array([0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24,
                                       0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]),
                    "consumption": np.array([0.28, 0.30, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24,
                                           0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14]),
                    "investment": np.array([0.12, 0.16, 0.19, 0.21, 0.22, 0.22, 0.21, 0.20, 0.19, 0.18,
                                          0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]),
                    "inflation": np.array([-0.14, -0.11, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.03,
                                         -0.02, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]),
                    "nominal_interest": np.array([-0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.03, -0.02, -0.02,
                                               -0.02, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
                }
            elif shock_name == "monetary":
                alt_results = {
                    "output": np.array([-0.28, -0.24, -0.20, -0.17, -0.14, -0.12, -0.10, -0.08, -0.07, -0.06,
                                      -0.05, -0.04, -0.03, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01]),
                    "consumption": np.array([-0.18, -0.16, -0.14, -0.12, -0.11, -0.09, -0.08, -0.07, -0.06, -0.05,
                                          -0.04, -0.03, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, 0.00]),
                    "investment": np.array([-0.38, -0.33, -0.29, -0.25, -0.22, -0.19, -0.16, -0.14, -0.12, -0.10,
                                         -0.09, -0.07, -0.06, -0.05, -0.04, -0.04, -0.03, -0.03, -0.02, -0.02]),
                    "inflation": np.array([-0.19, -0.14, -0.11, -0.09, -0.07, -0.05, -0.04, -0.03, -0.02, -0.02,
                                        -0.01, -0.01, -0.01, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
                    "nominal_interest": np.array([0.24, 0.19, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
                                              0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00])
                }
            else:
                alt_results = {}
        else:
            logger.warning(f"Unknown implementation: {implementation}")
            alt_results = {}
        
        # Extend results to match the requested periods if needed
        for var in alt_results:
            if len(alt_results[var]) < 40:
                # Extend with zeros
                extension = np.zeros(40 - len(alt_results[var]))
                alt_results[var] = np.concatenate([alt_results[var], extension])
        
        logger.info(f"Loaded alternative implementation results from {implementation} for {shock_name} shock")
        
        return alt_results


# 2. Comparison Metrics
class ComparisonMetrics:
    """Class for computing comparison metrics between model outputs."""
    
    @staticmethod
    def compute_rmse(model_output, benchmark, variables=None, periods=None):
        """Compute Root Mean Square Error (RMSE) between model output and benchmark."""
        logger.info("Computing RMSE")
        
        # Determine variables to compare
        if variables is None:
            variables = list(set(model_output.keys()) & set(benchmark.keys()))
        else:
            variables = [var for var in variables if var in model_output and var in benchmark]
        
        if not variables:
            logger.warning("No common variables found for RMSE computation")
            return {}
        
        # Determine periods to compare
        if periods is None:
            periods = min(
                min(len(model_output[var]) for var in variables),
                min(len(benchmark[var]) for var in variables)
            )
        else:
            periods = min(
                periods,
                min(len(model_output[var]) for var in variables),
                min(len(benchmark[var]) for var in variables)
            )
        
        # Compute RMSE
        rmse = {}
        for var in variables:
            model_values = model_output[var][:periods]
            benchmark_values = benchmark[var][:periods]
            
            # Compute RMSE
            rmse[var] = np.sqrt(np.mean((model_values - benchmark_values) ** 2))
        
        logger.info(f"RMSE computed for {len(variables)} variables over {periods} periods")
        
        return rmse
    
    @staticmethod
    def compute_correlation(model_output, benchmark, variables=None, periods=None):
        """Compute correlation between model output and benchmark."""
        logger.info("Computing correlation")
        
        # Determine variables to compare
        if variables is None:
            variables = list(set(model_output.keys()) & set(benchmark.keys()))
        else:
            variables = [var for var in variables if var in model_output and var in benchmark]
        
        if not variables:
            logger.warning("No common variables found for correlation computation")
            return {}
        
        # Determine periods to compare
        if periods is None:
            periods = min(
                min(len(model_output[var]) for var in variables),
                min(len(benchmark[var]) for var in variables)
            )
        else:
            periods = min(
                periods,
                min(len(model_output[var]) for var in variables),
                min(len(benchmark[var]) for var in variables)
            )
        
        # Compute correlation
        correlation = {}
        for var in variables:
            model_values = model_output[var][:periods]
            benchmark_values = benchmark[var][:periods]
            
            # Compute correlation
            correlation[var] = np.corrcoef(model_values, benchmark_values)[0, 1]
        
        logger.info(f"Correlation computed for {len(variables)} variables over {periods} periods")
        
        return correlation
    
    @staticmethod
    def compute_peak_timing_difference(model_output, benchmark, variables=None):
        """Compute difference in peak timing between model output and benchmark."""
        logger.info("Computing peak timing difference")
        
        # Determine variables to compare
        if variables is None:
            variables = list(set(model_output.keys()) & set(benchmark.keys()))
        else:
            variables = [var for var in variables if var in model_output and var in benchmark]
        
        if not variables:
            logger.warning("No common variables found for peak timing difference computation")
            return {}
        
        # Compute peak timing difference
        peak_diff = {}
        for var in variables:
            # Find peak in model output
            model_values = np.abs(model_output[var])
            model_peak_idx = np.argmax(model_values)
            
            # Find peak in benchmark
            benchmark_values = np.abs(benchmark[var])
            benchmark_peak_idx = np.argmax(benchmark_values)
            
            # Compute difference
            peak_diff[var] = model_peak_idx - benchmark_peak_idx
        
        logger.info(f"Peak timing difference computed for {len(variables)} variables")
        
        return peak_diff


# 3. Validation Report Generator
class ValidationReportGenerator:
    """Class for generating validation reports."""
    
    def __init__(self, output_dir="results/validation"):
        """Initialize the validation report generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized validation report generator with output directory: {output_dir}")
    
    def generate_comparison_report(self, model_output, benchmark, metrics, shock_name, benchmark_source, output_file="comparison_report.html"):
        """Generate a comparison report between model output and benchmark."""
        logger.info(f"Generating comparison report: {output_file}")
        
        # Create report path
        report_path = os.path.join(self.output_dir, output_file)
        
        # Create HTML
        with open(report_path, 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<title>DSGE Model Validation Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
            f.write("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n")
            f.write("h2 { color: #3498db; margin-top: 30px; }\n")
            f.write("h3 { color: #2980b9; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write(".good { color: green; }\n")
            f.write(".moderate { color: orange; }\n")
            f.write(".poor { color: red; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 10px 0; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            
            # Header
            f.write(f"<h1>DSGE Model Validation Report</h1>\n")
            f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"<p>Shock: {shock_name}</p>\n")
            f.write(f"<p>Benchmark source: {benchmark_source}</p>\n")
            
            # Summary
            f.write("<h2>Summary</h2>\n")
            
            # RMSE summary
            if "rmse" in metrics:
                f.write("<h3>Root Mean Square Error (RMSE)</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Variable</th><th>RMSE</th><th>Assessment</th></tr>\n")
                
                for var, value in metrics["rmse"].items():
                    # Assess RMSE
                    if value < 0.05:
                        assessment = "<span class='good'>Good</span>"
                    elif value < 0.1:
                        assessment = "<span class='moderate'>Moderate</span>"
                    else:
                        assessment = "<span class='poor'>Poor</span>"
                    
                    f.write(f"<tr><td>{var}</td><td>{value:.4f}</td><td>{assessment}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Correlation summary
            if "correlation" in metrics:
                f.write("<h3>Correlation</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Variable</th><th>Correlation</th><th>Assessment</th></tr>\n")
                
                for var, value in metrics["correlation"].items():
                    # Assess correlation
                    if value > 0.9:
                        assessment = "<span class='good'>Good</span>"
                    elif value > 0.7:
                        assessment = "<span class='moderate'>Moderate</span>"
                    else:
                        assessment = "<span class='poor'>Poor</span>"
                    
                    f.write(f"<tr><td>{var}</td><td>{value:.4f}</td><td>{assessment}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Peak timing difference summary
            if "peak_diff" in metrics:
                f.write("<h3>Peak Timing Difference</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Variable</th><th>Difference (periods)</th><th>Assessment</th></tr>\n")
                
                for var, value in metrics["peak_diff"].items():
                    # Assess peak timing difference
                    if abs(value) <= 1:
                        assessment = "<span class='good'>Good</span>"
                    elif abs(value) <= 3:
                        assessment = "<span class='moderate'>Moderate</span>"
                    else:
                        assessment = "<span class='poor'>Poor</span>"
                    
                    f.write(f"<tr><td>{var}</td><td>{value}</td><td>{assessment}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Detailed comparison
            f.write("<h2>Detailed Comparison</h2>\n")
            
            # Variables to compare
            variables = list(set(model_output.keys()) & set(benchmark.keys()))
            
            for var in variables:
                f.write(f"<h3>{var}</h3>\n")
                
                # Create plot
                plt.figure(figsize=(10, 6))
                
                # Time axis
                t = np.arange(min(len(model_output[var]), len(benchmark[var])))
                
                # Plot model output
                plt.plot(t, model_output[var][:len(t)], 'b-', label="Model")
                
                # Plot benchmark
                plt.plot(t, benchmark[var][:len(t)], 'r--', label="Benchmark")
                
                plt.title(f"{var} Response to {shock_name} Shock")
                plt.xlabel("Periods")
                plt.ylabel("Response")
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plot_file = f"{var}_{shock_name}.png"
                plot_path = os.path.join(self.output_dir, plot_file)
                plt.savefig(plot_path)
                plt.close()
                
                # Add plot to report
                f.write(f"<img src='{plot_file}' alt='{var} Response' />\n")
                
                # Add metrics
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                
                if "rmse" in metrics and var in metrics["rmse"]:
                    f.write(f"<tr><td>RMSE</td><td>{metrics['rmse'][var]:.4f}</td></tr>\n")
                
                if "correlation" in metrics and var in metrics["correlation"]:
                    f.write(f"<tr><td>Correlation</td><td>{metrics['correlation'][var]:.4f}</td></tr>\n")
                
                if "peak_diff" in metrics and var in metrics["peak_diff"]:
                    f.write(f"<tr><td>Peak Timing Difference</td><td>{metrics['peak_diff'][var]}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Conclusion
            f.write("<h2>Conclusion</h2>\n")
            
            # Compute overall assessment
            if "rmse" in metrics:
                avg_rmse = np.mean(list(metrics["rmse"].values()))
                if avg_rmse < 0.05:
                    rmse_assessment = "good"
                elif avg_rmse < 0.1:
                    rmse_assessment = "moderate"
                else:
                    rmse_assessment = "poor"
            else:
                rmse_assessment = "unknown"
            
            if "correlation" in metrics:
                avg_corr = np.mean(list(metrics["correlation"].values()))
                if avg_corr > 0.9:
                    corr_assessment = "good"
                elif avg_corr > 0.7:
                    corr_assessment = "moderate"
                else:
                    corr_assessment = "poor"
            else:
                corr_assessment = "unknown"
            
            # Generate conclusion text
            if rmse_assessment == "good" and corr_assessment == "good":
                conclusion = "The model shows excellent agreement with the benchmark results. "
                conclusion += "The impulse responses closely match in both magnitude and dynamics."
            elif rmse_assessment == "good" or corr_assessment == "good":
                conclusion = "The model shows good agreement with the benchmark results. "
                conclusion += "The impulse responses match reasonably well, with some minor differences."
            elif rmse_assessment == "moderate" and corr_assessment == "moderate":
                conclusion = "The model shows moderate agreement with the benchmark results. "
                conclusion += "There are some differences in the impulse responses that may warrant further investigation."
            else:
                conclusion = "The model shows significant differences from the benchmark results. "
                conclusion += "Further investigation and model refinement may be necessary."
            
            f.write(f"<p>{conclusion}</p>\n")
            
            f.write("</body>\n</html>")
        
        logger.info(f"Comparison report generated: {report_path}")
        
        return report_path


# 4. Sensitivity Analysis
class SensitivityAnalysis:
    """Class for sensitivity analysis of model parameters."""
    
    def __init__(self, base_config):
        """Initialize the sensitivity analysis."""
        self.base_config = base_config
        
        logger.info("Initialized sensitivity analysis")
    
    def analyze_parameter_sensitivity(self, parameter, values, shock_name, periods=40):
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


def main():
    """Main function demonstrating model validation techniques."""
    print("=== DSGE Model Validation Example ===")
    
    # Create output directories
    os.makedirs("results/validation", exist_ok=True)
    
    # 1. Replicating Published Results
    print("\n1. Replicating Published Results")
    
    # Create model
    config = ConfigManager()
    model = SmetsWoutersModel(config)
    
    # Compute impulse response for technology shock
    tech_irf = model.compute_impulse_response("technology", periods=40)
    
    # Load benchmark data
    benchmark_loader = BenchmarkDataLoader()
    sw_benchmark = benchmark_loader.load_published_results("smets_wouters_2007", "technology")
    
    # Compute comparison metrics
    metrics = {}
    metrics["rmse"] = ComparisonMetrics.compute_rmse(tech_irf, sw_benchmark)
    metrics["correlation"] = ComparisonMetrics.compute_correlation(tech_irf, sw_benchmark)
    metrics["peak_diff"] = ComparisonMetrics.compute_peak_timing_difference(tech_irf, sw_benchmark)
    
    # Generate comparison report
    report_generator = ValidationReportGenerator()
    report_path = report_generator.generate_comparison_report(
        tech_irf, sw_benchmark, metrics, "technology", "smets_wouters_2007"
    )
    
    print(f"Comparison report generated: {report_path}")
    
    # 2. Comparing with Alternative Implementations
    print("\n2. Comparing with Alternative Implementations")
    
    # Load alternative implementation results
    dynare_results = benchmark_loader.load_alternative_implementation("dynare", "technology")
    
    # Compute comparison metrics
    alt_metrics = {}
    alt_metrics["rmse"] = ComparisonMetrics.compute_rmse(tech_irf, dynare_results)
    alt_metrics["correlation"] = ComparisonMetrics.compute_correlation(tech_irf, dynare_results)
    alt_metrics["peak_diff"] = ComparisonMetrics.compute_peak_timing_difference(tech_irf, dynare_results)
    
    # Generate comparison report
    alt_report_path = report_generator.generate_comparison_report(
        tech_irf, dynare_results, alt_metrics, "technology", "dynare_implementation",
        output_file="alternative_comparison_report.html"
    )
    
    print(f"Alternative implementation comparison report generated: {alt_report_path}")
    
    # 3. Sensitivity Testing
    print("\n3. Sensitivity Testing")
    
    # Create sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalysis(config)
    
    # Analyze sensitivity to beta (discount factor)
    beta_sensitivity = sensitivity_analyzer.analyze_parameter_sensitivity(
        "base_model.beta", [0.97, 0.98, 0.99, 0.995], "technology"
    )
    
    # Analyze sensitivity to phi_pi (Taylor rule inflation response)
    phi_pi_sensitivity = sensitivity_analyzer.analyze_parameter_sensitivity(
        "base_model.phi_pi", [1.3, 1.5, 1.7, 2.0], "monetary"
    )
    
    # Print sensitivity results
    print("\nParameter Sensitivity Results:")
    print("\nBeta (discount factor):")
    for var, sensitivity in beta_sensitivity["sensitivity"].items():
        print(f"  {var}: {sensitivity:.4f}")
    
    print("\nPhi_pi (Taylor rule inflation response):")
    for var, sensitivity in phi_pi_sensitivity["sensitivity"].items():
        print(f"  {var}: {sensitivity:.4f}")
    
    print("\nModel validation example completed successfully.")


if __name__ == "__main__":
    main()
