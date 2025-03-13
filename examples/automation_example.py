#!/usr/bin/env python
"""
Automation Example

This script demonstrates automation techniques for DSGE model workflows:
1. Batch processing of estimations
2. Parameter sweep automation
3. Scheduled forecasting
4. Automated report generation
5. Notification systems

The example shows how to automate repetitive tasks and create efficient workflows.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import itertools
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import argparse
import sys
import traceback
import re
import shutil
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dsge_automation")


# Mock DSGE model classes for demonstration
class ConfigManager:
    """Mock configuration manager for DSGE model."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to a JSON configuration file.
        """
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
            "financial_extension": {
                "enabled": False
            },
            "open_economy_extension": {
                "enabled": False
            },
            "fiscal_extension": {
                "enabled": False
            },
            "solution": {
                "method": "perturbation",
                "perturbation_order": 1
            },
            "estimation": {
                "method": "bayesian",
                "mcmc_algorithm": "metropolis_hastings",
                "num_chains": 4,
                "num_draws": 10000
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
    
    def enable_extension(self, extension_name):
        """Enable a model extension."""
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}")
        
        self.set(f"{extension_name}.enabled", True)
    
    def disable_extension(self, extension_name):
        """Disable a model extension."""
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}")
        
        self.set(f"{extension_name}.enabled", False)


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
            "phi_dy": config.get("base_model.phi_dy", 0.125)
        }
        
        # Check which extensions are enabled
        self.extensions = {
            "financial": config.get("financial_extension.enabled", False),
            "open_economy": config.get("open_economy_extension.enabled", False),
            "fiscal": config.get("fiscal_extension.enabled", False)
        }
        
        logger.info(f"Initialized model with extensions: {self.extensions}")
    
    def solve(self):
        """Solve the model."""
        logger.info("Solving model...")
        time.sleep(0.5)  # Simulate computation time
        
        # Generate synthetic solution
        solution = {
            "status": "success",
            "method": self.config.get("solution.method"),
            "extensions": self.extensions,
            "computation_time": 0.5
        }
        
        logger.info(f"Model solved using {solution['method']}")
        
        return solution
    
    def estimate(self, data=None):
        """Estimate the model."""
        logger.info("Estimating model...")
        
        # Estimation settings
        method = self.config.get("estimation.method", "bayesian")
        mcmc_algorithm = self.config.get("estimation.mcmc_algorithm", "metropolis_hastings")
        num_chains = self.config.get("estimation.num_chains", 4)
        num_draws = self.config.get("estimation.num_draws", 10000)
        
        logger.info(f"Using {method} estimation with {mcmc_algorithm}, {num_chains} chains, {num_draws} draws")
        
        # Simulate computation time based on settings
        computation_time = 0.01 * num_chains * num_draws / 1000
        time.sleep(min(computation_time, 2.0))  # Cap at 2 seconds for demonstration
        
        # Generate synthetic estimation results
        np.random.seed(42)
        estimated_params = {}
        for param, value in self.params.items():
            # Add some random noise to the true parameter values
            estimated_params[param] = value * (1 + 0.1 * np.random.randn())
        
        # Add some diagnostics
        diagnostics = {
            "log_likelihood": -100.0 + 10.0 * np.random.randn(),
            "acceptance_rate": 0.2 + 0.1 * np.random.rand()
        }
        
        results = {
            "status": "success",
            "method": method,
            "mcmc_algorithm": mcmc_algorithm,
            "num_chains": num_chains,
            "num_draws": num_draws,
            "estimated_parameters": estimated_params,
            "diagnostics": diagnostics,
            "computation_time": computation_time
        }
        
        logger.info(f"Model estimation completed in {computation_time:.2f} seconds")
        
        return results
    
    def forecast(self, periods=40, num_simulations=100):
        """Generate forecasts."""
        logger.info(f"Generating forecasts for {periods} periods with {num_simulations} simulations...")
        
        # Simulate computation time
        computation_time = 0.01 * periods * num_simulations / 1000
        time.sleep(min(computation_time, 1.0))  # Cap at 1 second for demonstration
        
        # Generate synthetic forecast data
        np.random.seed(42)
        
        # Variables to forecast
        variables = ["output", "consumption", "investment", "inflation", "nominal_interest"]
        
        # Generate baseline forecast
        baseline = {}
        for var in variables:
            # Different trends for different variables
            if var == "output":
                trend = 0.005  # 0.5% growth per period
            elif var == "consumption":
                trend = 0.004  # 0.4% growth per period
            elif var == "investment":
                trend = 0.006  # 0.6% growth per period
            elif var == "inflation":
                trend = 0.0005  # 0.05% increase per period
            else:  # nominal_interest
                trend = 0.0002  # 0.02% increase per period
            
            # Generate baseline forecast with trend and cycle
            t = np.arange(periods)
            baseline[var] = trend * t + 0.1 * np.sin(t / 10) + 0.02 * np.random.randn(periods)
        
        # Generate simulations for uncertainty
        simulations = {}
        for var in variables:
            # Base forecast
            base = baseline[var]
            
            # Generate simulations with increasing uncertainty
            sim = np.zeros((periods, num_simulations))
            for t in range(periods):
                # Increasing uncertainty over time
                uncertainty = 0.02 * np.sqrt(t + 1)
                sim[t, :] = base[t] + uncertainty * np.random.randn(num_simulations)
            
            simulations[var] = sim
        
        # Compute confidence intervals
        confidence_intervals = {}
        for var in variables:
            sim = simulations[var]
            
            # Compute percentiles
            ci = {
                "median": np.median(sim, axis=1),
                "lower_90": np.percentile(sim, 5, axis=1),
                "upper_90": np.percentile(sim, 95, axis=1),
                "lower_68": np.percentile(sim, 16, axis=1),
                "upper_68": np.percentile(sim, 84, axis=1)
            }
            
            confidence_intervals[var] = ci
        
        results = {
            "status": "success",
            "periods": periods,
            "num_simulations": num_simulations,
            "baseline": baseline,
            "confidence_intervals": confidence_intervals,
            "computation_time": computation_time
        }
        
        logger.info(f"Forecasting completed in {computation_time:.2f} seconds")
        
        return results


# 1. Batch Processing
class BatchProcessor:
    """Class for batch processing of DSGE model operations."""
    
    def __init__(self, output_dir="results/automation/batch"):
        """Initialize the batch processor."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized batch processor with output directory: {output_dir}")
    
    def process_configs(self, config_files, operation, parallel=True, max_workers=None):
        """Process multiple configurations."""
        logger.info(f"Processing {len(config_files)} configurations with operation: {operation}")
        
        if parallel:
            max_workers = max_workers or mp.cpu_count()
            logger.info(f"Using parallel processing with {max_workers} workers")
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create tasks
                tasks = [(config_file, operation) for config_file in config_files]
                
                # Execute tasks
                results = list(executor.map(self._process_config_task, tasks))
        else:
            logger.info("Using sequential processing")
            
            # Process sequentially
            results = []
            for config_file in config_files:
                result = self._process_config(config_file, operation)
                results.append(result)
        
        logger.info(f"Completed batch processing of {len(config_files)} configurations")
        
        return results
    
    def _process_config_task(self, task):
        """Process a single configuration task (for parallel execution)."""
        config_file, operation = task
        return self._process_config(config_file, operation)
    
    def _process_config(self, config_file, operation):
        """Process a single configuration."""
        try:
            logger.info(f"Processing configuration: {config_file}")
            
            # Load configuration
            config = ConfigManager(config_file)
            
            # Create model
            model = SmetsWoutersModel(config)
            
            # Perform operation
            if operation == "solve":
                result = model.solve()
            elif operation == "estimate":
                result = model.estimate()
            elif operation == "forecast":
                result = model.forecast()
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Add metadata
            result["config_file"] = config_file
            result["operation"] = operation
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save result
            output_file = os.path.join(
                self.output_dir,
                f"{os.path.basename(config_file).replace('.json', '')}_{operation}_result.json"
            )
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved result to {output_file}")
            
            return {
                "status": "success",
                "config_file": config_file,
                "operation": operation,
                "result": result,
                "output_file": output_file
            }
        
        except Exception as e:
            logger.error(f"Error processing {config_file}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "config_file": config_file,
                "operation": operation,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def generate_batch_report(self, results, report_file="batch_report.html"):
        """Generate a report for batch processing results."""
        logger.info(f"Generating batch processing report: {report_file}")
        
        # Count successes and failures
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "error"]
        
        # Create report
        report_path = os.path.join(self.output_dir, report_file)
        
        with open(report_path, 'w') as f:
            f.write("<html>\n<head>\n")
            f.write("<title>DSGE Model Batch Processing Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1 { color: #2c3e50; }\n")
            f.write("h2 { color: #3498db; }\n")
            f.write("table { border-collapse: collapse; width: 100%; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write(".success { color: green; }\n")
            f.write(".error { color: red; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            
            # Header
            f.write(f"<h1>DSGE Model Batch Processing Report</h1>\n")
            f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"<p>Total configurations: {len(results)}</p>\n")
            f.write(f"<p>Successful: <span class='success'>{len(successes)}</span></p>\n")
            f.write(f"<p>Failed: <span class='error'>{len(failures)}</span></p>\n")
            
            # Successful results
            if successes:
                f.write("<h2>Successful Results</h2>\n")
                f.write("<table>\n")
                f.write("<tr><th>Configuration</th><th>Operation</th><th>Computation Time</th><th>Output File</th></tr>\n")
                
                for result in successes:
                    config_file = os.path.basename(result["config_file"])
                    operation = result["operation"]
                    computation_time = result["result"].get("computation_time", "N/A")
                    output_file = os.path.basename(result["output_file"])
                    
                    f.write(f"<tr>")
                    f.write(f"<td>{config_file}</td>")
                    f.write(f"<td>{operation}</td>")
                    f.write(f"<td>{computation_time:.2f} seconds</td>")
                    f.write(f"<td>{output_file}</td>")
                    f.write(f"</tr>\n")
                
                f.write("</table>\n")
            
            # Failed results
            if failures:
                f.write("<h2>Failed Results</h2>\n")
                f.write("<table>\n")
                f.write("<tr><th>Configuration</th><th>Operation</th><th>Error</th></tr>\n")
                
                for result in failures:
                    config_file = os.path.basename(result["config_file"])
                    operation = result["operation"]
                    error = result["error"]
                    
                    f.write(f"<tr>")
                    f.write(f"<td>{config_file}</td>")
                    f.write(f"<td>{operation}</td>")
                    f.write(f"<td class='error'>{error}</td>")
                    f.write(f"</tr>\n")
                
                f.write("</table>\n")
            
            f.write("</body>\n</html>")
        
        logger.info(f"Batch processing report generated: {report_path}")
        
        return report_path


# 2. Parameter Sweep
class ParameterSweep:
    """Class for parameter sweep automation."""
    
    def __init__(self, base_config, output_dir="results/automation/sweep"):
        """Initialize the parameter sweep."""
        self.base_config = base_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized parameter sweep with output directory: {output_dir}")
    
    def generate_configs(self, param_grid, prefix="sweep"):
        """Generate configurations for parameter sweep."""
        logger.info(f"Generating configurations for parameter sweep with {len(param_grid)} parameters")
        
        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        logger.info(f"Generated {len(combinations)} parameter combinations")
        
        # Create configurations
        config_files = []
        
        for i, combination in enumerate(combinations):
            # Create a new configuration
            config = ConfigManager()
            config.update(self.base_config.get())
            
            # Set parameter values
            param_dict = {}
            for name, value in zip(param_names, combination):
                config.set(name, value)
                param_dict[name] = value
            
            # Create a descriptive filename
            param_str = "_".join([f"{name.split('.')[-1]}_{value}" for name, value in zip(param_names, combination)])
            filename = f"{prefix}_{param_str}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save configuration
            config.save_config(filepath)
            config_files.append(filepath)
        
        logger.info(f"Generated {len(config_files)} configuration files")
        
        return config_files
    
    def run_sweep(self, operation, parallel=True, max_workers=None):
        """Run parameter sweep."""
        # Get all configuration files in the output directory
        config_files = glob.glob(os.path.join(self.output_dir, "sweep_*.json"))
        
        if not config_files:
            logger.warning("No configuration files found for parameter sweep")
            return []
        
        logger.info(f"Running parameter sweep with {len(config_files)} configurations")
        
        # Create batch processor
        batch_processor = BatchProcessor(output_dir=self.output_dir)
        
        # Process configurations
        results = batch_processor.process_configs(
            config_files=config_files,
            operation=operation,
            parallel=parallel,
            max_workers=max_workers
        )
        
        logger.info(f"Parameter sweep completed with {len(results)} results")
        
        return results
    
    def analyze_sweep_results(self, results, output_prefix="sweep_analysis"):
        """Analyze parameter sweep results."""
        logger.info(f"Analyzing parameter sweep results for {len(results)} configurations")
        
        # Filter successful results
        successes = [r for r in results if r["status"] == "success"]
        
        if not successes:
            logger.warning("No successful results to analyze")
            return {}
        
        # Extract parameters and metrics
        data = []
        
        for result in successes:
            # Load the configuration
            config_file = result["config_file"]
            config = ConfigManager(config_file)
            
            # Extract parameters of interest
            row = {}
            
            # Add parameters from filename
            filename = os.path.basename(config_file)
            param_str = filename.replace("sweep_", "").replace(".json", "")
            
            for param_value in param_str.split("_"):
                match = re.match(r"([a-zA-Z]+)_([\d\.]+)", param_value)
                if match:
                    param, value = match.groups()
                    try:
                        row[param] = float(value)
                    except ValueError:
                        row[param] = value
            
            # Add metrics from result
            if result["operation"] == "solve":
                row["computation_time"] = result["result"].get("computation_time", 0)
            
            elif result["operation"] == "estimate":
                row["computation_time"] = result["result"].get("computation_time", 0)
                row["log_likelihood"] = result["result"].get("diagnostics", {}).get("log_likelihood", 0)
                row["acceptance_rate"] = result["result"].get("diagnostics", {}).get("acceptance_rate", 0)
            
            elif result["operation"] == "forecast":
                row["computation_time"] = result["result"].get("computation_time", 0)
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save data
        csv_file = os.path.join(self.output_dir, f"{output_prefix}.csv")
        df.to_csv(csv_file, index=False)
        
        # Generate plots
        self._generate_sweep_plots(df, output_prefix)
        
        logger.info(f"Parameter sweep analysis completed. Results saved to {csv_file}")
        
        return {
            "data": df,
            "csv_file": csv_file
        }
    
    def _generate_sweep_plots(self, df, output_prefix):
        """Generate plots for parameter sweep analysis."""
        # Identify parameter columns and metric columns
        param_cols = [col for col in df.columns if col not in ["computation_time", "log_likelihood", "acceptance_rate"]]
        metric_cols = [col for col in df.columns if col in ["computation_time", "log_likelihood", "acceptance_rate"]]
        
        if not param_cols or not metric_cols:
            logger.warning("Not enough columns for plotting")
            return
        
        # Create plots for each metric
        for metric in metric_cols:
            plt.figure(figsize=(12, 8))
            
            # If we have one parameter, create a line plot
            if len(param_cols) == 1:
                param = param_cols[0]
                
                # Sort by parameter value
                plot_df = df.sort_values(param)
                
                plt.plot(plot_df[param], plot_df[metric], 'o-')
                plt.xlabel(param)
                plt.ylabel(metric)
                plt.title(f"{metric} vs {param}")
                plt.grid(True)
            
            # If we have two parameters, create a heatmap
            elif len(param_cols) == 2:
                param1, param2 = param_cols
                
                # Create pivot table
                pivot = df.pivot_table(
                    values=metric,
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )
                
                plt.imshow(pivot, cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label=metric)
                
                # Set tick labels
                plt.xticks(range(len(pivot.columns)), pivot.columns)
                plt.yticks(range(len(pivot.index)), pivot.index)
                
                plt.xlabel(param2)
                plt.ylabel(param1)
                plt.title(f"{metric} vs {param1} and {param2}")
            
            # Save plot
            plot_file = os.path.join(self.output_dir, f"{output_prefix}_{metric}.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Generated plot: {plot_file}")


# 3. Scheduled Forecasting
class ScheduledForecaster:
    """Class for scheduled forecasting."""
    
    def __init__(self, config, output_dir="results/automation/forecasts"):
        """Initialize the scheduled forecaster."""
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized scheduled forecaster with output directory: {output_dir}")
    
    def run_forecast(self, periods=40, num_simulations=100):
        """Run a forecast."""
        logger.info(f"Running forecast with {periods} periods and {num_simulations} simulations")
        
        # Create model
        model = SmetsWoutersModel(self.config)
        
        # Generate forecast
        forecast = model.forecast(periods=periods, num_simulations=num_simulations)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast["timestamp"] = timestamp
        
        # Save forecast
        forecast_file = os.path.join(self.output_dir, f"forecast_{timestamp}.json")
        
        with open(forecast_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_forecast = self._make_json_serializable(forecast)
            json.dump(serializable_forecast, f, indent=2)
        
        logger.info(f"Forecast saved to {forecast_file}")
        
        # Generate plots
        self._generate_forecast_plots(forecast, timestamp)
        
        return {
            "forecast": forecast,
            "forecast_file": forecast_file,
            "timestamp": timestamp
        }
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _generate_forecast_plots(self, forecast, timestamp):
        """Generate plots for forecast."""
        # Variables to plot
        variables = list(forecast["baseline"].keys())
        
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, f"plots_{timestamp}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate individual plots
        for var in variables:
            plt.figure(figsize=(10, 6))
            
            # Time axis
            t = np.arange(forecast["periods"])
            
            # Plot baseline
            plt.plot(t, forecast["baseline"][var], 'b-', label="Baseline")
            
            # Plot confidence intervals
            ci = forecast["confidence_intervals"][var]
            plt.fill_between(t, ci["lower_90"], ci["upper_90"], color='b', alpha=0.1, label="90% CI")
            plt.fill_between(t, ci["lower_68"], ci["upper_68"], color='b', alpha=0.2, label="68% CI")
            
            plt.title(f"Forecast: {var}")
            plt.xlabel("Periods")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{var}.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
        
        # Generate combined plot
        plt.figure(figsize=(15, 10))
        
        for i, var in enumerate(variables):
            plt.subplot(3, 2, i + 1)
            
            # Time axis
            t = np.arange(forecast["periods"])
            
            # Plot baseline
            plt.plot(t, forecast["baseline"][var], 'b-', label="Baseline")
            
            # Plot confidence intervals
            ci = forecast["confidence_intervals"][var]
            plt.fill_between(t, ci["lower_90"], ci["upper_90"], color='b', alpha=0.1, label="90% CI")
            plt.fill_between(t, ci["lower_68"], ci["upper_68"], color='b', alpha=0.2, label="68% CI")
            
            plt.title(var)
            plt.xlabel("Periods")
            plt.ylabel("Value")
            
            if i == 0:
                plt.legend()
            
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "combined.png"))
        plt.close()
        
        logger.info(f"Generated forecast plots in {plots_dir}")
    
    def schedule_forecasts(self, interval_hours=24, periods=40, num_simulations=100):
        """Schedule forecasts to run at regular intervals."""
        logger.info(f"Scheduling forecasts to run every {interval_hours} hours")
        
        # Create a mock schedule
        now = datetime.now()
        schedule = []
        
        for i in range(7):  # Schedule for a week
            run_time = now + timedelta(hours=i * interval_hours)
            schedule.append({
                "run_id": i + 1,
                "scheduled_time": run_time.strftime("%Y-%m-%d %H:%M:%S"),
                "periods": periods,
                "num_simulations": num_simulations
            })
        
        # Save schedule
        schedule_file = os.path.join(self.output_dir, "forecast_schedule.json")
        
        with open(schedule_file, 'w') as f:
            json.dump(schedule, f, indent=2)
        
        logger.info(f"Forecast schedule saved to {schedule_file}")
        
        # In a real implementation, you would set up a cron job or Windows Task Scheduler task
        # For demonstration, we'll just print the command that would be scheduled
        script_path = os.path.abspath(__file__)
        command = f"python {script_path} --run-forecast --periods {periods} --simulations {num_simulations}"
        
        logger.info(f"Command to schedule: {command}")
        
        return {
            "schedule": schedule,
            "schedule_file": schedule_file,
            "command": command
        }


# 4. Report Generation
class ReportGenerator:
    """Class for automated report generation."""
    
    def __init__(self, output_dir="results/automation/reports"):
        """Initialize the report generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized report generator with output directory: {output_dir}")
    
    def generate_html_report(self, title, sections, output_file):
        """Generate an HTML report."""
        logger.info(f"Generating HTML report: {output_file}")
        
        # Create report path
        report_path = os.path.join(self.output_dir, output_file)
        
        # Create HTML
        with open(report_path, 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"<title>{title}</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
            f.write("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n")
            f.write("h2 { color: #3498db; margin-top: 30px; }\n")
            f.write("h3 { color: #2980b9; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 10px 0; }\n")
            f.write(".toc { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }\n")
            f.write(".toc ul { list-style-type: none; padding-left: 20px; }\n")
            f.write(".toc li { margin: 5px 0; }\n")
            f.write(".footer { margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px; color: #7f8c8d; font-size: 0.9em; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            
            # Header
            f.write(f"<h1>{title}</h1>\n")
            f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Table of Contents
            f.write("<div class='toc'>\n")
            f.write("<h2>Table of Contents</h2>\n")
            f.write("<ul>\n")
            
            for i, section in enumerate(sections):
                section_id = f"section-{i+1}"
                f.write(f"<li><a href='#{section_id}'>{section['title']}</a></li>\n")
            
            f.write("</ul>\n")
            f.write("</div>\n")
            
            # Sections
            for i, section in enumerate(sections):
                section_id = f"section-{i+1}"
                f.write(f"<h2 id='{section_id}'>{section['title']}</h2>\n")
                
                # Content
                if "text" in section:
                    f.write(f"<p>{section['text']}</p>\n")
                
                # Table
                if "table" in section:
                    table = section["table"]
                    
                    f.write("<table>\n")
                    
                    # Header
                    if "header" in table:
                        f.write("<tr>\n")
                        for cell in table["header"]:
                            f.write(f"<th>{cell}</th>\n")
                        f.write("</tr>\n")
                    
                    # Rows
                    for row in table["rows"]:
                        f.write("<tr>\n")
                        for cell in row:
                            f.write(f"<td>{cell}</td>\n")
                        f.write("</tr>\n")
                    
                    f.write("</table>\n")
                
                # Image
                if "image" in section:
                    image = section["image"]
                    image_path = image["path"]
                    
                    # Copy image to report directory if it's not already there
                    if not os.path.dirname(image_path) == self.output_dir:
                        dest_path = os.path.join(self.output_dir, os.path.basename(image_path))
                        shutil.copy(image_path, dest_path)
                        image_path = os.path.basename(image_path)
                    
                    f.write(f"<img src='{image_path}' alt='{image.get('alt', 'Image')}' ")
                    if "width" in image:
                        f.write(f"width='{image['width']}' ")
                    if "height" in image:
                        f.write(f"height='{image['height']}' ")
                    f.write("/>\n")
            
            # Footer
            f.write("<div class='footer'>\n")
            f.write("<p>Generated by DSGE Model Automation Framework</p>\n")
            f.write("</div>\n")
            
            f.write("</body>\n</html>")
        
        logger.info(f"HTML report generated: {report_path}")
        
        return report_path


# 5. Notification System
class NotificationSystem:
    """Class for sending notifications about automation results."""
    
    def __init__(self, smtp_server=None, smtp_port=None, smtp_username=None, smtp_password=None):
        """Initialize the notification system."""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        
        logger.info("Initialized notification system")
    
    def send_email(self, to_email, subject, body, attachments=None):
        """Send an email notification."""
        if not self.smtp_server:
            logger.warning("SMTP server not configured. Email notification skipped.")
            return False
        
        logger.info(f"Sending email notification to {to_email}")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.smtp_username
            msg["To"] = to_email
            msg["Subject"] = subject
            
            # Add body
            msg.attach(MIMEText(body, "plain"))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    with open(file_path, "rb") as f:
                        attachment = MIMEApplication(f.read(), _subtype="octet-stream")
                        attachment.add_header(
                            "Content-Disposition", "attachment", filename=os.path.basename(file_path)
                        )
                        msg.attach(attachment)
            
            # Connect to server and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info("Email notification sent successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False


def main():
    """Main function demonstrating automation techniques."""
    # Create output directories
    os.makedirs("results/automation", exist_ok=True)
    
    print("=== DSGE Model Automation Example ===")
    
    # 1. Batch Processing
    print("\n1. Batch Processing")
    
    # Create sample configurations
    config_dir = "results/automation/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # Create base configuration
    base_config = ConfigManager()
    base_config.save_config(os.path.join(config_dir, "base_config.json"))
    
    # Create configurations with different extensions
    extensions_config = ConfigManager()
    extensions_config.enable_extension("financial_extension")
    extensions_config.save_config(os.path.join(config_dir, "financial_config.json"))
    
    open_economy_config = ConfigManager()
    open_economy_config.enable_extension("open_economy_extension")
    open_economy_config.save_config(os.path.join(config_dir, "open_economy_config.json"))
    
    fiscal_config = ConfigManager()
    fiscal_config.enable_extension("fiscal_extension")
    fiscal_config.save_config(os.path.join(config_dir, "fiscal_config.json"))
    
    all_extensions_config = ConfigManager()
    all_extensions_config.enable_extension("financial_extension")
    all_extensions_config.enable_extension("open_economy_extension")
    all_extensions_config.enable_extension("fiscal_extension")
    all_extensions_config.save_config(os.path.join(config_dir, "all_extensions_config.json"))
    
    # Create batch processor
    batch_processor = BatchProcessor()
    
    # Get all configuration files
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    
    # Process configurations
    results = batch_processor.process_configs(
        config_files=config_files,
        operation="solve",
        parallel=True
    )
    
    # Generate batch report
    report_path = batch_processor.generate_batch_report(results)
    print(f"Batch processing report generated: {report_path}")
    
    # 2. Parameter Sweep
    print("\n2. Parameter Sweep")
    
    # Create parameter sweep
    sweep = ParameterSweep(base_config)
    
    # Define parameter grid
    param_grid = {
        "base_model.beta": [0.97, 0.98, 0.99],
        "base_model.phi_pi": [1.3, 1.5, 1.7]
    }
    
    # Generate configurations
    sweep_configs = sweep.generate_configs(param_grid)
    
    # Run sweep
    sweep_results = sweep.run_sweep("solve")
    
    # Analyze results
    analysis = sweep.analyze_sweep_results(sweep_results)
    
    # 3. Scheduled Forecasting
    print("\n3. Scheduled Forecasting")
    
    # Create scheduled forecaster
    forecaster = ScheduledForecaster(base_config)
    
    # Run a forecast
    forecast_results = forecaster.run_forecast(periods=20, num_simulations=50)
    
    # Schedule forecasts
    schedule = forecaster.schedule_forecasts(interval_hours=24)
    print(f"Forecast schedule created: {schedule['schedule_file']}")
    print(f"Command to schedule: {schedule['command']}")
    
    # 4. Report Generation
    print("\n4. Report Generation")
    
    # Create report generator
    report_generator = ReportGenerator()
    
    # Generate forecast report
    forecast_report = report_generator.generate_html_report(
        title="DSGE Model Forecast Report",
        sections=[
            {
                "title": "Forecast Overview",
                "text": "This report presents the results of a DSGE model forecast."
            },
            {
                "title": "Forecast Results",
                "text": "The following chart shows the forecast for key macroeconomic variables:",
                "image": {
                    "path": os.path.join(forecaster.output_dir, f"plots_{forecast_results['timestamp']}", "combined.png"),
                    "alt": "Combined Forecast Chart"
                }
            }
        ],
        output_file="forecast_report.html"
    )
    print(f"Forecast report generated: {forecast_report}")
    
    # 5. Notification System
    print("\n5. Notification System")
    
    # Create notification system
    notification = NotificationSystem()
    
    # Simulate sending a notification
    print("Simulating email notification (SMTP server not configured)")
    print("In a real implementation, this would send an email with the forecast report attached")
    
    print("\nAutomation example completed successfully.")


if __name__ == "__main__":
    main()
