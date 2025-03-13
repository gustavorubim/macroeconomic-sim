"""
Uncertainty module for the DSGE model.

This module provides functions for quantifying and visualizing forecast uncertainty
in the DSGE model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from config.config_manager import ConfigManager
from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.perturbation import PerturbationSolver
from dsge.solution.projection import ProjectionSolver

# Set up logging
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Class for quantifying and visualizing forecast uncertainty.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the uncertainty quantifier.
        
        Args:
            model (SmetsWoutersModel): The DSGE model.
            config (Optional[Union[Dict[str, Any], ConfigManager]]): Configuration.
                If a dictionary is provided, it will be used as the configuration.
                If a ConfigManager is provided, its configuration will be used.
                If None, the default configuration will be used.
        """
        self.model = model
        
        # Initialize configuration
        if config is None:
            self.config = ConfigManager()
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.update(config)
        else:
            self.config = config
        
        # Initialize solver
        self.solver = None
        
        # Initialize simulations
        self.simulations = None
        
        # Initialize statistics
        self.statistics = None
    
    def solve_model(self) -> None:
        """
        Solve the model using the specified solution method.
        """
        # Get solution method from configuration
        solution_method = self.config.get("solution.method")
        
        # Solve the model
        if solution_method == "perturbation":
            perturbation_order = self.config.get("solution.perturbation_order")
            self.solver = PerturbationSolver(self.model, order=perturbation_order)
        else:  # projection
            projection_method = self.config.get("solution.projection_method")
            projection_nodes = self.config.get("solution.projection_nodes")
            self.solver = ProjectionSolver(self.model, method=projection_method, nodes=projection_nodes)
        
        # Solve the model
        self.solver.solve()
    
    def generate_simulations(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        n_simulations: int = 1000, 
        variable_names: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Monte Carlo simulations for forecast uncertainty.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            n_simulations (int): Number of simulations to run.
            variable_names (Optional[List[str]]): Names of variables to simulate.
                If None, all variables will be simulated.
            seed (Optional[int]): Random seed for reproducibility.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of simulations.
                Keys are variable names.
                Values are arrays of shape (n_simulations, forecast_periods).
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of historical periods
        n_hist = len(data)
        
        # Simulate model for historical period
        states_hist, controls_hist = self.solver.simulate(n_hist)
        
        # Get final state
        final_state = states_hist[-1, :]
        
        # Initialize simulations
        simulations = {}
        for var_name in variable_names:
            simulations[var_name] = np.zeros((n_simulations, forecast_periods))
        
        # Run simulations
        for i in range(n_simulations):
            # Generate random shocks
            shocks = np.random.normal(0, 1, (forecast_periods, 7))
            
            # Simulate model for forecast period
            states_fore, controls_fore = self.solver.simulate(
                forecast_periods,
                initial_states=final_state,
                shocks=shocks
            )
            
            # Store simulations
            for j, var_name in enumerate(variable_names):
                # In a real implementation, this would map model variables to observed variables
                # Here, we're assuming a one-to-one mapping for simplicity
                simulations[var_name][i, :] = controls_fore[:, j]
        
        # Store simulations
        self.simulations = simulations
        
        return simulations
    
    def compute_statistics(
        self, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics from the simulations.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to compute statistics for.
                If None, statistics for all variables will be computed.
                
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of statistics.
                First level keys are variable names.
                Second level keys are statistic names.
                Values are arrays of statistic values.
        """
        # If simulations not generated, raise error
        if self.simulations is None:
            raise ValueError("Simulations not generated. Call generate_simulations() first.")
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.simulations.keys())
        
        # Initialize statistics
        statistics = {}
        
        # For each variable
        for var_name in variable_names:
            statistics[var_name] = {}
            
            # Get simulations for this variable
            sims = self.simulations[var_name]
            
            # Compute mean
            statistics[var_name]["mean"] = np.mean(sims, axis=0)
            
            # Compute standard deviation
            statistics[var_name]["std"] = np.std(sims, axis=0)
            
            # Compute percentiles
            for p in [5, 10, 25, 50, 75, 90, 95]:
                statistics[var_name][f"p{p}"] = np.percentile(sims, p, axis=0)
        
        # Store statistics
        self.statistics = statistics
        
        return statistics
    
    def plot_fan_chart(
        self, 
        data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot fan chart for forecast uncertainty.
        
        Args:
            data (pd.DataFrame): Historical data.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            dates (Optional[pd.DatetimeIndex]): Dates for the historical data.
            forecast_dates (Optional[pd.DatetimeIndex]): Dates for the forecast.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If statistics not computed, compute them
        if self.statistics is None:
            self.compute_statistics()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.statistics.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Forecast Uncertainty"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get historical data
            hist = data[var_name].values
            
            # Get forecast statistics
            stats = self.statistics[var_name]
            
            # Create x-axis for historical data
            if dates is not None:
                x_hist = dates
            else:
                x_hist = np.arange(len(hist))
            
            # Create x-axis for forecast data
            if forecast_dates is not None:
                x_fore = forecast_dates
            else:
                x_fore = np.arange(len(hist), len(hist) + len(stats["mean"]))
            
            # Plot historical data
            ax.plot(x_hist, hist, 'b-', label="Historical", alpha=0.7)
            
            # Plot forecast mean
            ax.plot(x_fore, stats["mean"], 'r-', label="Mean Forecast", alpha=0.7)
            
            # Plot forecast percentiles
            ax.fill_between(x_fore, stats["p5"], stats["p95"], color='r', alpha=0.1, label="90% CI")
            ax.fill_between(x_fore, stats["p10"], stats["p90"], color='r', alpha=0.2)
            ax.fill_between(x_fore, stats["p25"], stats["p75"], color='r', alpha=0.3)
            
            # Add vertical line to separate historical and forecast periods
            ax.axvline(x_hist[-1], color='k', linestyle='--', alpha=0.5)
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_distribution(
        self, 
        variable_names: Optional[List[str]] = None, 
        periods: Optional[List[int]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of forecast values.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            periods (Optional[List[int]]): Forecast periods to plot.
                If None, periods [0, 4, 8] will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If simulations not generated, raise error
        if self.simulations is None:
            raise ValueError("Simulations not generated. Call generate_simulations() first.")
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.simulations.keys())
        
        # Set default periods if not provided
        if periods is None:
            # Get number of forecast periods
            n_periods = self.simulations[variable_names[0]].shape[1]
            
            # Set periods to [0, 1/3, 2/3] of forecast horizon
            periods = [0, n_periods // 3, 2 * n_periods // 3]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Forecast Distribution"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names) * len(periods)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        plot_idx = 1
        for var_name in variable_names:
            for period in periods:
                # Create subplot
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                plot_idx += 1
                
                # Get simulations for this variable and period
                values = self.simulations[var_name][:, period]
                
                # Plot distribution
                sns.histplot(values, kde=True, ax=ax)
                
                # Add vertical line for mean
                ax.axvline(np.mean(values), color='r', linestyle='--', alpha=0.7, label="Mean")
                
                # Add vertical lines for percentiles
                ax.axvline(np.percentile(values, 5), color='g', linestyle=':', alpha=0.7, label="5th/95th Percentile")
                ax.axvline(np.percentile(values, 95), color='g', linestyle=':', alpha=0.7)
                
                # Set title and labels
                ax.set_title(f"{var_name} (Period {period})")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                
                # Add legend to the first subplot
                if plot_idx == 2:
                    ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_forecast_error_bands(
        self, 
        variable_names: Optional[List[str]] = None, 
        confidence_level: float = 0.9
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute forecast error bands.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to compute error bands for.
                If None, error bands for all variables will be computed.
            confidence_level (float): Confidence level for the error bands.
                
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary of error bands.
                Keys are variable names.
                Values are tuples of (lower_band, upper_band).
        """
        # If statistics not computed, compute them
        if self.statistics is None:
            self.compute_statistics()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.statistics.keys())
        
        # Compute alpha
        alpha = 1 - confidence_level
        
        # Initialize error bands
        error_bands = {}
        
        # For each variable
        for var_name in variable_names:
            # Get mean and standard deviation
            mean = self.statistics[var_name]["mean"]
            std = self.statistics[var_name]["std"]
            
            # Compute error bands
            z = stats.norm.ppf(1 - alpha / 2)
            lower_band = mean - z * std
            upper_band = mean + z * std
            
            # Store error bands
            error_bands[var_name] = (lower_band, upper_band)
        
        return error_bands
    
    def plot_error_bands(
        self, 
        data: pd.DataFrame, 
        error_bands: Dict[str, Tuple[np.ndarray, np.ndarray]], 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot forecast error bands.
        
        Args:
            data (pd.DataFrame): Historical data.
            error_bands (Dict[str, Tuple[np.ndarray, np.ndarray]]): Dictionary of error bands.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            dates (Optional[pd.DatetimeIndex]): Dates for the historical data.
            forecast_dates (Optional[pd.DatetimeIndex]): Dates for the forecast.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If statistics not computed, compute them
        if self.statistics is None:
            self.compute_statistics()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(error_bands.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Forecast Error Bands"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get historical data
            hist = data[var_name].values
            
            # Get forecast mean
            mean = self.statistics[var_name]["mean"]
            
            # Get error bands
            lower_band, upper_band = error_bands[var_name]
            
            # Create x-axis for historical data
            if dates is not None:
                x_hist = dates
            else:
                x_hist = np.arange(len(hist))
            
            # Create x-axis for forecast data
            if forecast_dates is not None:
                x_fore = forecast_dates
            else:
                x_fore = np.arange(len(hist), len(hist) + len(mean))
            
            # Plot historical data
            ax.plot(x_hist, hist, 'b-', label="Historical", alpha=0.7)
            
            # Plot forecast mean
            ax.plot(x_fore, mean, 'r-', label="Mean Forecast", alpha=0.7)
            
            # Plot error bands
            ax.fill_between(x_fore, lower_band, upper_band, color='r', alpha=0.2, label="Error Band")
            
            # Add vertical line to separate historical and forecast periods
            ax.axvline(x_hist[-1], color='k', linestyle='--', alpha=0.5)
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_parameter_uncertainty(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        param_samples: Dict[str, np.ndarray], 
        variable_names: Optional[List[str]] = None,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute forecast uncertainty due to parameter uncertainty.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            param_samples (Dict[str, np.ndarray]): Dictionary of parameter samples.
                Keys are parameter names.
                Values are arrays of parameter samples.
            variable_names (Optional[List[str]]): Names of variables to compute uncertainty for.
                If None, uncertainty for all variables will be computed.
            n_samples (int): Number of parameter samples to use.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of simulations.
                Keys are variable names.
                Values are arrays of shape (n_samples, forecast_periods).
        """
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of historical periods
        n_hist = len(data)
        
        # Initialize simulations
        simulations = {}
        for var_name in variable_names:
            simulations[var_name] = np.zeros((n_samples, forecast_periods))
        
        # Get parameter names
        param_names = list(param_samples.keys())
        
        # Run simulations
        for i in range(n_samples):
            # Create a copy of the model parameters
            params = self.model.params.copy()
            
            # Set parameter values
            for param_name in param_names:
                # Get parameter sample
                if i < len(param_samples[param_name]):
                    params[param_name] = param_samples[param_name][i]
            
            # Create a new model with the modified parameters
            model = SmetsWoutersModel(params)
            
            # Solve the model
            solver = PerturbationSolver(model, order=1)
            solver.solve()
            
            # Simulate model for historical period
            states_hist, controls_hist = solver.simulate(n_hist)
            
            # Get final state
            final_state = states_hist[-1, :]
            
            # Simulate model for forecast period
            states_fore, controls_fore = solver.simulate(
                forecast_periods,
                initial_states=final_state,
                shocks=np.zeros((forecast_periods, 7))  # No shocks for parameter uncertainty
            )
            
            # Store simulations
            for j, var_name in enumerate(variable_names):
                # In a real implementation, this would map model variables to observed variables
                # Here, we're assuming a one-to-one mapping for simplicity
                simulations[var_name][i, :] = controls_fore[:, j]
        
        return simulations
    
    def save_simulations(self, path: str) -> None:
        """
        Save the simulations to a file.
        
        Args:
            path (str): Path to save the simulations.
        """
        # If simulations not generated, raise error
        if self.simulations is None:
            raise ValueError("Simulations not generated. Call generate_simulations() first.")
        
        # Convert simulations to a format suitable for saving
        simulations_dict = {}
        for var_name, sims in self.simulations.items():
            simulations_dict[var_name] = sims.tolist()
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(simulations_dict, f)
        
        logger.info(f"Simulations saved to {path}")
    
    @classmethod
    def load_simulations(cls, path: str, model: SmetsWoutersModel) -> 'UncertaintyQuantifier':
        """
        Load simulations from a file.
        
        Args:
            path (str): Path to load the simulations from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            UncertaintyQuantifier: Uncertainty quantifier object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            simulations_dict = json.load(f)
        
        # Convert simulations back to numpy arrays
        simulations = {}
        for var_name, sims in simulations_dict.items():
            simulations[var_name] = np.array(sims)
        
        # Create uncertainty quantifier object
        quantifier = cls(model)
        quantifier.simulations = simulations
        
        return quantifier