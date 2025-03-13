"""
Baseline forecasting module for the DSGE model.

This module provides functions for generating baseline forecasts
from the DSGE model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
import logging

from config.config_manager import ConfigManager
from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.perturbation import PerturbationSolver
from dsge.solution.projection import ProjectionSolver

# Set up logging
logger = logging.getLogger(__name__)


class BaselineForecaster:
    """
    Class for generating baseline forecasts from the DSGE model.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the baseline forecaster.
        
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
        
        # Initialize forecasts
        self.forecasts = None
    
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
    
    def generate_forecast(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate baseline forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts.
                Keys are variable names.
                Values are arrays of forecast values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of historical periods
        n_hist = len(data)
        
        # Simulate model for historical period
        states_hist, controls_hist = self.solver.simulate(n_hist)
        
        # Get final state
        final_state = states_hist[-1, :]
        
        # Simulate model for forecast period
        states_fore, controls_fore = self.solver.simulate(
            forecast_periods,
            initial_states=final_state,
            shocks=np.zeros((forecast_periods, 7))  # No shocks for baseline forecast
        )
        
        # Initialize forecasts
        forecasts = {}
        
        # For each variable
        for i, var_name in enumerate(variable_names):
            # Get historical data
            hist = data[var_name].values
            
            # Get forecast data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            fore = controls_fore[:, i]
            
            # Store forecast
            forecasts[var_name] = fore
        
        # Store forecasts
        self.forecasts = forecasts
        
        return forecasts
    
    def plot_forecast(
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
        Plot baseline forecast.
        
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
        # If forecasts not generated, raise error
        if self.forecasts is None:
            raise ValueError("Forecasts not generated. Call generate_forecast() first.")
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.forecasts.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Baseline Forecast"
        
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
            
            # Get forecast data
            fore = self.forecasts[var_name]
            
            # Create x-axis for historical data
            if dates is not None:
                x_hist = dates
            else:
                x_hist = np.arange(len(hist))
            
            # Create x-axis for forecast data
            if forecast_dates is not None:
                x_fore = forecast_dates
            else:
                x_fore = np.arange(len(hist), len(hist) + len(fore))
            
            # Plot historical data
            ax.plot(x_hist, hist, 'b-', label="Historical", alpha=0.7)
            
            # Plot forecast data
            ax.plot(x_fore, fore, 'r-', label="Forecast", alpha=0.7)
            
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
    
    def generate_conditional_forecast(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        conditions: Dict[str, List[float]], 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate conditional forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            conditions (Dict[str, List[float]]): Conditions for the forecast.
                Keys are variable names.
                Values are lists of values for each forecast period.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts.
                Keys are variable names.
                Values are arrays of forecast values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of historical periods
        n_hist = len(data)
        
        # Simulate model for historical period
        states_hist, controls_hist = self.solver.simulate(n_hist)
        
        # Get final state
        final_state = states_hist[-1, :]
        
        # In a real implementation, this would compute the shocks needed
        # to satisfy the conditions using the model solution
        # Here, we're providing a simplified placeholder
        
        # Initialize forecasts
        forecasts = {}
        
        # For each variable
        for i, var_name in enumerate(variable_names):
            # Get historical data
            hist = data[var_name].values
            
            # If variable is conditioned
            if var_name in conditions:
                # Use the condition as the forecast
                fore = np.array(conditions[var_name])
            else:
                # Generate random forecast
                # In a real implementation, this would be computed using the model solution
                fore = np.random.normal(hist[-1], 0.1, forecast_periods)
            
            # Store forecast
            forecasts[var_name] = fore
        
        # Store forecasts
        self.forecasts = forecasts
        
        return forecasts
    
    def compute_forecast_statistics(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        n_simulations: int = 1000, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute forecast statistics.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            n_simulations (int): Number of simulations to run.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
                
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of forecast statistics.
                First level keys are variable names.
                Second level keys are statistic names.
                Values are arrays of statistic values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of historical periods
        n_hist = len(data)
        
        # Simulate model for historical period
        states_hist, controls_hist = self.solver.simulate(n_hist)
        
        # Get final state
        final_state = states_hist[-1, :]
        
        # Initialize forecast simulations
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
                simulations[var_name][i, :] = controls_fore[:, j]
        
        # Compute statistics
        statistics = {}
        for var_name in variable_names:
            statistics[var_name] = {}
            
            # Mean
            statistics[var_name]["mean"] = np.mean(simulations[var_name], axis=0)
            
            # Standard deviation
            statistics[var_name]["std"] = np.std(simulations[var_name], axis=0)
            
            # Percentiles
            for p in [5, 10, 25, 50, 75, 90, 95]:
                statistics[var_name][f"p{p}"] = np.percentile(simulations[var_name], p, axis=0)
        
        return statistics
    
    def plot_forecast_fan_chart(
        self, 
        data: pd.DataFrame, 
        statistics: Dict[str, Dict[str, np.ndarray]], 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot forecast fan chart.
        
        Args:
            data (pd.DataFrame): Historical data.
            statistics (Dict[str, Dict[str, np.ndarray]]): Forecast statistics.
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
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(statistics.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Forecast Fan Chart"
        
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
            stats = statistics[var_name]
            
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
    
    def save_forecasts(self, path: str) -> None:
        """
        Save the forecasts to a file.
        
        Args:
            path (str): Path to save the forecasts.
        """
        # If forecasts not generated, raise error
        if self.forecasts is None:
            raise ValueError("Forecasts not generated. Call generate_forecast() first.")
        
        # Convert forecasts to a format suitable for saving
        forecasts_dict = {}
        for var_name, forecast in self.forecasts.items():
            forecasts_dict[var_name] = forecast.tolist()
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(forecasts_dict, f)
        
        logger.info(f"Forecasts saved to {path}")
    
    @classmethod
    def load_forecasts(cls, path: str, model: SmetsWoutersModel) -> 'BaselineForecaster':
        """
        Load forecasts from a file.
        
        Args:
            path (str): Path to load the forecasts from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            BaselineForecaster: Baseline forecaster object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            forecasts_dict = json.load(f)
        
        # Convert forecasts back to numpy arrays
        forecasts = {}
        for var_name, forecast in forecasts_dict.items():
            forecasts[var_name] = np.array(forecast)
        
        # Create baseline forecaster object
        forecaster = cls(model)
        forecaster.forecasts = forecasts
        
        return forecaster