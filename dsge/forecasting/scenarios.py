"""
Scenarios module for the DSGE model.

This module provides functions for generating alternative scenario forecasts
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
from dsge.forecasting.baseline import BaselineForecaster

# Set up logging
logger = logging.getLogger(__name__)


class ScenarioForecaster:
    """
    Class for generating alternative scenario forecasts from the DSGE model.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the scenario forecaster.
        
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
        
        # Initialize baseline forecaster
        self.baseline_forecaster = BaselineForecaster(model, config)
        
        # Initialize scenarios
        self.scenarios = {}
    
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
    
    def generate_baseline_forecast(
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
        # Generate baseline forecast
        return self.baseline_forecaster.generate_forecast(
            data=data,
            forecast_periods=forecast_periods,
            variable_names=variable_names
        )
    
    def generate_shock_scenario(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        shock_name: str, 
        shock_size: float, 
        shock_periods: Optional[List[int]] = None, 
        variable_names: Optional[List[str]] = None,
        scenario_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate shock scenario forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            shock_name (str): Name of the shock.
            shock_size (float): Size of the shock in standard deviations.
            shock_periods (Optional[List[int]]): Periods in which to apply the shock.
                If None, the shock will be applied in the first period only.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
            scenario_name (Optional[str]): Name of the scenario.
                If None, a name will be generated based on the shock.
                
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
        
        # Set default shock periods if not provided
        if shock_periods is None:
            shock_periods = [0]
        
        # Set default scenario name if not provided
        if scenario_name is None:
            scenario_name = f"{shock_name}_shock_{shock_size}sd"
        
        # Number of historical periods
        n_hist = len(data)
        
        # Simulate model for historical period
        states_hist, controls_hist = self.solver.simulate(n_hist)
        
        # Get final state
        final_state = states_hist[-1, :]
        
        # Get all shock names
        all_shock_names = [
            "technology",
            "preference",
            "investment",
            "government",
            "monetary",
            "price_markup",
            "wage_markup"
        ]
        
        # Get shock index
        shock_idx = all_shock_names.index(shock_name)
        
        # Get shock standard deviation
        shock_std = self.model.params.get(f"{shock_name}_sigma", 0.01)
        
        # Create shock vector
        shocks = np.zeros((forecast_periods, len(all_shock_names)))
        
        # Set shock values
        for period in shock_periods:
            if period < forecast_periods:
                shocks[period, shock_idx] = shock_size * shock_std
        
        # Simulate model for forecast period
        states_fore, controls_fore = self.solver.simulate(
            forecast_periods,
            initial_states=final_state,
            shocks=shocks
        )
        
        # Initialize forecasts
        forecasts = {}
        
        # For each variable
        for i, var_name in enumerate(variable_names):
            # Get forecast data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            fore = controls_fore[:, i]
            
            # Store forecast
            forecasts[var_name] = fore
        
        # Store scenario
        self.scenarios[scenario_name] = forecasts
        
        return forecasts
    
    def generate_parameter_scenario(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        param_name: str, 
        param_value: float, 
        variable_names: Optional[List[str]] = None,
        scenario_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate parameter scenario forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            param_name (str): Name of the parameter.
            param_value (float): Value of the parameter.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
            scenario_name (Optional[str]): Name of the scenario.
                If None, a name will be generated based on the parameter.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts.
                Keys are variable names.
                Values are arrays of forecast values.
        """
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Set default scenario name if not provided
        if scenario_name is None:
            scenario_name = f"{param_name}_{param_value}"
        
        # Create a copy of the model parameters
        params = self.model.params.copy()
        
        # Set parameter value
        params[param_name] = param_value
        
        # Create a new model with the modified parameters
        model = SmetsWoutersModel(params)
        
        # Create a new forecaster with the modified model
        forecaster = BaselineForecaster(model, self.config)
        
        # Generate forecast
        forecasts = forecaster.generate_forecast(
            data=data,
            forecast_periods=forecast_periods,
            variable_names=variable_names
        )
        
        # Store scenario
        self.scenarios[scenario_name] = forecasts
        
        return forecasts
    
    def generate_policy_scenario(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        policy_params: Dict[str, float], 
        variable_names: Optional[List[str]] = None,
        scenario_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate policy scenario forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            policy_params (Dict[str, float]): Policy parameters.
                Keys are parameter names.
                Values are parameter values.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
            scenario_name (Optional[str]): Name of the scenario.
                If None, a name will be generated based on the policy.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts.
                Keys are variable names.
                Values are arrays of forecast values.
        """
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Set default scenario name if not provided
        if scenario_name is None:
            scenario_name = "policy_scenario"
        
        # Create a copy of the model parameters
        params = self.model.params.copy()
        
        # Set policy parameters
        for param_name, param_value in policy_params.items():
            params[param_name] = param_value
        
        # Create a new model with the modified parameters
        model = SmetsWoutersModel(params)
        
        # Create a new forecaster with the modified model
        forecaster = BaselineForecaster(model, self.config)
        
        # Generate forecast
        forecasts = forecaster.generate_forecast(
            data=data,
            forecast_periods=forecast_periods,
            variable_names=variable_names
        )
        
        # Store scenario
        self.scenarios[scenario_name] = forecasts
        
        return forecasts
    
    def generate_custom_scenario(
        self, 
        data: pd.DataFrame, 
        forecast_periods: int, 
        scenario_func: Callable[[SmetsWoutersModel, pd.DataFrame, int], Dict[str, np.ndarray]], 
        variable_names: Optional[List[str]] = None,
        scenario_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate custom scenario forecast.
        
        Args:
            data (pd.DataFrame): Historical data.
            forecast_periods (int): Number of periods to forecast.
            scenario_func (Callable[[SmetsWoutersModel, pd.DataFrame, int], Dict[str, np.ndarray]]):
                Function that generates the scenario forecast.
                Takes the model, historical data, and forecast periods as arguments.
                Returns a dictionary of forecasts.
            variable_names (Optional[List[str]]): Names of variables to forecast.
                If None, all variables will be forecasted.
            scenario_name (Optional[str]): Name of the scenario.
                If None, a name will be generated.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts.
                Keys are variable names.
                Values are arrays of forecast values.
        """
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Set default scenario name if not provided
        if scenario_name is None:
            scenario_name = "custom_scenario"
        
        # Generate forecast using the custom function
        forecasts = scenario_func(self.model, data, forecast_periods)
        
        # Filter forecasts to include only the requested variables
        filtered_forecasts = {}
        for var_name in variable_names:
            if var_name in forecasts:
                filtered_forecasts[var_name] = forecasts[var_name]
        
        # Store scenario
        self.scenarios[scenario_name] = filtered_forecasts
        
        return filtered_forecasts
    
    def plot_scenarios(
        self, 
        data: pd.DataFrame, 
        scenario_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot scenario forecasts.
        
        Args:
            data (pd.DataFrame): Historical data.
            scenario_names (Optional[List[str]]): Names of scenarios to plot.
                If None, all scenarios will be plotted.
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
        # If no scenarios, raise error
        if not self.scenarios:
            raise ValueError("No scenarios generated. Generate scenarios first.")
        
        # Get all scenario names if not provided
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            # Get variables that are in all scenarios
            var_sets = [set(self.scenarios[name].keys()) for name in scenario_names]
            variable_names = list(set.intersection(*var_sets))
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Scenario Forecasts"
        
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
            
            # Create x-axis for historical data
            if dates is not None:
                x_hist = dates
            else:
                x_hist = np.arange(len(hist))
            
            # Plot historical data
            ax.plot(x_hist, hist, 'k-', label="Historical", alpha=0.7)
            
            # Plot each scenario
            for scenario_name in scenario_names:
                # Get forecast data
                fore = self.scenarios[scenario_name][var_name]
                
                # Create x-axis for forecast data
                if forecast_dates is not None:
                    x_fore = forecast_dates
                else:
                    x_fore = np.arange(len(hist), len(hist) + len(fore))
                
                # Plot forecast data
                ax.plot(x_fore, fore, label=scenario_name, alpha=0.7)
            
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
    
    def compute_scenario_differences(
        self, 
        baseline_name: str, 
        scenario_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute differences between scenarios and baseline.
        
        Args:
            baseline_name (str): Name of the baseline scenario.
            scenario_names (Optional[List[str]]): Names of scenarios to compare.
                If None, all scenarios except the baseline will be compared.
            variable_names (Optional[List[str]]): Names of variables to compare.
                If None, all variables will be compared.
                
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of differences.
                First level keys are scenario names.
                Second level keys are variable names.
                Values are arrays of differences.
        """
        # If no scenarios, raise error
        if not self.scenarios:
            raise ValueError("No scenarios generated. Generate scenarios first.")
        
        # If baseline not found, raise error
        if baseline_name not in self.scenarios:
            raise ValueError(f"Baseline scenario '{baseline_name}' not found.")
        
        # Get all scenario names if not provided
        if scenario_names is None:
            scenario_names = [name for name in self.scenarios.keys() if name != baseline_name]
        
        # Get all variable names if not provided
        if variable_names is None:
            # Get variables that are in all scenarios
            var_sets = [set(self.scenarios[name].keys()) for name in [baseline_name] + scenario_names]
            variable_names = list(set.intersection(*var_sets))
        
        # Initialize differences
        differences = {}
        
        # For each scenario
        for scenario_name in scenario_names:
            differences[scenario_name] = {}
            
            # For each variable
            for var_name in variable_names:
                # Get baseline and scenario forecasts
                baseline = self.scenarios[baseline_name][var_name]
                scenario = self.scenarios[scenario_name][var_name]
                
                # Compute difference
                diff = scenario - baseline
                
                # Store difference
                differences[scenario_name][var_name] = diff
        
        return differences
    
    def plot_scenario_differences(
        self, 
        differences: Dict[str, Dict[str, np.ndarray]], 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot differences between scenarios and baseline.
        
        Args:
            differences (Dict[str, Dict[str, np.ndarray]]): Dictionary of differences.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            forecast_dates (Optional[pd.DatetimeIndex]): Dates for the forecast.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get all scenario names
        scenario_names = list(differences.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            # Get variables that are in all scenarios
            var_sets = [set(differences[name].keys()) for name in scenario_names]
            variable_names = list(set.intersection(*var_sets))
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Scenario Differences"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Plot each scenario difference
            for scenario_name in scenario_names:
                # Get difference data
                diff = differences[scenario_name][var_name]
                
                # Create x-axis for difference data
                if forecast_dates is not None:
                    x_diff = forecast_dates
                else:
                    x_diff = np.arange(len(diff))
                
                # Plot difference data
                ax.plot(x_diff, diff, label=scenario_name, alpha=0.7)
            
            # Add horizontal line at zero
            ax.axhline(0, color='k', linestyle='-', alpha=0.2)
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Difference")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def save_scenarios(self, path: str) -> None:
        """
        Save the scenarios to a file.
        
        Args:
            path (str): Path to save the scenarios.
        """
        # If no scenarios, raise error
        if not self.scenarios:
            raise ValueError("No scenarios generated. Generate scenarios first.")
        
        # Convert scenarios to a format suitable for saving
        scenarios_dict = {}
        for scenario_name, forecasts in self.scenarios.items():
            scenarios_dict[scenario_name] = {}
            for var_name, forecast in forecasts.items():
                scenarios_dict[scenario_name][var_name] = forecast.tolist()
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(scenarios_dict, f)
        
        logger.info(f"Scenarios saved to {path}")
    
    @classmethod
    def load_scenarios(cls, path: str, model: SmetsWoutersModel) -> 'ScenarioForecaster':
        """
        Load scenarios from a file.
        
        Args:
            path (str): Path to load the scenarios from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            ScenarioForecaster: Scenario forecaster object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            scenarios_dict = json.load(f)
        
        # Convert scenarios back to numpy arrays
        scenarios = {}
        for scenario_name, forecasts in scenarios_dict.items():
            scenarios[scenario_name] = {}
            for var_name, forecast in forecasts.items():
                scenarios[scenario_name][var_name] = np.array(forecast)
        
        # Create scenario forecaster object
        forecaster = cls(model)
        forecaster.scenarios = scenarios
        
        return forecaster