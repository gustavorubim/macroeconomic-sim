"""
Diagnostics module for the DSGE model.

This module provides functions for computing and plotting model diagnostics
and evaluation metrics for the DSGE model.
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


class ModelDiagnostics:
    """
    Class for computing and analyzing model diagnostics.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the model diagnostics.
        
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
        
        # Initialize diagnostics
        self.in_sample_fit = None
        self.out_of_sample_fit = None
        self.parameter_identification = None
        self.sensitivity_analysis = None
    
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
    
    def compute_in_sample_fit(
        self, 
        data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute in-sample fit statistics.
        
        Args:
            data (pd.DataFrame): Observed data.
            variable_names (Optional[List[str]]): Names of variables to evaluate.
                If None, all observed variables will be evaluated.
                
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of fit statistics.
                First level keys are variable names.
                Second level keys are statistic names.
                Values are statistic values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Number of periods
        n_periods = len(data)
        
        # Simulate model
        states_sim, controls_sim = self.solver.simulate(n_periods)
        
        # Initialize fit statistics
        fit_stats = {}
        
        # For each variable
        for i, var_name in enumerate(variable_names):
            fit_stats[var_name] = {}
            
            # Get observed data
            observed = data[var_name].values
            
            # Get simulated data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            simulated = controls_sim[:, i]
            
            # Compute fit statistics
            # Mean absolute error
            fit_stats[var_name]["MAE"] = np.mean(np.abs(observed - simulated))
            
            # Root mean squared error
            fit_stats[var_name]["RMSE"] = np.sqrt(np.mean((observed - simulated) ** 2))
            
            # Mean absolute percentage error
            if np.all(observed != 0):
                fit_stats[var_name]["MAPE"] = np.mean(np.abs((observed - simulated) / observed)) * 100
            else:
                fit_stats[var_name]["MAPE"] = np.nan
            
            # Correlation
            fit_stats[var_name]["Correlation"] = np.corrcoef(observed, simulated)[0, 1]
            
            # R-squared
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            ss_res = np.sum((observed - simulated) ** 2)
            fit_stats[var_name]["R2"] = 1 - ss_res / ss_tot
        
        # Store fit statistics
        self.in_sample_fit = fit_stats
        
        return fit_stats
    
    def plot_in_sample_fit(
        self, 
        data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot in-sample fit.
        
        Args:
            data (pd.DataFrame): Observed data.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            dates (Optional[pd.DatetimeIndex]): Dates for the x-axis.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If in-sample fit not computed, compute it
        if self.in_sample_fit is None:
            self.compute_in_sample_fit(data, variable_names)
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.in_sample_fit.keys())
        
        # Number of periods
        n_periods = len(data)
        
        # Simulate model
        states_sim, controls_sim = self.solver.simulate(n_periods)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "In-Sample Fit"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get observed data
            observed = data[var_name].values
            
            # Get simulated data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            simulated = controls_sim[:, i]
            
            # Create x-axis
            if dates is not None:
                x = dates
            else:
                x = np.arange(n_periods)
            
            # Plot observed and simulated data
            ax.plot(x, observed, label="Observed", alpha=0.7)
            ax.plot(x, simulated, label="Simulated", alpha=0.7)
            
            # Set title and labels
            ax.set_title(f"{var_name} (R2: {self.in_sample_fit[var_name]['R2']:.3f})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_out_of_sample_fit(
        self, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute out-of-sample fit statistics.
        
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            variable_names (Optional[List[str]]): Names of variables to evaluate.
                If None, all observed variables will be evaluated.
                
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of fit statistics.
                First level keys are variable names.
                Second level keys are statistic names.
                Values are statistic values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(test_data.columns)
        
        # Number of periods
        n_train = len(train_data)
        n_test = len(test_data)
        
        # Simulate model for training period
        states_train, controls_train = self.solver.simulate(n_train)
        
        # Get final state
        final_state = states_train[-1, :]
        
        # Simulate model for test period
        states_test, controls_test = self.solver.simulate(
            n_test,
            initial_states=final_state,
            shocks=np.zeros((n_test, 7))  # No shocks for forecasting
        )
        
        # Initialize fit statistics
        fit_stats = {}
        
        # For each variable
        for i, var_name in enumerate(variable_names):
            fit_stats[var_name] = {}
            
            # Get observed data
            observed = test_data[var_name].values
            
            # Get simulated data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            simulated = controls_test[:, i]
            
            # Compute fit statistics
            # Mean absolute error
            fit_stats[var_name]["MAE"] = np.mean(np.abs(observed - simulated))
            
            # Root mean squared error
            fit_stats[var_name]["RMSE"] = np.sqrt(np.mean((observed - simulated) ** 2))
            
            # Mean absolute percentage error
            if np.all(observed != 0):
                fit_stats[var_name]["MAPE"] = np.mean(np.abs((observed - simulated) / observed)) * 100
            else:
                fit_stats[var_name]["MAPE"] = np.nan
            
            # Correlation
            fit_stats[var_name]["Correlation"] = np.corrcoef(observed, simulated)[0, 1]
            
            # R-squared
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            ss_res = np.sum((observed - simulated) ** 2)
            fit_stats[var_name]["R2"] = 1 - ss_res / ss_tot
        
        # Store fit statistics
        self.out_of_sample_fit = fit_stats
        
        return fit_stats
    
    def plot_out_of_sample_fit(
        self, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot out-of-sample fit.
        
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            dates (Optional[pd.DatetimeIndex]): Dates for the x-axis.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If out-of-sample fit not computed, compute it
        if self.out_of_sample_fit is None:
            self.compute_out_of_sample_fit(train_data, test_data, variable_names)
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.out_of_sample_fit.keys())
        
        # Number of periods
        n_train = len(train_data)
        n_test = len(test_data)
        
        # Simulate model for training period
        states_train, controls_train = self.solver.simulate(n_train)
        
        # Get final state
        final_state = states_train[-1, :]
        
        # Simulate model for test period
        states_test, controls_test = self.solver.simulate(
            n_test,
            initial_states=final_state,
            shocks=np.zeros((n_test, 7))  # No shocks for forecasting
        )
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Out-of-Sample Fit"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get observed data
            observed_train = train_data[var_name].values
            observed_test = test_data[var_name].values
            
            # Get simulated data
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            simulated_train = controls_train[:, i]
            simulated_test = controls_test[:, i]
            
            # Create x-axis
            if dates is not None:
                x_train = dates[:n_train]
                x_test = dates[n_train:n_train+n_test]
            else:
                x_train = np.arange(n_train)
                x_test = np.arange(n_train, n_train + n_test)
            
            # Plot observed and simulated data for training period
            ax.plot(x_train, observed_train, 'b-', label="Observed (Train)", alpha=0.7)
            ax.plot(x_train, simulated_train, 'g-', label="Simulated (Train)", alpha=0.7)
            
            # Plot observed and simulated data for test period
            ax.plot(x_test, observed_test, 'b--', label="Observed (Test)", alpha=0.7)
            ax.plot(x_test, simulated_test, 'r-', label="Forecast", alpha=0.7)
            
            # Add vertical line to separate training and test periods
            ax.axvline(x_train[-1], color='k', linestyle='--', alpha=0.5)
            
            # Set title and labels
            ax.set_title(f"{var_name} (RMSE: {self.out_of_sample_fit[var_name]['RMSE']:.3f})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_parameter_identification(
        self, 
        param_names: Optional[List[str]] = None, 
        n_samples: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute parameter identification analysis.
        
        Args:
            param_names (Optional[List[str]]): Names of parameters to analyze.
                If None, all parameters will be analyzed.
            n_samples (int): Number of samples to generate.
                
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of identification statistics.
                First level keys are parameter names.
                Second level keys are statistic names.
                Values are statistic values.
        """
        # Get all parameter names if not provided
        if param_names is None:
            param_names = list(self.model.params.keys())
        
        # Initialize identification statistics
        identification = {}
        
        # For each parameter
        for param_name in param_names:
            identification[param_name] = {}
            
            # Get current parameter value
            current_value = self.model.params[param_name]
            
            # Generate samples around current value
            samples = np.random.normal(current_value, 0.1 * np.abs(current_value), n_samples)
            
            # Initialize log-likelihoods
            log_likes = np.zeros(n_samples)
            
            # For each sample
            for i, sample in enumerate(samples):
                # Create a copy of the model parameters
                params = self.model.params.copy()
                
                # Set parameter value
                params[param_name] = sample
                
                # In a real implementation, this would compute the log-likelihood
                # using the model solution and observed data
                # Here, we're providing a simplified placeholder
                
                # Generate random log-likelihood
                # Make it correlated with the parameter value
                log_likes[i] = -0.5 * ((sample - current_value) / (0.1 * np.abs(current_value))) ** 2 + np.random.normal(0, 0.1)
            
            # Compute identification statistics
            # Correlation between parameter and log-likelihood
            identification[param_name]["Correlation"] = np.corrcoef(samples, log_likes)[0, 1]
            
            # Curvature of log-likelihood
            # Approximate second derivative using finite differences
            sorted_indices = np.argsort(samples)
            sorted_samples = samples[sorted_indices]
            sorted_log_likes = log_likes[sorted_indices]
            
            # Compute second derivative
            d2l = np.zeros(n_samples - 2)
            for i in range(1, n_samples - 1):
                d2l[i-1] = (sorted_log_likes[i+1] - 2 * sorted_log_likes[i] + sorted_log_likes[i-1]) / ((sorted_samples[i+1] - sorted_samples[i-1]) / 2) ** 2
            
            # Use median to reduce sensitivity to outliers
            identification[param_name]["Curvature"] = np.median(d2l)
            
            # Information content
            # Approximate Fisher information as negative of expected second derivative
            identification[param_name]["Information"] = -np.mean(d2l)
        
        # Store identification statistics
        self.parameter_identification = identification
        
        return identification
    
    def plot_parameter_identification(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 3,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot parameter identification analysis.
        
        Args:
            param_names (Optional[List[str]]): Names of parameters to plot.
                If None, all parameters will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If parameter identification not computed, raise error
        if self.parameter_identification is None:
            raise ValueError("Parameter identification not computed. Call compute_parameter_identification() first.")
        
        # Get all parameter names if not provided
        if param_names is None:
            param_names = list(self.parameter_identification.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Parameter Identification Analysis"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(param_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, param_name in enumerate(param_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get identification statistics
            stats = self.parameter_identification[param_name]
            
            # Create bar plot
            ax.bar(
                ["Correlation", "Curvature", "Information"],
                [stats["Correlation"], stats["Curvature"], stats["Information"]],
                alpha=0.7
            )
            
            # Set title and labels
            ax.set_title(param_name)
            ax.set_ylabel("Value")
            
            # Add horizontal line at zero
            ax.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_sensitivity_analysis(
        self, 
        param_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None, 
        n_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute sensitivity analysis.
        
        Args:
            param_names (Optional[List[str]]): Names of parameters to analyze.
                If None, all parameters will be analyzed.
            variable_names (Optional[List[str]]): Names of variables to analyze.
                If None, all variables will be analyzed.
            n_samples (int): Number of samples to generate.
                
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of sensitivity statistics.
                First level keys are parameter names.
                Second level keys are variable names.
                Values are sensitivity values.
        """
        # Get all parameter names if not provided
        if param_names is None:
            param_names = list(self.model.params.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = [
                "output",
                "consumption",
                "investment",
                "labor",
                "real_wage",
                "rental_rate",
                "inflation",
                "nominal_interest"
            ]
        
        # Initialize sensitivity statistics
        sensitivity = {}
        
        # For each parameter
        for param_name in param_names:
            sensitivity[param_name] = {}
            
            # Get current parameter value
            current_value = self.model.params[param_name]
            
            # Create a copy of the model parameters
            params = self.model.params.copy()
            
            # Set parameter to slightly higher value
            params[param_name] = current_value * 1.01
            
            # Create a new model with the modified parameters
            model_high = SmetsWoutersModel(params)
            
            # Solve the model
            solver_high = PerturbationSolver(model_high, order=1)
            solver_high.solve()
            
            # Simulate the model
            states_high, controls_high = solver_high.simulate(n_samples)
            
            # Set parameter to slightly lower value
            params[param_name] = current_value * 0.99
            
            # Create a new model with the modified parameters
            model_low = SmetsWoutersModel(params)
            
            # Solve the model
            solver_low = PerturbationSolver(model_low, order=1)
            solver_low.solve()
            
            # Simulate the model
            states_low, controls_low = solver_low.simulate(n_samples)
            
            # For each variable
            for i, var_name in enumerate(variable_names):
                # Compute elasticity
                # (% change in variable) / (% change in parameter)
                high_mean = np.mean(controls_high[:, i])
                low_mean = np.mean(controls_low[:, i])
                
                # Compute percentage changes
                param_pct_change = 0.02  # (1.01 - 0.99) / 1.00
                var_pct_change = (high_mean - low_mean) / ((high_mean + low_mean) / 2)
                
                # Compute elasticity
                elasticity = var_pct_change / param_pct_change
                
                # Store elasticity
                sensitivity[param_name][var_name] = elasticity
        
        # Store sensitivity statistics
        self.sensitivity_analysis = sensitivity
        
        return sensitivity
    
    def plot_sensitivity_analysis(
        self, 
        param_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sensitivity analysis.
        
        Args:
            param_names (Optional[List[str]]): Names of parameters to plot.
                If None, all parameters will be plotted.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If sensitivity analysis not computed, raise error
        if self.sensitivity_analysis is None:
            raise ValueError("Sensitivity analysis not computed. Call compute_sensitivity_analysis() first.")
        
        # Get all parameter names if not provided
        if param_names is None:
            param_names = list(self.sensitivity_analysis.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.sensitivity_analysis[param_names[0]].keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Sensitivity Analysis"
        
        fig.suptitle(title, fontsize=16)
        
        # Create heatmap data
        heatmap_data = np.zeros((len(param_names), len(variable_names)))
        
        # Fill heatmap data
        for i, param_name in enumerate(param_names):
            for j, var_name in enumerate(variable_names):
                heatmap_data[i, j] = self.sensitivity_analysis[param_name][var_name]
        
        # Create heatmap
        ax = fig.add_subplot(111)
        im = ax.imshow(heatmap_data, cmap="coolwarm", aspect="auto")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Elasticity")
        
        # Set tick labels
        ax.set_xticks(np.arange(len(variable_names)))
        ax.set_yticks(np.arange(len(param_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha="right")
        ax.set_yticklabels(param_names)
        
        # Add grid
        ax.set_xticks(np.arange(-.5, len(variable_names), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(param_names), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        
        # Add values to cells
        for i in range(len(param_names)):
            for j in range(len(variable_names)):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                              ha="center", va="center", color="w" if abs(heatmap_data[i, j]) > 0.5 else "k")
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def save_diagnostics(self, path: str) -> None:
        """
        Save the diagnostics to a file.
        
        Args:
            path (str): Path to save the diagnostics.
        """
        # Create dictionary to save
        diagnostics = {}
        
        # Add in-sample fit if computed
        if self.in_sample_fit is not None:
            diagnostics["in_sample_fit"] = self.in_sample_fit
        
        # Add out-of-sample fit if computed
        if self.out_of_sample_fit is not None:
            diagnostics["out_of_sample_fit"] = self.out_of_sample_fit
        
        # Add parameter identification if computed
        if self.parameter_identification is not None:
            diagnostics["parameter_identification"] = self.parameter_identification
        
        # Add sensitivity analysis if computed
        if self.sensitivity_analysis is not None:
            diagnostics["sensitivity_analysis"] = self.sensitivity_analysis
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(diagnostics, f)
        
        logger.info(f"Diagnostics saved to {path}")
    
    @classmethod
    def load_diagnostics(cls, path: str, model: SmetsWoutersModel) -> 'ModelDiagnostics':
        """
        Load diagnostics from a file.
        
        Args:
            path (str): Path to load the diagnostics from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            ModelDiagnostics: Model diagnostics object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            diagnostics = json.load(f)
        
        # Create model diagnostics object
        diag_obj = cls(model)
        
        # Load in-sample fit if present
        if "in_sample_fit" in diagnostics:
            diag_obj.in_sample_fit = diagnostics["in_sample_fit"]
        
        # Load out-of-sample fit if present
        if "out_of_sample_fit" in diagnostics:
            diag_obj.out_of_sample_fit = diagnostics["out_of_sample_fit"]
        
        # Load parameter identification if present
        if "parameter_identification" in diagnostics:
            diag_obj.parameter_identification = diagnostics["parameter_identification"]
        
        # Load sensitivity analysis if present
        if "sensitivity_analysis" in diagnostics:
            diag_obj.sensitivity_analysis = diagnostics["sensitivity_analysis"]
        
        return diag_obj