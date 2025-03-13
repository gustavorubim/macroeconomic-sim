"""
Decomposition module for the DSGE model.

This module provides functions for computing and plotting historical shock
decomposition and variance decomposition for the DSGE model.
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


class ShockDecomposition:
    """
    Class for computing and analyzing shock decompositions.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the shock decomposition.
        
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
        
        # Initialize decompositions
        self.historical_decomposition = None
        self.variance_decomposition = None
    
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
    
    def compute_historical_decomposition(
        self, 
        data: pd.DataFrame, 
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute historical shock decomposition.
        
        Args:
            data (pd.DataFrame): Observed data.
            variable_names (Optional[List[str]]): Names of variables to decompose.
                If None, all observed variables will be decomposed.
                
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of historical decompositions.
                First level keys are variable names.
                Second level keys are shock names.
                Values are arrays of contributions.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Get all shock names
        shock_names = [
            "technology",
            "preference",
            "investment",
            "government",
            "monetary",
            "price_markup",
            "wage_markup"
        ]
        
        # Number of periods
        n_periods = len(data)
        
        # Initialize historical decomposition
        historical_decomposition = {}
        
        # For each variable
        for var_name in variable_names:
            historical_decomposition[var_name] = {}
            
            # Initialize contributions
            for shock_name in shock_names:
                historical_decomposition[var_name][shock_name] = np.zeros(n_periods)
            
            # Add initial conditions
            historical_decomposition[var_name]["initial"] = np.zeros(n_periods)
            
            # Add residual
            historical_decomposition[var_name]["residual"] = np.zeros(n_periods)
        
        # In a real implementation, this would compute the historical decomposition
        # using the Kalman smoother or a similar method
        # Here, we're providing a simplified placeholder
        
        # For demonstration purposes, we'll generate random contributions
        for var_name in variable_names:
            # Get observed data for this variable
            observed = data[var_name].values
            
            # Generate random contributions
            for shock_name in shock_names:
                # Generate random contribution
                contribution = np.random.normal(0, 0.1, n_periods)
                
                # Ensure contributions sum to observed data
                contribution = contribution / np.sum(np.abs(contribution)) * np.sum(np.abs(observed)) * 0.8
                
                # Store contribution
                historical_decomposition[var_name][shock_name] = contribution
            
            # Set initial conditions
            historical_decomposition[var_name]["initial"] = np.ones(n_periods) * observed[0] * 0.1
            
            # Compute residual
            total_contribution = np.sum([
                historical_decomposition[var_name][shock_name]
                for shock_name in shock_names
            ], axis=0) + historical_decomposition[var_name]["initial"]
            
            historical_decomposition[var_name]["residual"] = observed - total_contribution
        
        # Store historical decomposition
        self.historical_decomposition = historical_decomposition
        
        return historical_decomposition
    
    def plot_historical_decomposition(
        self, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot historical shock decomposition.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            dates (Optional[pd.DatetimeIndex]): Dates for the x-axis.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If historical decomposition not computed, raise error
        if self.historical_decomposition is None:
            raise ValueError("Historical decomposition not computed. Call compute_historical_decomposition() first.")
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.historical_decomposition.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Historical Shock Decomposition"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get decomposition for this variable
            decomposition = self.historical_decomposition[var_name]
            
            # Get shock names
            shock_names = list(decomposition.keys())
            
            # Remove residual and initial from shock names
            if "residual" in shock_names:
                shock_names.remove("residual")
            if "initial" in shock_names:
                shock_names.remove("initial")
            
            # Get number of periods
            n_periods = len(decomposition[shock_names[0]])
            
            # Create x-axis
            if dates is not None:
                x = dates
            else:
                x = np.arange(n_periods)
            
            # Create stacked bar plot
            bottom = np.zeros(n_periods)
            for shock_name in shock_names:
                ax.bar(
                    x,
                    decomposition[shock_name],
                    bottom=bottom,
                    label=shock_name,
                    alpha=0.7
                )
                bottom += decomposition[shock_name]
            
            # Add initial conditions
            if "initial" in decomposition:
                ax.bar(
                    x,
                    decomposition["initial"],
                    bottom=bottom,
                    label="Initial",
                    alpha=0.7
                )
                bottom += decomposition["initial"]
            
            # Add residual
            if "residual" in decomposition:
                ax.bar(
                    x,
                    decomposition["residual"],
                    bottom=bottom,
                    label="Residual",
                    alpha=0.7
                )
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Contribution")
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_variance_decomposition(
        self, 
        variable_names: Optional[List[str]] = None, 
        horizons: List[int] = [1, 4, 8, 16, 32, 100]
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Compute variance decomposition.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to decompose.
                If None, all variables will be decomposed.
            horizons (List[int]): Horizons for the variance decomposition.
                
        Returns:
            Dict[str, Dict[int, Dict[str, float]]]: Dictionary of variance decompositions.
                First level keys are variable names.
                Second level keys are horizons.
                Third level keys are shock names.
                Values are contributions (fractions).
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
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
        
        # Get all shock names
        shock_names = [
            "technology",
            "preference",
            "investment",
            "government",
            "monetary",
            "price_markup",
            "wage_markup"
        ]
        
        # Initialize variance decomposition
        variance_decomposition = {}
        
        # For each variable
        for var_name in variable_names:
            variance_decomposition[var_name] = {}
            
            # For each horizon
            for horizon in horizons:
                variance_decomposition[var_name][horizon] = {}
                
                # Initialize contributions
                total_variance = 0.0
                
                # For each shock
                for shock_name in shock_names:
                    # In a real implementation, this would compute the variance decomposition
                    # using the model solution
                    # Here, we're providing a simplified placeholder
                    
                    # Generate random contribution
                    contribution = np.random.uniform(0, 1)
                    
                    # Store contribution
                    variance_decomposition[var_name][horizon][shock_name] = contribution
                    
                    # Add to total variance
                    total_variance += contribution
                
                # Normalize contributions
                for shock_name in shock_names:
                    variance_decomposition[var_name][horizon][shock_name] /= total_variance
        
        # Store variance decomposition
        self.variance_decomposition = variance_decomposition
        
        return variance_decomposition
    
    def plot_variance_decomposition(
        self, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot variance decomposition.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If variance decomposition not computed, raise error
        if self.variance_decomposition is None:
            raise ValueError("Variance decomposition not computed. Call compute_variance_decomposition() first.")
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.variance_decomposition.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Variance Decomposition"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get decomposition for this variable
            decomposition = self.variance_decomposition[var_name]
            
            # Get horizons
            horizons = sorted(list(decomposition.keys()))
            
            # Get shock names
            shock_names = list(decomposition[horizons[0]].keys())
            
            # Create stacked bar plot
            bottom = np.zeros(len(horizons))
            for shock_name in shock_names:
                # Get contributions for this shock
                contributions = [decomposition[horizon][shock_name] for horizon in horizons]
                
                # Plot contributions
                ax.bar(
                    np.arange(len(horizons)),
                    contributions,
                    bottom=bottom,
                    label=shock_name,
                    alpha=0.7
                )
                
                # Update bottom
                bottom += np.array(contributions)
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Contribution")
            
            # Set x-tick labels
            ax.set_xticks(np.arange(len(horizons)))
            ax.set_xticklabels([str(h) for h in horizons])
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_forecast_error_variance_decomposition(
        self, 
        variable_names: Optional[List[str]] = None, 
        horizons: List[int] = [1, 4, 8, 16, 32, 100]
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Compute forecast error variance decomposition.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to decompose.
                If None, all variables will be decomposed.
            horizons (List[int]): Horizons for the variance decomposition.
                
        Returns:
            Dict[str, Dict[int, Dict[str, float]]]: Dictionary of variance decompositions.
                First level keys are variable names.
                Second level keys are horizons.
                Third level keys are shock names.
                Values are contributions (fractions).
        """
        # This is similar to variance decomposition, but focuses on forecast errors
        # In a real implementation, this would compute the forecast error variance decomposition
        # using the model solution
        # Here, we're providing a simplified placeholder that just calls compute_variance_decomposition
        
        return self.compute_variance_decomposition(variable_names, horizons)
    
    def plot_forecast_error_variance_decomposition(
        self, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot forecast error variance decomposition.
        
        Args:
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If variance decomposition not computed, raise error
        if self.variance_decomposition is None:
            raise ValueError("Variance decomposition not computed. Call compute_variance_decomposition() first.")
        
        # Create title
        if title is None:
            title = "Forecast Error Variance Decomposition"
        
        # Call plot_variance_decomposition with the new title
        return self.plot_variance_decomposition(variable_names, figsize, n_cols, title)
    
    def save_decompositions(self, path: str) -> None:
        """
        Save the decompositions to a file.
        
        Args:
            path (str): Path to save the decompositions.
        """
        # Create dictionary to save
        decompositions = {}
        
        # Add historical decomposition if computed
        if self.historical_decomposition is not None:
            decompositions["historical"] = {}
            for var_name, var_decomp in self.historical_decomposition.items():
                decompositions["historical"][var_name] = {}
                for shock_name, contribution in var_decomp.items():
                    decompositions["historical"][var_name][shock_name] = contribution.tolist()
        
        # Add variance decomposition if computed
        if self.variance_decomposition is not None:
            decompositions["variance"] = {}
            for var_name, var_decomp in self.variance_decomposition.items():
                decompositions["variance"][var_name] = {}
                for horizon, horizon_decomp in var_decomp.items():
                    decompositions["variance"][var_name][str(horizon)] = horizon_decomp
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(decompositions, f)
        
        logger.info(f"Decompositions saved to {path}")
    
    @classmethod
    def load_decompositions(cls, path: str, model: SmetsWoutersModel) -> 'ShockDecomposition':
        """
        Load decompositions from a file.
        
        Args:
            path (str): Path to load the decompositions from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            ShockDecomposition: Shock decomposition object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            decompositions = json.load(f)
        
        # Create shock decomposition object
        decomp_obj = cls(model)
        
        # Load historical decomposition if present
        if "historical" in decompositions:
            historical = {}
            for var_name, var_decomp in decompositions["historical"].items():
                historical[var_name] = {}
                for shock_name, contribution in var_decomp.items():
                    historical[var_name][shock_name] = np.array(contribution)
            
            decomp_obj.historical_decomposition = historical
        
        # Load variance decomposition if present
        if "variance" in decompositions:
            variance = {}
            for var_name, var_decomp in decompositions["variance"].items():
                variance[var_name] = {}
                for horizon, horizon_decomp in var_decomp.items():
                    variance[var_name][int(horizon)] = horizon_decomp
            
            decomp_obj.variance_decomposition = variance
        
        return decomp_obj