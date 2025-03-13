"""
Impulse response functions module for the DSGE model.

This module provides functions for computing and plotting impulse response
functions for the DSGE model.
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


class ImpulseResponseFunctions:
    """
    Class for computing and analyzing impulse response functions.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the impulse response functions.
        
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
        
        # Initialize IRFs
        self.irfs = None
    
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
    
    def compute_irfs(
        self, 
        shock_names: Optional[List[str]] = None, 
        periods: int = 40, 
        shock_size: float = 1.0
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute impulse response functions.
        
        Args:
            shock_names (Optional[List[str]]): Names of shocks to compute IRFs for.
                If None, IRFs for all shocks will be computed.
            periods (int): Number of periods for the IRFs.
            shock_size (float): Size of the shock in standard deviations.
                
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of IRFs.
                First level keys are shock names.
                Second level keys are variable names.
                Values are arrays of IRF values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all shock names if not provided
        if shock_names is None:
            shock_names = [
                "technology",
                "preference",
                "investment",
                "government",
                "monetary",
                "price_markup",
                "wage_markup"
            ]
        
        # Initialize IRFs
        irfs = {}
        
        # Compute IRFs for each shock
        for shock_name in shock_names:
            # Get shock standard deviation
            shock_std = self.model.params.get(f"{shock_name}_sigma", 0.01)
            
            # Create shock vector
            shocks = np.zeros((periods, len(shock_names)))
            shock_idx = shock_names.index(shock_name)
            shocks[0, shock_idx] = shock_size * shock_std
            
            # Simulate model with the shock
            states_sim, controls_sim = self.solver.simulate(
                periods=periods,
                initial_states=np.zeros(len(shock_names) + 1),  # +1 for capital
                shocks=shocks
            )
            
            # Store IRFs
            irfs[shock_name] = {
                # States
                "capital": states_sim[:, 0],
                
                # Shocks
                "technology_shock": states_sim[:, 1],
                "preference_shock": states_sim[:, 2],
                "investment_shock": states_sim[:, 3],
                "government_shock": states_sim[:, 4],
                "monetary_shock": states_sim[:, 5],
                "price_markup_shock": states_sim[:, 6],
                "wage_markup_shock": states_sim[:, 7],
                
                # Controls
                "output": controls_sim[:, 0],
                "consumption": controls_sim[:, 1],
                "investment": controls_sim[:, 2],
                "labor": controls_sim[:, 3],
                "real_wage": controls_sim[:, 4],
                "rental_rate": controls_sim[:, 5],
                "inflation": controls_sim[:, 6],
                "nominal_interest": controls_sim[:, 7],
            }
        
        # Store IRFs
        self.irfs = irfs
        
        return irfs
    
    def plot_irfs(
        self, 
        shock_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 3,
        title: Optional[str] = None,
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None
    ) -> plt.Figure:
        """
        Plot impulse response functions.
        
        Args:
            shock_names (Optional[List[str]]): Names of shocks to plot.
                If None, all shocks will be plotted.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
            confidence_intervals (Optional[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]): 
                Confidence intervals for the IRFs.
                First level keys are shock names.
                Second level keys are variable names.
                Values are tuples of (lower, upper) arrays.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # If IRFs not computed, compute them
        if self.irfs is None:
            self.compute_irfs()
        
        # Get all shock names if not provided
        if shock_names is None:
            shock_names = list(self.irfs.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(self.irfs[shock_names[0]].keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Impulse Response Functions"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Plot IRFs for each shock
            for shock_name in shock_names:
                # Get IRF
                irf = self.irfs[shock_name][var_name]
                
                # Plot IRF
                ax.plot(irf, label=shock_name)
                
                # Plot confidence intervals if provided
                if (confidence_intervals is not None and
                    shock_name in confidence_intervals and
                    var_name in confidence_intervals[shock_name]):
                    lower, upper = confidence_intervals[shock_name][var_name]
                    ax.fill_between(
                        np.arange(len(irf)),
                        lower,
                        upper,
                        alpha=0.2
                    )
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Periods")
            ax.set_ylabel("Deviation from SS")
            
            # Add horizontal line at zero
            ax.axhline(0, color='k', linestyle='-', alpha=0.2)
            
            # Add legend to the first subplot
            if i == 0:
                ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_conditional_irfs(
        self, 
        shock_names: List[str], 
        shock_values: List[float], 
        periods: int = 40
    ) -> Dict[str, np.ndarray]:
        """
        Compute conditional impulse response functions.
        
        Args:
            shock_names (List[str]): Names of shocks to condition on.
            shock_values (List[float]): Values of the shocks.
            periods (int): Number of periods for the IRFs.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of conditional IRFs.
                Keys are variable names.
                Values are arrays of IRF values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
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
        
        # Create shock vector
        shocks = np.zeros((periods, len(all_shock_names)))
        
        # Set shock values
        for shock_name, shock_value in zip(shock_names, shock_values):
            shock_idx = all_shock_names.index(shock_name)
            shocks[0, shock_idx] = shock_value
        
        # Simulate model with the shocks
        states_sim, controls_sim = self.solver.simulate(
            periods=periods,
            initial_states=np.zeros(len(all_shock_names) + 1),  # +1 for capital
            shocks=shocks
        )
        
        # Store conditional IRFs
        conditional_irfs = {
            # States
            "capital": states_sim[:, 0],
            
            # Shocks
            "technology_shock": states_sim[:, 1],
            "preference_shock": states_sim[:, 2],
            "investment_shock": states_sim[:, 3],
            "government_shock": states_sim[:, 4],
            "monetary_shock": states_sim[:, 5],
            "price_markup_shock": states_sim[:, 6],
            "wage_markup_shock": states_sim[:, 7],
            
            # Controls
            "output": controls_sim[:, 0],
            "consumption": controls_sim[:, 1],
            "investment": controls_sim[:, 2],
            "labor": controls_sim[:, 3],
            "real_wage": controls_sim[:, 4],
            "rental_rate": controls_sim[:, 5],
            "inflation": controls_sim[:, 6],
            "nominal_interest": controls_sim[:, 7],
        }
        
        return conditional_irfs
    
    def plot_conditional_irfs(
        self, 
        conditional_irfs: Dict[str, np.ndarray], 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 3,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot conditional impulse response functions.
        
        Args:
            conditional_irfs (Dict[str, np.ndarray]): Conditional IRFs.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(conditional_irfs.keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Conditional Impulse Response Functions"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        for i, var_name in enumerate(variable_names):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Get IRF
            irf = conditional_irfs[var_name]
            
            # Plot IRF
            ax.plot(irf)
            
            # Set title and labels
            ax.set_title(var_name)
            ax.set_xlabel("Periods")
            ax.set_ylabel("Deviation from SS")
            
            # Add horizontal line at zero
            ax.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def compute_nonlinear_irfs(
        self, 
        shock_names: Optional[List[str]] = None, 
        periods: int = 40, 
        shock_sizes: List[float] = [0.5, 1.0, 2.0]
    ) -> Dict[str, Dict[str, Dict[float, np.ndarray]]]:
        """
        Compute nonlinear impulse response functions.
        
        Args:
            shock_names (Optional[List[str]]): Names of shocks to compute IRFs for.
                If None, IRFs for all shocks will be computed.
            periods (int): Number of periods for the IRFs.
            shock_sizes (List[float]): Sizes of the shocks in standard deviations.
                
        Returns:
            Dict[str, Dict[str, Dict[float, np.ndarray]]]: Dictionary of nonlinear IRFs.
                First level keys are shock names.
                Second level keys are variable names.
                Third level keys are shock sizes.
                Values are arrays of IRF values.
        """
        # If model not solved, solve it
        if self.solver is None:
            self.solve_model()
        
        # Get all shock names if not provided
        if shock_names is None:
            shock_names = [
                "technology",
                "preference",
                "investment",
                "government",
                "monetary",
                "price_markup",
                "wage_markup"
            ]
        
        # Initialize nonlinear IRFs
        nonlinear_irfs = {}
        
        # Compute nonlinear IRFs for each shock
        for shock_name in shock_names:
            nonlinear_irfs[shock_name] = {}
            
            # Get shock standard deviation
            shock_std = self.model.params.get(f"{shock_name}_sigma", 0.01)
            
            # Compute IRFs for each shock size
            for shock_size in shock_sizes:
                # Create shock vector
                shocks = np.zeros((periods, len(shock_names)))
                shock_idx = shock_names.index(shock_name)
                shocks[0, shock_idx] = shock_size * shock_std
                
                # Simulate model with the shock
                states_sim, controls_sim = self.solver.simulate(
                    periods=periods,
                    initial_states=np.zeros(len(shock_names) + 1),  # +1 for capital
                    shocks=shocks
                )
                
                # Store IRFs for each variable
                for var_idx, var_name in enumerate([
                    "output", "consumption", "investment", "labor",
                    "real_wage", "rental_rate", "inflation", "nominal_interest"
                ]):
                    if var_name not in nonlinear_irfs[shock_name]:
                        nonlinear_irfs[shock_name][var_name] = {}
                    
                    nonlinear_irfs[shock_name][var_name][shock_size] = controls_sim[:, var_idx]
        
        return nonlinear_irfs
    
    def plot_nonlinear_irfs(
        self, 
        nonlinear_irfs: Dict[str, Dict[str, Dict[float, np.ndarray]]], 
        shock_names: Optional[List[str]] = None, 
        variable_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 3,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot nonlinear impulse response functions.
        
        Args:
            nonlinear_irfs (Dict[str, Dict[str, Dict[float, np.ndarray]]]): Nonlinear IRFs.
            shock_names (Optional[List[str]]): Names of shocks to plot.
                If None, all shocks will be plotted.
            variable_names (Optional[List[str]]): Names of variables to plot.
                If None, all variables will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            title (Optional[str]): Title for the figure.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get all shock names if not provided
        if shock_names is None:
            shock_names = list(nonlinear_irfs.keys())
        
        # Get all variable names if not provided
        if variable_names is None:
            variable_names = list(nonlinear_irfs[shock_names[0]].keys())
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create title
        if title is None:
            title = "Nonlinear Impulse Response Functions"
        
        fig.suptitle(title, fontsize=16)
        
        # Number of subplots
        n_plots = len(variable_names) * len(shock_names)
        
        # Compute number of rows
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        plot_idx = 1
        for shock_name in shock_names:
            for var_name in variable_names:
                # Create subplot
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                plot_idx += 1
                
                # Plot IRFs for each shock size
                for shock_size, irf in nonlinear_irfs[shock_name][var_name].items():
                    ax.plot(irf, label=f"Size: {shock_size}")
                
                # Set title and labels
                ax.set_title(f"{shock_name} -> {var_name}")
                ax.set_xlabel("Periods")
                ax.set_ylabel("Deviation from SS")
                
                # Add horizontal line at zero
                ax.axhline(0, color='k', linestyle='-', alpha=0.2)
                
                # Add legend to the first subplot
                if plot_idx == 2:
                    ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def save_irfs(self, path: str) -> None:
        """
        Save the impulse response functions to a file.
        
        Args:
            path (str): Path to save the IRFs.
        """
        # If IRFs not computed, compute them
        if self.irfs is None:
            self.compute_irfs()
        
        # Convert IRFs to a format suitable for saving
        irfs_dict = {}
        for shock_name, shock_irfs in self.irfs.items():
            irfs_dict[shock_name] = {}
            for var_name, irf in shock_irfs.items():
                irfs_dict[shock_name][var_name] = irf.tolist()
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(irfs_dict, f)
        
        logger.info(f"IRFs saved to {path}")
    
    @classmethod
    def load_irfs(cls, path: str, model: SmetsWoutersModel) -> 'ImpulseResponseFunctions':
        """
        Load impulse response functions from a file.
        
        Args:
            path (str): Path to load the IRFs from.
            model (SmetsWoutersModel): The DSGE model.
            
        Returns:
            ImpulseResponseFunctions: Impulse response functions object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            irfs_dict = json.load(f)
        
        # Convert IRFs back to numpy arrays
        irfs = {}
        for shock_name, shock_irfs in irfs_dict.items():
            irfs[shock_name] = {}
            for var_name, irf in shock_irfs.items():
                irfs[shock_name][var_name] = np.array(irf)
        
        # Create impulse response functions object
        irf_obj = cls(model)
        irf_obj.irfs = irfs
        
        return irf_obj