"""
Prior distributions module for the DSGE model.

This module provides functions for defining and managing prior distributions
for the parameters of the DSGE model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PriorDistribution:
    """
    Class for representing prior distributions.
    """
    
    def __init__(
        self, 
        distribution: str, 
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a prior distribution.
        
        Args:
            distribution (str): Type of distribution.
                Options: "normal", "beta", "gamma", "inverse_gamma", "uniform".
            params (Dict[str, float]): Parameters of the distribution.
                For normal: {"mean": float, "std": float}
                For beta: {"alpha": float, "beta": float}
                For gamma: {"alpha": float, "beta": float}
                For inverse_gamma: {"alpha": float, "beta": float}
                For uniform: {"min": float, "max": float}
            bounds (Optional[Tuple[float, float]]): Bounds for the parameter.
                If None, default bounds will be used based on the distribution.
            description (Optional[str]): Description of the parameter.
                
        Raises:
            ValueError: If the distribution is not valid.
        """
        # Check distribution
        valid_distributions = ["normal", "beta", "gamma", "inverse_gamma", "uniform"]
        if distribution not in valid_distributions:
            raise ValueError(f"Invalid distribution: {distribution}. "
                            f"Must be one of: {', '.join(valid_distributions)}")
        
        self.distribution = distribution
        self.params = params
        self.description = description
        
        # Set bounds if not provided
        if bounds is None:
            if distribution == "normal":
                mean, std = params["mean"], params["std"]
                self.bounds = (mean - 4 * std, mean + 4 * std)
            elif distribution == "beta":
                self.bounds = (0.0, 1.0)
            elif distribution == "gamma":
                self.bounds = (0.0, float('inf'))
            elif distribution == "inverse_gamma":
                self.bounds = (0.0, float('inf'))
            elif distribution == "uniform":
                self.bounds = (params["min"], params["max"])
        else:
            self.bounds = bounds
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the probability density function (PDF) at x.
        
        Args:
            x (Union[float, np.ndarray]): Point(s) at which to evaluate the PDF.
            
        Returns:
            Union[float, np.ndarray]: PDF value(s) at x.
        """
        # Check if x is within bounds
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.bounds[0]) & (x <= self.bounds[1])
            x_valid = x[mask]
            
            # Compute PDF based on distribution
            if self.distribution == "normal":
                mean, std = self.params["mean"], self.params["std"]
                result[mask] = stats.norm.pdf(x_valid, loc=mean, scale=std)
            
            elif self.distribution == "beta":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.beta.pdf(x_valid, a=alpha, b=beta)
            
            elif self.distribution == "gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.gamma.pdf(x_valid, a=alpha, scale=1/beta)
            
            elif self.distribution == "inverse_gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.invgamma.pdf(x_valid, a=alpha, scale=beta)
            
            elif self.distribution == "uniform":
                min_val, max_val = self.params["min"], self.params["max"]
                result[mask] = stats.uniform.pdf(x_valid, loc=min_val, scale=max_val - min_val)
            
            return result
        
        else:
            # Check if x is within bounds
            if x < self.bounds[0] or x > self.bounds[1]:
                return 0.0
            
            # Compute PDF based on distribution
            if self.distribution == "normal":
                mean, std = self.params["mean"], self.params["std"]
                return stats.norm.pdf(x, loc=mean, scale=std)
            
            elif self.distribution == "beta":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.beta.pdf(x, a=alpha, b=beta)
            
            elif self.distribution == "gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.gamma.pdf(x, a=alpha, scale=1/beta)
            
            elif self.distribution == "inverse_gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.invgamma.pdf(x, a=alpha, scale=beta)
            
            elif self.distribution == "uniform":
                min_val, max_val = self.params["min"], self.params["max"]
                return stats.uniform.pdf(x, loc=min_val, scale=max_val - min_val)
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the log probability density function at x.
        
        Args:
            x (Union[float, np.ndarray]): Point(s) at which to evaluate the log PDF.
            
        Returns:
            Union[float, np.ndarray]: Log PDF value(s) at x.
        """
        # Check if x is within bounds
        if isinstance(x, np.ndarray):
            result = np.full_like(x, -float('inf'), dtype=float)
            mask = (x >= self.bounds[0]) & (x <= self.bounds[1])
            x_valid = x[mask]
            
            # Compute log PDF based on distribution
            if self.distribution == "normal":
                mean, std = self.params["mean"], self.params["std"]
                result[mask] = stats.norm.logpdf(x_valid, loc=mean, scale=std)
            
            elif self.distribution == "beta":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.beta.logpdf(x_valid, a=alpha, b=beta)
            
            elif self.distribution == "gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.gamma.logpdf(x_valid, a=alpha, scale=1/beta)
            
            elif self.distribution == "inverse_gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                result[mask] = stats.invgamma.logpdf(x_valid, a=alpha, scale=beta)
            
            elif self.distribution == "uniform":
                min_val, max_val = self.params["min"], self.params["max"]
                result[mask] = stats.uniform.logpdf(x_valid, loc=min_val, scale=max_val - min_val)
            
            return result
        
        else:
            # Check if x is within bounds
            if x < self.bounds[0] or x > self.bounds[1]:
                return -float('inf')
            
            # Compute log PDF based on distribution
            if self.distribution == "normal":
                mean, std = self.params["mean"], self.params["std"]
                return stats.norm.logpdf(x, loc=mean, scale=std)
            
            elif self.distribution == "beta":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.beta.logpdf(x, a=alpha, b=beta)
            
            elif self.distribution == "gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.gamma.logpdf(x, a=alpha, scale=1/beta)
            
            elif self.distribution == "inverse_gamma":
                alpha, beta = self.params["alpha"], self.params["beta"]
                return stats.invgamma.logpdf(x, a=alpha, scale=beta)
            
            elif self.distribution == "uniform":
                min_val, max_val = self.params["min"], self.params["max"]
                return stats.uniform.logpdf(x, loc=min_val, scale=max_val - min_val)
    
    def random(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Generate random samples from the prior distribution.
        
        Args:
            size (Optional[int]): Number of samples to generate.
                If None, a single sample will be returned.
                
        Returns:
            Union[float, np.ndarray]: Random sample(s) from the prior distribution.
        """
        # Generate random samples based on distribution
        if self.distribution == "normal":
            mean, std = self.params["mean"], self.params["std"]
            samples = stats.truncnorm.rvs(
                (self.bounds[0] - mean) / std,
                (self.bounds[1] - mean) / std,
                loc=mean,
                scale=std,
                size=size
            )
        
        elif self.distribution == "beta":
            alpha, beta = self.params["alpha"], self.params["beta"]
            samples = stats.beta.rvs(a=alpha, b=beta, size=size)
        
        elif self.distribution == "gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            samples = stats.gamma.rvs(a=alpha, scale=1/beta, size=size)
        
        elif self.distribution == "inverse_gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            samples = stats.invgamma.rvs(a=alpha, scale=beta, size=size)
        
        elif self.distribution == "uniform":
            min_val, max_val = self.params["min"], self.params["max"]
            samples = stats.uniform.rvs(loc=min_val, scale=max_val - min_val, size=size)
        
        return samples
    
    def mean(self) -> float:
        """
        Compute the mean of the prior distribution.
        
        Returns:
            float: Mean of the prior distribution.
        """
        if self.distribution == "normal":
            return self.params["mean"]
        
        elif self.distribution == "beta":
            alpha, beta = self.params["alpha"], self.params["beta"]
            return alpha / (alpha + beta)
        
        elif self.distribution == "gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            return alpha / beta
        
        elif self.distribution == "inverse_gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            if alpha > 1:
                return beta / (alpha - 1)
            else:
                return float('inf')
        
        elif self.distribution == "uniform":
            min_val, max_val = self.params["min"], self.params["max"]
            return (min_val + max_val) / 2
    
    def mode(self) -> float:
        """
        Compute the mode of the prior distribution.
        
        Returns:
            float: Mode of the prior distribution.
        """
        if self.distribution == "normal":
            return self.params["mean"]
        
        elif self.distribution == "beta":
            alpha, beta = self.params["alpha"], self.params["beta"]
            if alpha > 1 and beta > 1:
                return (alpha - 1) / (alpha + beta - 2)
            elif alpha < 1 and beta > 1:
                return 0.0
            elif alpha > 1 and beta < 1:
                return 1.0
            else:
                # Both alpha and beta less than 1, or one of them equals 1
                # Mode is not unique or at the boundary
                return np.nan
        
        elif self.distribution == "gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            if alpha >= 1:
                return (alpha - 1) / beta
            else:
                return 0.0
        
        elif self.distribution == "inverse_gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            return beta / (alpha + 1)
        
        elif self.distribution == "uniform":
            # Mode is not unique for uniform distribution
            return np.nan
    
    def variance(self) -> float:
        """
        Compute the variance of the prior distribution.
        
        Returns:
            float: Variance of the prior distribution.
        """
        if self.distribution == "normal":
            return self.params["std"] ** 2
        
        elif self.distribution == "beta":
            alpha, beta = self.params["alpha"], self.params["beta"]
            return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        
        elif self.distribution == "gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            return alpha / (beta ** 2)
        
        elif self.distribution == "inverse_gamma":
            alpha, beta = self.params["alpha"], self.params["beta"]
            if alpha > 2:
                return (beta ** 2) / ((alpha - 1) ** 2 * (alpha - 2))
            else:
                return float('inf')
        
        elif self.distribution == "uniform":
            min_val, max_val = self.params["min"], self.params["max"]
            return (max_val - min_val) ** 2 / 12
    
    def plot(
        self, 
        ax: Optional[plt.Axes] = None, 
        n_points: int = 1000, 
        **kwargs
    ) -> plt.Axes:
        """
        Plot the prior distribution.
        
        Args:
            ax (Optional[plt.Axes]): Matplotlib axes to plot on.
                If None, a new figure and axes will be created.
            n_points (int): Number of points to use for plotting.
            **kwargs: Additional keyword arguments to pass to plt.plot().
                
        Returns:
            plt.Axes: Matplotlib axes with the plot.
        """
        # Create figure and axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values
        if self.distribution == "normal":
            mean, std = self.params["mean"], self.params["std"]
            x_min = max(self.bounds[0], mean - 4 * std)
            x_max = min(self.bounds[1], mean + 4 * std)
        elif self.distribution in ["beta", "uniform"]:
            x_min, x_max = self.bounds
        else:  # gamma, inverse_gamma
            x_min = self.bounds[0]
            x_max = min(self.bounds[1], self.mean() + 4 * np.sqrt(self.variance()))
        
        x = np.linspace(x_min, x_max, n_points)
        
        # Compute PDF values
        y = self.pdf(x)
        
        # Plot
        ax.plot(x, y, **kwargs)
        
        # Add vertical lines for mean and mode
        mean_val = self.mean()
        mode_val = self.mode()
        
        if np.isfinite(mean_val) and x_min <= mean_val <= x_max:
            ax.axvline(mean_val, color='r', linestyle='--', alpha=0.5, label='Mean')
        
        if not np.isnan(mode_val) and np.isfinite(mode_val) and x_min <= mode_val <= x_max:
            ax.axvline(mode_val, color='g', linestyle=':', alpha=0.5, label='Mode')
        
        # Add labels and legend
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        return ax


class PriorSet:
    """
    Class for managing a set of prior distributions.
    """
    
    def __init__(self):
        """
        Initialize an empty set of prior distributions.
        """
        self.priors = {}
    
    def add_prior(
        self, 
        param_name: str, 
        distribution: str, 
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a prior distribution for a parameter.
        
        Args:
            param_name (str): Name of the parameter.
            distribution (str): Type of distribution.
                Options: "normal", "beta", "gamma", "inverse_gamma", "uniform".
            params (Dict[str, float]): Parameters of the distribution.
            bounds (Optional[Tuple[float, float]]): Bounds for the parameter.
            description (Optional[str]): Description of the parameter.
        """
        self.priors[param_name] = PriorDistribution(
            distribution=distribution,
            params=params,
            bounds=bounds,
            description=description
        )
    
    def get_prior(self, param_name: str) -> PriorDistribution:
        """
        Get the prior distribution for a parameter.
        
        Args:
            param_name (str): Name of the parameter.
            
        Returns:
            PriorDistribution: Prior distribution for the parameter.
            
        Raises:
            KeyError: If the parameter is not found.
        """
        if param_name not in self.priors:
            raise KeyError(f"Prior not found for parameter: {param_name}")
        
        return self.priors[param_name]
    
    def log_prior(self, params: Dict[str, float]) -> float:
        """
        Compute the log prior density for a set of parameters.
        
        Args:
            params (Dict[str, float]): Dictionary of parameter values.
            
        Returns:
            float: Log prior density.
        """
        log_prior = 0.0
        
        # Sum log prior densities for each parameter
        for param_name, value in params.items():
            if param_name in self.priors:
                log_prior += self.priors[param_name].log_pdf(value)
            else:
                # If no prior is specified, use a uniform prior
                log_prior += 0.0  # log(1) = 0
        
        return log_prior
    
    def random_draw(self) -> Dict[str, float]:
        """
        Generate a random draw from the prior distributions.
        
        Returns:
            Dict[str, float]: Dictionary of parameter values.
        """
        params = {}
        
        # Generate random values for each parameter
        for param_name, prior in self.priors.items():
            params[param_name] = prior.random()
        
        return params
    
    def plot_priors(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 3
    ) -> plt.Figure:
        """
        Plot the prior distributions.
        
        Args:
            param_names (Optional[List[str]]): List of parameter names to plot.
                If None, all priors will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get parameter names to plot
        if param_names is None:
            param_names = list(self.priors.keys())
        
        # Number of parameters
        n_params = len(param_names)
        
        # Compute number of rows
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes if needed
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Plot each prior
        for i, param_name in enumerate(param_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # Get prior
            prior = self.priors[param_name]
            
            # Plot prior
            prior.plot(ax=ax, label=param_name)
            
            # Set title
            ax.set_title(param_name)
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the prior set to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with prior information.
        """
        # Initialize data
        data = {
            "Parameter": [],
            "Distribution": [],
            "Mean": [],
            "Mode": [],
            "Std": [],
            "Lower Bound": [],
            "Upper Bound": [],
            "Description": []
        }
        
        # Add each prior
        for param_name, prior in self.priors.items():
            data["Parameter"].append(param_name)
            data["Distribution"].append(prior.distribution)
            data["Mean"].append(prior.mean())
            data["Mode"].append(prior.mode())
            data["Std"].append(np.sqrt(prior.variance()))
            data["Lower Bound"].append(prior.bounds[0])
            data["Upper Bound"].append(prior.bounds[1])
            data["Description"].append(prior.description)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df


def create_default_priors() -> PriorSet:
    """
    Create a set of default priors for the Smets and Wouters model.
    
    Returns:
        PriorSet: Set of default priors.
    """
    # Create prior set
    priors = PriorSet()
    
    # Household parameters
    priors.add_prior(
        param_name="beta",
        distribution="beta",
        params={"alpha": 99, "beta": 1},
        bounds=(0.9, 0.999),
        description="Discount factor"
    )
    
    priors.add_prior(
        param_name="sigma_c",
        distribution="normal",
        params={"mean": 1.5, "std": 0.375},
        bounds=(0.25, 3),
        description="Intertemporal elasticity of substitution"
    )
    
    priors.add_prior(
        param_name="h",
        distribution="beta",
        params={"alpha": 7, "beta": 3},
        bounds=(0.1, 0.95),
        description="Habit persistence"
    )
    
    priors.add_prior(
        param_name="sigma_l",
        distribution="normal",
        params={"mean": 2.0, "std": 0.75},
        bounds=(0.25, 5),
        description="Labor supply elasticity"
    )
    
    # Production parameters
    priors.add_prior(
        param_name="alpha",
        distribution="normal",
        params={"mean": 0.3, "std": 0.05},
        bounds=(0.1, 0.5),
        description="Capital share"
    )
    
    priors.add_prior(
        param_name="delta",
        distribution="beta",
        params={"alpha": 3, "beta": 10},
        bounds=(0.01, 0.15),
        description="Depreciation rate"
    )
    
    priors.add_prior(
        param_name="epsilon_p",
        distribution="gamma",
        params={"alpha": 6, "beta": 1},
        bounds=(1, 20),
        description="Price elasticity of demand"
    )
    
    priors.add_prior(
        param_name="xi_p",
        distribution="beta",
        params={"alpha": 6, "beta": 2},
        bounds=(0.1, 0.95),
        description="Calvo price stickiness"
    )
    
    priors.add_prior(
        param_name="iota_p",
        distribution="beta",
        params={"alpha": 5, "beta": 5},
        bounds=(0.01, 0.99),
        description="Price indexation"
    )
    
    # Wage setting parameters
    priors.add_prior(
        param_name="epsilon_w",
        distribution="gamma",
        params={"alpha": 6, "beta": 1},
        bounds=(1, 20),
        description="Wage elasticity of labor demand"
    )
    
    priors.add_prior(
        param_name="xi_w",
        distribution="beta",
        params={"alpha": 6, "beta": 2},
        bounds=(0.1, 0.95),
        description="Calvo wage stickiness"
    )
    
    priors.add_prior(
        param_name="iota_w",
        distribution="beta",
        params={"alpha": 5, "beta": 5},
        bounds=(0.01, 0.99),
        description="Wage indexation"
    )
    
    # Monetary policy parameters
    priors.add_prior(
        param_name="rho_r",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Interest rate smoothing"
    )
    
    priors.add_prior(
        param_name="phi_pi",
        distribution="normal",
        params={"mean": 1.5, "std": 0.25},
        bounds=(1.01, 3),
        description="Response to inflation"
    )
    
    priors.add_prior(
        param_name="phi_y",
        distribution="normal",
        params={"mean": 0.125, "std": 0.05},
        bounds=(0.01, 0.5),
        description="Response to output gap"
    )
    
    priors.add_prior(
        param_name="phi_dy",
        distribution="normal",
        params={"mean": 0.125, "std": 0.05},
        bounds=(0.01, 0.5),
        description="Response to output growth"
    )
    
    # Steady state parameters
    priors.add_prior(
        param_name="pi_bar",
        distribution="gamma",
        params={"alpha": 16, "beta": 10},
        bounds=(1.0, 1.02),
        description="Steady state inflation (quarterly)"
    )
    
    priors.add_prior(
        param_name="r_bar",
        distribution="gamma",
        params={"alpha": 16, "beta": 10},
        bounds=(1.0, 1.02),
        description="Steady state real interest rate (quarterly)"
    )
    
    # Shock persistence parameters
    priors.add_prior(
        param_name="technology_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Technology shock persistence"
    )
    
    priors.add_prior(
        param_name="preference_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Preference shock persistence"
    )
    
    priors.add_prior(
        param_name="investment_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Investment shock persistence"
    )
    
    priors.add_prior(
        param_name="government_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Government spending shock persistence"
    )
    
    priors.add_prior(
        param_name="monetary_rho",
        distribution="beta",
        params={"alpha": 5, "beta": 5},
        bounds=(0.1, 0.99),
        description="Monetary policy shock persistence"
    )
    
    priors.add_prior(
        param_name="price_markup_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Price markup shock persistence"
    )
    
    priors.add_prior(
        param_name="wage_markup_rho",
        distribution="beta",
        params={"alpha": 8, "beta": 2},
        bounds=(0.1, 0.99),
        description="Wage markup shock persistence"
    )
    
    # Shock standard deviations
    priors.add_prior(
        param_name="technology_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Technology shock standard deviation"
    )
    
    priors.add_prior(
        param_name="preference_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Preference shock standard deviation"
    )
    
    priors.add_prior(
        param_name="investment_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Investment shock standard deviation"
    )
    
    priors.add_prior(
        param_name="government_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Government spending shock standard deviation"
    )
    
    priors.add_prior(
        param_name="monetary_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Monetary policy shock standard deviation"
    )
    
    priors.add_prior(
        param_name="price_markup_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Price markup shock standard deviation"
    )
    
    priors.add_prior(
        param_name="wage_markup_sigma",
        distribution="inverse_gamma",
        params={"alpha": 3, "beta": 0.01},
        bounds=(0.001, 0.1),
        description="Wage markup shock standard deviation"
    )
    
    return priors