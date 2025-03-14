"""
Posterior analysis module for the DSGE model.

This module provides functions for analyzing the posterior distributions
of the parameters of the DSGE model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PosteriorAnalysis:
    """
    Class for analyzing posterior distributions from MCMC estimation.
    """
    
    def __init__(
        self, 
        mcmc_results: Dict[str, Any],
        prior_means: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the posterior analysis.
        
        Args:
            mcmc_results (Dict[str, Any]): Results from MCMC estimation.
                Should contain:
                - "samples": Array of parameter samples (n_samples, n_params)
                - "chains": List of chains, each an array (n_samples_per_chain, n_params)
                - "param_names": List of parameter names
                - "mean": Dictionary of posterior means
                - "std": Dictionary of posterior standard deviations
                - "percentiles": Dictionary of percentiles
            prior_means (Optional[Dict[str, float]]): Dictionary of prior means.
                If provided, will be used for computing shrinkage.
        """
        self.mcmc_results = mcmc_results
        self.prior_means = prior_means
        
        # Extract key components
        self.samples = mcmc_results["samples"]
        self.chains = mcmc_results["chains"]
        self.param_names = mcmc_results["param_names"]
        self.n_params = len(self.param_names)
        
        # Compute additional statistics if not already present
        if "mean" not in mcmc_results:
            self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """
        Compute basic statistics for the posterior distributions.
        """
        # Compute mean and standard deviation
        mean = np.mean(self.samples, axis=0)
        std = np.std(self.samples, axis=0)
        
        # Compute percentiles
        percentiles = np.percentile(self.samples, [2.5, 25, 50, 75, 97.5], axis=0)
        
        # Store results
        self.mcmc_results["mean"] = {
            self.param_names[i]: mean[i] for i in range(self.n_params)
        }
        self.mcmc_results["std"] = {
            self.param_names[i]: std[i] for i in range(self.n_params)
        }
        self.mcmc_results["percentiles"] = {
            "2.5%": {self.param_names[i]: percentiles[0, i] for i in range(self.n_params)},
            "25%": {self.param_names[i]: percentiles[1, i] for i in range(self.n_params)},
            "50%": {self.param_names[i]: percentiles[2, i] for i in range(self.n_params)},
            "75%": {self.param_names[i]: percentiles[3, i] for i in range(self.n_params)},
            "97.5%": {self.param_names[i]: percentiles[4, i] for i in range(self.n_params)},
        }
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary of the posterior distributions.
        
        Returns:
            pd.DataFrame: Summary statistics for each parameter.
        """
        # Initialize data
        data = {
            "Parameter": self.param_names,
            "Mean": [self.mcmc_results["mean"][p] for p in self.param_names],
            "Std": [self.mcmc_results["std"][p] for p in self.param_names],
            "2.5%": [self.mcmc_results["percentiles"]["2.5%"][p] for p in self.param_names],
            "25%": [self.mcmc_results["percentiles"]["25%"][p] for p in self.param_names],
            "50%": [self.mcmc_results["percentiles"]["50%"][p] for p in self.param_names],
            "75%": [self.mcmc_results["percentiles"]["75%"][p] for p in self.param_names],
            "97.5%": [self.mcmc_results["percentiles"]["97.5%"][p] for p in self.param_names],
        }
        
        # Add shrinkage if prior means are available
        if self.prior_means is not None:
            shrinkage = []
            for p in self.param_names:
                if p in self.prior_means:
                    prior_mean = self.prior_means[p]
                    posterior_mean = self.mcmc_results["mean"][p]
                    posterior_std = self.mcmc_results["std"][p]
                    
                    # Compute shrinkage (1 - posterior_var / prior_var)
                    # Here we approximate prior_var using the distance from prior to posterior mean
                    prior_var = (posterior_mean - prior_mean) ** 2
                    posterior_var = posterior_std ** 2
                    
                    if prior_var > 0:
                        shrink = 1 - posterior_var / prior_var
                        shrink = max(0, min(1, shrink))  # Ensure between 0 and 1
                    else:
                        shrink = np.nan
                    
                    shrinkage.append(shrink)
                else:
                    shrinkage.append(np.nan)
            
            data["Shrinkage"] = shrinkage
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def plot_trace(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2
    ) -> plt.Figure:
        """
        Plot trace plots for the MCMC chains.
        
        Args:
            param_names (Optional[List[str]]): List of parameter names to plot.
                If None, all parameters will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get parameter names to plot
        if param_names is None:
            param_names = self.param_names
        
        # Number of parameters
        n_params = len(param_names)
        
        # Compute number of rows
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is always a properly indexable array
        if n_rows == 1 and n_cols == 1:
            # For a single subplot, make it a 1D array with one element
            axes = np.array([axes]).flatten()
        elif n_rows == 1 or n_cols == 1:
            # For a single row or column, flatten to 1D
            axes = axes.flatten()
        
        # Plot each parameter
        for i, param_name in enumerate(param_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # Get parameter index
            param_idx = self.param_names.index(param_name)
            
            # Plot each chain
            for j, chain in enumerate(self.chains):
                ax.plot(chain[:, param_idx], alpha=0.7, label=f"Chain {j+1}")
            
            # Add horizontal line for posterior mean
            ax.axhline(
                self.mcmc_results["mean"][param_name],
                color='r',
                linestyle='--',
                alpha=0.5
            )
            
            # Set title and labels
            ax.set_title(param_name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.set_visible(False)
        
        # Add legend to the first subplot
        if n_params > 0:
            # Get the first subplot - works whether axes is 1D or 2D
            first_ax = axes.flat[0] if hasattr(axes, 'flat') else axes[0]
            first_ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_posterior(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        kde: bool = True,
        hist: bool = True,
        prior: Optional[Dict[str, Callable]] = None
    ) -> plt.Figure:
        """
        Plot posterior distributions.
        
        Args:
            param_names (Optional[List[str]]): List of parameter names to plot.
                If None, all parameters will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            kde (bool): Whether to plot kernel density estimate.
            hist (bool): Whether to plot histogram.
            prior (Optional[Dict[str, Callable]]): Dictionary mapping parameter names
                to functions that compute the prior PDF.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get parameter names to plot
        if param_names is None:
            param_names = self.param_names
        
        # Number of parameters
        n_params = len(param_names)
        
        # Compute number of rows
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is always a properly indexable array
        if n_rows == 1 and n_cols == 1:
            # For a single subplot, make it a 1D array with one element
            axes = np.array([axes]).flatten()
        elif n_rows == 1 or n_cols == 1:
            # For a single row or column, flatten to 1D
            axes = axes.flatten()
        
        # Plot each parameter
        for i, param_name in enumerate(param_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # Get parameter index
            param_idx = self.param_names.index(param_name)
            
            # Get samples for this parameter
            samples = self.samples[:, param_idx]
            
            # Plot histogram and KDE
            if hist and kde:
                sns.histplot(samples, kde=True, ax=ax, stat="density", alpha=0.5)
            elif hist:
                sns.histplot(samples, kde=False, ax=ax, stat="density", alpha=0.5)
            elif kde:
                sns.kdeplot(samples, ax=ax)
            
            # Add vertical lines for posterior mean and credible interval
            ax.axvline(
                self.mcmc_results["mean"][param_name],
                color='r',
                linestyle='--',
                alpha=0.5,
                label="Mean"
            )
            
            ax.axvline(
                self.mcmc_results["percentiles"]["2.5%"][param_name],
                color='g',
                linestyle=':',
                alpha=0.5,
                label="95% CI"
            )
            
            ax.axvline(
                self.mcmc_results["percentiles"]["97.5%"][param_name],
                color='g',
                linestyle=':',
                alpha=0.5
            )
            
            # Plot prior if provided
            if prior is not None and param_name in prior:
                # Generate x values
                x_min, x_max = ax.get_xlim()
                x = np.linspace(x_min, x_max, 1000)
                
                # Compute prior PDF
                prior_pdf = prior[param_name](x)
                
                # Plot prior
                ax.plot(x, prior_pdf, 'k--', alpha=0.5, label="Prior")
            
            # Set title and labels
            ax.set_title(param_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.set_visible(False)
        
        # Add legend to the first subplot
        if n_params > 0:
            # Get the first subplot - works whether axes is 1D or 2D
            first_ax = axes.flat[0] if hasattr(axes, 'flat') else axes[0]
            first_ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_pair(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (10, 10),
        kind: str = "scatter",
        diag_kind: str = "kde"
    ) -> plt.Figure:
        """
        Plot pairwise relationships between parameters.
        
        Args:
            param_names (Optional[List[str]]): List of parameter names to plot.
                If None, all parameters will be plotted.
            figsize (Tuple[float, float]): Figure size.
            kind (str): Kind of plot for off-diagonal plots.
                Options: "scatter", "kde", "hex".
            diag_kind (str): Kind of plot for diagonal plots.
                Options: "kde", "hist".
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get parameter names to plot
        if param_names is None:
            param_names = self.param_names
        
        # Number of parameters
        n_params = len(param_names)
        
        # Create DataFrame with samples
        data = {}
        for i, param_name in enumerate(param_names):
            param_idx = self.param_names.index(param_name)
            data[param_name] = self.samples[:, param_idx]
        
        df = pd.DataFrame(data)
        
        # Create pairplot
        g = sns.pairplot(
            df,
            kind=kind,
            diag_kind=diag_kind,
            plot_kws={"alpha": 0.5},
            height=figsize[0] / n_params
        )
        
        # Set title
        g.fig.suptitle("Posterior Parameter Relationships", y=1.02)
        
        return g.fig
    
    def compute_convergence_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute convergence diagnostics for the MCMC chains.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of convergence diagnostics.
        """
        # Initialize results
        results = {
            "gelman_rubin": {},
            "effective_sample_size": {},
            "geweke": {}
        }
        
        # Compute diagnostics for each parameter
        for i, param_name in enumerate(self.param_names):
            # Extract chains for this parameter
            param_chains = [chain[:, i] for chain in self.chains]
            
            # Compute Gelman-Rubin statistic
            results["gelman_rubin"][param_name] = self._gelman_rubin(param_chains)
            
            # Compute effective sample size
            results["effective_sample_size"][param_name] = self._effective_sample_size(param_chains)
            
            # Compute Geweke statistic
            results["geweke"][param_name] = self._geweke(np.concatenate(param_chains))
        
        return results
    
    def _gelman_rubin(self, chains: List[np.ndarray]) -> float:
        """
        Compute the Gelman-Rubin statistic for a parameter.
        
        Args:
            chains (List[np.ndarray]): List of chains for a parameter.
            
        Returns:
            float: Gelman-Rubin statistic.
        """
        # Number of chains
        m = len(chains)
        
        # Number of samples per chain
        n = chains[0].shape[0]
        
        # Compute chain means
        chain_means = np.array([np.mean(chain) for chain in chains])
        
        # Compute overall mean
        overall_mean = np.mean(chain_means)
        
        # Compute between-chain variance
        B = n * np.sum((chain_means - overall_mean) ** 2) / (m - 1)
        
        # Compute within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in chains])
        
        # Compute variance estimate
        var_estimate = ((n - 1) / n) * W + B / n
        
        # Compute potential scale reduction factor
        R = np.sqrt(var_estimate / W)
        
        return R
    
    def _effective_sample_size(self, chains: List[np.ndarray]) -> float:
        """
        Compute the effective sample size for a parameter.
        
        Args:
            chains (List[np.ndarray]): List of chains for a parameter.
            
        Returns:
            float: Effective sample size.
        """
        # Number of chains
        m = len(chains)
        
        # Number of samples per chain
        n = chains[0].shape[0]
        
        # Concatenate chains
        samples = np.concatenate(chains)
        
        # Compute autocorrelation
        acf = self._autocorrelation(samples)
        
        # Compute effective sample size
        ess = m * n / (1 + 2 * np.sum(acf[1:]))
        
        return ess
    
    def _autocorrelation(self, x: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """
        Compute autocorrelation function.
        
        Args:
            x (np.ndarray): Time series.
            max_lag (int): Maximum lag.
            
        Returns:
            np.ndarray: Autocorrelation function.
        """
        # Mean and variance
        mean = np.mean(x)
        var = np.var(x, ddof=1)
        
        # Compute autocorrelation
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean((x[:-lag] - mean) * (x[lag:] - mean)) / var
        
        return acf
    
    def _geweke(self, x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
        """
        Compute the Geweke statistic for a parameter.
        
        Args:
            x (np.ndarray): MCMC samples.
            first (float): Fraction of the chain to use for the first part.
            last (float): Fraction of the chain to use for the last part.
            
        Returns:
            float: Geweke statistic.
        """
        # Number of samples
        n = len(x)
        
        # Compute indices for first and last parts
        n_first = int(first * n)
        n_last = int(last * n)
        
        # Extract first and last parts
        x_first = x[:n_first]
        x_last = x[-n_last:]
        
        # Compute means
        mean_first = np.mean(x_first)
        mean_last = np.mean(x_last)
        
        # Compute spectral density estimates
        # In a real implementation, this would use a proper spectral density estimator
        # Here, we're using a simplified version
        var_first = np.var(x_first, ddof=1) / n_first
        var_last = np.var(x_last, ddof=1) / n_last
        
        # Compute Geweke statistic
        z = (mean_first - mean_last) / np.sqrt(var_first + var_last)
        
        return z
    
    def convergence_summary(self) -> pd.DataFrame:
        """
        Generate a summary of convergence diagnostics.
        
        Returns:
            pd.DataFrame: Summary of convergence diagnostics.
        """
        # Compute diagnostics
        diagnostics = self.compute_convergence_diagnostics()
        
        # Initialize data
        data = {
            "Parameter": self.param_names,
            "Gelman-Rubin": [diagnostics["gelman_rubin"][p] for p in self.param_names],
            "ESS": [diagnostics["effective_sample_size"][p] for p in self.param_names],
            "Geweke": [diagnostics["geweke"][p] for p in self.param_names],
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def plot_autocorrelation(
        self, 
        param_names: Optional[List[str]] = None, 
        figsize: Tuple[float, float] = (12, 8),
        n_cols: int = 2,
        max_lag: int = 50
    ) -> plt.Figure:
        """
        Plot autocorrelation functions for the MCMC chains.
        
        Args:
            param_names (Optional[List[str]]): List of parameter names to plot.
                If None, all parameters will be plotted.
            figsize (Tuple[float, float]): Figure size.
            n_cols (int): Number of columns in the subplot grid.
            max_lag (int): Maximum lag to plot.
                
        Returns:
            plt.Figure: Matplotlib figure with the plots.
        """
        # Get parameter names to plot
        if param_names is None:
            param_names = self.param_names
        
        # Number of parameters
        n_params = len(param_names)
        
        # Compute number of rows
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is always a properly indexable array
        if n_rows == 1 and n_cols == 1:
            # For a single subplot, make it a 1D array with one element
            axes = np.array([axes]).flatten()
        elif n_rows == 1 or n_cols == 1:
            # For a single row or column, flatten to 1D
            axes = axes.flatten()
        
        # Plot each parameter
        for i, param_name in enumerate(param_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # Get parameter index
            param_idx = self.param_names.index(param_name)
            
            # Concatenate chains
            samples = np.concatenate([chain[:, param_idx] for chain in self.chains])
            
            # Compute autocorrelation
            acf = self._autocorrelation(samples, max_lag=max_lag)
            
            # Plot autocorrelation
            ax.bar(np.arange(len(acf)), acf, alpha=0.7)
            
            # Add horizontal lines for significance
            ax.axhline(0, color='k', linestyle='-', alpha=0.2)
            ax.axhline(1.96 / np.sqrt(len(samples)), color='r', linestyle='--', alpha=0.5)
            ax.axhline(-1.96 / np.sqrt(len(samples)), color='r', linestyle='--', alpha=0.5)
            
            # Set title and labels
            ax.set_title(param_name)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
            
            # Set x-axis limits
            ax.set_xlim(-0.5, max_lag + 0.5)
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_results(self, path: str) -> None:
        """
        Save the MCMC results to a file.
        
        Args:
            path (str): Path to save the results.
        """
        # Create a copy of the results
        results = self.mcmc_results.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        results["samples"] = results["samples"].tolist()
        results["chains"] = [chain.tolist() for chain in results["chains"]]
        
        # Save to file
        import json
        with open(path, 'w') as f:
            json.dump(results, f)
        
        logger.info(f"MCMC results saved to {path}")
    
    @classmethod
    def load_results(cls, path: str) -> 'PosteriorAnalysis':
        """
        Load MCMC results from a file.
        
        Args:
            path (str): Path to load the results from.
            
        Returns:
            PosteriorAnalysis: Posterior analysis object.
        """
        # Load from file
        import json
        with open(path, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays
        results["samples"] = np.array(results["samples"])
        results["chains"] = [np.array(chain) for chain in results["chains"]]
        
        # Create posterior analysis object
        return cls(results)