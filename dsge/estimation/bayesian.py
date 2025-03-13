"""
Bayesian estimation module for the DSGE model.

This module implements Bayesian estimation methods for the DSGE model,
including Metropolis-Hastings MCMC algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import scipy.stats as stats
from scipy.optimize import minimize
import logging
import time
from multiprocessing import Pool, cpu_count

from config.config_manager import ConfigManager
from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.perturbation import PerturbationSolver
from dsge.solution.projection import ProjectionSolver
from dsge.data.processor import DataProcessor

# Set up logging
logger = logging.getLogger(__name__)


class Prior:
    """
    Class for representing prior distributions.
    """
    
    def __init__(
        self, 
        distribution: str, 
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None
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
    
    def pdf(self, x: float) -> float:
        """
        Compute the probability density function (PDF) at x.
        
        Args:
            x (float): Point at which to evaluate the PDF.
            
        Returns:
            float: PDF value at x.
        """
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
    
    def log_pdf(self, x: float) -> float:
        """
        Compute the log probability density function at x.
        
        Args:
            x (float): Point at which to evaluate the log PDF.
            
        Returns:
            float: Log PDF value at x.
        """
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


class BayesianEstimator:
    """
    Class for Bayesian estimation of DSGE models.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        data: Union[pd.DataFrame, DataProcessor],
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the Bayesian estimator.
        
        Args:
            model (SmetsWoutersModel): The DSGE model to estimate.
            data (Union[pd.DataFrame, DataProcessor]): Data for estimation.
                If a DataFrame is provided, it should contain the observed variables.
                If a DataProcessor is provided, its processed data will be used.
            config (Optional[Union[Dict[str, Any], ConfigManager]]): Configuration.
                If a dictionary is provided, it will be used as the configuration.
                If a ConfigManager is provided, its configuration will be used.
                If None, the default configuration will be used.
        """
        self.model = model
        
        # Extract data
        if isinstance(data, DataProcessor):
            self.data = data.get_processed_data()
        else:
            self.data = data
        
        # Initialize configuration
        if config is None:
            self.config = ConfigManager()
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.update(config)
        else:
            self.config = config
        
        # Extract estimation parameters
        self.estimation_params = self.config.get("estimation")
        
        # Initialize priors
        self.priors = {}
        
        # Initialize MCMC results
        self.mcmc_results = None
    
    def set_prior(
        self, 
        param_name: str, 
        distribution: str, 
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Set a prior distribution for a parameter.
        
        Args:
            param_name (str): Name of the parameter.
            distribution (str): Type of distribution.
                Options: "normal", "beta", "gamma", "inverse_gamma", "uniform".
            params (Dict[str, float]): Parameters of the distribution.
            bounds (Optional[Tuple[float, float]]): Bounds for the parameter.
        """
        self.priors[param_name] = Prior(distribution, params, bounds)
    
    def set_default_priors(self) -> None:
        """
        Set default priors for the model parameters.
        """
        # Household parameters
        self.set_prior("beta", "beta", {"alpha": 99, "beta": 1}, (0.9, 0.999))
        self.set_prior("sigma_c", "normal", {"mean": 1.5, "std": 0.375}, (0.25, 3))
        self.set_prior("h", "beta", {"alpha": 7, "beta": 3}, (0.1, 0.95))
        self.set_prior("sigma_l", "normal", {"mean": 2.0, "std": 0.75}, (0.25, 5))
        
        # Production parameters
        self.set_prior("alpha", "normal", {"mean": 0.3, "std": 0.05}, (0.1, 0.5))
        self.set_prior("delta", "beta", {"alpha": 3, "beta": 10}, (0.01, 0.15))
        self.set_prior("epsilon_p", "gamma", {"alpha": 6, "beta": 1}, (1, 20))
        self.set_prior("xi_p", "beta", {"alpha": 6, "beta": 2}, (0.1, 0.95))
        self.set_prior("iota_p", "beta", {"alpha": 5, "beta": 5}, (0.01, 0.99))
        
        # Wage setting parameters
        self.set_prior("epsilon_w", "gamma", {"alpha": 6, "beta": 1}, (1, 20))
        self.set_prior("xi_w", "beta", {"alpha": 6, "beta": 2}, (0.1, 0.95))
        self.set_prior("iota_w", "beta", {"alpha": 5, "beta": 5}, (0.01, 0.99))
        
        # Monetary policy parameters
        self.set_prior("rho_r", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("phi_pi", "normal", {"mean": 1.5, "std": 0.25}, (1.01, 3))
        self.set_prior("phi_y", "normal", {"mean": 0.125, "std": 0.05}, (0.01, 0.5))
        self.set_prior("phi_dy", "normal", {"mean": 0.125, "std": 0.05}, (0.01, 0.5))
        
        # Steady state parameters
        self.set_prior("pi_bar", "gamma", {"alpha": 16, "beta": 10}, (1.0, 1.02))
        self.set_prior("r_bar", "gamma", {"alpha": 16, "beta": 10}, (1.0, 1.02))
        
        # Shock persistence parameters
        self.set_prior("technology_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("preference_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("investment_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("government_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("monetary_rho", "beta", {"alpha": 5, "beta": 5}, (0.1, 0.99))
        self.set_prior("price_markup_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        self.set_prior("wage_markup_rho", "beta", {"alpha": 8, "beta": 2}, (0.1, 0.99))
        
        # Shock standard deviations
        self.set_prior("technology_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("preference_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("investment_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("government_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("monetary_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("price_markup_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
        self.set_prior("wage_markup_sigma", "inverse_gamma", {"alpha": 3, "beta": 0.01}, (0.001, 0.1))
    
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
    
    def log_likelihood(
        self, 
        params: Dict[str, float], 
        data: pd.DataFrame
    ) -> float:
        """
        Compute the log likelihood for a set of parameters.
        
        Args:
            params (Dict[str, float]): Dictionary of parameter values.
            data (pd.DataFrame): Observed data.
            
        Returns:
            float: Log likelihood.
        """
        # Update model parameters
        self.model.params.update(params)
        
        # Solve the model
        solution_method = self.config.get("solution.method")
        if solution_method == "perturbation":
            perturbation_order = self.config.get("solution.perturbation_order")
            solver = PerturbationSolver(self.model, order=perturbation_order)
        else:  # projection
            projection_method = self.config.get("solution.projection_method")
            projection_nodes = self.config.get("solution.projection_nodes")
            solver = ProjectionSolver(self.model, method=projection_method, nodes=projection_nodes)
        
        try:
            # Solve the model
            solver.solve()
            
            # Simulate the model
            n_periods = len(data)
            states_sim, controls_sim = solver.simulate(n_periods)
            
            # Extract observed variables
            # In a real implementation, this would map model variables to observed variables
            # Here, we're assuming a one-to-one mapping for simplicity
            simulated = pd.DataFrame(controls_sim, columns=data.columns)
            
            # Compute log likelihood
            # In a real implementation, this would use a proper likelihood function
            # such as a multivariate normal with measurement errors
            # Here, we're using a simplified version
            residuals = data - simulated
            
            # Compute covariance matrix of residuals
            cov_matrix = np.cov(residuals.T)
            
            # Add a small value to the diagonal to ensure positive definiteness
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            
            # Compute log likelihood using multivariate normal
            log_like = 0.0
            for t in range(n_periods):
                log_like += stats.multivariate_normal.logpdf(
                    residuals.iloc[t].values,
                    mean=np.zeros(len(data.columns)),
                    cov=cov_matrix
                )
            
            return log_like
        
        except Exception as e:
            logger.warning(f"Error computing likelihood: {str(e)}")
            return -float('inf')
    
    def log_posterior(
        self, 
        params: Dict[str, float], 
        data: pd.DataFrame
    ) -> float:
        """
        Compute the log posterior density for a set of parameters.
        
        Args:
            params (Dict[str, float]): Dictionary of parameter values.
            data (pd.DataFrame): Observed data.
            
        Returns:
            float: Log posterior density.
        """
        # Compute log prior
        log_prior = self.log_prior(params)
        
        # If prior is -inf, return -inf (parameter out of bounds)
        if log_prior == -float('inf'):
            return -float('inf')
        
        # Compute log likelihood
        log_like = self.log_likelihood(params, data)
        
        # Compute log posterior
        log_post = log_prior + log_like
        
        return log_post
    
    def metropolis_hastings(
        self, 
        initial_params: Dict[str, float], 
        n_draws: int, 
        burn_in: int, 
        tune: int, 
        target_acceptance: float = 0.234, 
        proposal_scale: float = 0.1, 
        n_chains: int = 4, 
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the Metropolis-Hastings MCMC algorithm.
        
        Args:
            initial_params (Dict[str, float]): Initial parameter values.
            n_draws (int): Number of draws per chain.
            burn_in (int): Number of burn-in draws to discard.
            tune (int): Number of tuning iterations.
            target_acceptance (float): Target acceptance rate.
            proposal_scale (float): Initial proposal scale.
            n_chains (int): Number of MCMC chains.
            seed (Optional[int]): Random seed.
            
        Returns:
            Dict[str, Any]: MCMC results.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Parameter names and initial values
        param_names = list(initial_params.keys())
        n_params = len(param_names)
        
        # Initialize chains
        chains = []
        acceptance_rates = []
        
        # Run chains in parallel
        with Pool(processes=min(n_chains, cpu_count())) as pool:
            # Define arguments for each chain
            chain_args = []
            for i in range(n_chains):
                # Set different random seed for each chain
                chain_seed = None if seed is None else seed + i
                
                # Add arguments
                chain_args.append((
                    i,
                    initial_params,
                    param_names,
                    n_draws,
                    burn_in,
                    tune,
                    target_acceptance,
                    proposal_scale,
                    self.data,
                    chain_seed
                ))
            
            # Run chains in parallel
            results = pool.map(self._run_chain, chain_args)
            
            # Extract results
            for chain_samples, chain_acceptance in results:
                chains.append(chain_samples)
                acceptance_rates.append(chain_acceptance)
        
        # Combine chains
        all_samples = np.concatenate(chains, axis=0)
        
        # Compute statistics
        mean = np.mean(all_samples, axis=0)
        std = np.std(all_samples, axis=0)
        percentiles = np.percentile(all_samples, [2.5, 25, 50, 75, 97.5], axis=0)
        
        # Create results dictionary
        results = {
            "samples": all_samples,
            "chains": chains,
            "acceptance_rates": acceptance_rates,
            "mean": {param_names[i]: mean[i] for i in range(n_params)},
            "std": {param_names[i]: std[i] for i in range(n_params)},
            "percentiles": {
                "2.5%": {param_names[i]: percentiles[0, i] for i in range(n_params)},
                "25%": {param_names[i]: percentiles[1, i] for i in range(n_params)},
                "50%": {param_names[i]: percentiles[2, i] for i in range(n_params)},
                "75%": {param_names[i]: percentiles[3, i] for i in range(n_params)},
                "97.5%": {param_names[i]: percentiles[4, i] for i in range(n_params)},
            },
            "param_names": param_names,
        }
        
        # Store results
        self.mcmc_results = results
        
        return results
    
    def _run_chain(
        self, 
        args: Tuple
    ) -> Tuple[np.ndarray, float]:
        """
        Run a single MCMC chain.
        
        Args:
            args (Tuple): Tuple of arguments:
                - chain_id (int): Chain identifier.
                - initial_params (Dict[str, float]): Initial parameter values.
                - param_names (List[str]): List of parameter names.
                - n_draws (int): Number of draws.
                - burn_in (int): Number of burn-in draws.
                - tune (int): Number of tuning iterations.
                - target_acceptance (float): Target acceptance rate.
                - proposal_scale (float): Initial proposal scale.
                - data (pd.DataFrame): Observed data.
                - seed (Optional[int]): Random seed.
                
        Returns:
            Tuple[np.ndarray, float]:
                - Samples from the chain (n_draws - burn_in, n_params).
                - Acceptance rate.
        """
        # Extract arguments
        chain_id, initial_params, param_names, n_draws, burn_in, tune, target_acceptance, proposal_scale, data, seed = args
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Number of parameters
        n_params = len(param_names)
        
        # Initialize chain
        chain = np.zeros((n_draws, n_params))
        
        # Initialize current state
        current_params = initial_params.copy()
        current_log_post = self.log_posterior(current_params, data)
        
        # Initialize proposal covariance matrix
        proposal_cov = np.eye(n_params) * proposal_scale**2
        
        # Initialize acceptance counter
        n_accepted = 0
        
        # Run chain
        for i in range(n_draws):
            # Propose new parameters
            proposal = np.random.multivariate_normal(
                mean=[current_params[name] for name in param_names],
                cov=proposal_cov
            )
            
            # Convert to dictionary
            proposal_params = {name: proposal[j] for j, name in enumerate(param_names)}
            
            # Compute log posterior
            proposal_log_post = self.log_posterior(proposal_params, data)
            
            # Compute acceptance probability
            log_alpha = min(0, proposal_log_post - current_log_post)
            
            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                # Accept
                current_params = proposal_params
                current_log_post = proposal_log_post
                n_accepted += 1
            
            # Store current parameters
            for j, name in enumerate(param_names):
                chain[i, j] = current_params[name]
            
            # Tune proposal covariance during tuning phase
            if i < tune and i > 0 and (i + 1) % 100 == 0:
                # Compute acceptance rate
                acceptance_rate = n_accepted / (i + 1)
                
                # Adjust proposal scale
                if acceptance_rate < target_acceptance:
                    proposal_scale *= 0.9  # Decrease scale
                else:
                    proposal_scale *= 1.1  # Increase scale
                
                # Update proposal covariance
                if i > 100:
                    # Use empirical covariance of chain so far
                    proposal_cov = np.cov(chain[:i, :].T) * proposal_scale**2
                else:
                    # Use scaled identity matrix
                    proposal_cov = np.eye(n_params) * proposal_scale**2
                
                logger.info(f"Chain {chain_id}: Iteration {i+1}, Acceptance rate: {acceptance_rate:.3f}, Proposal scale: {proposal_scale:.6f}")
        
        # Compute final acceptance rate
        acceptance_rate = n_accepted / n_draws
        
        # Discard burn-in
        chain = chain[burn_in:, :]
        
        return chain, acceptance_rate
    
    def estimate(self) -> Dict[str, Any]:
        """
        Estimate the model using Bayesian methods.
        
        Returns:
            Dict[str, Any]: Estimation results.
        """
        # Set default priors if not set
        if not self.priors:
            self.set_default_priors()
        
        # Get initial parameters
        initial_params = {}
        for param_name in self.priors:
            # Use prior mean or mode as initial value
            if self.priors[param_name].distribution == "normal":
                initial_params[param_name] = self.priors[param_name].params["mean"]
            elif self.priors[param_name].distribution == "beta":
                alpha = self.priors[param_name].params["alpha"]
                beta = self.priors[param_name].params["beta"]
                initial_params[param_name] = (alpha - 1) / (alpha + beta - 2)
            elif self.priors[param_name].distribution == "gamma":
                alpha = self.priors[param_name].params["alpha"]
                beta = self.priors[param_name].params["beta"]
                initial_params[param_name] = (alpha - 1) / beta
            elif self.priors[param_name].distribution == "inverse_gamma":
                alpha = self.priors[param_name].params["alpha"]
                beta = self.priors[param_name].params["beta"]
                initial_params[param_name] = beta / (alpha + 1)
            elif self.priors[param_name].distribution == "uniform":
                min_val = self.priors[param_name].params["min"]
                max_val = self.priors[param_name].params["max"]
                initial_params[param_name] = (min_val + max_val) / 2
        
        # Get MCMC parameters
        mcmc_algorithm = self.estimation_params.get("mcmc_algorithm", "metropolis_hastings")
        n_chains = self.estimation_params.get("num_chains", 4)
        n_draws = self.estimation_params.get("num_draws", 10000)
        burn_in = self.estimation_params.get("burn_in", 5000)
        tune = self.estimation_params.get("tune", 1000)
        target_acceptance = self.estimation_params.get("target_acceptance", 0.234)
        
        # Run MCMC
        if mcmc_algorithm == "metropolis_hastings":
            start_time = time.time()
            results = self.metropolis_hastings(
                initial_params=initial_params,
                n_draws=n_draws,
                burn_in=burn_in,
                tune=tune,
                target_acceptance=target_acceptance,
                n_chains=n_chains
            )
            end_time = time.time()
            
            # Add timing information
            results["time_elapsed"] = end_time - start_time
            
            return results
        else:
            raise ValueError(f"Invalid MCMC algorithm: {mcmc_algorithm}")
    
    def get_posterior_mean(self) -> Dict[str, float]:
        """
        Get the posterior mean of the parameters.
        
        Returns:
            Dict[str, float]: Posterior mean.
            
        Raises:
            ValueError: If MCMC has not been run.
        """
        if self.mcmc_results is None:
            raise ValueError("MCMC has not been run. Call estimate() first.")
        
        return self.mcmc_results["mean"]
    
    def get_credible_interval(
        self, 
        param_name: str, 
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Get the credible interval for a parameter.
        
        Args:
            param_name (str): Name of the parameter.
            alpha (float): Significance level (default: 0.05 for 95% CI).
            
        Returns:
            Tuple[float, float]: Lower and upper bounds of the credible interval.
            
        Raises:
            ValueError: If MCMC has not been run or parameter not found.
        """
        if self.mcmc_results is None:
            raise ValueError("MCMC has not been run. Call estimate() first.")
        
        if param_name not in self.mcmc_results["param_names"]:
            raise ValueError(f"Parameter not found: {param_name}")
        
        # Get parameter index
        param_idx = self.mcmc_results["param_names"].index(param_name)
        
        # Get samples
        samples = self.mcmc_results["samples"][:, param_idx]
        
        # Compute credible interval
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def get_posterior_distribution(
        self, 
        param_name: str
    ) -> np.ndarray:
        """
        Get the posterior distribution for a parameter.
        
        Args:
            param_name (str): Name of the parameter.
            
        Returns:
            np.ndarray: Samples from the posterior distribution.
            
        Raises:
            ValueError: If MCMC has not been run or parameter not found.
        """
        if self.mcmc_results is None:
            raise ValueError("MCMC has not been run. Call estimate() first.")
        
        if param_name not in self.mcmc_results["param_names"]:
            raise ValueError(f"Parameter not found: {param_name}")
        
        # Get parameter index
        param_idx = self.mcmc_results["param_names"].index(param_name)
        
        # Get samples
        samples = self.mcmc_results["samples"][:, param_idx]
        
        return samples
    
    def compute_marginal_likelihood(self) -> float:
        """
        Compute the marginal likelihood of the model.
        
        Returns:
            float: Marginal likelihood.
            
        Raises:
            ValueError: If MCMC has not been run.
        """
        if self.mcmc_results is None:
            raise ValueError("MCMC has not been run. Call estimate() first.")
        
        # In a real implementation, this would compute the marginal likelihood
        # using methods like bridge sampling or harmonic mean estimator
        # Here, we're providing a placeholder
        
        return 0.0