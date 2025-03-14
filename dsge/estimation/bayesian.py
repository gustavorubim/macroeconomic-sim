"""
Bayesian estimation for DSGE models.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import scipy.stats as stats
import logging

from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.base_solver import BaseSolver

logger = logging.getLogger(__name__)

class Prior:
    """Class representing a prior distribution for a parameter."""
    def __init__(
        self,
        name: str,
        distribution: str,
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None
    ):
        self.name = name
        self.distribution = distribution.lower()
        self.params = params
        self.bounds = bounds
        self.dist = self._create_distribution()
        
    def _create_distribution(self) -> stats.rv_continuous:
        """Create scipy distribution object."""
        if self.distribution == 'normal':
            # Handle both parameterizations for normal
            if 'mean' in self.params and 'std' in self.params:
                return stats.norm(loc=self.params['mean'], scale=self.params['std'])
            return stats.norm(**self.params)
            
        elif self.distribution == 'beta':
            # Convert mean and std to beta parameters a and b
            if 'mean' in self.params and 'std' in self.params:
                mean = self.params['mean']
                std = self.params['std']
                var = std ** 2
                a = ((1 - mean) / var - 1 / mean) * mean ** 2
                b = a * (1 / mean - 1)
                return stats.beta(a=a, b=b)
            elif 'loc' in self.params and 'scale' in self.params:
                # Convert loc and scale to mean/std
                mean = self.params['loc']
                std = self.params['scale']
                var = std ** 2
                a = ((1 - mean) / var - 1 / mean) * mean ** 2
                b = a * (1 / mean - 1)
                return stats.beta(a=a, b=b)
            elif 'a' in self.params and 'b' in self.params:
                return stats.beta(a=self.params['a'], b=self.params['b'])
            else:
                raise ValueError("Beta distribution requires valid parameters")
                
        elif self.distribution == 'gamma':
            # Handle both parameterizations for gamma
            if 'mean' in self.params and 'std' in self.params:
                mean = self.params['mean']
                std = self.params['std']
                shape = (mean / std) ** 2
                scale = std ** 2 / mean
                return stats.gamma(a=shape, scale=scale)
            return stats.gamma(**self.params)
            
        elif self.distribution == 'uniform':
            return stats.uniform(**self.params)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
            
    def log_pdf(self, value: float) -> float:
        """Compute log probability density."""
        if self.bounds is not None:
            if value < self.bounds[0] or value > self.bounds[1]:
                return -np.inf
        return self.dist.logpdf(value)
        
    def random(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw random samples."""
        samples = self.dist.rvs(size=size)
        if self.bounds is not None:
            # Reject samples outside bounds
            if size is not None:
                mask = (samples >= self.bounds[0]) & (samples <= self.bounds[1])
                while not np.all(mask):
                    new_samples = self.dist.rvs(size=np.sum(~mask))
                    samples[~mask] = new_samples
                    mask = (samples >= self.bounds[0]) & (samples <= self.bounds[1])
            else:
                while samples < self.bounds[0] or samples > self.bounds[1]:
                    samples = self.dist.rvs()
        return samples

class PriorSet:
    """Collection of priors for all model parameters."""
    def __init__(self, priors: Dict[str, Prior]):
        self.priors = priors
        
    def log_joint(self, params: Dict[str, float]) -> float:
        """Compute joint log prior probability."""
        return sum(
            self.priors[name].log_pdf(value)
            for name, value in params.items()
            if name in self.priors
        )
        
    def random(self) -> Dict[str, float]:
        """Draw random samples from all priors."""
        return {
            name: prior.random()
            for name, prior in self.priors.items()
        }

class BayesianEstimator:
    """Bayesian estimation for DSGE models using MCMC methods."""
    
    def __init__(self, model: SmetsWoutersModel, data: Any, config: Dict[str, Any]):
        self.model = model
        self.data = data
        self.config = config
        self.solver = None
        
    def set_solver(self, solver: BaseSolver) -> None:
        """Set the solver for model solution."""
        self.solver = solver
        
    def log_likelihood(self, params: List[float]) -> float:
        """Compute log likelihood for parameters as list."""
        param_dict = {name: value for name, value in zip(self.model.params.keys(), params)}
        return self.evaluate_likelihood(param_dict)
    
    def log_posterior(self, params: List[float]) -> float:
        """Compute log posterior for parameters as list."""
        param_dict = {name: value for name, value in zip(self.model.params.keys(), params)}
        return self.evaluate_posterior(param_dict)
    
    def evaluate_likelihood(self, params: Dict[str, float]) -> float:
        """Evaluate model likelihood for parameters as dictionary."""
        if self.solver is None:
            raise ValueError("Solver must be set before evaluating likelihood")
            
        # Update model parameters
        self.model.update_parameters(params)
        
        # Solve model
        solution = self.solver.solve()
        if solution is None:
            return -np.inf
            
        # Compute likelihood using state space representation
        try:
            likelihood = self._compute_likelihood(solution)
            return likelihood
        except Exception as e:
            logger.warning(f"Likelihood evaluation failed: {str(e)}")
            return -np.inf
    
    def evaluate_posterior(self, params: Dict[str, float]) -> float:
        """Evaluate posterior density for parameters as dictionary."""
        log_prior = self._evaluate_prior(params)
        if log_prior == -np.inf:
            return -np.inf
            
        log_likelihood = self.evaluate_likelihood(params)
        return log_prior + log_likelihood
    
    def _evaluate_prior(self, params: Dict[str, float]) -> float:
        """Evaluate joint prior density."""
        prior_set = self.config.get('estimation.priors', None)
        if prior_set is None:
            logger.warning("No priors configured, using default (improper) prior")
            return 0.0
            
        if not isinstance(prior_set, PriorSet):
            try:
                priors = {
                    name: Prior(
                        name=name,
                        distribution=spec['distribution'],
                        params=spec['params'],
                        bounds=spec.get('bounds', None)
                    )
                    for name, spec in prior_set.items()
                }
                prior_set = PriorSet(priors)
            except Exception as e:
                logger.error(f"Failed to create priors from config: {str(e)}")
                return -np.inf
                
        try:
            return prior_set.log_joint(params)
        except Exception as e:
            logger.error(f"Error evaluating prior density: {str(e)}")
            return -np.inf
    
    def _compute_likelihood(self, solution: Dict[str, np.ndarray]) -> float:
        """Compute likelihood using Kalman filter."""
        # Extract state space matrices
        F = solution.get('F', None)  # State transition
        P = solution.get('P', None)  # Observation
        
        if F is None or P is None:
            logger.warning("Missing state space matrices")
            return -np.inf
            
        # Set up noise covariances
        n_states = F.shape[0]
        n_obs = P.shape[0]
        Q = np.eye(n_states) * 0.1  # State noise covariance
        R = np.eye(n_obs) * 0.1  # Measurement noise covariance
        
        # Initialize Kalman filter
        data = self.data.values
        n_periods = len(data)
        log_lik = 0.0
        
        # Initial state
        state = np.zeros(n_states)
        state_cov = np.eye(n_states)
        
        try:
            # Run filter
            for t in range(n_periods):
                # Predict
                state_pred = F @ state
                state_cov_pred = F @ state_cov @ F.T + Q
                
                # Update
                obs_pred = P @ state_pred
                innovation = data[t] - obs_pred
                innovation_cov = P @ state_cov_pred @ P.T + R
                
                # Kalman gain
                try:
                    innovation_cov_inv = np.linalg.inv(innovation_cov)
                except np.linalg.LinAlgError:
                    logger.warning("Singular innovation covariance")
                    return -np.inf
                    
                gain = state_cov_pred @ P.T @ innovation_cov_inv
                
                # State update
                state = state_pred + gain @ innovation
                state_cov = state_cov_pred - gain @ P @ state_cov_pred
                
                # Log likelihood contribution
                log_det = np.log(np.linalg.det(innovation_cov))
                if not np.isfinite(log_det):
                    logger.warning("Non-finite determinant in likelihood computation")
                    return -np.inf
                    
                quad_form = innovation.T @ innovation_cov_inv @ innovation
                log_lik += -0.5 * (n_obs * np.log(2 * np.pi) + log_det + quad_form)
                
            return float(log_lik)
            
        except Exception as e:
            logger.error(f"Error in Kalman filter: {str(e)}")
            return -np.inf