"""
Baseline forecasting implementation for DSGE models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging

from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.base_solver import BaseSolver
from dsge.solution.perturbation import PerturbationSolver

logger = logging.getLogger(__name__)

class BaselineForecaster:
    """
    Baseline forecasting implementation for DSGE models.
    """
    
    def __init__(self, model: SmetsWoutersModel, config: Dict[str, Any]):
        """
        Initialize the forecaster.
        
        Args:
            model: The DSGE model to use for forecasting
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.solver = PerturbationSolver(model)
        self.solution = None
        
    def solve_model(self) -> None:
        """
        Solve the model using the configured solver.
        """
        self.solution = self.solver.solve()
        if self.solution is None:
            raise RuntimeError("Failed to solve model")
            
    def generate_forecast(
        self, 
        data: pd.DataFrame, 
        horizon: int,
        n_draws: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts from current state.
        
        Args:
            data: Historical data
            horizon: Forecast horizon
            n_draws: Number of Monte Carlo draws
            seed: Random seed for reproducibility
            
        Returns:
            Dict[str, pd.DataFrame]: Forecast distributions
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self.solution is None:
            self.solve_model()
            
        # Get last observation as initial state
        initial_state = self._get_initial_state(data)
        
        # Generate multiple paths
        forecasts = []
        for _ in range(n_draws):
            # Simulate one path
            states, controls = self._simulate_path(initial_state, horizon)
            forecasts.append(self._combine_variables(states, controls))
            
        # Compute statistics
        forecast_stats = self._compute_forecast_stats(forecasts)
        
        return forecast_stats
    
    def _get_initial_state(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract initial state from data.
        
        Args:
            data: Historical data
            
        Returns:
            np.ndarray: Initial state vector
        """
        # Get last observation
        last_obs = data.iloc[-1]
        
        # Map observables to state variables
        n_states = len(self.model.variables.state)
        initial_state = np.zeros(n_states)
        
        # Map observed variables to states
        state_map = {
            'output': 'technology_shock',
            'consumption': 'preference_shock',
            'investment': 'investment_shock',
            'labor': 'wage_markup_shock',
            'inflation': 'price_markup_shock',
            'interest_rate': 'monetary_shock'
        }
        
        for obs_name, state_name in state_map.items():
            if obs_name in last_obs.index and state_name in self.model.variables.state:
                state_idx = self.model.variables.state.index(state_name)
                initial_state[state_idx] = last_obs[obs_name]
                
        return initial_state
    
    def _simulate_path(
        self, 
        initial_state: np.ndarray, 
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one forecast path.
        
        Args:
            initial_state: Initial state vector
            horizon: Forecast horizon
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Simulated states and controls
        """
        n_states = len(self.model.variables.state)
        n_controls = len(self.model.variables.control)
        
        states = np.zeros((horizon, n_states))
        controls = np.zeros((horizon, n_controls))
        
        # Initialize
        states[0] = initial_state
        controls[0] = self.solution["P"] @ states[0]
        
        # Generate future path
        for t in range(1, horizon):
            # State transition with shock
            shock = np.random.normal(0, 1, n_states)
            states[t] = self.solution["F"] @ states[t-1] + shock
            
            # Control response
            controls[t] = self.solution["P"] @ states[t]
            
        return states, controls
    
    def _combine_variables(
        self, 
        states: np.ndarray, 
        controls: np.ndarray
    ) -> pd.DataFrame:
        """
        Combine state and control variables into a single DataFrame.
        
        Args:
            states: Simulated states
            controls: Simulated controls
            
        Returns:
            pd.DataFrame: Combined variables
        """
        # Create DataFrame with all variables
        data = np.column_stack([states, controls])
        columns = self.model.variables.state + self.model.variables.control
        
        return pd.DataFrame(data, columns=columns)
    
    def _compute_forecast_stats(
        self, 
        forecasts: List[pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute forecast statistics.
        
        Args:
            forecasts: List of simulated paths
            
        Returns:
            Dict[str, pd.DataFrame]: Forecast statistics
        """
        # Stack all forecasts
        forecast_stack = pd.concat(forecasts, axis=1, keys=range(len(forecasts)))
        
        # Compute statistics
        stats = {
            'mean': forecast_stack.mean(axis=1).unstack(),
            'median': forecast_stack.median(axis=1).unstack(),
            'std': forecast_stack.std(axis=1).unstack(),
            'lower': forecast_stack.quantile(0.025, axis=1).unstack(),
            'upper': forecast_stack.quantile(0.975, axis=1).unstack()
        }
        
        return stats