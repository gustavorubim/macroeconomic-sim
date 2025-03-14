"""
Base solver class for DSGE models.
"""

from typing import Dict, Any, Optional
import numpy as np

class BaseSolver:
    """
    Base class for DSGE model solvers.
    
    This class defines the interface that all solver implementations
    should follow.
    """
    
    def __init__(self, model):
        """
        Initialize the solver.
        
        Args:
            model: The DSGE model to solve
        """
        self.model = model
        self.solution = None
    
    def solve(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Solve the model.
        
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary containing solution matrices,
            or None if solution fails
        """
        raise NotImplementedError("Solver implementations must override solve()")
    
    def simulate(self, periods: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate the model using the current solution.
        
        Args:
            periods (int): Number of periods to simulate
            seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing simulated time series
        """
        raise NotImplementedError("Solver implementations must override simulate()")
    
    def compute_moments(self) -> Dict[str, float]:
        """
        Compute theoretical moments from the model solution.
        
        Returns:
            Dict[str, float]: Dictionary containing theoretical moments
        """
        raise NotImplementedError("Solver implementations must override compute_moments()")
    
    def impulse_response(
        self, 
        shock: str, 
        periods: int, 
        scale: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute impulse response functions.
        
        Args:
            shock (str): Name of shock to perturb
            periods (int): Number of periods to compute
            scale (float): Scale factor for shock size
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing IRF time series
        """
        raise NotImplementedError("Solver implementations must override impulse_response()")
    
    def check_solution(self) -> bool:
        """
        Check if the current solution is valid.
        
        Returns:
            bool: True if solution is valid, False otherwise
        """
        raise NotImplementedError("Solver implementations must override check_solution()")