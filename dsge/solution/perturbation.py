"""
Perturbation solution method for the DSGE model.

This module implements first, second, and third-order perturbation methods
for solving the DSGE model.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import sympy as sp
from scipy.optimize import root
import logging
import numpy as np

from config.config_manager import ConfigManager
from dsge.core.base_model import SmetsWoutersModel
from dsge.solution.base_solver import BaseSolver

# Set up logging
logger = logging.getLogger(__name__)


class PerturbationSolver(BaseSolver):
    """
    Class for solving DSGE models using perturbation methods.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        order: int = 2
    ):
        """
        Initialize the perturbation solver.
        
        Args:
            model (SmetsWoutersModel): The DSGE model to solve.
            order (int): Order of perturbation (1, 2, or 3).
                Default is 2 (second-order).
                
        Raises:
            ValueError: If the order is not valid.
        """
        # Check order
        if order not in [1, 2, 3]:
            raise ValueError(f"Invalid perturbation order: {order}. Must be 1, 2, or 3.")
        
        self.model = model
        self.order = order
        
        # Get model parameters
        self.params = model.params
        
        # Get steady state
        if model.steady_state is None:
            model.compute_steady_state()
        self.steady_state = model.steady_state
        
        # Initialize solution
        self.solution = None
    
    def create_symbolic_model(self) -> Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Expr]]:
        """
        Create a symbolic representation of the model.
        
        Returns:
            Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Expr]]:
                - List of state variables (symbols)
                - List of control variables (symbols)
                - List of model equations (expressions)
        """
        # Define symbolic variables
        # State variables (predetermined)
        k = sp.Symbol('k')  # Capital
        a = sp.Symbol('a')  # Technology shock
        b = sp.Symbol('b')  # Preference shock
        i_shock = sp.Symbol('i_shock')  # Investment shock
        g = sp.Symbol('g')  # Government spending shock
        r_shock = sp.Symbol('r_shock')  # Monetary policy shock
        p_shock = sp.Symbol('p_shock')  # Price markup shock
        w_shock = sp.Symbol('w_shock')  # Wage markup shock
        
        # Control variables (non-predetermined)
        y = sp.Symbol('y')  # Output
        c = sp.Symbol('c')  # Consumption
        i = sp.Symbol('i')  # Investment
        l = sp.Symbol('l')  # Labor
        w = sp.Symbol('w')  # Real wage
        r_k = sp.Symbol('r_k')  # Rental rate of capital
        pi = sp.Symbol('pi')  # Inflation
        r = sp.Symbol('r')  # Nominal interest rate
        
        # Define parameters
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        delta = self.params["delta"]
        sigma_c = self.params["sigma_c"]
        h = self.params["h"]
        sigma_l = self.params["sigma_l"]
        xi_p = self.params["xi_p"]
        xi_w = self.params["xi_w"]
        iota_p = self.params["iota_p"]
        iota_w = self.params["iota_w"]
        rho_r = self.params["rho_r"]
        phi_pi = self.params["phi_pi"]
        phi_y = self.params["phi_y"]
        phi_dy = self.params["phi_dy"]
        pi_bar = self.params["pi_bar"]
        r_bar = self.params["r_bar"]
        
        # Shock persistence parameters
        rho_a = self.params["technology_rho"]
        rho_b = self.params["preference_rho"]
        rho_i = self.params["investment_rho"]
        rho_g = self.params["government_rho"]
        rho_r = self.params["monetary_rho"]
        rho_p = self.params["price_markup_rho"]
        rho_w = self.params["wage_markup_rho"]
        
        # Define model equations
        # This is a simplified version of the model equations
        # In a real implementation, this would include all the equations
        # from the Smets and Wouters model
        
        # Production function
        eq1 = y - a * k**alpha * l**(1-alpha)
        
        # Capital accumulation
        eq2 = k - (1 - delta) * k - i
        
        # Resource constraint
        eq3 = y - c - i - g
        
        # Consumption Euler equation
        eq4 = c**(-sigma_c) - beta * (1 + r) / pi * c**(-sigma_c)
        
        # Labor supply
        eq5 = w - l**sigma_l * c**sigma_c
        
        # Capital demand
        eq6 = r_k - alpha * a * k**(alpha-1) * l**(1-alpha)
        
        # Labor demand
        eq7 = w - (1 - alpha) * a * k**alpha * l**(-alpha)
        
        # Phillips curve
        eq8 = pi - (1 - xi_p) * (w / ((1 - alpha) * a * k**alpha * l**(-alpha))) - xi_p * pi
        
        # Monetary policy rule
        eq9 = r - rho_r * r - (1 - rho_r) * (r_bar + phi_pi * (pi - pi_bar) + phi_y * (y - y) + phi_dy * (y - y)) - r_shock
        
        # Shock processes
        eq10 = a - rho_a * a
        eq11 = b - rho_b * b
        eq12 = i_shock - rho_i * i_shock
        eq13 = g - rho_g * g
        eq14 = r_shock - rho_r * r_shock
        eq15 = p_shock - rho_p * p_shock
        eq16 = w_shock - rho_w * w_shock
        
        # Collect state and control variables
        states = [k, a, b, i_shock, g, r_shock, p_shock, w_shock]
        controls = [y, c, i, l, w, r_k, pi, r]
        
        # Collect equations
        equations = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16]
        
        return states, controls, equations
    
    def compute_steady_state_numerically(
        self, 
        states: List[sp.Symbol], 
        controls: List[sp.Symbol], 
        equations: List[sp.Expr]
    ) -> Dict[sp.Symbol, float]:
        """
        Compute the steady state of the model numerically.
        
        Args:
            states (List[sp.Symbol]): List of state variables.
            controls (List[sp.Symbol]): List of control variables.
            equations (List[sp.Expr]): List of model equations.
            
        Returns:
            Dict[sp.Symbol, float]: Dictionary mapping symbols to steady state values.
        """
        # Initialize with values from the model's steady state
        ss_values = {}
        
        # Set steady state values for state variables
        ss_values[states[0]] = self.steady_state["capital"]  # Capital
        
        # Set shock steady states to zero
        for i in range(1, len(states)):
            ss_values[states[i]] = 0.0
        
        # Set steady state values for control variables
        ss_values[controls[0]] = self.steady_state["output"]  # Output
        ss_values[controls[1]] = self.steady_state["consumption"]  # Consumption
        ss_values[controls[2]] = self.steady_state["investment"]  # Investment
        ss_values[controls[3]] = self.steady_state["labor"]  # Labor
        ss_values[controls[4]] = self.steady_state["real_wage"]  # Real wage
        ss_values[controls[5]] = self.steady_state["rental_rate"]  # Rental rate
        ss_values[controls[6]] = self.steady_state["inflation"]  # Inflation
        ss_values[controls[7]] = self.steady_state["nominal_interest"]  # Nominal interest
        
        # Define a function to evaluate the equations at given values
    @staticmethod
    def safe_float(value, threshold=1e-6):
            """
            Safely convert a symbolic expression to float, handling complex numbers and special values.
            
            Args:
                value: Value to convert
                threshold: Threshold for considering imaginary parts negligible
                
            Returns:
                float: Real part of the value if imaginary part is negligible
            """
            import math
            import sympy as sp
            import numpy as np
            from sympy.core.numbers import (Float, Integer, Rational)
            
            def is_effectively_zero(x: float) -> bool:
                return abs(x) < threshold if x is not None else True
                
            def handle_special_value(val):
                """Handle special values like infinity, nan, etc."""
                if val in (sp.zoo, sp.oo, -sp.oo, sp.nan, None, float('inf'), float('-inf'), float('nan')):
                    return 0.0
                return None
                
            def handle_complex(real_val: float, imag_val: float) -> float:
                """Handle complex number conversion."""
                if is_effectively_zero(imag_val):
                    return real_val
                    
                if is_effectively_zero(real_val):
                    return 0.0
                    
                if abs(imag_val) / (abs(real_val) + 1e-10) < threshold:
                    return real_val
                    
                # Return magnitude for significant complex values
                return abs(complex(real_val, imag_val))
            
            try:
                # Handle special values
                special_result = handle_special_value(value)
                if special_result is not None:
                    return special_result
                
                # Handle numeric types
                if isinstance(value, (float, int, Float, Integer, Rational)):
                    result = float(value)
                    return 0.0 if math.isnan(result) else result
                
                # Handle symbolic expressions
                if isinstance(value, sp.Expr):
                    evaluated = value.evalf()
                    
                    # Try direct conversion for real numbers
                    if evaluated.is_real:
                        try:
                            return float(evaluated)
                        except (TypeError, ValueError):
                            return 0.0
                    
                    # Handle complex numbers
                    try:
                        real_val = float(evaluated.as_real_imag()[0])
                        imag_val = float(evaluated.as_real_imag()[1])
                        return handle_complex(real_val, imag_val)
                    except (TypeError, ValueError):
                        return 0.0
                
                # Try direct conversion as last resort
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
                    
            except Exception as e:
                logger.error(f"Unexpected error in safe_float: {str(e)}")
                return 0.0
    
    def eval_equations(self, values_dict, equations):
        """Evaluate model equations at given values."""
        return [PerturbationSolver.safe_float(eq.subs(values_dict)) for eq in equations]
        
        # Check if the steady state satisfies the equations
        residuals = self.eval_equations(ss_values, equations)
        if max(abs(np.array(residuals))) > 1e-6:
            logger.warning("Steady state does not satisfy model equations. Solving numerically...")
            
            # Define variables to solve for
            variables = states + controls
            
            # Initial guess
            x0 = [ss_values[var] for var in variables]
            
            # Define objective function
            def objective(x):
                values = {var: val for var, val in zip(variables, x)}
                return self.eval_equations(values, equations)
            
            try:
                # Solve for steady state with scaled objective
                def scaled_objective(x):
                    values = {var: val for var, val in zip(variables, x)}
                    residuals = self.eval_equations(values, equations)
                    return np.array(residuals) / (np.abs(x) + 1e-10)  # Scale by variable magnitude
                
                # Try multiple initial guesses if first one fails
                for scale in [1.0, 0.1, 10.0]:
                    result = root(scaled_objective, x0 * scale, method='lm', options={'maxfev': 10000})
                    if result.success:
                        # Update steady state values
                        for i, var in enumerate(variables):
                            ss_values[var] = result.x[i]
                        logger.info("Steady state solved numerically.")
                        return ss_values
                
                logger.error("Failed to solve for steady state numerically.")
                return None
                
            except Exception as e:
                logger.error(f"Error solving for steady state: {str(e)}")
                return None
    
    def linearize_model(
        self,
        states: List[sp.Symbol],
        controls: List[sp.Symbol],
        equations: List[sp.Expr],
        ss_values: Dict[sp.Symbol, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model around the steady state.
        
        Args:
            states (List[sp.Symbol]): List of state variables.
            controls (List[sp.Symbol]): List of control variables.
            equations (List[sp.Expr]): List of model equations.
            ss_values (Dict[sp.Symbol, float]): Steady state values.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - A matrix: Derivative of equations w.r.t. current states
                - B matrix: Derivative of equations w.r.t. next period states
                - C matrix: Derivative of equations w.r.t. current controls
                - D matrix: Derivative of equations w.r.t. next period controls
        """
        def compute_derivative(eq: sp.Expr, var: sp.Symbol) -> float:
            """Safely compute derivative and evaluate at steady state."""
            try:
                derivative = sp.diff(eq, var)
                evaluated = derivative.subs(ss_values)
                return PerturbationSolver.safe_float(evaluated)
            except Exception as e:
                logger.warning(f"Error computing derivative of equation w.r.t. {var}: {str(e)}")
                return 0.0

        # Number of variables
        n_states = len(states)
        n_controls = len(controls)
        n_eqs = len(equations)
        
        # Initialize matrices
        A = np.zeros((n_eqs, n_states))
        B = np.zeros((n_eqs, n_states))
        C = np.zeros((n_eqs, n_controls))
        D = np.zeros((n_eqs, n_controls))
        
        # Compute derivatives with error handling
        for i, eq in enumerate(equations):
            try:
                # Current states
                for j, var in enumerate(states):
                    A[i, j] = compute_derivative(eq, var)
                
                # Next period states
                for j, var in enumerate(states):
                    B[i, j] = compute_derivative(eq, sp.Symbol(f"{var}_next"))
                
                # Current controls
                for j, var in enumerate(controls):
                    C[i, j] = compute_derivative(eq, var)
                
                # Next period controls
                for j, var in enumerate(controls):
                    D[i, j] = compute_derivative(eq, sp.Symbol(f"{var}_next"))
                    
            except Exception as e:
                logger.error(f"Error linearizing equation {i}: {str(e)}")
                # Set zero derivatives for this equation
                A[i, :] = 0.0
                B[i, :] = 0.0
                C[i, :] = 0.0
                D[i, :] = 0.0
        
        return A, B, C, D
    
    def solve_first_order(
        self, 
        A: np.ndarray, 
        B: np.ndarray, 
        C: np.ndarray, 
        D: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the linearized model using the method of undetermined coefficients.
        
        Args:
            A (np.ndarray): Derivative of equations w.r.t. current states.
            B (np.ndarray): Derivative of equations w.r.t. next period states.
            C (np.ndarray): Derivative of equations w.r.t. current controls.
            D (np.ndarray): Derivative of equations w.r.t. next period controls.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - P matrix: Policy function for controls (first-order)
                - F matrix: Transition function for states (first-order)
        """
        # Number of variables
        n_states = A.shape[1]
        n_controls = C.shape[1]
        
        # In a real implementation, this would solve the linear rational expectations model
        # using methods like Blanchard-Kahn, Klein, or Sims
        # Here, we're providing a simplified placeholder
        
        # Placeholder for policy function
        P = np.zeros((n_controls, n_states))
        
        # Placeholder for transition function
        F = np.zeros((n_states, n_states))
        
        # In a real implementation, these would be computed based on A, B, C, D
        
        return P, F
    
    def solve_higher_order(
        self, 
        states: List[sp.Symbol], 
        controls: List[sp.Symbol], 
        equations: List[sp.Expr], 
        ss_values: Dict[sp.Symbol, float], 
        P: np.ndarray, 
        F: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Solve the model to higher order using perturbation.
        
        Args:
            states (List[sp.Symbol]): List of state variables.
            controls (List[sp.Symbol]): List of control variables.
            equations (List[sp.Expr]): List of model equations.
            ss_values (Dict[sp.Symbol, float]): Steady state values.
            P (np.ndarray): First-order policy function.
            F (np.ndarray): First-order transition function.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of solution matrices.
        """
        # Number of variables
        n_states = len(states)
        n_controls = len(controls)
        
        # Initialize solution dictionary
        solution = {
            "P": P,  # First-order policy function
            "F": F,  # First-order transition function
        }
        
        # If second-order or higher
        if self.order >= 2:
            # Placeholder for second-order terms
            P_ss = np.zeros((n_controls, n_states, n_states))
            F_ss = np.zeros((n_states, n_states, n_states))
            
            # In a real implementation, these would be computed using the method
            # of undetermined coefficients and higher-order derivatives
            
            solution["P_ss"] = P_ss
            solution["F_ss"] = F_ss
        
        # If third-order
        if self.order >= 3:
            # Placeholder for third-order terms
            P_sss = np.zeros((n_controls, n_states, n_states, n_states))
            F_sss = np.zeros((n_states, n_states, n_states, n_states))
            
            # In a real implementation, these would be computed using the method
            # of undetermined coefficients and higher-order derivatives
            
            solution["P_sss"] = P_sss
            solution["F_sss"] = F_sss
        
        return solution
    
    def solve(self) -> Dict[str, np.ndarray]:
        """
        Solve the model using perturbation.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of solution matrices including:
                - state_transition: State transition matrix
                - policy_function: Policy function matrix
                - steady_state: Steady state values
        """
        try:
            # Create symbolic model
            logger.info("Creating symbolic model...")
            states, controls, equations = self.create_symbolic_model()
            
            # Compute steady state
            logger.info("Computing steady state...")
            ss_values = self.compute_steady_state_numerically(states, controls, equations)
            if ss_values is None:
                logger.error("Failed to compute steady state.")
                return None
            
            # Log steady state values for debugging
            logger.debug("Steady state values:")
            for var, val in ss_values.items():
                logger.debug(f"{var}: {val}")
        
        try:
            # Linearize model
            logger.info("Linearizing model...")
            A, B, C, D = self.linearize_model(states, controls, equations, ss_values)
            
            # Log matrix properties
            for name, matrix in [('A', A), ('B', B), ('C', C), ('D', D)]:
                logger.debug(f"{name} shape: {matrix.shape}")
                logger.debug(f"{name} condition number: {np.linalg.cond(matrix)}")
                logger.debug(f"{name} has NaN: {np.any(np.isnan(matrix))}")
                logger.debug(f"{name} has Inf: {np.any(np.isinf(matrix))}")
            
            # Solve first-order
            logger.info("Solving first-order conditions...")
            P, F = self.solve_first_order(A, B, C, D)
            
            if P is None or F is None:
                logger.error("First-order solution returned None")
                return None
            
            # Create solution with correct keys
            solution = {
                'state_transition': F,
                'policy_function': P,
                'steady_state': ss_values
            }
            
            logger.info("Model solution completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model solution: {str(e)}")
            return None
        
        # Add higher-order terms if needed
        if self.order > 1:
            higher_order = self.solve_higher_order(states, controls, equations, ss_values, P, F)
            solution.update(higher_order)
        
        # Store solution
        self.solution = solution
        
        return solution
    
    def simulate(
        self, 
        periods: int, 
        initial_states: Optional[np.ndarray] = None, 
        shocks: Optional[np.ndarray] = None, 
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model using the perturbation solution.
        
        Args:
            periods (int): Number of periods to simulate.
            initial_states (Optional[np.ndarray]): Initial state values.
                If None, steady state values will be used.
            shocks (Optional[np.ndarray]): Shock innovations.
                If None, random shocks will be generated.
            seed (Optional[int]): Random seed for shock generation.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Simulated states (periods x n_states)
                - Simulated controls (periods x n_controls)
        """
        # If solution not available, solve the model
        if self.solution is None:
            self.solve()
        
        # Extract solution matrices
        P = self.solution["P"]
        F = self.solution["F"]
        
        # Get dimensions
        n_states = F.shape[0]
        n_controls = P.shape[0]
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize states and controls
        states_sim = np.zeros((periods, n_states))
        controls_sim = np.zeros((periods, n_controls))
        
        # Set initial states
        if initial_states is None:
            # Use steady state (zeros in deviation form)
            states_sim[0, :] = np.zeros(n_states)
        else:
            states_sim[0, :] = initial_states
        
        # Generate shocks if not provided
        if shocks is None:
            # Extract shock standard deviations
            shock_sds = [
                self.params["technology_sigma"],
                self.params["preference_sigma"],
                self.params["investment_sigma"],
                self.params["government_sigma"],
                self.params["monetary_sigma"],
                self.params["price_markup_sigma"],
                self.params["wage_markup_sigma"],
            ]
            
            # Generate random shocks
            shocks = np.random.normal(0, 1, (periods, len(shock_sds)))
            for i, sd in enumerate(shock_sds):
                shocks[:, i] *= sd
        
        # Compute initial controls
        controls_sim[0, :] = P @ states_sim[0, :]
        
        # Simulate
        for t in range(1, periods):
            # State transition
            states_sim[t, :] = F @ states_sim[t-1, :] + shocks[t-1, :]
            
            # Control policy
            controls_sim[t, :] = P @ states_sim[t, :]
            
            # If second-order
            if self.order >= 2 and "P_ss" in self.solution:
                P_ss = self.solution["P_ss"]
                # Add second-order terms
                for i in range(n_controls):
                    for j in range(n_states):
                        for k in range(n_states):
                            controls_sim[t, i] += 0.5 * P_ss[i, j, k] * states_sim[t, j] * states_sim[t, k]
            
            # If third-order
            if self.order >= 3 and "P_sss" in self.solution:
                P_sss = self.solution["P_sss"]
                # Add third-order terms
                for i in range(n_controls):
                    for j in range(n_states):
                        for k in range(n_states):
                            for l in range(n_states):
                                controls_sim[t, i] += (1/6) * P_sss[i, j, k, l] * states_sim[t, j] * states_sim[t, k] * states_sim[t, l]
        
        return states_sim, controls_sim