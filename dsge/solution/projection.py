"""
Projection solution method for the DSGE model.

This module implements projection methods for solving the DSGE model,
including Chebyshev polynomial collocation and finite elements methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import sympy as sp
from scipy.optimize import minimize
import logging

from config.config_manager import ConfigManager
from dsge.core.base_model import SmetsWoutersModel

# Set up logging
logger = logging.getLogger(__name__)


class ProjectionSolver:
    """
    Class for solving DSGE models using projection methods.
    """
    
    def __init__(
        self, 
        model: SmetsWoutersModel,
        method: str = "chebyshev",
        nodes: int = 5
    ):
        """
        Initialize the projection solver.
        
        Args:
            model (SmetsWoutersModel): The DSGE model to solve.
            method (str): Projection method to use.
                Options: "chebyshev" or "finite_elements".
            nodes (int): Number of nodes per dimension.
                
        Raises:
            ValueError: If the method is not valid.
        """
        # Check method
        valid_methods = ["chebyshev", "finite_elements"]
        if method not in valid_methods:
            raise ValueError(f"Invalid projection method: {method}. "
                            f"Must be one of: {', '.join(valid_methods)}")
        
        self.model = model
        self.method = method
        self.nodes = nodes
        
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
    
    def generate_chebyshev_nodes(
        self, 
        n_dims: int, 
        n_nodes: int, 
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Generate Chebyshev nodes for the projection method.
        
        Args:
            n_dims (int): Number of dimensions (state variables).
            n_nodes (int): Number of nodes per dimension.
            bounds (List[Tuple[float, float]]): Bounds for each dimension.
                
        Returns:
            np.ndarray: Array of Chebyshev nodes.
        """
        # Generate Chebyshev nodes in [-1, 1]
        nodes_1d = np.cos(np.pi * (2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes))
        
        # Scale nodes to bounds
        scaled_nodes = []
        for i in range(n_dims):
            a, b = bounds[i]
            scaled = 0.5 * (a + b) + 0.5 * (b - a) * nodes_1d
            scaled_nodes.append(scaled)
        
        # Generate grid of nodes
        grid = np.meshgrid(*scaled_nodes, indexing='ij')
        
        # Reshape to (n_points, n_dims)
        points = np.column_stack([g.flatten() for g in grid])
        
        return points
    
    def generate_finite_element_nodes(
        self, 
        n_dims: int, 
        n_nodes: int, 
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Generate finite element nodes for the projection method.
        
        Args:
            n_dims (int): Number of dimensions (state variables).
            n_nodes (int): Number of nodes per dimension.
            bounds (List[Tuple[float, float]]): Bounds for each dimension.
                
        Returns:
            np.ndarray: Array of finite element nodes.
        """
        # Generate evenly spaced nodes
        nodes_1d = []
        for i in range(n_dims):
            a, b = bounds[i]
            nodes = np.linspace(a, b, n_nodes)
            nodes_1d.append(nodes)
        
        # Generate grid of nodes
        grid = np.meshgrid(*nodes_1d, indexing='ij')
        
        # Reshape to (n_points, n_dims)
        points = np.column_stack([g.flatten() for g in grid])
        
        return points
    
    def chebyshev_basis(
        self, 
        x: np.ndarray, 
        degree: int, 
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Compute Chebyshev basis functions.
        
        Args:
            x (np.ndarray): Points at which to evaluate the basis functions.
            degree (int): Degree of the Chebyshev polynomials.
            bounds (List[Tuple[float, float]]): Bounds for each dimension.
                
        Returns:
            np.ndarray: Array of basis function values.
        """
        # Number of dimensions
        n_dims = x.shape[1]
        
        # Scale x to [-1, 1]
        x_scaled = np.zeros_like(x)
        for i in range(n_dims):
            a, b = bounds[i]
            x_scaled[:, i] = 2 * (x[:, i] - a) / (b - a) - 1
        
        # Compute Chebyshev polynomials for each dimension
        T = []
        for i in range(n_dims):
            T_i = []
            for j in range(degree + 1):
                if j == 0:
                    T_i.append(np.ones_like(x_scaled[:, i]))
                elif j == 1:
                    T_i.append(x_scaled[:, i])
                else:
                    T_i.append(2 * x_scaled[:, i] * T_i[j-1] - T_i[j-2])
            T.append(T_i)
        
        # Compute tensor product basis
        n_points = x.shape[0]
        n_basis = (degree + 1) ** n_dims
        basis = np.ones((n_points, n_basis))
        
        # Generate multi-indices
        multi_indices = np.zeros((n_basis, n_dims), dtype=int)
        for i in range(n_basis):
            idx = i
            for j in range(n_dims):
                multi_indices[i, j] = idx % (degree + 1)
                idx //= (degree + 1)
        
        # Compute basis functions
        for i in range(n_basis):
            for j in range(n_dims):
                basis[:, i] *= T[j][multi_indices[i, j]]
        
        return basis
    
    def finite_element_basis(
        self, 
        x: np.ndarray, 
        nodes: np.ndarray, 
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Compute finite element basis functions.
        
        Args:
            x (np.ndarray): Points at which to evaluate the basis functions.
            nodes (np.ndarray): Nodes of the finite element grid.
            bounds (List[Tuple[float, float]]): Bounds for each dimension.
                
        Returns:
            np.ndarray: Array of basis function values.
        """
        # Number of dimensions
        n_dims = x.shape[1]
        
        # Number of nodes per dimension
        n_nodes = int(np.round(nodes.shape[0] ** (1 / n_dims)))
        
        # Compute basis functions
        n_points = x.shape[0]
        n_basis = nodes.shape[0]
        basis = np.zeros((n_points, n_basis))
        
        # For each point
        for i in range(n_points):
            # Find the cell containing the point
            cell_idx = []
            for j in range(n_dims):
                a, b = bounds[j]
                idx = int(np.floor((x[i, j] - a) / (b - a) * (n_nodes - 1)))
                idx = max(0, min(idx, n_nodes - 2))
                cell_idx.append(idx)
            
            # Compute basis function values
            for j in range(n_basis):
                # Check if the node is a vertex of the cell
                is_vertex = True
                for k in range(n_dims):
                    node_idx = j // (n_nodes ** k) % n_nodes
                    if node_idx != cell_idx[k] and node_idx != cell_idx[k] + 1:
                        is_vertex = False
                        break
                
                if is_vertex:
                    # Compute the value of the basis function
                    value = 1.0
                    for k in range(n_dims):
                        node_idx = j // (n_nodes ** k) % n_nodes
                        a, b = bounds[k]
                        h = (b - a) / (n_nodes - 1)
                        if node_idx == cell_idx[k]:
                            value *= 1 - (x[i, k] - (a + cell_idx[k] * h)) / h
                        else:  # node_idx == cell_idx[k] + 1
                            value *= (x[i, k] - (a + cell_idx[k] * h)) / h
                    
                    basis[i, j] = value
        
        return basis
    
    def solve_projection(
        self, 
        states: List[sp.Symbol], 
        controls: List[sp.Symbol], 
        equations: List[sp.Expr], 
        ss_values: Dict[sp.Symbol, float]
    ) -> Dict[str, Any]:
        """
        Solve the model using projection methods.
        
        Args:
            states (List[sp.Symbol]): List of state variables.
            controls (List[sp.Symbol]): List of control variables.
            equations (List[sp.Expr]): List of model equations.
            ss_values (Dict[sp.Symbol, float]): Steady state values.
            
        Returns:
            Dict[str, Any]: Dictionary of solution components.
        """
        # Number of variables
        n_states = len(states)
        n_controls = len(controls)
        
        # Define bounds for state variables
        # In a real implementation, these would be chosen based on the model
        bounds = []
        for i, var in enumerate(states):
            ss_val = ss_values[var]
            # Set bounds to +/- 30% around steady state
            bounds.append((ss_val * 0.7, ss_val * 1.3))
        
        # Generate nodes
        if self.method == "chebyshev":
            nodes = self.generate_chebyshev_nodes(n_states, self.nodes, bounds)
        else:  # finite_elements
            nodes = self.generate_finite_element_nodes(n_states, self.nodes, bounds)
        
        # Number of basis functions
        n_basis = nodes.shape[0]
        
        # Initialize coefficients
        coeffs = np.zeros((n_controls, n_basis))
        
        # Define the residual function
        def residual_function(coeffs_flat):
            # Reshape coefficients
            coeffs = coeffs_flat.reshape((n_controls, n_basis))
            
            # Compute residuals at each node
            residuals = []
            for i in range(nodes.shape[0]):
                # State values at this node
                state_vals = {var: nodes[i, j] for j, var in enumerate(states)}
                
                # Compute control values using the approximation
                if self.method == "chebyshev":
                    basis = self.chebyshev_basis(nodes[i:i+1, :], self.nodes - 1, bounds)
                else:  # finite_elements
                    basis = self.finite_element_basis(nodes[i:i+1, :], nodes, bounds)
                
                control_vals = {}
                for j, var in enumerate(controls):
                    control_vals[var] = np.dot(coeffs[j, :], basis[0, :])
                
                # Combine state and control values
                vals = {**state_vals, **control_vals}
                
                # Evaluate residuals
                for eq in equations:
                    residuals.append(float(eq.subs(vals)))
            
            # Return sum of squared residuals
            return np.sum(np.array(residuals) ** 2)
        
        # Optimize to find coefficients
        result = minimize(
            residual_function,
            coeffs.flatten(),
            method='BFGS',
            options={'disp': True}
        )
        
        # Reshape solution
        coeffs = result.x.reshape((n_controls, n_basis))
        
        # Store solution
        solution = {
            "method": self.method,
            "nodes": nodes,
            "bounds": bounds,
            "coeffs": coeffs,
            "states": states,
            "controls": controls,
        }
        
        return solution
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the model using projection methods.
        
        Returns:
            Dict[str, Any]: Dictionary of solution components.
        """
        # Create symbolic model
        states, controls, equations = self.create_symbolic_model()
        
        # Compute steady state
        ss_values = {}
        for var in states + controls:
            if var in self.steady_state:
                ss_values[var] = self.steady_state[var]
            else:
                # For variables not in steady_state, set to 0 (for shocks)
                ss_values[var] = 0.0
        
        # Solve using projection
        solution = self.solve_projection(states, controls, equations, ss_values)
        
        # Store solution
        self.solution = solution
        
        return solution
    
    def evaluate_solution(
        self, 
        state_values: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the solution at given state values.
        
        Args:
            state_values (np.ndarray): State values at which to evaluate the solution.
                Shape: (n_points, n_states)
                
        Returns:
            np.ndarray: Control values.
                Shape: (n_points, n_controls)
        """
        # If solution not available, solve the model
        if self.solution is None:
            self.solve()
        
        # Extract solution components
        method = self.solution["method"]
        nodes = self.solution["nodes"]
        bounds = self.solution["bounds"]
        coeffs = self.solution["coeffs"]
        
        # Number of controls
        n_controls = coeffs.shape[0]
        
        # Number of points
        n_points = state_values.shape[0]
        
        # Compute basis functions
        if method == "chebyshev":
            basis = self.chebyshev_basis(state_values, self.nodes - 1, bounds)
        else:  # finite_elements
            basis = self.finite_element_basis(state_values, nodes, bounds)
        
        # Compute control values
        control_values = np.zeros((n_points, n_controls))
        for i in range(n_controls):
            control_values[:, i] = np.dot(basis, coeffs[i, :])
        
        return control_values
    
    def simulate(
        self, 
        periods: int, 
        initial_states: Optional[np.ndarray] = None, 
        shocks: Optional[np.ndarray] = None, 
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model using the projection solution.
        
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
        
        # Extract solution components
        states = self.solution["states"]
        
        # Number of state variables
        n_states = len(states)
        
        # Get number of controls
        n_controls = self.solution["coeffs"].shape[0]
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize states and controls
        states_sim = np.zeros((periods, n_states))
        controls_sim = np.zeros((periods, n_controls))
        
        # Set initial states
        if initial_states is None:
            # Use steady state values
            for i, var in enumerate(states):
                if var in self.steady_state:
                    states_sim[0, i] = self.steady_state[var]
                else:
                    # For variables not in steady_state, set to 0 (for shocks)
                    states_sim[0, i] = 0.0
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
        controls_sim[0, :] = self.evaluate_solution(states_sim[0:1, :])[0, :]
        
        # Simulate
        for t in range(1, periods):
            # Extract shock persistence parameters
            rho_a = self.params["technology_rho"]
            rho_b = self.params["preference_rho"]
            rho_i = self.params["investment_rho"]
            rho_g = self.params["government_rho"]
            rho_r = self.params["monetary_rho"]
            rho_p = self.params["price_markup_rho"]
            rho_w = self.params["wage_markup_rho"]
            
            # State transition
            # Capital (state 0)
            states_sim[t, 0] = (1 - self.params["delta"]) * states_sim[t-1, 0] + controls_sim[t-1, 2]  # Investment
            
            # Shock processes (states 1-7)
            states_sim[t, 1] = rho_a * states_sim[t-1, 1] + shocks[t-1, 0]  # Technology
            states_sim[t, 2] = rho_b * states_sim[t-1, 2] + shocks[t-1, 1]  # Preference
            states_sim[t, 3] = rho_i * states_sim[t-1, 3] + shocks[t-1, 2]  # Investment
            states_sim[t, 4] = rho_g * states_sim[t-1, 4] + shocks[t-1, 3]  # Government
            states_sim[t, 5] = rho_r * states_sim[t-1, 5] + shocks[t-1, 4]  # Monetary
            states_sim[t, 6] = rho_p * states_sim[t-1, 6] + shocks[t-1, 5]  # Price markup
            states_sim[t, 7] = rho_w * states_sim[t-1, 7] + shocks[t-1, 6]  # Wage markup
            
            # Compute controls
            controls_sim[t, :] = self.evaluate_solution(states_sim[t:t+1, :])[0, :]
        
        return states_sim, controls_sim