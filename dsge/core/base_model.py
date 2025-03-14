"""
Base Smets and Wouters (2007) DSGE model implementation.

This module implements the core equations of the Smets and Wouters (2007) model,
which is a medium-scale New Keynesian DSGE model with various frictions.

References:
    Smets, F., & Wouters, R. (2007). Shocks and frictions in US business cycles: 
    A Bayesian DSGE approach. American Economic Review, 97(3), 586-606.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass

from config.config_manager import ConfigManager


@dataclass
class ModelVariables:
    """
    Class to store model variables.
    
    This class provides a structured way to store and access the variables
    of the DSGE model.
    """
    # Endogenous variables
    output: np.ndarray  # Output
    consumption: np.ndarray  # Consumption
    investment: np.ndarray  # Investment
    capital: np.ndarray  # Capital stock
    capital_utilization: np.ndarray  # Capital utilization rate
    labor: np.ndarray  # Labor hours
    real_wage: np.ndarray  # Real wage
    rental_rate: np.ndarray  # Rental rate of capital
    inflation: np.ndarray  # Inflation rate
    price_markup: np.ndarray  # Price markup
    wage_markup: np.ndarray  # Wage markup
    nominal_interest: np.ndarray  # Nominal interest rate
    real_interest: np.ndarray  # Real interest rate
    
    # Exogenous shock processes
    technology_shock: np.ndarray  # Technology shock
    preference_shock: np.ndarray  # Preference shock
    investment_shock: np.ndarray  # Investment adjustment cost shock
    government_shock: np.ndarray  # Government spending shock
    monetary_shock: np.ndarray  # Monetary policy shock
    price_markup_shock: np.ndarray  # Price markup shock
    wage_markup_shock: np.ndarray  # Wage markup shock
    
    @classmethod
    def create_empty(cls, periods: int) -> 'ModelVariables':
        """
        Create an empty ModelVariables instance with arrays of zeros.
        
        Args:
            periods (int): Number of time periods.
            
        Returns:
            ModelVariables: An instance with arrays of zeros.
        """
        return cls(
            # Endogenous variables
            output=np.zeros(periods),
            consumption=np.zeros(periods),
            investment=np.zeros(periods),
            capital=np.zeros(periods),
            capital_utilization=np.zeros(periods),
            labor=np.zeros(periods),
            real_wage=np.zeros(periods),
            rental_rate=np.zeros(periods),
            inflation=np.zeros(periods),
            price_markup=np.zeros(periods),
            wage_markup=np.zeros(periods),
            nominal_interest=np.zeros(periods),
            real_interest=np.zeros(periods),
            
            # Exogenous shock processes
            technology_shock=np.zeros(periods),
            preference_shock=np.zeros(periods),
            investment_shock=np.zeros(periods),
            government_shock=np.zeros(periods),
            monetary_shock=np.zeros(periods),
            price_markup_shock=np.zeros(periods),
            wage_markup_shock=np.zeros(periods),
        )
@dataclass
class ModelCategories:
    state: list = None
    control: list = None
    shock: list = None
    equations: list = None


class SmetsWoutersModel:
    """
    Implementation of the Smets and Wouters (2007) DSGE model.
    
    This class implements the core equations of the Smets and Wouters (2007) model,
    which is a medium-scale New Keynesian DSGE model with various frictions.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigManager]] = None):
        """
        Initialize the Smets and Wouters model.
        
        Args:
            config (Optional[Union[Dict[str, Any], ConfigManager]]): Model configuration.
                If a dictionary is provided, it will be used as the configuration.
                If a ConfigManager is provided, its configuration will be used.
                If None, the default configuration will be used.
        """
        # Initialize configuration
        if config is None:
            self.config = ConfigManager()
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.update(config)
        else:
            self.config = config
        
        # Extract model parameters from configuration
        self.params = self._extract_parameters()
        
        # Initialize model variables and steady state values
        self.steady_state = None
        self.variables = ModelCategories(
            state=['capital', 'technology_shock', 'preference_shock', 'investment_shock',
                  'government_shock', 'monetary_shock', 'price_markup_shock', 'wage_markup_shock'],
            control=['output', 'consumption', 'investment', 'labor', 'real_wage',
                    'rental_rate', 'inflation', 'nominal_interest'],
            shock=['technology', 'preference', 'investment', 'government',
                  'monetary', 'price_markup', 'wage_markup']
        )
        
        # Initialize model equations
        self.equations = self.get_model_equations()
        
        # Add extension variables if enabled
        if self.config.get('model.extensions.financial', False):
            self.variables.control.extend(['spread', 'net_worth', 'leverage'])
            
            # Update steady state calculation for financial variables
            if self.steady_state is not None:
                self.steady_state.update({
                    'spread': 0.0,  # In steady state, risk spreads are typically zero
                    'net_worth': 1.0,  # Normalized net worth
                    'leverage': 1.0,  # Normalized leverage
                })
                
        if self.config.get('model.extensions.open_economy', False):
            self.variables.state.extend(['exports', 'imports', 'exchange_rate', 'foreign_output'])
            
            # Update steady state calculation for open economy variables
            if self.steady_state is not None:
                self.steady_state.update({
                    'exports': 0.2,  # 20% of GDP in steady state
                    'imports': 0.2,  # Balanced trade in steady state
                    'exchange_rate': 1.0,  # Normalized exchange rate
                    'foreign_output': 1.0,  # Normalized foreign output
                })
    
    def _extract_parameters(self) -> Dict[str, float]:
        """
        Extract model parameters from the configuration.
        
        Returns:
            Dict[str, float]: Dictionary of model parameters.
        """
        # Get base model parameters
        params = self.config.get("base_model").copy()
        
        # Add shock parameters
        shock_params = self.config.get("shocks")
        for shock_name, shock_config in shock_params.items():
            params[f"{shock_name}_rho"] = shock_config["rho"]
            params[f"{shock_name}_sigma"] = shock_config["sigma"]
        
        return params
    
    def compute_steady_state(self) -> Dict[str, float]:
        """
        Compute the steady state of the model.
        
        Returns:
            Dict[str, float]: Dictionary of steady state values.
        """
        # Extract parameters
        beta = self.params["beta"]
        delta = self.params["delta"]
        alpha = self.params["alpha"]
        g_y = self.params["g_y"]
        pi_bar = self.params["pi_bar"]
        
        # Compute steady state values
        r_k_ss = 1/beta - (1 - delta)  # Rental rate of capital
        w_ss = 1.0  # Normalized real wage
        l_ss = 1.0  # Normalized labor
        k_l_ratio = alpha / (1 - alpha) * w_ss / r_k_ss  # Capital-labor ratio
        k_ss = k_l_ratio * l_ss  # Capital stock
        y_ss = k_ss**alpha * l_ss**(1-alpha)  # Output
        i_ss = delta * k_ss  # Investment
        c_ss = y_ss - i_ss  # Consumption
        r_ss = 1/beta * pi_bar  # Nominal interest rate
        
        # Store steady state values
        self.steady_state = {
            "output": y_ss,
            "consumption": c_ss,
            "investment": i_ss,
            "capital": k_ss,
            "capital_utilization": 1.0,
            "labor": l_ss,
            "real_wage": w_ss,
            "rental_rate": r_k_ss,
            "inflation": pi_bar,
            "price_markup": 1.0,
            "wage_markup": 1.0,
            "nominal_interest": r_ss,
            "real_interest": r_ss / pi_bar,
        }
        
        # Add financial extension steady states if enabled
        if self.config.get('model.extensions.financial', False):
            self.steady_state.update({
                'spread': 0.0,  # Zero spread in steady state
                'net_worth': y_ss,  # Net worth normalized to output
                'leverage': 1.0,  # Unit leverage in steady state
            })
        
        # Add open economy extension steady states if enabled
        if self.config.get('model.extensions.open_economy', False):
            exports_ss = 0.2 * y_ss  # 20% of GDP
            imports_ss = exports_ss  # Balanced trade
            
            self.steady_state.update({
                'exports': exports_ss,
                'imports': imports_ss,
                'exchange_rate': 1.0,  # Normalized exchange rate
                'foreign_output': y_ss,  # Foreign output normalized to domestic
            })
        
        return self.steady_state
    
    def initialize_variables(self, periods: int) -> ModelVariables:
        """
        Initialize model variables.
        
        Args:
            periods (int): Number of time periods.
            
        Returns:
            ModelVariables: Initialized model variables.
        """
        # Create empty variables
        self.variables = ModelVariables.create_empty(periods)
        
        # If steady state has not been computed, compute it
        if self.steady_state is None:
            self.compute_steady_state()
        
        # Initialize variables to steady state values
        for t in range(periods):
            self.variables.output[t] = self.steady_state["output"]
            self.variables.consumption[t] = self.steady_state["consumption"]
            self.variables.investment[t] = self.steady_state["investment"]
            self.variables.capital[t] = self.steady_state["capital"]
            self.variables.capital_utilization[t] = self.steady_state["capital_utilization"]
            self.variables.labor[t] = self.steady_state["labor"]
            self.variables.real_wage[t] = self.steady_state["real_wage"]
            self.variables.rental_rate[t] = self.steady_state["rental_rate"]
            self.variables.inflation[t] = self.steady_state["inflation"]
            self.variables.price_markup[t] = self.steady_state["price_markup"]
            self.variables.wage_markup[t] = self.steady_state["wage_markup"]
            self.variables.nominal_interest[t] = self.steady_state["nominal_interest"]
            self.variables.real_interest[t] = self.steady_state["real_interest"]
        
        return self.variables
    
    def update_parameters(self, new_params: Dict[str, float]) -> None:
        """
        Update model parameters.
        
        Args:
            new_params: Dictionary of parameter names and values to update
        """
        # Validate parameters
        for name, value in new_params.items():
            if name not in self.params:
                raise ValueError(f"Unknown parameter: {name}")
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {name} must be numeric, got {type(value)}")
                
        # Update parameters
        self.params.update(new_params)
        
        # Reset cached computations
        if hasattr(self, 'steady_state'):
            delattr(self, 'steady_state')
            
    def generate_shock_processes(self, periods: int, seed: Optional[int] = None) -> None:
        """
        Generate exogenous shock processes.
        
        Args:
            periods (int): Number of time periods.
            seed (Optional[int]): Random seed for reproducibility.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # If variables have not been initialized, initialize them
        if self.variables is None:
            self.initialize_variables(periods)
        
        # Generate shock processes
        for shock_name in ["technology", "preference", "investment", "government", 
                          "monetary", "price_markup", "wage_markup"]:
            rho = self.params[f"{shock_name}_rho"]
            sigma = self.params[f"{shock_name}_sigma"]
            
            # Generate innovations
            innovations = np.random.normal(0, sigma, periods)
            
            # Initialize shock process
            shock = np.zeros(periods)
            
            # Generate AR(1) process
            for t in range(1, periods):
                shock[t] = rho * shock[t-1] + innovations[t]
            
            # Store shock process
            setattr(self.variables, f"{shock_name}_shock", shock)
    
    def get_model_equations(self) -> List[Callable]:
        """
        Get the model equations as a list of callable functions.
        
        Returns:
            List[Callable]: List of equation functions.
        """
        beta = self.params["beta"]
        delta = self.params["delta"]
        alpha = self.params["alpha"]
        epsilon_p = self.params.get("epsilon_p", 6.0)
        epsilon_w = self.params.get("epsilon_w", 6.0)
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
        
        # Define equation functions
        def euler_equation(vars_t, vars_tp1):
            """Consumption Euler equation"""
            return (vars_t['lambda_t'] - beta * (1 + vars_t['r_t']) *
                   vars_tp1['lambda_t'] / vars_tp1['pi_t'])
        
        def capital_accumulation(vars_t, vars_tm1):
            """Capital accumulation equation"""
            return vars_t['k_t'] - (1 - delta) * vars_tm1['k_t'] - vars_t['i_t']
        
        def production_function(vars_t, vars_tm1):
            """Production function"""
            return (vars_t['y_t'] - vars_t['a_t'] *
                   vars_tm1['k_t']**alpha * vars_t['l_t']**(1-alpha))
        
        def rental_rate(vars_t, vars_tm1):
            """Rental rate of capital"""
            return (vars_t['r_k_t'] - alpha * vars_t['mc_t'] *
                   vars_t['y_t'] / vars_tm1['k_t'])
        
        def wage_equation(vars_t):
            """Real wage equation"""
            return (vars_t['w_t'] - (1 - alpha) * vars_t['mc_t'] *
                   vars_t['y_t'] / vars_t['l_t'])
        
        def price_phillips(vars_t, vars_tm1):
            """Price Phillips curve"""
            return (vars_t['pi_t'] - (1 - xi_p) * vars_t['pi_star_t'] -
                   xi_p * vars_tm1['pi_t']**iota_p * pi_bar**(1-iota_p))
        
        def wage_phillips(vars_t, vars_tm1):
            """Wage Phillips curve"""
            return (vars_t['w_t'] - (1 - xi_w) * vars_t['w_star_t'] -
                   xi_w * vars_tm1['w_t'] * vars_tm1['pi_t']**iota_w *
                   pi_bar**(1-iota_w) / vars_t['pi_t'])
        
        def monetary_policy(vars_t, vars_tm1):
            """Monetary policy rule"""
            return (vars_t['r_t'] - rho_r * vars_tm1['r_t'] -
                   (1 - rho_r) * (r_bar + phi_pi * (vars_t['pi_t'] - pi_bar) +
                   phi_y * (vars_t['y_t'] - self.steady_state['output']) +
                   phi_dy * (vars_t['y_t'] - vars_tm1['y_t'])) -
                   vars_t['r_shock_t'])
        
        def market_clearing(vars_t):
            """Market clearing condition"""
            return vars_t['y_t'] - vars_t['c_t'] - vars_t['i_t'] - vars_t['g_t']
        
        equations = [
            euler_equation,
            capital_accumulation,
            production_function,
            rental_rate,
            wage_equation,
            price_phillips,
            wage_phillips,
            monetary_policy,
            market_clearing
        ]
        
        return equations
    
    def log_linearize(self) -> List[str]:
        """
        Get the log-linearized model equations.
        
        Returns:
            List[str]: List of log-linearized model equations.
        """
        # Log-linearized equations
        equations = [
            # Consumption Euler equation
            "c_t = (h/(1+h)) * c_{t-1} + (1/(1+h)) * E_t[c_{t+1}] + ((sigma_c-1)*w*l/c)/(sigma_c*(1+h)) * (l_t - E_t[l_{t+1}]) - (1-h)/(sigma_c*(1+h)) * (r_t - E_t[pi_{t+1}] + b_t)",
            
            # Investment Euler equation
            "i_t = (1/(1+beta)) * i_{t-1} + (beta/(1+beta)) * E_t[i_{t+1}] + (1/(1+beta)) * (1/gamma) * q_t + e_i_t",
            
            # Tobin's Q
            "q_t = -(r_t - E_t[pi_{t+1}]) + ((1-delta)/(r_k+1-delta)) * E_t[q_{t+1}] + (r_k/(r_k+1-delta)) * E_t[r_k_{t+1}] + b_t",
            
            # Capital accumulation
            "k_t = (1-delta) * k_{t-1} + delta * i_t",
            
            # Production function
            "y_t = phi * (alpha * k_{t-1} + (1-alpha) * l_t + e_a_t)",
            
            # Capital-labor ratio
            "k_{t-1} - l_t = w_t - r_k_t",
            
            # Phillips curve
            "pi_t = (iota_p/(1+beta*iota_p)) * pi_{t-1} + (beta/(1+beta*iota_p)) * E_t[pi_{t+1}] + ((1-xi_p)*(1-beta*xi_p)/((1+beta*iota_p)*xi_p)) * mc_t + e_p_t",
            
            # Marginal cost
            "mc_t = alpha * r_k_t + (1-alpha) * w_t - e_a_t",
            
            # Wage Phillips curve
            "w_t = (1/(1+beta)) * w_{t-1} + (beta/(1+beta)) * E_t[w_{t+1}] + (iota_w/(1+beta)) * pi_{t-1} - ((1+beta*iota_w)/(1+beta)) * pi_t + (beta/(1+beta)) * E_t[pi_{t+1}] + ((1-xi_w)*(1-beta*xi_w)/((1+beta)*xi_w)) * (mrs_t - w_t) + e_w_t",
            
            # Labor supply
            "mrs_t = sigma_l * l_t + (1/(1-h)) * (c_t - h*c_{t-1}) - b_t",
            
            # Monetary policy rule
            "r_t = rho_r * r_{t-1} + (1-rho_r) * (phi_pi * pi_t + phi_y * y_gap_t) + phi_dy * (y_gap_t - y_gap_{t-1}) + e_r_t",
            
            # Output gap
            "y_gap_t = y_t - y_t^n",
            
            # Resource constraint
            "y_t = (c/y) * c_t + (i/y) * i_t + e_g_t",
            
            # Natural output
            "y_t^n = ...",  # This would be derived from the model without nominal rigidities
        ]
        
        return equations
    
    def simulate(self, periods: int, seed: Optional[int] = None) -> ModelVariables:
        """
        Simulate the model.
        
        This method is a placeholder. The actual simulation would be implemented
        using the solution method specified in the configuration.
        
        Args:
            periods (int): Number of time periods to simulate.
            seed (Optional[int]): Random seed for reproducibility.
            
        Returns:
            ModelVariables: Simulated model variables.
        """
        # Initialize variables
        self.initialize_variables(periods)
        
        # Generate shock processes
        self.generate_shock_processes(periods, seed)
        
        # Placeholder for actual simulation
        # In a real implementation, this would use the solution method
        # specified in the configuration to solve and simulate the model.
        
        return self.variables