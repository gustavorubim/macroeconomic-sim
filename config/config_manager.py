"""
Configuration manager for the DSGE model.
This module provides a class for managing the configuration of the DSGE model.
"""

import os
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from config.default_config import get_default_config, update_config


class ConfigManager:
    """
    Configuration manager for the DSGE model.
    
    This class provides methods for loading, saving, and updating the configuration
    of the DSGE model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (Optional[str]): Path to a JSON configuration file.
                If provided, the configuration will be loaded from this file.
                If not provided, the default configuration will be used.
        """
        # Start with the default configuration
        self.config = get_default_config()
        
        # If a config path is provided, load and apply it
        if config_path is not None:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to a JSON configuration file.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Update the current configuration with the loaded one
        self.config = update_config(self.config, loaded_config)
    
    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path (str): Path where to save the configuration.
        """
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with the provided updates.
        
        Args:
            updates (Dict[str, Any]): The updates to apply to the configuration.
        """
        self.config = update_config(self.config, updates)
    
    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key (Optional[str]): The key to get. If None, the entire configuration is returned.
            default (Any): The default value to return if the key is not found.
        
        Returns:
            Any: The configuration value.
        """
        if key is None:
            return self.config
        
        # Handle nested keys with dot notation (e.g., "base_model.beta")
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key (str): The key to set.
            value (Any): The value to set.
        """
        # Handle nested keys with dot notation (e.g., "base_model.beta")
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def enable_extension(self, extension_name: str) -> None:
        """
        Enable a model extension.
        
        Args:
            extension_name (str): The name of the extension to enable.
                Must be one of: "financial_extension", "open_economy_extension", "fiscal_extension".
        
        Raises:
            ValueError: If the extension name is not valid.
        """
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}. "
                            f"Must be one of: {', '.join(valid_extensions)}")
        
        self.set(f"{extension_name}.enabled", True)
    
    def disable_extension(self, extension_name: str) -> None:
        """
        Disable a model extension.
        
        Args:
            extension_name (str): The name of the extension to disable.
                Must be one of: "financial_extension", "open_economy_extension", "fiscal_extension".
        
        Raises:
            ValueError: If the extension name is not valid.
        """
        valid_extensions = ["financial_extension", "open_economy_extension", "fiscal_extension"]
        if extension_name not in valid_extensions:
            raise ValueError(f"Invalid extension name: {extension_name}. "
                            f"Must be one of: {', '.join(valid_extensions)}")
        
        self.set(f"{extension_name}.enabled", False)
    
    def set_solution_method(self, method: str, **kwargs) -> None:
        """
        Set the solution method.
        
        Args:
            method (str): The solution method to use.
                Must be one of: "perturbation", "projection".
            **kwargs: Additional parameters for the solution method.
                For perturbation: perturbation_order (int): The order of perturbation (1, 2, or 3).
                For projection: projection_method (str): The projection method ("chebyshev" or "finite_elements").
                               projection_nodes (int): The number of nodes per dimension.
        
        Raises:
            ValueError: If the method is not valid.
        """
        valid_methods = ["perturbation", "projection"]
        if method not in valid_methods:
            raise ValueError(f"Invalid solution method: {method}. "
                            f"Must be one of: {', '.join(valid_methods)}")
        
        updates = {"solution": {"method": method}}
        
        if method == "perturbation" and "perturbation_order" in kwargs:
            order = kwargs["perturbation_order"]
            if order not in [1, 2, 3]:
                raise ValueError(f"Invalid perturbation order: {order}. Must be 1, 2, or 3.")
            updates["solution"]["perturbation_order"] = order
        
        elif method == "projection":
            if "projection_method" in kwargs:
                proj_method = kwargs["projection_method"]
                valid_proj_methods = ["chebyshev", "finite_elements"]
                if proj_method not in valid_proj_methods:
                    raise ValueError(f"Invalid projection method: {proj_method}. "
                                    f"Must be one of: {', '.join(valid_proj_methods)}")
                updates["solution"]["projection_method"] = proj_method
            
            if "projection_nodes" in kwargs:
                nodes = kwargs["projection_nodes"]
                if not isinstance(nodes, int) or nodes < 2:
                    raise ValueError(f"Invalid number of nodes: {nodes}. Must be an integer >= 2.")
                updates["solution"]["projection_nodes"] = nodes
        
        self.update(updates)
    
    def set_data_range(self, start_date: str, end_date: str) -> None:
        """
        Set the date range for data.
        
        Args:
            start_date (str): The start date in ISO format (YYYY-MM-DD).
            end_date (str): The end date in ISO format (YYYY-MM-DD).
        """
        self.update({
            "data": {
                "start_date": start_date,
                "end_date": end_date
            }
        })
    
    def set_estimation_params(self, **kwargs) -> None:
        """
        Set estimation parameters.
        
        Args:
            **kwargs: Estimation parameters to set.
                method (str): The estimation method ("bayesian" or "maximum_likelihood").
                mcmc_algorithm (str): The MCMC algorithm for Bayesian estimation.
                num_chains (int): The number of MCMC chains.
                num_draws (int): The number of draws per chain.
                burn_in (int): The number of burn-in draws.
                tune (int): The number of tuning iterations.
                target_acceptance (float): The target acceptance rate.
        """
        updates = {"estimation": {}}
        
        for key, value in kwargs.items():
            if key == "method":
                valid_methods = ["bayesian", "maximum_likelihood"]
                if value not in valid_methods:
                    raise ValueError(f"Invalid estimation method: {value}. "
                                    f"Must be one of: {', '.join(valid_methods)}")
            
            updates["estimation"][key] = value
        
        self.update(updates)