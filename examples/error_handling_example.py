#!/usr/bin/env python
"""
Error Handling Example

This script demonstrates robust error handling and logging strategies:
1. Comprehensive exception handling
2. Detailed logging framework
3. Input validation patterns
4. Recovery strategies
5. Debugging tools

The example shows how to build robust and reliable DSGE model applications.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import logging
import traceback
from datetime import datetime
import re
import argparse
from functools import wraps
import inspect
import tempfile
import shutil
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type

# Configure logging
def setup_logging(log_dir="logs", log_level=logging.INFO, log_to_console=True):
    """
    Set up logging with file rotation and optional console output.
    
    Args:
        log_dir (str): Directory for log files
        log_level (int): Logging level
        log_to_console (bool): Whether to log to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("dsge_model")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create file handler with rotation
    log_file = os.path.join(log_dir, f"dsge_model_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create logger
logger = setup_logging()


# Custom exception hierarchy
class DSGEError(Exception):
    """Base class for all DSGE model exceptions."""
    pass

class ConfigurationError(DSGEError):
    """Exception raised for configuration errors."""
    pass

class ValidationError(DSGEError):
    """Exception raised for validation errors."""
    pass

class ModelError(DSGEError):
    """Exception raised for model-related errors."""
    pass

class SolutionError(ModelError):
    """Exception raised for solution-related errors."""
    pass

class EstimationError(ModelError):
    """Exception raised for estimation-related errors."""
    pass

class DataError(DSGEError):
    """Exception raised for data-related errors."""
    pass

class IOError(DSGEError):
    """Exception raised for input/output errors."""
    pass


# Input validation decorators and functions
def validate_types(**type_specs):
    """
    Decorator to validate function argument types.
    
    Args:
        **type_specs: Mapping of parameter names to expected types
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_specs.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_range(param_name, min_value=None, max_value=None, inclusive=True):
    """
    Decorator to validate that a parameter is within a specified range.
    
    Args:
        param_name (str): Name of the parameter to validate
        min_value (float, optional): Minimum allowed value
        max_value (float, optional): Maximum allowed value
        inclusive (bool): Whether the range is inclusive
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Get parameter value
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                
                # Check minimum value
                if min_value is not None:
                    if inclusive and value < min_value:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be >= {min_value}, got {value}"
                        )
                    elif not inclusive and value <= min_value:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be > {min_value}, got {value}"
                        )
                
                # Check maximum value
                if max_value is not None:
                    if inclusive and value > max_value:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be <= {max_value}, got {value}"
                        )
                    elif not inclusive and value >= max_value:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be < {max_value}, got {value}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_config(config, schema):
    """
    Validate a configuration against a schema.
    
    Args:
        config (dict): Configuration to validate
        schema (dict): Schema to validate against
        
    Raises:
        ValidationError: If validation fails
    """
    def validate_object(obj, schema_obj, path=""):
        # Check required fields
        if "required" in schema_obj:
            for field in schema_obj["required"]:
                if field not in obj:
                    raise ValidationError(f"Missing required field: {path + field}")
        
        # Check properties
        if "properties" in schema_obj:
            for field, field_schema in schema_obj["properties"].items():
                if field in obj:
                    field_path = f"{path}{field}."
                    field_value = obj[field]
                    
                    # Check type
                    if "type" in field_schema:
                        expected_type = field_schema["type"]
                        if expected_type == "object" and not isinstance(field_value, dict):
                            raise ValidationError(
                                f"Field {path + field} must be an object, got {type(field_value).__name__}"
                            )
                        elif expected_type == "array" and not isinstance(field_value, list):
                            raise ValidationError(
                                f"Field {path + field} must be an array, got {type(field_value).__name__}"
                            )
                        elif expected_type == "string" and not isinstance(field_value, str):
                            raise ValidationError(
                                f"Field {path + field} must be a string, got {type(field_value).__name__}"
                            )
                        elif expected_type == "number" and not isinstance(field_value, (int, float)):
                            raise ValidationError(
                                f"Field {path + field} must be a number, got {type(field_value).__name__}"
                            )
                        elif expected_type == "boolean" and not isinstance(field_value, bool):
                            raise ValidationError(
                                f"Field {path + field} must be a boolean, got {type(field_value).__name__}"
                            )
                    
                    # Check enum
                    if "enum" in field_schema and field_value not in field_schema["enum"]:
                        raise ValidationError(
                            f"Field {path + field} must be one of {field_schema['enum']}, got {field_value}"
                        )
                    
                    # Check minimum/maximum
                    if "minimum" in field_schema and field_value < field_schema["minimum"]:
                        raise ValidationError(
                            f"Field {path + field} must be >= {field_schema['minimum']}, got {field_value}"
                        )
                    if "maximum" in field_schema and field_value > field_schema["maximum"]:
                        raise ValidationError(
                            f"Field {path + field} must be <= {field_schema['maximum']}, got {field_value}"
                        )
                    
                    # Recursively validate objects
                    if "type" in field_schema and field_schema["type"] == "object":
                        validate_object(field_value, field_schema, field_path)
                    
                    # Validate array items
                    if "type" in field_schema and field_schema["type"] == "array" and "items" in field_schema:
                        for i, item in enumerate(field_value):
                            if "type" in field_schema["items"] and field_schema["items"]["type"] == "object":
                                validate_object(item, field_schema["items"], f"{field_path}[{i}].")
    
    # Start validation
    validate_object(config, schema)


# Error handling and recovery strategies
@contextmanager
def error_context(error_type=Exception, recovery_func=None, max_retries=3, retry_delay=1.0):
    """
    Context manager for error handling with optional recovery.
    
    Args:
        error_type (Type[Exception]): Type of error to catch
        recovery_func (Callable, optional): Function to call for recovery
        max_retries (int): Maximum number of retries
        retry_delay (float): Delay between retries in seconds
        
    Yields:
        None
    """
    retries = 0
    while True:
        try:
            yield
            break  # Success, exit the loop
        except error_type as e:
            retries += 1
            logger.warning(f"Error occurred: {str(e)}")
            
            if retries >= max_retries:
                logger.error(f"Maximum retries ({max_retries}) reached. Giving up.")
                raise
            
            logger.info(f"Retry {retries}/{max_retries} after {retry_delay} seconds...")
            
            if recovery_func:
                try:
                    recovery_func(e, retries)
                except Exception as recovery_error:
                    logger.error(f"Recovery function failed: {str(recovery_error)}")
            
            time.sleep(retry_delay)

def safe_file_operation(operation, *args, backup=True, **kwargs):
    """
    Safely perform a file operation with backup.
    
    Args:
        operation (Callable): File operation function
        *args: Arguments for the operation
        backup (bool): Whether to create a backup
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Any: Result of the operation
    """
    # Check if the first argument is a file path
    if args and isinstance(args[0], str) and os.path.exists(args[0]):
        file_path = args[0]
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            try:
                shutil.copy2(file_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
    
    # Perform the operation
    try:
        result = operation(*args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"File operation failed: {str(e)}")
        
        # Restore from backup if available
        if backup and 'file_path' in locals() and os.path.exists(f"{file_path}.bak"):
            try:
                shutil.copy2(f"{file_path}.bak", file_path)
                logger.info(f"Restored from backup: {file_path}")
            except Exception as restore_error:
                logger.error(f"Failed to restore from backup: {str(restore_error)}")
        
        raise


# Performance monitoring and debugging
def timeit(func):
    """
    Decorator to measure and log function execution time.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        
        return result
    return wrapper

def memory_usage(func):
    """
    Decorator to measure and log memory usage of a function.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # Memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run function
            result = func(*args, **kwargs)
            
            # Memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            logger.info(f"Function {func.__name__} memory usage: {mem_after - mem_before:.2f} MB")
            
            return result
        
        except ImportError:
            logger.warning("psutil not available. Memory usage monitoring skipped.")
            return func(*args, **kwargs)
    
    return wrapper

def debug_args(func):
    """
    Decorator to log function arguments for debugging.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(func).args
        arg_values = {name: value for name, value in zip(arg_names, args)}
        arg_values.update(kwargs)
        
        # Format argument values for logging
        arg_str = ", ".join(f"{name}={repr(value)}" for name, value in arg_values.items())
        
        logger.debug(f"Calling {func.__name__}({arg_str})")
        
        result = func(*args, **kwargs)
        
        logger.debug(f"Function {func.__name__} returned {repr(result)}")
        
        return result
    
    return wrapper


# Mock DSGE model classes for demonstration
class ConfigManager:
    """Mock configuration manager for DSGE model."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to a JSON configuration file.
        """
        # Default configuration
        self.config = {
            "base_model": {
                "beta": 0.99,
                "alpha": 0.33,
                "delta": 0.025,
                "sigma_c": 1.5,
                "h": 0.7,
                "sigma_l": 2.0,
                "xi_p": 0.75,
                "xi_w": 0.75,
                "iota_p": 0.5,
                "iota_w": 0.5,
                "rho_r": 0.8,
                "phi_pi": 1.5,
                "phi_y": 0.125,
                "phi_dy": 0.125,
                "pi_bar": 1.005,
                "r_bar": 1.0101
            },
            "solution": {
                "method": "perturbation",
                "perturbation_order": 1
            }
        }
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to a JSON configuration file.
            
        Raises:
            IOError: If the file cannot be read
            ConfigurationError: If the configuration is invalid
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Validate configuration
            self._validate_config(loaded_config)
            
            # Update configuration
            self._update_config(self.config, loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise IOError(error_msg)
        
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def save_config(self, config_path):
        """
        Save configuration to a JSON file.
        
        Args:
            config_path (str): Path to save the configuration.
            
        Raises:
            IOError: If the file cannot be written
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Use safe file operation
            safe_file_operation(
                self._write_config,
                config_path,
                self.config
            )
            
            logger.info(f"Saved configuration to {config_path}")
        
        except Exception as e:
            error_msg = f"Error saving configuration: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
    
    def _write_config(self, config_path, config):
        """
        Write configuration to a file.
        
        Args:
            config_path (str): Path to write to
            config (dict): Configuration to write
        """
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _validate_config(self, config):
        """
        Validate configuration.
        
        Args:
            config (dict): Configuration to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "base_model": {
                    "type": "object",
                    "properties": {
                        "beta": {"type": "number", "minimum": 0, "maximum": 1},
                        "alpha": {"type": "number", "minimum": 0, "maximum": 1},
                        "delta": {"type": "number", "minimum": 0, "maximum": 1},
                        "sigma_c": {"type": "number", "minimum": 0},
                        "h": {"type": "number", "minimum": 0, "maximum": 1},
                        "sigma_l": {"type": "number", "minimum": 0},
                        "xi_p": {"type": "number", "minimum": 0, "maximum": 1},
                        "xi_w": {"type": "number", "minimum": 0, "maximum": 1},
                        "iota_p": {"type": "number", "minimum": 0, "maximum": 1},
                        "iota_w": {"type": "number", "minimum": 0, "maximum": 1},
                        "rho_r": {"type": "number", "minimum": 0, "maximum": 1},
                        "phi_pi": {"type": "number", "minimum": 0},
                        "phi_y": {"type": "number"},
                        "phi_dy": {"type": "number"},
                        "pi_bar": {"type": "number", "minimum": 0},
                        "r_bar": {"type": "number", "minimum": 0}
                    }
                },
                "solution": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["perturbation", "projection"]},
                        "perturbation_order": {"type": "number", "enum": [1, 2, 3]},
                        "projection_method": {"type": "string", "enum": ["chebyshev", "finite_elements"]},
                        "projection_nodes": {"type": "number", "minimum": 2}
                    }
                }
            }
        }
        
        # Validate against schema
        validate_config(config, schema)
    
    def _update_config(self, base_config, updates):
        """
        Update a configuration dictionary with another one.
        
        Args:
            base_config (dict): Base configuration to update
            updates (dict): Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                # Recursively update nested dictionaries
                self._update_config(base_config[key], value)
            else:
                # Replace or add values
                base_config[key] = value
    
    def get(self, key=None, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str, optional): The key to get. If None, the entire configuration is returned.
            default (any, optional): The default value to return if the key is not found.
            
        Returns:
            any: The configuration value.
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
    
    @validate_types(key=str)
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): The key to set.
            value (any): The value to set.
            
        Raises:
            ValidationError: If validation fails
        """
        # Handle nested keys with dot notation (e.g., "base_model.beta")
        if '.' in key:
            parts = key.split('.')
            config = self.config
            
            # Navigate to the nested dictionary
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                elif not isinstance(config[part], dict):
                    raise ValidationError(f"Cannot set nested key '{key}': '{part}' is not a dictionary")
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
            
            # Validate the updated configuration
            if parts[0] == "base_model":
                self._validate_base_model_param(parts[-1], value)
            elif parts[0] == "solution":
                self._validate_solution_param(parts[-1], value)
        else:
            # Set top-level key
            self.config[key] = value
    
    def _validate_base_model_param(self, param, value):
        """
        Validate a base model parameter.
        
        Args:
            param (str): Parameter name
            value (any): Parameter value
            
        Raises:
            ValidationError: If validation fails
        """
        if param in ["beta", "alpha", "delta", "h", "xi_p", "xi_w", "iota_p", "iota_w", "rho_r"]:
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValidationError(f"Parameter '{param}' must be a number between 0 and 1")
        
        elif param in ["sigma_c", "sigma_l", "phi_pi", "pi_bar", "r_bar"]:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValidationError(f"Parameter '{param}' must be a non-negative number")
        
        elif param in ["phi_y", "phi_dy"]:
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Parameter '{param}' must be a number")
    
    def _validate_solution_param(self, param, value):
        """
        Validate a solution parameter.
        
        Args:
            param (str): Parameter name
            value (any): Parameter value
            
        Raises:
            ValidationError: If validation fails
        """
        if param == "method":
            if value not in ["perturbation", "projection"]:
                raise ValidationError(f"Solution method must be 'perturbation' or 'projection', got '{value}'")
        
        elif param == "perturbation_order":
            if value not in [1, 2, 3]:
                raise ValidationError(f"Perturbation order must be 1, 2, or 3, got {value}")
        
        elif param == "projection_method":
            if value not in ["chebyshev", "finite_elements"]:
                raise ValidationError(f"Projection method must be 'chebyshev' or 'finite_elements', got '{value}'")
        
        elif param == "projection_nodes":
            if not isinstance(value, int) or value < 2:
                raise ValidationError(f"Projection nodes must be an integer >= 2, got {value}")


class SmetsWoutersModel:
    """Mock Smets-Wouters model for demonstration."""
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config (ConfigManager): Model configuration.
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, ConfigManager):
            raise ValidationError("config must be a ConfigManager instance")
        
        self.config = config
        
        # Extract parameters
        try:
            self.params = {
                "beta": config.get("base_model.beta", 0.99),
                "alpha": config.get("base_model.alpha", 0.33),
                "delta": config.get("base_model.delta", 0.025),
                "sigma_c": config.get("base_model.sigma_c", 1.5),
                "h": config.get("base_model.h", 0.7),
                "sigma_l": config.get("base_model.sigma_l", 2.0),
                "xi_p": config.get("base_model.xi_p", 0.75),
                "xi_w": config.get("base_model.xi_w", 0.75),
                "iota_p": config.get("base_model.iota_p", 0.5),
                "iota_w": config.get("base_model.iota_w", 0.5),
                "rho_r": config.get("base_model.rho_r", 0.8),
                "phi_pi": config.get("base_model.phi_pi", 1.5),
                "phi_y": config.get("base_model.phi_y", 0.125),
                "phi_dy": config.get("base_model.phi_dy", 0.125),
                "pi_bar": config.get("base_model.pi_bar", 1.005),
                "r_bar": config.get("base_model.r_bar", 1.0101)
            }
            
            logger.info("Model parameters extracted from configuration")
        
        except Exception as e:
            error_msg = f"Error extracting parameters from configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    @timeit
    @validate_range("shock_size", min_value=0)
    def compute_impulse_response(self, shock_name, periods=40, shock_size=1.0):
        """
        Compute impulse response function.
        
        Args:
            shock_name (str): Name of the shock
            periods (int): Number of periods
            shock_size (float): Size of the shock
            
        Returns:
            dict: Impulse response function
            
        Raises:
            ValidationError: If inputs are invalid
            ModelError: If computation fails
        """
        logger.info(f"Computing impulse response for {shock_name} shock")
        
        # Validate inputs
        valid_shocks = ["technology", "preference", "investment", "government", "monetary", "price_markup", "wage_markup"]
        if shock_name not in valid_shocks:
            error_msg = f"Invalid shock name: {shock_name}. Must be one of: {', '.join(valid_shocks)}"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        if not isinstance(periods, int) or periods <= 0:
            error_msg = f"Periods must be a positive integer, got {periods}"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        try:
            # Simulate computation
            time.sleep(0.1)
            
            # Generate synthetic IRF
            np.random.seed(42)
            
            # Different decay rates for different variables and shocks
            if shock_name == "technology":
                decay_rates = {
                    "output": 20,
                    "consumption": 25,
                    "investment": 15,
                    "inflation": 10,
                    "nominal_interest": 8
                }
                signs = {
                    "output": 1,
                    "consumption": 1,
                    "investment": 1,
                    "inflation": -1,
                    "nominal_interest": -0.5
                }
            elif shock_name == "monetary":
                decay_rates = {
                    "output": 10,
                    "consumption": 12,
                    "investment": 8,
                    "inflation": 15,
                    "nominal_interest": 5
                }
                signs = {
                    "output": -1,
                    "consumption": -1,
                    "investment": -1,
                    "inflation": -1,
                    "nominal_interest": 1
                }
            else:  # Generic shock
                decay_rates = {
                    "output": 15,
                    "consumption": 18,
                    "investment": 12,
                    "inflation": 10,
                    "nominal_interest": 8
                }
                signs = {
                    "output": 1,
                    "consumption": 1,
                    "investment": 1,
                    "inflation": 1,
                    "nominal_interest": 1
                }
            
            # Generate IRFs
            irf = {}
            for var, decay in decay_rates.items():
                t = np.arange(periods)
                response = signs[var] * shock_size * np.exp(-t / decay)
                
                # Add some noise
                response += 0.05 * np.random.randn(periods)
                
                irf[var] = response.tolist()  # Convert to list for JSON serialization
            
            logger.info(f"Impulse response computation completed for {shock_name} shock")
            
            return irf
        
        except Exception as e:
            error_msg = f"Error computing impulse response: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise ModelError(error_msg)
    
    @timeit
    @memory_usage
    def solve(self):
        """
        Solve the model.
        
        Returns:
            dict: Solution results
            
        Raises:
            SolutionError: If solution fails
        """
        logger.info("Solving model...")
        
        try:
            # Get solution method
            method = self.config.get("solution.method", "perturbation")
            
            if method == "perturbation":
                order = self.config.get("solution.perturbation_order", 1)
                logger.info(f"Using perturbation method with order {order}")
                
                # Simulate computation
                time.sleep(0.2)
                
                # Generate synthetic solution
                solution = {
                    "status": "success",
                    "method": "perturbation",
                    "order": order,
                    "computation_time": 0.2
                }
            
            elif method == "projection":
                proj_method = self.config.get("solution.projection_method", "chebyshev")
                nodes = self.config.get("solution.projection_nodes", 5)
                logger.info(f"Using projection method ({proj_method}) with {nodes} nodes")
                
                # Simulate computation
                time.sleep(0.5)
                
                # Generate synthetic solution
                solution = {
                    "status": "success",
                    "method": "projection",
                    "projection_method": proj_method,
                    "nodes": nodes,
                    "computation_time": 0.5
                }
            
            else:
                error_msg = f"Unknown solution method: {method}"
                logger.error(error_msg)
                raise SolutionError(error_msg)
            
            logger.info(f"Model solved successfully using {method} method")
            
            return solution
        
        except Exception as e:
            error_msg = f"Error solving model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise SolutionError(error_msg)


class DataProcessor:
    """Mock data processor for demonstration."""
    
    def __init__(self, data=None):
        """
        Initialize the data processor.
        
        Args:
            data (pd.DataFrame, optional): Data to process
        """
        self.data = data
    
    @validate_types(data=pd.DataFrame)
    def set_data(self, data):
        """
        Set data to process.
        
        Args:
            data (pd.DataFrame): Data to process
            
        Raises:
            ValidationError: If data is invalid
        """
        self.data = data
        logger.info(f"Data set with shape {data.shape}")
    
    def load_data(self, file_path):
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the data file
            
        Raises:
            IOError: If file cannot be read
            DataError: If data is invalid
        """
        try:
            # Check file extension
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise IOError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded data from {file_path} with shape {self.data.shape}")
        
        except FileNotFoundError:
            error_msg = f"Data file not found: {file_path}"
            logger.error(error_msg)
            raise IOError(error_msg)
        
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg)
    
    def save_data(self, file_path):
        """
        Save data to a file.
        
        Args:
            file_path (str): Path to save the data
            
        Raises:
            ValidationError: If data is not set
            IOError: If file cannot be written
        """
        if self.data is None:
            error_msg = "No data to save"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check file extension
            if file_path.endswith('.csv'):
                self.data.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data.to_excel(file_path, index=False)
            elif file_path.endswith('.json'):
                self.data.to_json(file_path, orient='records')
            else:
                raise IOError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Saved data to {file_path}")
        
        except Exception as e:
            error_msg = f"Error saving data: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
    
    @validate_types(column=str)
    def apply_transformation(self, column, transformation, **kwargs):
        """
        Apply a transformation to a data column.
        
        Args:
            column (str): Column name
            transformation (str): Transformation type
            **kwargs: Additional parameters for the transformation
            
        Returns:
            pd.Series: Transformed data
            
        Raises:
            ValidationError: If inputs are invalid
            DataError: If transformation fails
        """
        if self.data is None:
            error_msg = "No data to transform"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        if column not in self.data.columns:
            error_msg = f"Column not found: {column}"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        logger.info(f"Applying {transformation} transformation to column {column}")
        
        try:
            # Apply transformation
            if transformation == "log":
                if (self.data[column] <= 0).any():
                    error_msg = f"Cannot apply log transformation to non-positive values in column {column}"
                    logger.error(error_msg)
                    raise DataError(error_msg)
                
                self.data[column] = np.log(self.data[column])
            
            elif transformation == "diff":
                periods = kwargs.get('periods', 1)
                self.data[column] = self.data[column].diff(periods)
            
            elif transformation == "pct_change":
                periods = kwargs.get('periods', 1)
                self.data[column] = self.data[column].pct_change(periods) * 100
            
            elif transformation == "standardize":
                mean = self.data[column].mean()
                std = self.data[column].std()
                
                if std == 0:
                    error_msg = f"Cannot standardize column {column} with zero standard deviation"
                    logger.error(error_msg)
                    raise DataError(error_msg)
                
                self.data[column] = (self.data[column] - mean) / std
            
            elif transformation == "winsorize":
                lower = kwargs.get('lower', 0.05)
                upper = kwargs.get('upper', 0.95)
                
                if not 0 <= lower < upper <= 1:
                    error_msg = f"Invalid winsorization bounds: lower={lower}, upper={upper}"
                    logger.error(error_msg)
                    raise ValidationError(error_msg)
                
                lower_bound = self.data[column].quantile(lower)
                upper_bound = self.data[column].quantile(upper)
                
                self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
            
            else:
                error_msg = f"Unknown transformation: {transformation}"
                logger.error(error_msg)
                raise ValidationError(error_msg)
            
            logger.info(f"Transformation {transformation} applied to column {column}")
            
            return self.data[column]
        
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            
            error_msg = f"Error applying transformation: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg)


# Example usage with error handling
def example_configuration_error_handling():
    """Example of configuration error handling."""
    print("\n=== Configuration Error Handling Example ===")
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an invalid configuration file
        invalid_config_path = os.path.join(temp_dir, "invalid_config.json")
        with open(invalid_config_path, 'w') as f:
            f.write('{"base_model": {"beta": 1.5}}')  # Invalid beta value
        
        # Try to load the invalid configuration
        try:
            config = ConfigManager(invalid_config_path)
            print("Configuration loaded successfully")
        except ConfigurationError as e:
            print(f"Configuration error caught: {str(e)}")
        
        # Create a valid configuration file
        valid_config_path = os.path.join(temp_dir, "valid_config.json")
        with open(valid_config_path, 'w') as f:
            f.write('{"base_model": {"beta": 0.95}}')
        
        # Load the valid configuration
        try:
            config = ConfigManager(valid_config_path)
            print("Configuration loaded successfully")
            
            # Try to set an invalid parameter
            try:
                config.set("base_model.beta", 1.5)  # Invalid beta value
                print("Parameter set successfully")
            except ValidationError as e:
                print(f"Validation error caught: {str(e)}")
            
            # Set a valid parameter
            config.set("base_model.beta", 0.98)
            print("Parameter set successfully")
            
            # Save the configuration
            config_save_path = os.path.join(temp_dir, "saved_config.json")
            config.save_config(config_save_path)
            print(f"Configuration saved to {config_save_path}")
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


def example_model_error_handling():
    """Example of model error handling."""
    print("\n=== Model Error Handling Example ===")
    
    # Create a configuration
    config = ConfigManager()
    
    # Create a model
    try:
        model = SmetsWoutersModel(config)
        print("Model created successfully")
        
        # Try to compute an impulse response with an invalid shock name
        try:
            irf = model.compute_impulse_response("invalid_shock", periods=40, shock_size=1.0)
            print("Impulse response computed successfully")
        except ValidationError as e:
            print(f"Validation error caught: {str(e)}")
        
        # Compute an impulse response with a valid shock name
        try:
            irf = model.compute_impulse_response("technology", periods=40, shock_size=1.0)
            print("Impulse response computed successfully")
            print(f"IRF contains {len(irf)} variables")
        except ModelError as e:
            print(f"Model error caught: {str(e)}")
        
        # Solve the model
        try:
            solution = model.solve()
            print("Model solved successfully")
            print(f"Solution method: {solution['method']}")
        except SolutionError as e:
            print(f"Solution error caught: {str(e)}")
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def example_data_error_handling():
    """Example of data error handling."""
    print("\n=== Data Error Handling Example ===")
    
    # Create a data processor
    processor = DataProcessor()
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample data file
        data_path = os.path.join(temp_dir, "sample_data.csv")
        
        # Create sample data
        data = pd.DataFrame({
            "gdp": [100, 102, 105, 103, 107],
            "inflation": [2.0, 2.1, 2.3, 2.0, 1.9],
            "interest_rate": [3.0, 3.0, 3.25, 3.25, 3.0]
        })
        
        # Save sample data
        data.to_csv(data_path, index=False)
        
        # Load the data
        try:
            processor.load_data(data_path)
            print("Data loaded successfully")
            print(f"Data shape: {processor.data.shape}")
            
            # Try to apply a transformation to a non-existent column
            try:
                processor.apply_transformation("non_existent", "log")
                print("Transformation applied successfully")
            except ValidationError as e:
                print(f"Validation error caught: {str(e)}")
            
            # Apply a valid transformation
            try:
                processor.apply_transformation("gdp", "log")
                print("Transformation applied successfully")
                
                # Apply another transformation that would fail
                try:
                    processor.apply_transformation("inflation", "log")  # This might have negative values
                    print("Transformation applied successfully")
                except DataError as e:
                    print(f"Data error caught: {str(e)}")
                
                # Save the processed data
                processed_path = os.path.join(temp_dir, "processed_data.csv")
                processor.save_data(processed_path)
                print(f"Processed data saved to {processed_path}")
            
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


def example_recovery_strategies():
    """Example of recovery strategies."""
    print("\n=== Recovery Strategies Example ===")
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample configuration file
        config_path = os.path.join(temp_dir, "config.json")
        
        # Create a valid configuration
        config = ConfigManager()
        config.save_config(config_path)
        print(f"Configuration saved to {config_path}")
        
        # Define a function that might fail
        def process_config(config_path, retry_count=0):
            """Process a configuration file."""
            # Simulate a failure on the first attempt
            if retry_count == 0:
                raise IOError("Simulated file access error")
            
            # Load the configuration
            config = ConfigManager(config_path)
            return config
        
        # Define a recovery function
        def recovery_func(error, retry_count):
            """Recovery function for error_context."""
            print(f"Recovery attempt {retry_count}: {str(error)}")
            
            # Simulate recovery actions
            print("Performing recovery actions...")
            time.sleep(1)
        
        # Use error_context for automatic retries with recovery
        try:
            with error_context(IOError, recovery_func=recovery_func, max_retries=3, retry_delay=0.5):
                config = process_config(config_path, retry_count=0)  # This will fail
            
            print("This line should not be reached")
        
        except IOError as e:
            print(f"Error context caught IOError after max retries: {str(e)}")
        
        # Try again with a successful second attempt
        try:
            retry_count = [0]  # Use a list to allow modification in the inner function
            
            def process_with_retry(config_path):
                """Process with retry simulation."""
                retry_count[0] += 1
                return process_config(config_path, retry_count=retry_count[0])
            
            with error_context(IOError, recovery_func=recovery_func, max_retries=3, retry_delay=0.5):
                config = process_with_retry(config_path)  # This will succeed on the second attempt
            
            print("Configuration processed successfully after recovery")
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


def example_safe_file_operations():
    """Example of safe file operations."""
    print("\n=== Safe File Operations Example ===")
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample file
        file_path = os.path.join(temp_dir, "sample.txt")
        
        # Write initial content
        with open(file_path, 'w') as f:
            f.write("Initial content")
        
        print(f"Created sample file: {file_path}")
        
        # Define a function that modifies a file safely
        def modify_file(file_path, new_content):
            """Modify a file with new content."""
            with open(file_path, 'w') as f:
                f.write(new_content)
            return True
        
        # Define a function that will fail
        def modify_file_with_error(file_path, new_content):
            """Modify a file with new content, but raise an error."""
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            # Simulate an error after writing
            raise IOError("Simulated error after writing")
        
        # Safely modify the file
        try:
            result = safe_file_operation(
                modify_file,
                file_path,
                "Modified content"
            )
            
            print("File modified successfully")
            
            # Read the modified content
            with open(file_path, 'r') as f:
                content = f.read()
            
            print(f"File content: {content}")
            
            # Try to modify with an error
            try:
                result = safe_file_operation(
                    modify_file_with_error,
                    file_path,
                    "This will cause an error"
                )
                
                print("This line should not be reached")
            
            except IOError as e:
                print(f"Safe file operation caught error: {str(e)}")
                
                # Read the content after error (should be restored from backup)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                print(f"File content after error: {content}")
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


def example_performance_monitoring():
    """Example of performance monitoring."""
    print("\n=== Performance Monitoring Example ===")
    
    # Create a configuration
    config = ConfigManager()
    
    # Create a model
    model = SmetsWoutersModel(config)
    
    # Solve the model (with timing decorator)
    solution = model.solve()
    print(f"Model solved in {solution['computation_time']:.2f} seconds")
    
    # Define a function with debug_args decorator
    @debug_args
    def analyze_solution(solution, detail_level=1):
        """Analyze a solution with debug logging."""
        print(f"Analyzing solution with detail level {detail_level}")
        time.sleep(0.5)
        return {"status": "success", "detail_level": detail_level}
    
    # Call the function
    result = analyze_solution(solution, detail_level=2)
    print(f"Analysis result: {result}")


def main():
    """Main function demonstrating error handling techniques."""
    print("=== DSGE Model Error Handling Example ===")
    
    # Run examples
    example_configuration_error_handling()
    example_model_error_handling()
    example_data_error_handling()
    example_recovery_strategies()
    example_safe_file_operations()
    example_performance_monitoring()
    
    print("\nError handling example completed successfully.")


if __name__ == "__main__":
    main()