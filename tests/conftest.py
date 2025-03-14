"""
Configuration file for pytest.

This file contains fixtures and configuration settings for pytest.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path to import modules
sys.path.insert(0, os.path.abspath('.'))

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config():
    """Return a ConfigManager instance with default configuration."""
    return ConfigManager()


@pytest.fixture(scope="session")
def model(config):
    """Return a SmetsWoutersModel instance with default configuration."""
    return SmetsWoutersModel(config)


@pytest.fixture(scope="session")
def sample_data():
    """Return a sample dataset for testing."""
    # Create a sample dataset with random values
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2000-01-01', periods=40, freq='Q')
    
    # Create a DataFrame with sample data
    data = pd.DataFrame({
        'output': np.random.normal(0, 1, size=40),
        'consumption': np.random.normal(0, 0.8, size=40),
        'investment': np.random.normal(0, 2, size=40),
        'labor': np.random.normal(0, 0.5, size=40),
        'real_wage': np.random.normal(0, 0.7, size=40),
        'inflation': np.random.normal(0, 0.3, size=40),
        'interest_rate': np.random.normal(0, 0.4, size=40),
    }, index=dates)
    
    return data


@pytest.fixture(scope="session")
def steady_state(model):
    """Return the steady state of the model."""
    model.compute_steady_state()
    return model.steady_state


@pytest.fixture(scope="function")
def temp_dir(tmpdir):
    """Return a temporary directory for test files."""
    return tmpdir


@pytest.fixture(scope="function")
def random_seed():
    """Set a random seed for reproducibility."""
    np.random.seed(42)
    return 42