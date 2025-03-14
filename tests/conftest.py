import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dsge.core import SmetsWoutersModel
from config.config_manager import ConfigManager

@pytest.fixture(scope="function")
def config():
    """Create a test configuration."""
    base_config = {
        "base_model": {
            "alpha": 0.33,
            "beta": 0.99,
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
            "r_bar": 1.01,
            "g_y": 0.2
        },
        "shocks": {
            "technology": {"rho": 0.95, "sigma": 0.01},
            "preference": {"rho": 0.9, "sigma": 0.01},
            "investment": {"rho": 0.9, "sigma": 0.01},
            "government": {"rho": 0.9, "sigma": 0.01},
            "monetary": {"rho": 0.5, "sigma": 0.005},
            "price_markup": {"rho": 0.8, "sigma": 0.01},
            "wage_markup": {"rho": 0.8, "sigma": 0.01}
        },
        "model": {
            "extensions": {
                "financial": False,
                "open_economy": False
            }
        }
    }
    config_manager = ConfigManager()
    config_manager.update(base_config)
    return config_manager

@pytest.fixture(scope="function")
def model(config):
    """Create a test model instance."""
    return SmetsWoutersModel(config)

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_outputs")

@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 12345

@pytest.fixture
def steady_state(request):
    """Provide steady state values for testing."""
    return {
        "output": 1.0,
        "consumption": 0.6,
        "investment": 0.2,
        "capital": 4.0,
        "labor": 1.0,
        "real_wage": 1.0,
        "rental_rate": 0.03,
        "inflation": 1.005,
        "nominal_interest": 1.01,
    }

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Create sample time series data
    dates = pd.date_range(start='2000-01-01', periods=40, freq='QE')
    np.random.seed(12345)
    data = pd.DataFrame(
        np.random.randn(40, 7),
        columns=['output', 'consumption', 'investment', 'labor', 'real_wage', 'inflation', 'interest_rate'],
        index=dates
    )
    return data

@pytest.fixture
def benchmark():
    """Create a benchmark fixture for performance testing."""
    class Benchmark:
        def __init__(self):
            self.times = []
            self.stats = {}
        
        def __call__(self, func, *args, **kwargs):
            """Run function and collect timing statistics."""
            import time
            rounds = kwargs.pop('rounds', 3) if 'rounds' in kwargs else 3
            warmup = kwargs.pop('warmup', 1) if 'warmup' in kwargs else 1
            
            # Warmup rounds
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Timed rounds
            times = []
            for _ in range(rounds):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                times.append(end - start)
            
            self.times = times
            self.stats = {
                'min': min(times),
                'max': max(times),
                'mean': sum(times) / len(times),
                'rounds': rounds
            }
            return result
        
        def pedantic(self, func, *args, **kwargs):
            """Alias for __call__ for compatibility."""
            return self(func, *args, **kwargs)
    
    return Benchmark()