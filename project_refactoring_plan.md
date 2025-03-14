# Project Refactoring and Testing Framework Implementation Plan

## 1. Project Structure Refactoring

### 1.1 Directory Structure to Create

```
tests/
├── __init__.py             # Already exists
├── unit/                   # Unit tests for individual components
│   ├── __init__.py
│   ├── core/               # Tests for dsge/core modules
│   │   ├── __init__.py
│   │   ├── test_base_model.py
│   │   └── test_steady_state.py
│   ├── data/               # Tests for dsge/data modules
│   │   ├── __init__.py
│   │   ├── test_fetcher.py
│   │   └── test_processor.py
│   ├── solution/           # Tests for dsge/solution modules
│   │   ├── __init__.py
│   │   ├── test_perturbation.py
│   │   └── test_projection.py
│   ├── estimation/         # Tests for dsge/estimation modules
│   │   ├── __init__.py
│   │   ├── test_bayesian.py
│   │   ├── test_priors.py
│   │   └── test_posteriors.py
│   ├── analysis/           # Tests for dsge/analysis modules
│   │   ├── __init__.py
│   │   ├── test_impulse_response.py
│   │   ├── test_decomposition.py
│   │   └── test_diagnostics.py
│   ├── forecasting/        # Tests for dsge/forecasting modules
│   │   ├── __init__.py
│   │   ├── test_baseline.py
│   │   ├── test_scenarios.py
│   │   └── test_uncertainty.py
│   └── visualization/      # Tests for dsge/visualization modules
│       ├── __init__.py
│       ├── test_plots.py
│       └── test_publication.py
├── integration/            # Tests for component interactions
│   ├── __init__.py
│   ├── test_model_solution.py
│   ├── test_data_estimation.py
│   └── test_estimation_forecasting.py
├── functional/             # End-to-end workflow tests
│   ├── __init__.py
│   ├── test_estimation_workflow.py
│   ├── test_forecasting_workflow.py
│   └── test_scenario_analysis.py
├── performance/            # Performance benchmarks
│   ├── __init__.py
│   ├── test_solution_speed.py
│   ├── test_memory_usage.py
│   └── test_scalability.py
├── utils/                  # Test utilities and helpers
│   ├── __init__.py
│   ├── mock_data.py
│   └── test_helpers.py
├── debug/                  # Relocated debugging scripts
│   ├── __init__.py
│   ├── debug_steady_state.py
│   ├── test_import.py
│   ├── test_core_import.py
│   └── fix_corruption.py
├── conftest.py             # pytest configuration and fixtures
└── master_test.py          # Master test script
```

### 1.2 Files to Move from Root to tests/debug/

1. `debug_steady_state.py` → `tests/debug/debug_steady_state.py`
2. `test_import.py` → `tests/debug/test_import.py`
3. `test_core_import.py` → `tests/debug/test_core_import.py`
4. `fix_corruption.py` → `tests/debug/fix_corruption.py`

### 1.3 Path Updates Required

When moving the debug scripts, relative imports may need to be updated. For example, in `debug_steady_state.py`:

```python
# Add the project root to Python path to import modules
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
```

Should be updated to:

```python
# Add the project root to Python path to import modules
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
```

## 2. Testing Framework Implementation

### 2.1 Test Utilities (tests/utils/)

#### 2.1.1 Mock Data Generator (tests/utils/mock_data.py)

```python
"""
Utilities for generating mock data for tests.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_time_series(
    start_date="2000-01-01",
    periods=100,
    variables=None,
    include_missing=False,
    include_outliers=False,
    random_seed=None
):
    """
    Generate mock time series data for testing.
    
    Parameters
    ----------
    start_date : str
        Starting date for the time series
    periods : int
        Number of periods to generate
    variables : list
        List of variable names. If None, defaults to basic macroeconomic variables
    include_missing : bool
        Whether to include missing values (NaN) in the data
    include_outliers : bool
        Whether to include outliers in the data
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pandas.DataFrame
        Mock time series data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if variables is None:
        variables = [
            "gdp", "consumption", "investment", "government_spending",
            "inflation", "interest_rate", "hours_worked", "wages"
        ]
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='Q')
    
    # Generate data for each variable
    data = {}
    for var in variables:
        # Generate random walk with drift
        if var in ["gdp", "consumption", "investment", "government_spending"]:
            # Real variables tend to have positive drift and be more persistent
            drift = 0.005
            persistence = 0.8
            volatility = 0.01
        elif var in ["inflation", "interest_rate"]:
            # Nominal variables tend to be less persistent
            drift = 0.001
            persistence = 0.6
            volatility = 0.005
        else:
            # Other variables
            drift = 0.002
            persistence = 0.7
            volatility = 0.008
        
        # Generate base series
        shocks = np.random.normal(drift, volatility, periods)
        series = np.zeros(periods)
        
        # AR(1) process
        series[0] = np.random.normal(0, volatility)
        for t in range(1, periods):
            series[t] = persistence * series[t-1] + shocks[t]
        
        # Add trend for certain variables
        if var in ["gdp", "consumption", "investment"]:
            trend = np.linspace(0, 0.5, periods)
            series += trend
        
        # Make all series positive by shifting if needed
        min_val = min(series)
        if min_val < 0:
            series -= min_val * 1.1  # Add a 10% buffer
        
        # Add outliers if requested
        if include_outliers:
            outlier_indices = np.random.choice(
                range(periods), size=int(periods * 0.05), replace=False
            )
            for idx in outlier_indices:
                # Add or subtract a large value
                series[idx] += np.random.choice([-1, 1]) * np.abs(series[idx]) * 2
        
        # Add missing values if requested
        if include_missing:
            missing_indices = np.random.choice(
                range(periods), size=int(periods * 0.03), replace=False
            )
            for idx in missing_indices:
                series[idx] = np.nan
        
        data[var] = series
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df


def generate_shock_series(n_variables=7, n_periods=100, ar_params=None, cov_matrix=None, random_seed=None):
    """
    Generate correlated shock series for testing.
    
    Parameters
    ----------
    n_variables : int
        Number of shock variables
    n_periods : int
        Number of periods
    ar_params : list
        List of AR(1) coefficients for each shock
    cov_matrix : numpy.ndarray
        Covariance matrix for the shocks
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    numpy.ndarray
        Array of shock series with shape (n_periods, n_variables)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if ar_params is None:
        ar_params = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2][:n_variables]
    
    if cov_matrix is None:
        # Default: diagonal covariance matrix
        cov_matrix = np.eye(n_variables)
    
    # Initialize shock series
    shocks = np.zeros((n_periods, n_variables))
    
    # Generate initial values
    shocks[0, :] = np.random.multivariate_normal(np.zeros(n_variables), cov_matrix)
    
    # Generate remaining values using AR(1) process
    for t in range(1, n_periods):
        innovations = np.random.multivariate_normal(np.zeros(n_variables), cov_matrix)
        for i in range(n_variables):
            shocks[t, i] = ar_params[i] * shocks[t-1, i] + innovations[i]
    
    return shocks


def generate_steady_state_values():
    """
    Generate plausible steady state values for a DSGE model.
    
    Returns
    -------
    dict
        Dictionary with steady state values
    """
    return {
        "capital": 10.0,
        "output": 1.0,
        "consumption": 0.65,
        "investment": 0.2,
        "government_spending": 0.15,
        "labor": 0.33,
        "real_wage": 0.75,
        "rental_rate": 0.03,
        "inflation": 1.005,  # Quarterly inflation rate
        "nominal_interest": 1.01   # Quarterly nominal interest rate
    }


def generate_model_parameters():
    """
    Generate plausible parameter values for a DSGE model.
    
    Returns
    -------
    dict
        Dictionary with parameter values
    """
    return {
        # Structural parameters
        "alpha": 0.33,          # Capital share
        "beta": 0.99,           # Discount factor
        "delta": 0.025,         # Depreciation rate
        "sigma_c": 1.5,         # Elasticity of intertemporal substitution
        "h": 0.7,               # Habit formation
        "sigma_l": 2.0,         # Elasticity of labor supply
        "xi_p": 0.75,           # Price stickiness
        "xi_w": 0.75,           # Wage stickiness
        "iota_p": 0.5,          # Price indexation
        "iota_w": 0.5,          # Wage indexation
        
        # Monetary policy
        "rho_r": 0.8,           # Interest rate smoothing
        "phi_pi": 1.5,          # Response to inflation
        "phi_y": 0.125,         # Response to output
        "phi_dy": 0.125,        # Response to output growth
        "pi_bar": 1.005,        # Inflation target
        "r_bar": 1.01,          # Steady state nominal interest rate
        
        # Shock persistence
        "technology_rho": 0.95,
        "preference_rho": 0.9,
        "investment_rho": 0.85,
        "government_rho": 0.8,
        "monetary_rho": 0.5,
        "price_markup_rho": 0.75,
        "wage_markup_rho": 0.75,
        
        # Shock volatilities
        "technology_sigma": 0.01,
        "preference_sigma": 0.01,
        "investment_sigma": 0.01,
        "government_sigma": 0.01,
        "monetary_sigma": 0.01,
        "price_markup_sigma": 0.01,
        "wage_markup_sigma": 0.01
    }


def generate_mock_config():
    """
    Generate a mock configuration for testing.
    
    Returns
    -------
    dict
        Dictionary with configuration values
    """
    return {
        "model": {
            "name": "smets_wouters",
            "financial_frictions": False,
            "open_economy": False,
            "fiscal_policy": False
        },
        "solution": {
            "method": "perturbation",
            "order": 1,
            "use_jax": False
        },
        "estimation": {
            "method": "bayesian",
            "num_draws": 1000,
            "num_chains": 2,
            "burn_in": 200,
            "tune": 100
        },
        "data": {
            "start_date": "2000-01-01",
            "end_date": "2020-12-31",
            "frequency": "quarterly",
            "series": [
                "gdp", "consumption", "investment", "government_spending",
                "inflation", "interest_rate", "hours_worked", "wages"
            ],
            "transform": {
                "gdp": "log_diff",
                "consumption": "log_diff",
                "investment": "log_diff",
                "government_spending": "log_diff",
                "inflation": "level",
                "interest_rate": "level",
                "hours_worked": "log",
                "wages": "log_diff"
            }
        },
        "forecast": {
            "horizon": 12,
            "simulations": 1000,
            "show_uncertainty": True,
            "scenarios": ["baseline", "monetary_shock", "fiscal_shock"]
        }
    }
```

#### 2.1.2 Test Helpers (tests/utils/test_helpers.py)

```python
"""
Helper functions for tests.
"""

import os
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
import contextlib
import shutil


@contextlib.contextmanager
def tempdir():
    """
    Context manager for creating a temporary directory.
    
    Yields
    ------
    pathlib.Path
        Path to the temporary directory
    """
    path = tempfile.mkdtemp()
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path)


def create_temp_config_file(config_dict, prefix="test_config_", suffix=".json"):
    """
    Create a temporary configuration file.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary
    prefix : str
        Prefix for the temporary file
    suffix : str
        Suffix for the temporary file
    
    Returns
    -------
    str
        Path to the temporary configuration file
    """
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False, mode='w') as f:
        json.dump(config_dict, f, indent=2)
        return f.name


def assert_matrices_close(A, B, rtol=1e-05, atol=1e-08):
    """
    Assert that two matrices are approximately equal.
    
    Parameters
    ----------
    A : array_like
        First matrix
    B : array_like
        Second matrix
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    """
    A = np.asarray(A)
    B = np.asarray(B)
    
    assert A.shape == B.shape, f"Matrices have different shapes: {A.shape} vs {B.shape}"
    assert np.allclose(A, B, rtol=rtol, atol=atol), f"Matrices are not close"


def assert_series_stationary(series, alpha=0.05):
    """
    Assert that a time series is stationary.
    
    Parameters
    ----------
    series : array_like
        Time series
    alpha : float
        Significance level for the test
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Run Augmented Dickey-Fuller test
    result = adfuller(series)
    
    # Check if p-value is less than alpha (reject null hypothesis of non-stationarity)
    assert result[1] < alpha, f"Series is not stationary (p-value: {result[1]})"


def assert_convergence(draws, threshold=1.1):
    """
    Assert that MCMC draws have converged based on Gelman-Rubin statistic.
    
    Parameters
    ----------
    draws : array_like
        MCMC draws with shape (chains, draws, parameters)
    threshold : float
        Threshold for convergence
    """
    from statsmodels.stats.diagnostic import gelman_rubin
    
    n_chains, n_draws, n_params = draws.shape
    
    for i in range(n_params):
        chains = [draws[j, :, i] for j in range(n_chains)]
        r_stat = gelman_rubin(chains)
        assert r_stat < threshold, f"Parameter {i} has not converged (R={r_stat})"


def assert_posterior_contains_true(posterior_means, posterior_stds, true_values, alpha=0.05):
    """
    Assert that posterior distributions contain the true parameter values.
    
    Parameters
    ----------
    posterior_means : array_like
        Posterior means
    posterior_stds : array_like
        Posterior standard deviations
    true_values : array_like
        True parameter values
    alpha : float
        Significance level
    """
    import scipy.stats as stats
    
    for i, (mean, std, true) in enumerate(zip(posterior_means, posterior_stds, true_values)):
        # Compute z-score
        z = (true - mean) / std
        
        # Compute p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        assert p_value > alpha, f"Parameter {i} is significantly different from true value (p={p_value})"
```

### 2.2 pytest Configuration (tests/conftest.py)

```python
"""
pytest configuration and fixtures.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Add project root to Python path to import modules
sys.path.insert(0, os.path.abspath('..'))

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver
from tests.utils.mock_data import (
    generate_time_series,
    generate_shock_series,
    generate_steady_state_values,
    generate_model_parameters,
    generate_mock_config
)


@pytest.fixture
def mock_config():
    """
    Fixture for mock configuration.
    """
    config = ConfigManager()
    mock_dict = generate_mock_config()
    config.load_dict(mock_dict)
    return config


@pytest.fixture
def mock_data():
    """
    Fixture for mock time series data.
    """
    return generate_time_series(periods=100, random_seed=42)


@pytest.fixture
def mock_model(mock_config):
    """
    Fixture for mock model.
    """
    return SmetsWoutersModel(mock_config)


@pytest.fixture
def mock_solver(mock_model):
    """
    Fixture for mock solver.
    """
    return PerturbationSolver(mock_model, order=1)


@pytest.fixture
def mock_solution(mock_solver):
    """
    Fixture for mock solution.
    """
    return mock_solver.solve()


@pytest.fixture
def mock_shocks():
    """
    Fixture for mock shock series.
    """
    return generate_shock_series(n_variables=7, n_periods=100, random_seed=42)


@pytest.fixture
def mock_parameters():
    """
    Fixture for mock model parameters.
    """
    return generate_model_parameters()


@pytest.fixture
def mock_steady_state():
    """
    Fixture for mock steady state values.
    """
    return generate_steady_state_values()
```

### 2.3 Master Test Script (tests/master_test.py)

```python
#!/usr/bin/env python
"""
Master test script for running and managing all tests.

This script:
1. Discovers and runs tests
2. Measures code coverage
3. Generates reports
4. Validates expected behaviors
"""

import os
import sys
import argparse
import time
import datetime
import json
import glob
import subprocess
import shutil
from pathlib import Path
import importlib.util
import importlib.metadata
import pytest
import coverage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests and generate reports.')
    
    parser.add_argument(
        '--unit', 
        action='store_true',
        help='Run unit tests only'
    )
    
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run integration tests only'
    )
    
    parser.add_argument(
        '--functional', 
        action='store_true',
        help='Run functional tests only'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all tests'
    )
    
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Measure code coverage'
    )
    
    parser.add_argument(
        '--report-dir', 
        type=str,
        default='test_reports',
        help='Directory for test reports'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--fail-fast', 
        action='store_true',
        help='Stop on first failure'
    )
    
    return parser.parse_args()


def get_test_paths(args):
    """
    Get test paths based on command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    
    Returns
    -------
    list
        List of test paths
    """
    # Default: run all tests
    if not (args.unit or args.integration or args.functional or args.performance) or args.all:
        return ['tests/unit', 'tests/integration', 'tests/functional', 'tests/performance']
    
    paths = []
    if args.unit:
        paths.append('tests/unit')
    if args.integration:
        paths.append('tests/integration')
    if args.functional:
        paths.append('tests/functional')
    if args.performance:
        paths.append('tests/performance')
    
    return paths


def run_tests(paths, report_dir, verbose=False, fail_fast=False, measure_coverage=False):
    """
    Run tests using pytest.
    
    Parameters
    ----------
    paths : list
        List of test paths
    report_dir : str
        Directory for test reports
    verbose : bool
        Verbose output
    fail_fast : bool
        Stop on first failure
    measure_coverage : bool
        Measure code coverage
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    dict
        Test results
    """
    # Create report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Build pytest arguments
    pytest_args = paths.copy()
    
    # Add options
    if verbose:
        pytest_args.append('-v')
    if fail_fast:
        pytest_args.append('-x')
    
    # Add test result options
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    junit_report = os.path.join(report_dir, f'test_results_{timestamp}.xml')
    html_report = os.path.join(report_dir, f'test_report_{timestamp}.html')
    
    pytest_args.extend([
        f'--junitxml={junit_report}',
        f'--html={html_report}',
        '--self-contained-html'
    ])
    
    # Initialize results dictionary
    results = {
        'timestamp': timestamp,
        'paths': paths,
        'junit_report': junit_report,
        'html_report': html_report
    }
    
    # Measure code coverage if requested
    if measure_coverage:
        cov = coverage.Coverage(
            source=['dsge'],
            omit=['*/__pycache__/*', '*/tests/*', '*/.tox/*', '*/.venv/*']
        )
        cov.start()
    
    # Run tests
    start_time = time.time()
    exit_code = pytest.main(pytest_args)
    execution_time = time.time() - start_time
    
    # Stop coverage measurement and generate report
    if measure_coverage:
        cov.stop()
        coverage_report = os.path.join(report_dir, f'coverage_{timestamp}')
        os.makedirs(coverage_report, exist_ok=True)
        
        # Generate reports
        cov.html_report(directory=coverage_report)
        cov.xml_report(outfile=os.path.join(coverage_report, 'coverage.xml'))
        
        # Save coverage data
        results['coverage_report'] = coverage_report
        results['coverage_percentage'] = cov.report()
    
    # Update results
    results['exit_code'] = exit_code
    results['execution_time'] = execution_time
    
    return exit_code, results


def generate_summary(results):
    """
    Generate a summary of test results.
    
    Parameters
    ----------
    results : dict
        Test results
    
    Returns
    -------
    str
        Summary of test results
    """
    summary = []
    summary.append("# Test Execution Summary")
    summary.append("")
    summary.append(f"Timestamp: {results['timestamp']}")
    summary.append(f"Test paths: {', '.join(results['paths'])}")
    summary.append(f"Execution time: {results['execution_time']:.2f} seconds")
    summary.append(f"Exit code: {results['exit_code']} {'(SUCCESS)' if results['exit_code'] == 0 else '(FAILURE)'}")
    summary.append("")
    
    if 'coverage_percentage' in results:
        summary.append(f"Coverage percentage: {results['coverage_percentage']:.2f}%")
    
    summary.append("")
    summary.append("## Reports")
    summary.append("")
    summary.append(f"JUnit XML report: {results['junit_report']}")
    summary.append(f"HTML report: {results['html_report']}")
    
    if 'coverage_report' in results:
        summary.append(f"Coverage report: {results['coverage_report']}")
    
    return "\n".join(summary)


def main():
    """Main function."""
    args = parse_args()
    
    # Get test paths
    paths = get_test_paths(args)
    
    # Run tests
    exit_code, results = run_tests(
        paths=paths,
        report_dir=args.report_dir,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        measure_coverage=args.coverage
    )
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save summary
    summary_path = os.path.join(args.report_dir, f"summary_{results['timestamp']}.md")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    # Print summary
    print("\n" + summary)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
```

### 2.4 Example Unit Test (tests/unit/core/test_base_model.py)

```python
"""
Unit tests for the base model.
"""

import pytest
import numpy as np
from dsge.core import SmetsWoutersModel


class TestSmetsWoutersModel:
    """Tests for the SmetsWoutersModel class."""
    
    def test_initialization(self, mock_config):
        """Test model initialization."""
        model = SmetsWoutersModel(mock_config)
        
        # Check that the model has been initialized
        assert model is not None
        
        # Check that the parameters have been set
        assert hasattr(model, 'params')
        assert len(model.params) > 0
        
        # Check critical parameters
        assert 'alpha' in model.params
        assert 'beta' in model.params
        assert 'delta' in model.params
    
    def test_compute_steady_state(self, mock_model):
        """Test computation of steady state."""
        # Compute steady state
        mock_model.compute_steady_state()
        
        # Check that the steady state has been computed
        assert hasattr(mock_model, 'steady_state')
        assert len(mock_model.steady_state) > 0
        
        # Check critical steady state values
        assert 'output' in mock_model.steady_state
        assert 'consumption' in mock_model.steady_state
        assert 'investment' in mock_model.steady_state
        assert 'capital' in mock_model.steady_state
        assert 'labor' in mock_model.steady_state
        
        # Check that the steady state values are positive
        for key, value in mock_model.steady_state.items():
            assert value > 0, f"Steady state value for {key} is not positive: {value}"
    
    def test_get_state_variables(self, mock_model):
        """Test retrieval of state variables."""
        state_vars = mock_model.get_state_variables()
        
        # Check that we have state variables
        assert len(state_vars) > 0
        
        # Check that critical state variables are included
        assert 'capital' in state_vars
        
        # Check for shocks
        assert any('shock' in var for var in state_vars)
    
    def test_get_control_variables(self, mock_model):
        """Test retrieval of control variables."""
        control_vars = mock_model.get_control_variables()
        
        # Check that we have control variables
        assert len(control_vars) > 0
        
        # Check that critical control variables are included
        assert 'output' in control_vars
        assert 'consumption' in control_vars
        assert 'investment' in control_vars
        assert 'labor' in control_vars
    
    def test_get_observable_variables(self, mock_model):
        """Test retrieval of observable variables."""
        observable_vars = mock_model.get_observable_variables()
        
        # Check that we have observable variables
        assert len(observable_vars) > 0
    
    def test_get_equations(self, mock_model):
        """Test retrieval of model equations."""
        equations = mock_model.get_equations()
        
        # Check that we have equations
        assert len(equations) > 0
        
        # The number of equations should match the number of control variables
        # plus the law of motion for state variables
        n_controls = len(mock_model.get_control_variables())
        n_states = len(mock_model.get_state_variables())
        assert len(equations) >= n_controls
    
    def test_set_parameters(self, mock_model):
        """Test setting model parameters."""
        # Original parameters
        original_params = mock_model.params.copy()
        
        # New parameters (change a few values)
        new_params = original_params.copy()
        new_params['alpha'] = 0.4  # Capital share
        new_params['beta'] = 0.98  # Discount factor
        
        # Set new parameters
        mock_model.set_parameters(new_params)
        
        # Check that the parameters have been updated
        assert mock_model.params['alpha'] == 0.4
        assert mock_model.params['beta'] == 0.98
        
        # Check that other parameters remain unchanged
        for key, value in original_params.items():
            if key not in ['alpha', 'beta']:
                assert mock_model.params[key] == value
    
    def test_parameter_bounds(self, mock_model):
        """Test parameter bounds."""
        # Get parameter bounds
        bounds = mock_model.get_parameter_bounds()
        
        # Check that we have bounds
        assert len(bounds) > 0
        
        # Check that bounds are valid (lower <= upper)
        for param, (lower, upper) in bounds.items():
            assert lower <= upper, f"Invalid bounds for {param}: [{lower}, {upper}]"
            
            # Check that current parameter values are within bounds
            if param in mock_model.params:
                assert lower <= mock_model.params[param] <= upper, \
                    f"Parameter {param} = {mock_model.params[param]} is outside bounds [{lower}, {upper}]"


# Additional tests can be added for specific model features, extensions, etc.
```

### 2.5 Example Integration Test (tests/integration/test_model_solution.py)

```python
"""
Integration tests for the model solution process.
"""

import pytest
import numpy as np
from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver


class TestModelSolution:
    """Tests for the interaction between model and solution methods."""
    
    def test_model_perturbation_integration(self, mock_model):
        """Test integration of model with perturbation solver."""
        # Create solver
        solver = PerturbationSolver(mock_model, order=1)
        
        # Solve model
        solution = solver.solve()
        
        # Check that the solution exists
        assert solution is not None
        
        # Check that the solution contains the necessary components
        assert 'policy_function' in solution
        assert 'transition_function' in solution
        assert 'state_names' in solution
        assert 'control_names' in solution
        
        # Check dimensions of policy and transition functions
        n_states = len(mock_model.get_state_variables())
        n_controls = len(mock_model.get_control_variables())
        
        # First-order approximation: linear functions
        assert solution['policy_function'].shape[1] == n_states
        assert solution['transition_function'].shape[1] == n_states
        
        # Number of rows should match the number of variables
        assert solution['policy_function'].shape[0] == n_controls
        assert solution['transition_function'].shape[0] == n_states
    
    def test_simulation(self, mock_model):
        """Test simulation of the solved model."""
        # Create solver
        solver = PerturbationSolver(mock_model, order=1)
        
        # Solve model
        solution = solver.solve()
        
        # Simulate model
        periods = 100
        states, controls = solver.simulate(periods=periods)
        
        # Check dimensions
        n_states = len(mock_model.get_state_variables())
        n_controls = len(mock_model.get_control_variables())
        
        assert states.shape == (periods, n_states)
        assert controls.shape == (periods, n_controls)
        
        # Check that the simulation is not degenerate
        # (variables should change over time due to shocks)
        assert np.std(states, axis=0).mean() > 0
        assert np.std(controls, axis=0).mean() > 0
    
    def test_impulse_responses(self, mock_model):
        """Test computation of impulse responses."""
        from dsge.analysis import ImpulseResponseFunctions
        
        # Create solver and solve model
        solver = PerturbationSolver(mock_model, order=1)
        solution = solver.solve()
        
        # Create IRF analyzer
        irf = ImpulseResponseFunctions(mock_model)
        
        # Compute IRFs for technology shock
        shock_name = "technology"
        periods = 40
        irfs = irf.compute_irfs(
            shock_names=[shock_name],
            periods=periods,
            shock_size=1.0
        )
        
        # Check that the IRFs contain the shock
        assert shock_name in irfs
        
        # Check that the IRFs contain the main variables
        for var in ["output", "consumption", "investment", "inflation"]:
            assert var in irfs[shock_name]
            
            # Check dimensions
            assert len(irfs[shock_name][var]) == periods
            
            # Check that the IRFs are not degenerate
            assert np.std(irfs[shock_name][var]) > 0
    
    def test_higher_order_perturbation(self, mock_model):
        """Test higher-order perturbation methods."""
        # Test 2nd order perturbation
        solver_2nd = PerturbationSolver(mock_model, order=2)
        solution_2nd = solver_2nd.solve()
        
        # Check that the solution exists and contains the necessary components
        assert solution_2nd is not None
        assert 'policy_function' in solution_2nd
        assert 'transition_function' in solution_2nd
        
        # The higher-order terms should exist for 2nd order
        assert 'policy_function_2nd' in solution_2nd
        assert 'transition_function_2nd' in solution_2nd
        
        # Ensure simulation works with higher-order solution
        periods = 50
        states_2nd, controls_2nd = solver_2nd.simulate(periods=periods)
        
        # Check dimensions
        n_states = len(mock_model.get_state_variables())
        n_controls = len(mock_model.get_control_variables())
        
        assert states_2nd.shape == (periods, n_states)
        assert controls_2nd.shape == (periods, n_controls)
```

### 2.6 Example Functional Test (tests/functional/test_estimation_workflow.py)

```python
"""
Functional tests for the end-to-end estimation workflow.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import json
from pathlib import Path

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel
from dsge.data import DataProcessor
from dsge.estimation import BayesianEstimator, PriorSet, create_default_priors
from tests.utils.mock_data import generate_time_series


class TestEstimationWorkflow:
    """Tests for the end-to-end estimation workflow."""
    
    def test_end_to_end_estimation(self, mock_config, tmp_path):
        """Test the end-to-end estimation workflow."""
        # Generate mock data
        data = generate_time_series(periods=50, random_seed=42)
        data_path = tmp_path / "test_data.csv"
        data.to_csv(data_path)
        
        # Create model
        model = SmetsWoutersModel(mock_config)
        
        # Load and process data
        processor = DataProcessor(config=mock_config)
        processed_data = processor.load_data(data_path)
        
        # Create estimator
        estimator = BayesianEstimator(model, processed_data, mock_config)
        
        # Set priors
        priors = create_default_priors()
        for param_name, prior in priors.priors.items():
            estimator.set_prior(
                param_name=param_name,
                distribution=prior.distribution,
                params=prior.params,
                bounds=prior.bounds
            )
        
        # Run short estimation (for testing only)
        mock_config.set("estimation.num_draws", 50)
        mock_config.set("estimation.num_chains", 1)
        mock_config.set("estimation.burn_in", 10)
        mock_config.set("estimation.tune", 10)
        
        # Run estimation
        results = estimator.estimate()
        
        # Check results
        assert results is not None
        assert "samples" in results
        assert "chains" in results
        assert "param_names" in results
        
        # Save results
        output_dir = tmp_path / "estimation_output"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = results.copy()
            results_json["samples"] = results["samples"].tolist()
            results_json["chains"] = [chain.tolist() for chain in results["chains"]]
            json.dump(results_json, f, indent=2)
        
        # Load results
        with open(output_dir / "results.json", "r") as f:
            loaded_results = json.load(f)
        
        # Check loaded results
        assert loaded_results is not None
        assert "samples" in loaded_results
        assert "param_names" in loaded_results
        
        # Convert back to numpy arrays
        loaded_results["samples"] = np.array(loaded_results["samples"])
        loaded_results["chains"] = [np.array(chain) for chain in loaded_results["chains"]]
        
        # Verify that params can be extracted
        params = {}
        for i, param_name in enumerate(loaded_results["param_names"]):
            params[param_name] = loaded_results["samples"][-1, i]
        
        # Ensure all parameters are present
        for param_name in model.get_parameter_names():
            if param_name in loaded_results["param_names"]:
                assert param_name in params
```

### 2.7 Example Performance Test (tests/performance/test_solution_speed.py)

```python
"""
Performance tests for model solution speed.
"""

import pytest
import numpy as np
import time
import pandas as pd
from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver


class TestSolutionSpeed:
    """Tests for solution method performance."""
    
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_perturbation_speed(self, mock_model, order):
        """Test the speed of perturbation methods of different orders."""
        # Create solver
        solver = PerturbationSolver(mock_model, order=order)
        
        # Measure solution time
        start_time = time.time()
        solution = solver.solve()
        end_time = time.time()
        
        # Solution time
        solution_time = end_time - start_time
        
        # Check that solution is valid
        assert solution is not None
        
        # Print performance metrics
        print(f"\nPerturbation order {order} solution time: {solution_time:.6f} seconds")
        
        # Higher-order solutions should take longer, but still be reasonable
        if order > 1:
            assert solution_time > 0.001, "Higher-order solution seems too fast"
        
        # Solution time should be within reasonable limits
        assert solution_time < 10, f"Solution time too long: {solution_time:.6f} seconds"
    
    def test_simulation_speed(self, mock_model):
        """Test the speed of model simulation."""
        # Create solver and solve model
        solver = PerturbationSolver(mock_model, order=1)
        solution = solver.solve()
        
        # Define simulation sizes
        sizes = [100, 1000, 10000]
        
        # Measure simulation time for each size
        times = []
        for size in sizes:
            # Measure simulation time
            start_time = time.time()
            states, controls = solver.simulate(periods=size)
            end_time = time.time()
            
            # Simulation time
            sim_time = end_time - start_time
            times.append(sim_time)
            
            # Check that simulation is valid
            assert states.shape[0] == size
            assert controls.shape[0] == size
            
            # Print performance metrics
            print(f"\nSimulation with {size} periods: {sim_time:.6f} seconds")
        
        # Create performance table
        performance = pd.DataFrame({
            'periods': sizes,
            'time_seconds': times,
            'periods_per_second': [size / time for size, time in zip(sizes, times)]
        })
        
        print("\nSimulation Performance:")
        print(performance)
        
        # Simulation speed should scale approximately linearly
        # (Check ratio of time per period is within 50% for different sizes)
        for i in range(1, len(sizes)):
            time_ratio = (times[i] / sizes[i]) / (times[i-1] / sizes[i-1])
            assert 0.5 <= time_ratio <= 2.0, f"Simulation speed scaling is not approximately linear: {time_ratio}"
```

## 3. Implementation Steps

### 3.1 Step 1: Creating the Directory Structure

Create all necessary directories as outlined in section 1.1.

### 3.2 Step 2: Moving Debug Scripts

1. Copy the contents of `debug_steady_state.py`, `test_import.py`, `test_core_import.py`, and `fix_corruption.py` from the root directory to the `tests/debug/` directory.
2. Update the imports and paths in the copied files as needed.
3. Create an `__init__.py` file in the `tests/debug/` directory.

### 3.3 Step 3: Creating Test Utilities

1. Create the test utilities outlined in section 2.1.
2. Create the pytest configuration file as outlined in section 2.2.

### 3.4 Step 4: Creating the Master Test Script

Create the master test script as outlined in section 2.3.

### 3.5 Step 5: Creating Example Tests

1. Create example unit tests as outlined in section 2.4.
2. Create example integration tests as outlined in section 2.5.
3. Create example functional tests as outlined in section 2.6.
4. Create example performance tests as outlined in section 2.7.

### 3.6 Step 6: Adding Documentation

1. Add docstrings to all test files explaining their purpose and scope.
2. Create a README file in the tests directory explaining the testing framework.

## 4. Next Steps

After implementing the test framework, the following additional steps are recommended:

1. Expand the test coverage to include all modules and features.
2. Create CI/CD integration to run tests automatically on code changes.
3. Set up regular performance testing to track performance over time.
4. Add more detailed documentation on how to write and run tests.

## 5. Testing Framework Summary

The testing framework consists of:

1. **Unit Tests**: Testing individual classes and functions in isolation.
2. **Integration Tests**: Testing interactions between components.
3. **Functional Tests**: Testing end-to-end workflows.
4. **Performance Tests**: Measuring and benchmarking performance.
5. **Test Utilities**: Helper functions and mock data generators.
6. **Master Test Script**: Discovering, running, and reporting on tests.

This framework provides comprehensive test coverage, from individual components to full system workflows, ensuring the reliability and correctness of the codebase.