# Unit Testing Guide

## Overview

Unit tests are the foundation of our testing framework, focusing on testing individual components in isolation. This document provides detailed guidance for implementing effective unit tests for the DSGE model.

## Principles of Unit Testing

1. **Isolation**: Test individual components without dependencies on other parts of the system
2. **Deterministic**: Tests should always produce the same result given the same input
3. **Fast**: Unit tests should execute quickly to enable rapid feedback
4. **Complete**: Cover normal cases, edge cases, and error conditions
5. **Independent**: No test should depend on another test's execution

## Test Structure

### Directory Structure

Each module should have corresponding test files with the following structure:

```
tests/unit/
├── core/
│   ├── __init__.py
│   ├── test_base_model.py     # Tests for base_model.py
│   └── test_steady_state.py   # Tests for steady_state.py
├── data/
│   ├── __init__.py
│   ├── test_fetcher.py        # Tests for fetcher.py
│   └── test_processor.py      # Tests for processor.py
└── ...
```

### File Naming

- Test files should be named `test_<module_name>.py`
- Use lowercase with underscores
- Names should clearly indicate what is being tested

### Class and Method Naming

- Test classes should be named `Test<ClassName>` (e.g., `TestSmetsWoutersModel`)
- Test methods should be named `test_<function_or_method>_<scenario>` (e.g., `test_compute_steady_state_normal`)
- Use descriptive names that indicate what is being tested and under what conditions

## Test Implementation

### Basic Test Structure

```python
import pytest
import numpy as np
from dsge.core import SmetsWoutersModel

class TestSmetsWoutersModel:
    """Tests for the SmetsWoutersModel class."""
    
    def test_initialization(self, mock_config):
        """Test model initialization with default configuration."""
        # Arrange
        model = SmetsWoutersModel(mock_config)
        
        # Act - nothing specific here since we're testing initialization
        
        # Assert
        assert model is not None
        assert hasattr(model, 'params')
        assert len(model.params) > 0
```

### Using Fixtures

We use pytest fixtures to set up test dependencies and reuse common test objects:

```python
@pytest.fixture
def mock_model(mock_config):
    """Fixture providing a configured model."""
    return SmetsWoutersModel(mock_config)

def test_compute_steady_state(self, mock_model):
    """Test computation of steady state."""
    # Act
    mock_model.compute_steady_state()
    
    # Assert
    assert hasattr(mock_model, 'steady_state')
    assert 'output' in mock_model.steady_state
    assert mock_model.steady_state['output'] > 0
```

### Parameterized Tests

Use parameterization to test multiple scenarios with the same test function:

```python
@pytest.mark.parametrize("alpha,expected_capital_share", [
    (0.3, 0.3),
    (0.33, 0.33),
    (0.4, 0.4),
])
def test_capital_share(self, mock_config, alpha, expected_capital_share):
    """Test capital share parameter is correctly set."""
    # Arrange
    mock_config.set('model.alpha', alpha)
    model = SmetsWoutersModel(mock_config)
    
    # Act - nothing specific, testing initialization
    
    # Assert
    assert model.params['alpha'] == expected_capital_share
```

### Exception Testing

Test that appropriate exceptions are raised in error conditions:

```python
def test_invalid_parameter(self, mock_config):
    """Test that invalid parameters raise appropriate exceptions."""
    # Arrange
    mock_config.set('model.alpha', -0.5)  # Negative capital share is invalid
    
    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        model = SmetsWoutersModel(mock_config)
    
    # Verify the exception message
    assert "alpha must be between 0 and 1" in str(excinfo.value)
```

## Test Categories

Unit tests for the DSGE model should cover the following categories:

### 1. Module: dsge.core

#### 1.1 SmetsWoutersModel

- Test initialization with different configurations
- Test parameter validation
- Test steady state computation
- Test retrieval of state, control, and observable variables
- Test parameter bounds enforcement
- Test model equations generation

#### 1.2 Steady State Solver

- Test steady state computation for base model
- Test steady state computation with extensions
- Test validation of steady state values
- Test handling of degenerate cases
- Test sensitivity to initial guesses

### 2. Module: dsge.solution

#### 2.1 PerturbationSolver

- Test initialization with different orders
- Test solution computation
- Test stability checking
- Test policy function properties
- Test error handling for indeterminate systems

#### 2.2 ProjectionSolver

- Test initialization with different basis functions
- Test solution accuracy
- Test handling of occasionally binding constraints
- Test convergence properties

### 3. Module: dsge.data

#### 3.1 DataFetcher

- Test API connections
- Test data retrieval
- Test error handling for API issues
- Test caching behavior

#### 3.2 DataProcessor

- Test data transformation
- Test handling of missing values
- Test detrending methods
- Test data validation

### 4. Module: dsge.estimation

#### 4.1 BayesianEstimator

- Test prior specification
- Test likelihood computation
- Test posterior sampling
- Test convergence diagnostics

#### 4.2 Priors and Posteriors

- Test prior distribution creation
- Test posterior analysis
- Test marginal likelihood computation
- Test model comparison metrics

### 5. Module: dsge.analysis

#### 5.1 ImpulseResponseFunctions

- Test IRF computation
- Test shock propagation
- Test permanent vs. temporary shocks
- Test conditional IRFs

#### 5.2 Decomposition

- Test shock decomposition
- Test variance decomposition
- Test historical decomposition

### 6. Module: dsge.forecasting

#### 6.1 BaselineForecaster

- Test forecast generation
- Test forecast evaluation
- Test conditioning on observables

#### 6.2 ScenarioForecaster

- Test alternative scenario creation
- Test counterfactual analysis
- Test policy experiment simulation

## Example Test Cases

### Example 1: Testing Model Initialization

```python
def test_model_initialization_default(self, mock_config):
    """Test model initialization with default configuration."""
    # Arrange & Act
    model = SmetsWoutersModel(mock_config)
    
    # Assert
    assert model is not None
    assert hasattr(model, 'params')
    assert 'alpha' in model.params
    assert 'beta' in model.params
    assert model.extensions['financial_frictions'] == mock_config.get('model.financial_frictions')

def test_model_initialization_with_extensions(self, mock_config):
    """Test model initialization with extensions enabled."""
    # Arrange
    mock_config.set('model.financial_frictions', True)
    
    # Act
    model = SmetsWoutersModel(mock_config)
    
    # Assert
    assert model.extensions['financial_frictions'] is True
    assert 'spread' in model.get_control_variables()
```

### Example 2: Testing Steady State Computation

```python
def test_compute_steady_state_base_model(self, mock_model):
    """Test computation of steady state for base model."""
    # Act
    mock_model.compute_steady_state()
    
    # Assert
    assert hasattr(mock_model, 'steady_state')
    assert 'output' in mock_model.steady_state
    assert 'consumption' in mock_model.steady_state
    assert 'investment' in mock_model.steady_state
    assert 'capital' in mock_model.steady_state
    assert 'labor' in mock_model.steady_state
    
    # Check balance conditions
    ss = mock_model.steady_state
    # Output equals consumption plus investment
    assert abs(ss['output'] - (ss['consumption'] + ss['investment'] + ss['government_spending'])) < 1e-6
    # Capital accumulation balances
    assert abs(ss['investment'] - mock_model.params['delta'] * ss['capital']) < 1e-6

def test_steady_state_with_financial_frictions(self, mock_config):
    """Test steady state computation with financial frictions enabled."""
    # Arrange
    mock_config.set('model.financial_frictions', True)
    model = SmetsWoutersModel(mock_config)
    
    # Act
    model.compute_steady_state()
    
    # Assert
    assert 'spread' in model.steady_state
    assert model.steady_state['spread'] > 0
```

### Example 3: Testing Solution Methods

```python
def test_perturbation_first_order(self, mock_model):
    """Test first-order perturbation solution."""
    # Arrange
    solver = PerturbationSolver(mock_model, order=1)
    
    # Act
    solution = solver.solve()
    
    # Assert
    assert solution is not None
    assert 'policy_function' in solution
    assert 'transition_function' in solution
    
    # Check dimensions
    n_states = len(mock_model.get_state_variables())
    n_controls = len(mock_model.get_control_variables())
    assert solution['policy_function'].shape == (n_controls, n_states)
    assert solution['transition_function'].shape == (n_states, n_states)
    
    # Check stability (eigenvalues of transition matrix are < 1 in modulus)
    eigenvalues = np.linalg.eigvals(solution['transition_function'])
    assert np.max(np.abs(eigenvalues)) < 1.0

@pytest.mark.parametrize("order", [1, 2, 3])
def test_perturbation_different_orders(self, mock_model, order):
    """Test perturbation solutions of different orders."""
    # Arrange
    solver = PerturbationSolver(mock_model, order=order)
    
    # Act
    solution = solver.solve()
    
    # Assert
    assert solution is not None
    
    # Higher-order terms should exist for order > 1
    if order > 1:
        assert f'policy_function_{order}nd' in solution
        assert f'transition_function_{order}nd' in solution
```

## Testing for Numerical Stability

DSGE models often involve numerical operations that may lead to stability issues. Tests should check for:

1. **Convergence**: Ensure iterative algorithms converge
2. **Precision**: Check that numerical errors are within acceptable bounds
3. **Stability**: Verify that small perturbations don't cause large changes

Example:

```python
def test_numerical_stability(self, mock_model):
    """Test numerical stability of steady state computation."""
    # Arrange
    original_params = mock_model.params.copy()
    
    # Act - compute steady state
    mock_model.compute_steady_state()
    original_ss = mock_model.steady_state.copy()
    
    # Slightly perturb a parameter
    perturbed_params = original_params.copy()
    perturbed_params['alpha'] *= 1.001  # 0.1% change
    mock_model.set_parameters(perturbed_params)
    mock_model.compute_steady_state()
    perturbed_ss = mock_model.steady_state
    
    # Assert - changes should be small and proportionate
    for key in original_ss:
        rel_change = abs((perturbed_ss[key] - original_ss[key]) / original_ss[key])
        assert rel_change < 0.01, f"Excessive sensitivity in {key}: {rel_change:.4f}"
```

## Mock Data Generation

Unit tests should use mock data to avoid dependencies on external data sources. The `tests/utils/mock_data.py` module provides functions for generating realistic mock data:

```python
from tests.utils.mock_data import generate_time_series

def test_data_processor(self):
    """Test data processing functionality."""
    # Arrange - generate mock data
    data = generate_time_series(
        start_date="2000-01-01",
        periods=100,
        variables=["gdp", "inflation", "interest_rate"],
        include_missing=True,
        random_seed=42
    )
    
    # Act
    processor = DataProcessor()
    processed_data = processor.process(data)
    
    # Assert
    assert processed_data is not None
    assert len(processed_data) == len(data)
    assert "gdp_growth" in processed_data.columns
```

## Code Coverage Goals

The unit test suite should aim for the following code coverage targets:

- **Overall coverage**: >80%
- **Core modules**: >90%
- **Utility functions**: >70%
- **Error handling paths**: >95%

Coverage should be measured using the `--coverage` flag with the master test script:

```bash
python tests/master_test.py --unit --coverage
```

## Mocking Dependencies

To ensure true unit testing, external dependencies should be mocked:

```python
from unittest.mock import patch, MagicMock

def test_data_fetcher_api_call(self):
    """Test FRED API data fetching."""
    # Arrange
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=100, freq='Q'),
        'value': np.random.randn(100)
    })
    
    # Mock the pandas_datareader.data.DataReader function
    with patch('pandas_datareader.data.DataReader') as mock_reader:
        mock_reader.return_value = mock_data
        
        # Act
        fetcher = DataFetcher()
        result = fetcher.fetch_series('GDP', 'fred')
        
        # Assert
        assert result is not None
        mock_reader.assert_called_once()
        assert len(result) == len(mock_data)
```

## Test Documentation

Tests should be well-documented to explain their purpose and approach:

1. **Class docstrings**: Explain what part of the system is being tested
2. **Method docstrings**: Describe the specific functionality being tested
3. **Code comments**: Explain complex test logic

Example:

```python
class TestPerturbationSolver:
    """
    Tests for the PerturbationSolver class.
    
    These tests verify that the perturbation method correctly solves the DSGE model
    using first, second, and third-order approximations.
    """
    
    def test_solve_indeterminate_model(self):
        """
        Test solver behavior with an indeterminate model.
        
        An indeterminate model has multiple equilibria and should raise
        an IndeterminacyError when solved.
        """
        # Test implementation...
```

## Testing Best Practices

1. **Arrange-Act-Assert pattern**: Structure tests with clear separation between setup, execution, and verification
2. **Single responsibility**: Each test should verify one specific behavior
3. **Descriptive failure messages**: Provide clear information when assertions fail
4. **Test independence**: No test should depend on the state from a previous test
5. **Consistent naming**: Follow the naming conventions consistently

## Conclusion

Effective unit tests are essential for maintaining code quality and preventing regressions. By following these guidelines, we can build a robust unit test suite that verifies the correctness of each component of the DSGE model implementation.

The unit tests establish a foundation upon which integration and functional tests can build, providing confidence in the overall system behavior.