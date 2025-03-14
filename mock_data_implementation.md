# Mock Data Implementation Guide

## Overview

Mock data generation is a critical component of our testing framework. This document provides detailed guidance for implementing robust mock data utilities that enable deterministic, realistic testing across all layers of the DSGE model.

## Purpose of Mock Data

Mock data serves several important purposes in testing:

1. **Isolation**: Tests can run without external dependencies
2. **Determinism**: Tests produce consistent results with fixed random seeds
3. **Coverage**: We can generate edge cases that might be rare in real data
4. **Performance**: Mock data can be optimized for test execution speed
5. **Portability**: Tests can run in any environment without special setup

## Types of Mock Data Needed

For comprehensive testing of a DSGE model, we need several types of mock data:

1. **Time Series Data**: Macroeconomic variables with realistic properties
2. **Shock Series**: Persistent shock processes with correlation structure
3. **Model Parameters**: Parameter sets with economically meaningful values
4. **Steady State Values**: Consistent steady state values for model testing
5. **Configuration Objects**: Mock configurations for different model setups

## Implementation Details

### 1. Time Series Generator

The time series generator creates realistic macroeconomic data with configurable properties:

```python
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
```

#### Implementation Approach

1. Set a random seed if provided for reproducibility
2. Define default variables if none are provided
3. Create a date range with quarterly frequency
4. Generate data for each variable with:
   - Realistic persistence (AR process)
   - Appropriate volatility
   - Trends for certain variables
   - Correlations between related variables
5. Add missing values or outliers if requested
6. Return as a pandas DataFrame with dates as index

#### Key Features

- **Realistic Autocorrelation**: Use AR(1) processes with different persistence parameters
- **Variable-Specific Properties**: Different means, volatilities, and trends for different types of variables
- **Correlations**: Impose realistic correlations between variables
- **Controlled Irregularities**: Option to include missing values and outliers

### 2. Shock Series Generator

The shock series generator creates correlated shock processes for model simulations:

```python
def generate_shock_series(
    n_variables=7,
    n_periods=100,
    ar_params=None,
    cov_matrix=None,
    random_seed=None
):
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
```

#### Implementation Approach

1. Set random seed if provided
2. Define default AR parameters if none are provided (with realistic persistence)
3. Define default covariance matrix if none is provided
4. Initialize array for shock series
5. Generate multivariate normal innovations using the covariance matrix
6. Apply AR(1) processes to propagate shocks
7. Return array of shock series

#### Key Features

- **Persistence**: Each shock follows an AR(1) process with configurable coefficient
- **Correlation Structure**: Shocks can have contemporaneous correlations
- **Standard SW Shocks**: Default setup matches the seven standard shocks in the Smets-Wouters model

### 3. Model Parameter Generator

The model parameter generator creates sets of economically plausible parameters:

```python
def generate_model_parameters():
    """
    Generate plausible parameter values for a DSGE model.
    
    Returns
    -------
    dict
        Dictionary with parameter values
    """
```

#### Implementation Approach

1. Define parameters based on standard calibrations in the literature
2. Ensure parameters are consistent with each other
3. Cover all parameter types:
   - Structural parameters (alpha, beta, etc.)
   - Monetary policy parameters
   - Shock persistence parameters
   - Shock volatility parameters
4. Return as a dictionary mapping parameter names to values

#### Key Features

- **Economic Consistency**: Parameters should make economic sense (e.g., 0 < beta < 1)
- **Plausible Values**: Values should be close to those in empirical literature
- **Complete Coverage**: Include all parameters needed for model solution

### 4. Steady State Value Generator

The steady state value generator creates consistent steady state values:

```python
def generate_steady_state_values():
    """
    Generate plausible steady state values for a DSGE model.
    
    Returns
    -------
    dict
        Dictionary with steady state values
    """
```

#### Implementation Approach

1. Define steady state values for key model variables
2. Ensure values are consistent with steady state relationships
3. Cover all major model variables
4. Return as a dictionary mapping variable names to values

#### Key Features

- **Consistency**: Values should satisfy model steady state equations
- **Realistic Levels**: Values should correspond to realistic economic quantities
- **Complete Coverage**: Include all variables needed for model initialization

### 5. Configuration Generator

The configuration generator creates mock configuration objects:

```python
def generate_mock_config():
    """
    Generate a mock configuration for testing.
    
    Returns
    -------
    dict
        Dictionary with configuration values
    """
```

#### Implementation Approach

1. Define configuration dictionary with all necessary sections
2. Include settings for different model components:
   - Model specification
   - Solution method
   - Estimation settings
   - Data handling
   - Forecasting options
3. Return as a nested dictionary

#### Key Features

- **Comprehensive Settings**: Cover all configuration options
- **Realistic Values**: Settings should be usable for actual model operations
- **Flexible Structure**: Support different model configurations

## Example Implementation

Here's a detailed example for the time series generator:

```python
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
```

## Usage in Tests

### Using Time Series Data

```python
def test_data_processor():
    """Test data processing functionality."""
    # Arrange - generate mock data
    data = generate_time_series(
        periods=100,
        variables=["gdp", "inflation", "interest_rate"],
        random_seed=42  # For reproducibility
    )
    
    # Act
    processor = DataProcessor()
    processed_data = processor.process(data)
    
    # Assert
    assert processed_data is not None
    assert len(processed_data) == len(data)
    # Further assertions...
```

### Using Model Parameters

```python
def test_model_with_custom_parameters():
    """Test model behavior with specific parameters."""
    # Arrange
    params = generate_model_parameters()
    # Modify specific parameters for the test
    params["alpha"] = 0.4
    params["beta"] = 0.98
    
    # Act
    model = SmetsWoutersModel(params=params)
    
    # Assert
    assert model.params["alpha"] == 0.4
    assert model.params["beta"] == 0.98
    # Further assertions...
```

### Using Shock Series

```python
def test_impulse_responses():
    """Test impulse response computation."""
    # Arrange
    model = mock_model()
    shocks = generate_shock_series(
        n_variables=1,  # Just one shock for simplicity
        n_periods=40,
        random_seed=42
    )
    
    # Act
    irf = ImpulseResponseFunctions(model)
    responses = irf.compute_responses(shocks)
    
    # Assert
    assert responses is not None
    assert responses.shape[0] == 40  # Same length as shocks
    # Further assertions...
```

## Recommended Extensions

### Time-Varying Parameters

For more advanced testing, implement time-varying parameters:

```python
def generate_time_varying_parameters(
    base_params,
    changing_params,
    periods=100,
    random_seed=None
):
    """Generate time-varying parameters for testing regime changes."""
    # Implementation...
```

### Structural Breaks

Implement functionality to create structural breaks in data:

```python
def add_structural_breaks(
    data,
    break_points,
    break_variables,
    shift_magnitudes
):
    """Add structural breaks to time series data."""
    # Implementation...
```

### Rich Correlation Structures

Create more complex correlation patterns between variables:

```python
def impose_correlation_structure(
    data,
    correlation_matrix
):
    """Impose a specific correlation structure on the data."""
    # Implementation...
```

## Best Practices

1. **Always Use Fixed Seeds**: Use fixed random seeds in tests for reproducibility
2. **Validate Generated Data**: Verify that generated data has the expected properties
3. **Keep Dependencies Minimal**: Mock data utilities should have minimal dependencies
4. **Provide Default Values**: Default parameters should be reasonable for most tests
5. **Document Economic Meaning**: Comment on the economic interpretation of values
6. **Test the Testing Utilities**: Write tests for your mock data generators
7. **Maintain Realistic Properties**: Ensure mock data has similar statistical properties to real data

## Common Pitfalls

1. **Unrealistic Values**: Parameters outside of plausible ranges
2. **Inconsistent Steady States**: Steady state values that don't satisfy model equations
3. **Independence Issues**: Generating the same "random" data across tests
4. **Over-Engineering**: Creating more complexity than needed for testing
5. **Insufficient Edge Cases**: Not testing boundary conditions

## Integration with Test Fixtures

The mock data generators should be integrated with pytest fixtures:

```python
@pytest.fixture
def mock_data():
    """Fixture for mock time series data."""
    return generate_time_series(periods=100, random_seed=42)

@pytest.fixture
def mock_parameters():
    """Fixture for mock model parameters."""
    return generate_model_parameters()

@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    config = ConfigManager()
    mock_dict = generate_mock_config()
    config.load_dict(mock_dict)
    return config
```

## Conclusion

Robust mock data generation is essential for effective testing of DSGE models. By implementing the utilities described in this guide, we can create tests that are deterministic, realistic, and comprehensive, without requiring access to external data sources.

These tools allow us to:
- Test model behavior under various scenarios
- Verify the correctness of data handling
- Ensure solution methods work as expected
- Test estimation and forecasting workflows

The mock data utilities form a foundation upon which our entire testing framework rests, enabling trustworthy verification of all aspects of the DSGE model implementation.