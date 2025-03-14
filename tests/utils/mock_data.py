"""
Mock data generator for testing.

This module provides functions for generating mock data for testing the DSGE model.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_random_time_series(start_date='2000-01-01', periods=40, freq='Q', seed=42):
    """
    Generate random time series data for testing.
    
    Parameters
    ----------
    start_date : str
        Start date for the time series.
    periods : int
        Number of periods to generate.
    freq : str
        Frequency of the time series (e.g., 'Q' for quarterly, 'M' for monthly).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with random time series data.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create DataFrame with random data
    data = pd.DataFrame({
        'output': np.random.normal(0, 1, size=periods),
        'consumption': np.random.normal(0, 0.8, size=periods),
        'investment': np.random.normal(0, 2, size=periods),
        'labor': np.random.normal(0, 0.5, size=periods),
        'real_wage': np.random.normal(0, 0.7, size=periods),
        'inflation': np.random.normal(0, 0.3, size=periods),
        'interest_rate': np.random.normal(0, 0.4, size=periods),
    }, index=dates)
    
    return data


def generate_model_consistent_data(model, solution, periods=40, shock_std=0.01, seed=42):
    """
    Generate model-consistent data for testing.
    
    Parameters
    ----------
    model : SmetsWoutersModel
        The DSGE model.
    solution : dict
        The solution of the model.
    periods : int
        Number of periods to generate.
    shock_std : float
        Standard deviation of the shocks.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with model-consistent data.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get dimensions
    n_states = len(model.variables.state)
    n_controls = len(model.variables.control)
    n_shocks = len(model.variables.shock)
    
    # Initialize state
    state = np.zeros(n_states)
    
    # Simulate
    states = [state]
    for t in range(periods):
        # Generate shocks
        shocks = np.random.normal(0, shock_std, n_shocks)
        
        # Update state
        state = solution['state_transition'][:, :n_states] @ state + \
               solution['state_transition'][:, n_states:] @ shocks
        
        states.append(state)
    
    # Convert to array
    states = np.array(states)
    
    # Compute controls
    controls = np.zeros((periods + 1, n_controls))
    for t in range(periods + 1):
        controls[t] = solution['observation_equation'] @ states[t]
    
    # Create date range
    dates = pd.date_range(start='2000-01-01', periods=periods + 1, freq='Q')
    
    # Create DataFrame
    state_names = model.variables.state
    control_names = model.variables.control
    
    # Combine states and controls
    data = np.concatenate([states, controls], axis=1)
    columns = state_names + control_names
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    return df


def generate_structural_shock_data(model, solution, periods=40, shock_type='monetary', shock_size=1.0, seed=42):
    """
    Generate data with a structural shock for testing.
    
    Parameters
    ----------
    model : SmetsWoutersModel
        The DSGE model.
    solution : dict
        The solution of the model.
    periods : int
        Number of periods to generate.
    shock_type : str
        Type of shock to generate (e.g., 'monetary', 'technology').
    shock_size : float
        Size of the shock.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with data including a structural shock.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get dimensions
    n_states = len(model.variables.state)
    n_controls = len(model.variables.control)
    n_shocks = len(model.variables.shock)
    
    # Get shock index
    shock_names = model.variables.shock
    try:
        shock_index = shock_names.index(shock_type)
    except ValueError:
        raise ValueError(f"Shock type '{shock_type}' not found in model shocks: {shock_names}")
    
    # Initialize state
    state = np.zeros(n_states)
    
    # Simulate
    states = [state]
    for t in range(periods):
        # Generate shocks
        shocks = np.zeros(n_shocks)
        
        # Add structural shock at t=10
        if t == 10:
            shocks[shock_index] = shock_size
        
        # Update state
        state = solution['state_transition'][:, :n_states] @ state + \
               solution['state_transition'][:, n_states:] @ shocks
        
        states.append(state)
    
    # Convert to array
    states = np.array(states)
    
    # Compute controls
    controls = np.zeros((periods + 1, n_controls))
    for t in range(periods + 1):
        controls[t] = solution['observation_equation'] @ states[t]
    
    # Create date range
    dates = pd.date_range(start='2000-01-01', periods=periods + 1, freq='Q')
    
    # Create DataFrame
    state_names = model.variables.state
    control_names = model.variables.control
    
    # Combine states and controls
    data = np.concatenate([states, controls], axis=1)
    columns = state_names + control_names
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    return df


def save_mock_data(data, path):
    """
    Save mock data to a file.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to save.
    path : str or Path
        The path to save the data to.
    """
    # Create directory if it doesn't exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data.to_csv(path)


def load_mock_data(path):
    """
    Load mock data from a file.
    
    Parameters
    ----------
    path : str or Path
        The path to load the data from.
    
    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    # Load data
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    
    return data


def generate_test_dataset(model, solution, output_dir, seed=42):
    """
    Generate a complete test dataset for testing.
    
    Parameters
    ----------
    model : SmetsWoutersModel
        The DSGE model.
    solution : dict
        The solution of the model.
    output_dir : str or Path
        The directory to save the data to.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary with paths to the generated data files.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate random data
    random_data = generate_random_time_series(seed=seed)
    random_path = output_dir / 'random_data.csv'
    save_mock_data(random_data, random_path)
    
    # Generate model-consistent data
    model_data = generate_model_consistent_data(model, solution, seed=seed)
    model_path = output_dir / 'model_data.csv'
    save_mock_data(model_data, model_path)
    
    # Generate monetary shock data
    monetary_data = generate_structural_shock_data(
        model, solution, shock_type='monetary', seed=seed
    )
    monetary_path = output_dir / 'monetary_shock_data.csv'
    save_mock_data(monetary_data, monetary_path)
    
    # Generate technology shock data
    technology_data = generate_structural_shock_data(
        model, solution, shock_type='technology', seed=seed
    )
    technology_path = output_dir / 'technology_shock_data.csv'
    save_mock_data(technology_data, technology_path)
    
    # Return paths
    return {
        'random': random_path,
        'model': model_path,
        'monetary': monetary_path,
        'technology': technology_path
    }