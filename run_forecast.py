#!/usr/bin/env python
"""
Forecasting script for the DSGE model.

This script generates forecasts from the DSGE model.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from config.config_manager import ConfigManager
from dsge.core import SmetsWoutersModel
from dsge.data import DataFetcher, DataProcessor
from dsge.forecasting import BaselineForecaster, ScenarioForecaster, UncertaintyQuantifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate forecasts from the DSGE model.')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/forecast_config.json',
        help='Path to the configuration file.'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/processed/forecast_data.csv',
        help='Path to the data file.'
    )
    
    parser.add_argument(
        '--params', 
        type=str, 
        default='results/estimation/results.json',
        help='Path to the estimated parameters file.'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/forecast',
        help='Path to the output directory.'
    )
    
    parser.add_argument(
        '--periods', 
        type=int, 
        default=20,
        help='Number of periods to forecast.'
    )
    
    parser.add_argument(
        '--scenarios', 
        action='store_true',
        help='Generate alternative scenarios.'
    )
    
    parser.add_argument(
        '--uncertainty', 
        action='store_true',
        help='Quantify forecast uncertainty.'
    )
    
    parser.add_argument(
        '--simulations', 
        type=int, 
        default=1000,
        help='Number of simulations for uncertainty quantification.'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed.'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a file."""
    config = ConfigManager()
    
    try:
        config.load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
    
    return config


def load_data(data_path):
    """Load data from a file."""
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise


def load_parameters(params_path):
    """Load estimated parameters from a file."""
    try:
        logger.info(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            results = json.load(f)
        
        # Extract posterior means
        if "mean" in results:
            params = results["mean"]
        else:
            # If mean not available, use the last sample from the first chain
            params = {}
            for i, param_name in enumerate(results["param_names"]):
                params[param_name] = results["samples"][-1][i]
        
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file not found: {params_path}")
        raise


def create_model(params, config):
    """Create the DSGE model with estimated parameters."""
    logger.info("Creating model")
    model = SmetsWoutersModel(params)
    return model


def generate_baseline_forecast(model, data, periods, config, output_dir):
    """Generate baseline forecast."""
    logger.info("Generating baseline forecast")
    
    # Create forecaster
    forecaster = BaselineForecaster(model, config)
    
    # Generate forecast
    forecasts = forecaster.generate_forecast(data, periods)
    
    # Plot forecast
    logger.info("Plotting baseline forecast")
    fig = forecaster.plot_forecast(data)
    fig.savefig(os.path.join(output_dir, 'baseline_forecast.png'))
    plt.close(fig)
    
    # Save forecast
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(os.path.join(output_dir, 'baseline_forecast.csv'))
    
    return forecasts


def generate_scenarios(model, data, periods, config, output_dir):
    """Generate alternative scenarios."""
    logger.info("Generating alternative scenarios")
    
    # Create scenario forecaster
    forecaster = ScenarioForecaster(model, config)
    
    # Generate baseline forecast
    baseline = forecaster.generate_baseline_forecast(data, periods)
    
    # Generate monetary policy shock scenario
    logger.info("Generating monetary policy shock scenario")
    forecaster.generate_shock_scenario(
        data=data,
        forecast_periods=periods,
        shock_name="monetary",
        shock_size=1.0,
        scenario_name="monetary_shock"
    )
    
    # Generate technology shock scenario
    logger.info("Generating technology shock scenario")
    forecaster.generate_shock_scenario(
        data=data,
        forecast_periods=periods,
        shock_name="technology",
        shock_size=1.0,
        scenario_name="technology_shock"
    )
    
    # Generate fiscal policy scenario
    logger.info("Generating fiscal policy scenario")
    forecaster.generate_shock_scenario(
        data=data,
        forecast_periods=periods,
        shock_name="government",
        shock_size=1.0,
        scenario_name="fiscal_policy"
    )
    
    # Generate alternative monetary policy rule scenario
    logger.info("Generating alternative monetary policy rule scenario")
    forecaster.generate_parameter_scenario(
        data=data,
        forecast_periods=periods,
        param_name="phi_pi",
        param_value=2.0,  # Stronger response to inflation
        scenario_name="stronger_inflation_response"
    )
    
    # Plot scenarios
    logger.info("Plotting scenarios")
    fig = forecaster.plot_scenarios(data)
    fig.savefig(os.path.join(output_dir, 'scenarios.png'))
    plt.close(fig)
    
    # Compute differences from baseline
    differences = forecaster.compute_scenario_differences("baseline")
    
    # Plot differences
    logger.info("Plotting scenario differences")
    fig = forecaster.plot_scenario_differences(differences)
    fig.savefig(os.path.join(output_dir, 'scenario_differences.png'))
    plt.close(fig)
    
    # Save scenarios
    forecaster.save_scenarios(os.path.join(output_dir, 'scenarios.json'))
    
    return forecaster.scenarios


def quantify_uncertainty(model, data, periods, n_simulations, config, output_dir):
    """Quantify forecast uncertainty."""
    logger.info("Quantifying forecast uncertainty")
    
    # Create uncertainty quantifier
    quantifier = UncertaintyQuantifier(model, config)
    
    # Generate simulations
    logger.info(f"Generating {n_simulations} simulations")
    simulations = quantifier.generate_simulations(
        data=data,
        forecast_periods=periods,
        n_simulations=n_simulations
    )
    
    # Compute statistics
    statistics = quantifier.compute_statistics()
    
    # Plot fan chart
    logger.info("Plotting fan chart")
    fig = quantifier.plot_fan_chart(data)
    fig.savefig(os.path.join(output_dir, 'fan_chart.png'))
    plt.close(fig)
    
    # Plot distribution
    logger.info("Plotting distribution")
    fig = quantifier.plot_distribution()
    fig.savefig(os.path.join(output_dir, 'distribution.png'))
    plt.close(fig)
    
    # Compute forecast error bands
    error_bands = quantifier.compute_forecast_error_bands()
    
    # Plot error bands
    logger.info("Plotting error bands")
    fig = quantifier.plot_error_bands(data, error_bands)
    fig.savefig(os.path.join(output_dir, 'error_bands.png'))
    plt.close(fig)
    
    # Save simulations
    quantifier.save_simulations(os.path.join(output_dir, 'simulations.json'))
    
    return simulations, statistics


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load data
    data = load_data(args.data)
    
    # Load parameters
    params = load_parameters(args.params)
    
    # Create model
    model = create_model(params, config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate baseline forecast
    baseline = generate_baseline_forecast(model, data, args.periods, config, args.output)
    
    # Generate alternative scenarios
    if args.scenarios:
        scenarios = generate_scenarios(model, data, args.periods, config, args.output)
    
    # Quantify forecast uncertainty
    if args.uncertainty:
        simulations, statistics = quantify_uncertainty(
            model, data, args.periods, args.simulations, config, args.output
        )
    
    logger.info(f"Forecasts saved to {args.output}")


if __name__ == '__main__':
    main()