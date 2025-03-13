#!/usr/bin/env python
"""
Estimation script for the DSGE model.

This script estimates the DSGE model using Bayesian methods.
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
from dsge.estimation import BayesianEstimator, PriorSet, create_default_priors
from dsge.estimation import PosteriorAnalysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Estimate the DSGE model.')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/estimation_config.json',
        help='Path to the configuration file.'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/processed/estimation_data.csv',
        help='Path to the data file.'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/estimation',
        help='Path to the output directory.'
    )
    
    parser.add_argument(
        '--fetch-data', 
        action='store_true',
        help='Fetch data from FRED.'
    )
    
    parser.add_argument(
        '--mcmc-draws', 
        type=int, 
        default=None,
        help='Number of MCMC draws.'
    )
    
    parser.add_argument(
        '--mcmc-chains', 
        type=int, 
        default=None,
        help='Number of MCMC chains.'
    )
    
    parser.add_argument(
        '--mcmc-burn', 
        type=int, 
        default=None,
        help='Number of burn-in draws.'
    )
    
    parser.add_argument(
        '--mcmc-tune', 
        type=int, 
        default=None,
        help='Number of tuning iterations.'
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


def load_or_fetch_data(args, config):
    """Load data from a file or fetch it from FRED."""
    if args.fetch_data:
        # Fetch data from FRED
        logger.info("Fetching data from FRED")
        fetcher = DataFetcher(config)
        data = fetcher.fetch_all_series()
        
        # Save raw data
        os.makedirs('data/raw', exist_ok=True)
        fetcher.save_data('data/raw/fred_data.csv')
        
        # Process data
        logger.info("Processing data")
        processor = DataProcessor(data, config)
        processed_data = processor.process_all_series()
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        processor.save_processed_data(args.data)
        
        return processed_data
    else:
        # Load data from file
        try:
            logger.info(f"Loading data from {args.data}")
            processor = DataProcessor(config=config)
            data = processor.load_processed_data(args.data)
            return data
        except FileNotFoundError:
            logger.error(f"Data file not found: {args.data}")
            logger.info("Use --fetch-data to fetch data from FRED")
            raise


def create_model(config):
    """Create the DSGE model."""
    logger.info("Creating model")
    model = SmetsWoutersModel(config)
    return model


def estimate_model(model, data, config, args):
    """Estimate the model using Bayesian methods."""
    logger.info("Estimating model")
    
    # Create estimator
    estimator = BayesianEstimator(model, data, config)
    
    # Set priors
    logger.info("Setting priors")
    priors = create_default_priors()
    for param_name, prior in priors.priors.items():
        estimator.set_prior(
            param_name=param_name,
            distribution=prior.distribution,
            params=prior.params,
            bounds=prior.bounds
        )
    
    # Update MCMC parameters from command line arguments
    if args.mcmc_draws is not None:
        config.set("estimation.num_draws", args.mcmc_draws)
    
    if args.mcmc_chains is not None:
        config.set("estimation.num_chains", args.mcmc_chains)
    
    if args.mcmc_burn is not None:
        config.set("estimation.burn_in", args.mcmc_burn)
    
    if args.mcmc_tune is not None:
        config.set("estimation.tune", args.mcmc_tune)
    
    # Run estimation
    logger.info("Running MCMC")
    results = estimator.estimate()
    
    return results


def analyze_results(results, data, config, output_dir):
    """Analyze the estimation results."""
    logger.info("Analyzing results")
    
    # Create posterior analysis
    analysis = PosteriorAnalysis(results)
    
    # Generate summary
    summary = analysis.summary()
    
    # Save summary
    summary.to_csv(os.path.join(output_dir, 'summary.csv'))
    
    # Plot trace
    logger.info("Plotting trace")
    fig = analysis.plot_trace()
    fig.savefig(os.path.join(output_dir, 'trace.png'))
    plt.close(fig)
    
    # Plot posterior
    logger.info("Plotting posterior")
    fig = analysis.plot_posterior()
    fig.savefig(os.path.join(output_dir, 'posterior.png'))
    plt.close(fig)
    
    # Plot pair
    logger.info("Plotting pair")
    fig = analysis.plot_pair()
    fig.savefig(os.path.join(output_dir, 'pair.png'))
    plt.close(fig)
    
    # Compute convergence diagnostics
    logger.info("Computing convergence diagnostics")
    diagnostics = analysis.compute_convergence_diagnostics()
    
    # Save diagnostics
    with open(os.path.join(output_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    # Plot autocorrelation
    logger.info("Plotting autocorrelation")
    fig = analysis.plot_autocorrelation()
    fig.savefig(os.path.join(output_dir, 'autocorrelation.png'))
    plt.close(fig)
    
    return analysis


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load or fetch data
    data = load_or_fetch_data(args, config)
    
    # Create model
    model = create_model(config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Estimate model
    results = estimate_model(model, data, config, args)
    
    # Save results
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = results.copy()
        results_json["samples"] = results["samples"].tolist()
        results_json["chains"] = [chain.tolist() for chain in results["chains"]]
        json.dump(results_json, f, indent=2)
    
    # Analyze results
    analysis = analyze_results(results, data, config, args.output)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()