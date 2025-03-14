"""
Functional tests for the forecasting module.

This module contains tests that verify the end-to-end functionality of the
forecasting module of the DSGE model.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver
from dsge.forecasting import BaselineForecaster, ScenarioForecaster, UncertaintyQuantifier


class TestForecasting:
    """Tests for the forecasting module."""

    def test_baseline_forecast(self, model, sample_data, config, temp_dir):
        """Test that the baseline forecaster works correctly."""
        # Initialize forecaster
        forecaster = BaselineForecaster(model, config)
        
        # Generate forecast
        forecast_periods = 8
        forecasts = forecaster.generate_forecast(sample_data, forecast_periods)
        
        # Check that forecasts exist
        assert forecasts is not None
        assert isinstance(forecasts, dict)
        
        # Check that essential forecast variables exist
        essential_vars = [
            'output', 'consumption', 'investment', 'labor', 
            'real_wage', 'inflation', 'nominal_interest'
        ]
        for var in essential_vars:
            assert var in forecasts
            assert isinstance(forecasts[var], np.ndarray)
            assert len(forecasts[var]) == forecast_periods
            assert not np.any(np.isnan(forecasts[var]))
            assert not np.any(np.isinf(forecasts[var]))
        
        # Plot forecast
        fig = forecaster.plot_forecast(sample_data)
        assert fig is not None
        
        # Save plot
        plot_path = Path(temp_dir) / 'baseline_forecast.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save forecast data
        forecast_df = pd.DataFrame(forecasts)
        forecast_path = Path(temp_dir) / 'baseline_forecast.csv'
        forecast_df.to_csv(forecast_path)
        
        # Check that forecast file exists
        assert forecast_path.exists()

    def test_scenario_forecasts(self, model, sample_data, config, temp_dir):
        """Test that the scenario forecaster works correctly."""
        # Initialize forecaster
        forecaster = ScenarioForecaster(model, config)
        
        # Generate baseline forecast
        forecast_periods = 8
        baseline = forecaster.generate_baseline_forecast(sample_data, forecast_periods)
        
        # Check that baseline forecast exists
        assert baseline is not None
        assert isinstance(baseline, dict)
        
        # Generate monetary policy shock scenario
        forecaster.generate_shock_scenario(
            data=sample_data,
            forecast_periods=forecast_periods,
            shock_name="monetary",
            shock_size=1.0,
            scenario_name="monetary_shock"
        )
        
        # Generate technology shock scenario
        forecaster.generate_shock_scenario(
            data=sample_data,
            forecast_periods=forecast_periods,
            shock_name="technology",
            shock_size=1.0,
            scenario_name="technology_shock"
        )
        
        # Check that scenarios exist
        assert "monetary_shock" in forecaster.scenarios
        assert "technology_shock" in forecaster.scenarios
        
        # Plot scenarios
        fig = forecaster.plot_scenarios(sample_data)
        assert fig is not None
        
        # Save plot
        plot_path = Path(temp_dir) / 'scenarios.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Compute differences from baseline
        differences = forecaster.compute_scenario_differences("baseline")
        
        # Check that differences exist
        assert differences is not None
        assert isinstance(differences, dict)
        assert "monetary_shock" in differences
        assert "technology_shock" in differences
        
        # Plot differences
        fig = forecaster.plot_scenario_differences(differences)
        assert fig is not None
        
        # Save plot
        plot_path = Path(temp_dir) / 'scenario_differences.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save scenarios
        scenario_path = Path(temp_dir) / 'scenarios.json'
        forecaster.save_scenarios(scenario_path)
        
        # Check that scenario file exists
        assert scenario_path.exists()

    def test_uncertainty_quantification(self, model, sample_data, config, temp_dir):
        """Test that the uncertainty quantifier works correctly."""
        # Initialize uncertainty quantifier
        quantifier = UncertaintyQuantifier(model, config)
        
        # Generate simulations
        forecast_periods = 8
        n_simulations = 50  # Use a small number for testing
        simulations = quantifier.generate_simulations(
            data=sample_data,
            forecast_periods=forecast_periods,
            n_simulations=n_simulations
        )
        
        # Check that simulations exist
        assert simulations is not None
        assert isinstance(simulations, dict)
        
        # Check that essential simulation variables exist
        essential_vars = [
            'output', 'consumption', 'investment', 'labor', 
            'real_wage', 'inflation', 'nominal_interest'
        ]
        for var in essential_vars:
            assert var in simulations
            assert isinstance(simulations[var], np.ndarray)
            assert simulations[var].shape == (n_simulations, forecast_periods)
            assert not np.any(np.isnan(simulations[var]))
            assert not np.any(np.isinf(simulations[var]))
        
        # Compute statistics
        statistics = quantifier.compute_statistics()
        
        # Check that statistics exist
        assert statistics is not None
        assert isinstance(statistics, dict)
        assert "mean" in statistics
        assert "median" in statistics
        assert "std" in statistics
        assert "percentiles" in statistics
        
        # Plot fan chart
        fig = quantifier.plot_fan_chart(sample_data)
        assert fig is not None
        
        # Save plot
        plot_path = Path(temp_dir) / 'fan_chart.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Compute forecast error bands
        error_bands = quantifier.compute_forecast_error_bands()
        
        # Check that error bands exist
        assert error_bands is not None
        assert isinstance(error_bands, dict)
        
        # Plot error bands
        fig = quantifier.plot_error_bands(sample_data, error_bands)
        assert fig is not None
        
        # Save plot
        plot_path = Path(temp_dir) / 'error_bands.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save simulations
        simulation_path = Path(temp_dir) / 'simulations.json'
        quantifier.save_simulations(simulation_path)
        
        # Check that simulation file exists
        assert simulation_path.exists()


class TestForecastingWorkflow:
    """Tests for the forecasting workflow."""

    def test_end_to_end_forecasting_workflow(self, model, sample_data, config, temp_dir):
        """Test the end-to-end forecasting workflow."""
        # Create output directory
        output_dir = Path(temp_dir) / 'forecast_results'
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Generate baseline forecast
        print("Generating baseline forecast...")
        forecaster = BaselineForecaster(model, config)
        baseline = forecaster.generate_forecast(sample_data, 8)
        
        # Save baseline forecast
        baseline_df = pd.DataFrame(baseline)
        baseline_df.to_csv(output_dir / 'baseline_forecast.csv')
        
        # Plot baseline forecast
        fig = forecaster.plot_forecast(sample_data)
        fig.savefig(output_dir / 'baseline_forecast.png')
        plt.close(fig)
        
        # Step 2: Generate alternative scenarios
        print("Generating alternative scenarios...")
        scenario_forecaster = ScenarioForecaster(model, config)
        
        # Generate baseline forecast
        scenario_forecaster.generate_baseline_forecast(sample_data, 8)
        
        # Generate monetary policy shock scenario
        scenario_forecaster.generate_shock_scenario(
            data=sample_data,
            forecast_periods=8,
            shock_name="monetary",
            shock_size=1.0,
            scenario_name="monetary_shock"
        )
        
        # Generate technology shock scenario
        scenario_forecaster.generate_shock_scenario(
            data=sample_data,
            forecast_periods=8,
            shock_name="technology",
            shock_size=1.0,
            scenario_name="technology_shock"
        )
        
        # Plot scenarios
        fig = scenario_forecaster.plot_scenarios(sample_data)
        fig.savefig(output_dir / 'scenarios.png')
        plt.close(fig)
        
        # Save scenarios
        scenario_forecaster.save_scenarios(output_dir / 'scenarios.json')
        
        # Step 3: Quantify forecast uncertainty
        print("Quantifying forecast uncertainty...")
        quantifier = UncertaintyQuantifier(model, config)
        
        # Generate simulations
        simulations = quantifier.generate_simulations(
            data=sample_data,
            forecast_periods=8,
            n_simulations=50
        )
        
        # Plot fan chart
        fig = quantifier.plot_fan_chart(sample_data)
        fig.savefig(output_dir / 'fan_chart.png')
        plt.close(fig)
        
        # Save simulations
        quantifier.save_simulations(output_dir / 'simulations.json')
        
        # Check that all output files exist
        assert (output_dir / 'baseline_forecast.csv').exists()
        assert (output_dir / 'baseline_forecast.png').exists()
        assert (output_dir / 'scenarios.png').exists()
        assert (output_dir / 'scenarios.json').exists()
        assert (output_dir / 'fan_chart.png').exists()
        assert (output_dir / 'simulations.json').exists()
        
        print("Forecasting workflow completed successfully.")
