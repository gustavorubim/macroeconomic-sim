"""
Performance tests for the DSGE model.

This module contains tests that measure the performance of various components
of the DSGE model.
"""

import pytest
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver, ProjectionSolver
from dsge.estimation import BayesianEstimator
from dsge.forecasting import BaselineForecaster


class TestModelPerformance:
    """Tests for the performance of the model."""

    def test_steady_state_computation_performance(self, model, benchmark):
        """Test the performance of steady state computation."""
        # Benchmark steady state computation
        benchmark(model.compute_steady_state)

    def test_model_solution_performance(self, model, benchmark):
        """Test the performance of model solution."""
        # Initialize solver
        solver = PerturbationSolver(model)
        
        # Benchmark model solution
        benchmark(solver.solve)

    def test_model_simulation_performance(self, model, benchmark):
        """Test the performance of model simulation."""
        # Initialize solver
        solver = PerturbationSolver(model)
        
        # Solve model
        solution = solver.solve()
        
        # Define simulation function
        def simulate():
            # Generate simulated data
            np.random.seed(42)
            n_periods = 100
            n_states = len(model.variables.state)
            n_shocks = len(model.variables.shock)
            
            # Initialize state
            state = np.zeros(n_states)
            
            # Simulate
            states = [state]
            for t in range(n_periods):
                # Generate shocks
                shocks = np.random.normal(0, 0.01, n_shocks)
                
                # Update state
                state = solution['state_transition'][:, :n_states] @ state + \
                       solution['state_transition'][:, n_states:] @ shocks
                
                states.append(state)
            
            return np.array(states)
        
        # Benchmark simulation
        benchmark(simulate)


class TestSolverPerformance:
    """Tests for the performance of different solvers."""

    def test_perturbation_solver_performance(self, model, benchmark):
        """Test the performance of the perturbation solver."""
        # Initialize solver
        solver = PerturbationSolver(model)
        
        # Benchmark solver
        benchmark(solver.solve)

    def test_projection_solver_performance(self, model, benchmark):
        """Test the performance of the projection solver."""
        # Initialize solver
        solver = ProjectionSolver(model)
        
        # Benchmark solver
        benchmark(solver.solve)

    def test_solver_comparison(self, model, temp_dir):
        """Compare the performance of different solvers."""
        # Initialize solvers
        perturbation_solver = PerturbationSolver(model)
        projection_solver = ProjectionSolver(model)
        
        # Measure perturbation solver performance
        start_time = time.time()
        perturbation_solution = perturbation_solver.solve()
        perturbation_time = time.time() - start_time
        
        # Measure projection solver performance
        start_time = time.time()
        projection_solution = projection_solver.solve()
        projection_time = time.time() - start_time
        
        # Print results
        print(f"Perturbation solver: {perturbation_time:.4f} seconds")
        print(f"Projection solver: {projection_time:.4f} seconds")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        solvers = ['Perturbation', 'Projection']
        times = [perturbation_time, projection_time]
        ax.bar(solvers, times)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Solver Performance Comparison')
        
        # Save plot
        plot_path = Path(temp_dir) / 'solver_comparison.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save results
        results = pd.DataFrame({
            'Solver': solvers,
            'Time (seconds)': times
        })
        results_path = Path(temp_dir) / 'solver_comparison.csv'
        results.to_csv(results_path, index=False)
        
        # Check that results file exists
        assert results_path.exists()


class TestEstimationPerformance:
    """Tests for the performance of the estimation module."""

    def test_likelihood_evaluation_performance(self, model, sample_data, config, benchmark):
        """Test the performance of likelihood evaluation."""
        # Initialize estimator
        estimator = BayesianEstimator(model, sample_data, config)
        
        # Set solver
        solver = PerturbationSolver(model)
        estimator.set_solver(solver)
        
        # Define likelihood function
        def evaluate_likelihood():
            return estimator.log_likelihood(list(model.params.values()))
        
        # Benchmark likelihood evaluation
        benchmark(evaluate_likelihood)

    def test_posterior_evaluation_performance(self, model, sample_data, config, benchmark):
        """Test the performance of posterior evaluation."""
        # Initialize estimator
        estimator = BayesianEstimator(model, sample_data, config)
        
        # Set solver
        solver = PerturbationSolver(model)
        estimator.set_solver(solver)
        
        # Define posterior function
        def evaluate_posterior():
            return estimator.log_posterior(list(model.params.values()))
        
        # Benchmark posterior evaluation
        benchmark(evaluate_posterior)


class TestForecastingPerformance:
    """Tests for the performance of the forecasting module."""

    def test_baseline_forecast_performance(self, model, sample_data, config, benchmark):
        """Test the performance of baseline forecasting."""
        # Initialize forecaster
        forecaster = BaselineForecaster(model, config)
        
        # Define forecast function
        def generate_forecast():
            return forecaster.generate_forecast(sample_data, 8)
        
        # Benchmark forecast generation
        benchmark(generate_forecast)

    def test_forecast_scaling(self, model, sample_data, config, temp_dir):
        """Test how forecast performance scales with forecast horizon."""
        # Initialize forecaster
        forecaster = BaselineForecaster(model, config)
        
        # Define forecast horizons
        horizons = [4, 8, 12, 16, 20]
        times = []
        
        # Measure forecast time for each horizon
        for horizon in horizons:
            start_time = time.time()
            forecaster.generate_forecast(sample_data, horizon)
            forecast_time = time.time() - start_time
            times.append(forecast_time)
            print(f"Forecast horizon {horizon}: {forecast_time:.4f} seconds")
        
        # Plot scaling
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(horizons, times, marker='o')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Forecast Performance Scaling')
        
        # Save plot
        plot_path = Path(temp_dir) / 'forecast_scaling.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save results
        results = pd.DataFrame({
            'Horizon': horizons,
            'Time (seconds)': times
        })
        results_path = Path(temp_dir) / 'forecast_scaling.csv'
        results.to_csv(results_path, index=False)
        
        # Check that results file exists
        assert results_path.exists()


class TestScalabilityPerformance:
    """Tests for the scalability of the model."""

    def test_model_size_scaling(self, config, temp_dir):
        """Test how performance scales with model size."""
        # Define model sizes (number of shocks)
        sizes = [1, 2, 3, 4, 5, 6, 7]
        solution_times = []
        simulation_times = []
        
        # Measure performance for each model size
        for size in sizes:
            # Configure model size
            config.set('model.num_shocks', size)
            
            # Initialize model
            model = SmetsWoutersModel(config)
            
            # Measure solution time
            solver = PerturbationSolver(model)
            start_time = time.time()
            solution = solver.solve()
            solution_time = time.time() - start_time
            solution_times.append(solution_time)
            
            # Measure simulation time
            n_periods = 100
            n_states = len(model.variables.state)
            n_shocks = len(model.variables.shock)
            
            start_time = time.time()
            # Initialize state
            state = np.zeros(n_states)
            
            # Simulate
            states = [state]
            for t in range(n_periods):
                # Generate shocks
                shocks = np.random.normal(0, 0.01, n_shocks)
                
                # Update state
                state = solution['state_transition'][:, :n_states] @ state + \
                       solution['state_transition'][:, n_states:] @ shocks
                
                states.append(state)
            
            simulation_time = time.time() - start_time
            simulation_times.append(simulation_time)
            
            print(f"Model size {size}: Solution {solution_time:.4f}s, Simulation {simulation_time:.4f}s")
        
        # Plot scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(sizes, solution_times, marker='o')
        ax1.set_xlabel('Model Size (Number of Shocks)')
        ax1.set_ylabel('Solution Time (seconds)')
        ax1.set_title('Solution Performance Scaling')
        
        ax2.plot(sizes, simulation_times, marker='o')
        ax2.set_xlabel('Model Size (Number of Shocks)')
        ax2.set_ylabel('Simulation Time (seconds)')
        ax2.set_title('Simulation Performance Scaling')
        
        fig.tight_layout()
        
        # Save plot
        plot_path = Path(temp_dir) / 'model_size_scaling.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Check that plot file exists
        assert plot_path.exists()
        
        # Save results
        results = pd.DataFrame({
            'Model Size': sizes,
            'Solution Time (seconds)': solution_times,
            'Simulation Time (seconds)': simulation_times
        })
        results_path = Path(temp_dir) / 'model_size_scaling.csv'
        results.to_csv(results_path, index=False)
        
        # Check that results file exists
        assert results_path.exists()