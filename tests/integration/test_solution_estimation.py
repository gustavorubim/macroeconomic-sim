"""
Integration tests for the solution and estimation modules.

This module contains tests that verify the interaction between the solution and
estimation modules of the DSGE model.
"""

import pytest
import numpy as np
import pandas as pd
from dsge.core import SmetsWoutersModel
from dsge.solution import PerturbationSolver
from dsge.estimation import BayesianEstimator, PriorSet, create_default_priors


class TestSolutionEstimation:
    """Tests for the integration of solution and estimation modules."""

    def test_solver_initialization_with_model(self, model):
        """Test that the solver initializes correctly with a model."""
        solver = PerturbationSolver(model)
        assert solver is not None
        assert solver.model is model

    def test_solver_solution(self, model):
        """Test that the solver produces a valid solution."""
        # Initialize solver
        solver = PerturbationSolver(model)
        
        # Solve model
        solution = solver.solve()
        
        # Check solution exists and has correct format
        assert solution is not None
        assert isinstance(solution, dict)
        
        # Check required keys
        required_keys = ['state_transition', 'policy_function', 'steady_state']
        for key in required_keys:
            assert key in solution, f"Missing required key: {key}"
            
        # Check matrices have correct type and shape
        F = solution['state_transition']
        P = solution['policy_function']
        ss = solution['steady_state']
        
        assert isinstance(F, np.ndarray), "State transition matrix must be numpy array"
        assert isinstance(P, np.ndarray), "Policy function matrix must be numpy array"
        assert isinstance(ss, dict), "Steady state must be dictionary"
        assert 'observation_equation' in solution
        
        # Check that matrices have correct dimensions
        n_states = len(model.variables.state)
        n_controls = len(model.variables.control)
        n_shocks = len(model.variables.shock)
        
        assert solution['state_transition'].shape == (n_states, n_states + n_shocks)
        assert solution['observation_equation'].shape == (n_controls, n_states)

    def test_estimator_initialization_with_model_and_data(self, model, sample_data, config):
        """Test that the estimator initializes correctly with a model and data."""
        # Initialize estimator
        estimator = BayesianEstimator(model, sample_data, config)
        
        # Check that estimator exists
        assert estimator is not None
        assert estimator.model is model
        assert estimator.data is sample_data

    def test_estimator_with_solver(self, model, sample_data, config):
        """Test that the estimator works correctly with a solver."""
        # Initialize estimator
        estimator = BayesianEstimator(model, sample_data, config)
        
        # Set priors
        priors = create_default_priors()
        for param_name, prior in priors.priors.items():
            estimator.set_prior(
                param_name=param_name,
                distribution=prior.distribution,
                params=prior.params,
                bounds=prior.bounds
            )
        
        # Set solver
        solver = PerturbationSolver(model)
        estimator.set_solver(solver)
        
        # Check that solver is set
        assert estimator.solver is solver
        
        # Check that likelihood function can be evaluated
        log_likelihood = estimator.log_likelihood(list(model.params.values()))
        assert isinstance(log_likelihood, float)
        assert not np.isnan(log_likelihood)
        assert not np.isinf(log_likelihood)

    def test_estimation_with_mock_data(self, model, config):
        """Test that the estimation works correctly with mock data."""
        # Generate mock data from the model
        solver = PerturbationSolver(model)
        solution = solver.solve()
        
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
        
        # Convert to DataFrame
        states = np.array(states)
        dates = pd.date_range(start='2000-01-01', periods=n_periods+1, freq='Q')
        state_names = model.variables.state
        mock_data = pd.DataFrame(states, index=dates, columns=state_names)
        
        # Initialize estimator
        estimator = BayesianEstimator(model, mock_data, config)
        
        # Set priors
        priors = create_default_priors()
        for param_name, prior in priors.priors.items():
            estimator.set_prior(
                param_name=param_name,
                distribution=prior.distribution,
                params=prior.params,
                bounds=prior.bounds
            )
        
        # Set solver
        estimator.set_solver(solver)
        
        # Check that posterior can be evaluated
        log_posterior = estimator.log_posterior(list(model.params.values()))
        assert isinstance(log_posterior, float)
        assert not np.isnan(log_posterior)
        assert not np.isinf(log_posterior)
        
        # Check that maximum likelihood estimation works
        # Note: This is a simplified test that only runs a few iterations
        config.set('estimation.max_iter', 5)  # Limit iterations for testing
        results = estimator.maximize_likelihood()
        
        # Check that results exist
        assert results is not None
        assert 'params' in results
        assert 'log_likelihood' in results
        assert isinstance(results['log_likelihood'], float)
        assert not np.isnan(results['log_likelihood'])
        assert not np.isinf(results['log_likelihood'])