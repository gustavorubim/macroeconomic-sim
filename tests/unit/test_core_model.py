"""
Unit tests for the core model module.

This module contains tests for the SmetsWoutersModel class and related functionality.
"""

import pytest
import numpy as np
from dsge.core import SmetsWoutersModel


class TestSmetsWoutersModel:
    """Tests for the SmetsWoutersModel class."""

    def test_model_initialization(self, config):
        """Test that the model initializes correctly."""
        model = SmetsWoutersModel(config)
        assert model is not None
        assert hasattr(model, 'params')
        assert isinstance(model.params, dict)

    def test_model_parameters(self, model):
        """Test that the model parameters are set correctly."""
        # Check that essential parameters exist
        essential_params = [
            'alpha', 'beta', 'delta', 'sigma_c', 'sigma_l', 
            'phi_pi', 'phi_y', 'rho_r', 'pi_bar', 'r_bar'
        ]
        for param in essential_params:
            assert param in model.params
            assert isinstance(model.params[param], (int, float))

    def test_steady_state_computation(self, model):
        """Test that the steady state is computed correctly."""
        # Compute steady state
        model.compute_steady_state()
        
        # Check that steady state exists
        assert hasattr(model, 'steady_state')
        assert isinstance(model.steady_state, dict)
        
        # Check that essential steady state variables exist
        essential_vars = [
            'output', 'consumption', 'investment', 'labor', 
            'real_wage', 'rental_rate', 'inflation', 'nominal_interest'
        ]
        for var in essential_vars:
            assert var in model.steady_state
            assert isinstance(model.steady_state[var], (int, float))
            assert not np.isnan(model.steady_state[var])
            assert not np.isinf(model.steady_state[var])

    def test_steady_state_consistency(self, model):
        """Test that the steady state is consistent with model equations."""
        # Compute steady state
        model.compute_steady_state()
        ss = model.steady_state
        
        # Check resource constraint: Y = C + I + G
        assert abs(ss['output'] - (ss['consumption'] + ss['investment'] + ss.get('government', 0))) < 1e-6
        
        # Check capital accumulation: K = (1-delta)*K + I
        # In steady state, I = delta*K
        assert abs(ss['investment'] - model.params['delta'] * ss['capital']) < 1e-6
        
        # Check production function: Y = A * K^alpha * L^(1-alpha)
        # Assuming technology A = 1 in steady state
        expected_output = ss['capital']**model.params['alpha'] * ss['labor']**(1-model.params['alpha'])
        assert abs(ss['output'] - expected_output) < 1e-6

    def test_model_variables(self, model):
        """Test that the model variables are set correctly."""
        # Check that model variables exist
        assert hasattr(model, 'variables')
        
        # Check that essential variable categories exist
        essential_categories = ['state', 'control', 'shock']
        for category in essential_categories:
            assert hasattr(model.variables, category)
            assert isinstance(getattr(model.variables, category), list)
            assert len(getattr(model.variables, category)) > 0

    def test_model_equations(self, model):
        """Test that the model equations are set correctly."""
        # Check that model equations exist
        assert hasattr(model, 'equations')
        assert isinstance(model.equations, list)
        assert len(model.equations) > 0
        
        # Check that each equation is a callable
        for eq in model.equations:
            assert callable(eq)


class TestModelExtensions:
    """Tests for model extensions."""

    def test_financial_extension(self, config):
        """Test that the financial extension can be enabled."""
        # Enable financial extension
        config.set('model.extensions.financial', True)
        
        # Initialize model with financial extension
        model = SmetsWoutersModel(config)
        
        # Check that financial variables exist
        financial_vars = ['spread', 'net_worth', 'leverage']
        for var in financial_vars:
            assert var in model.variables.control
        
        # Compute steady state
        model.compute_steady_state()
        
        # Check that financial steady state variables exist
        for var in financial_vars:
            assert var in model.steady_state
            assert isinstance(model.steady_state[var], (int, float))
            assert not np.isnan(model.steady_state[var])
            assert not np.isinf(model.steady_state[var])

    def test_open_economy_extension(self, config):
        """Test that the open economy extension can be enabled."""
        # Enable open economy extension
        config.set('model.extensions.open_economy', True)
        
        # Initialize model with open economy extension
        model = SmetsWoutersModel(config)
        
        # Check that open economy variables exist
        open_economy_vars = ['exports', 'imports', 'exchange_rate', 'foreign_output']
        for var in open_economy_vars:
            assert var in model.variables.control or var in model.variables.state
        
        # Compute steady state
        model.compute_steady_state()
        
        # Check that open economy steady state variables exist
        for var in open_economy_vars:
            assert var in model.steady_state
            assert isinstance(model.steady_state[var], (int, float))
            assert not np.isnan(model.steady_state[var])
            assert not np.isinf(model.steady_state[var])