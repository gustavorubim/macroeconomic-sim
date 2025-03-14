# DSGE Model Testing Framework

This directory contains a comprehensive testing framework for the DSGE model. The framework includes unit tests, integration tests, functional tests, performance tests, and debugging scripts.

## Directory Structure

- `unit/`: Unit tests for individual components of the DSGE model
- `integration/`: Integration tests for interactions between components
- `functional/`: Functional tests for end-to-end workflows
- `performance/`: Performance tests for measuring execution time and resource usage
- `debug/`: Debugging scripts for identifying and fixing issues
- `utils/`: Utility functions for testing

## Running Tests

The testing framework uses pytest for running tests. To run all tests, use the master test script:

```bash
python tests/run_tests.py --all
```

### Running Specific Test Types

You can run specific types of tests using the following options:

```bash
# Run unit tests only
python tests/run_tests.py --unit

# Run integration tests only
python tests/run_tests.py --integration

# Run functional tests only
python tests/run_tests.py --functional

# Run performance tests only
python tests/run_tests.py --performance
```

### Measuring Code Coverage

To measure code coverage, use the `--coverage` option:

```bash
python tests/run_tests.py --all --coverage
```

This will generate a coverage report in the `coverage_report` directory.

### Generating Test Reports

The test script automatically generates a test report in JSON format. You can specify the output path using the `--report` option:

```bash
python tests/run_tests.py --all --report my_test_report.json
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`. These fixtures include:

- `project_root`: The root directory of the project
- `config`: A ConfigManager instance with default configuration
- `model`: A SmetsWoutersModel instance with default configuration
- `sample_data`: A sample dataset for testing
- `steady_state`: The steady state of the model
- `temp_dir`: A temporary directory for test files
- `random_seed`: A random seed for reproducibility

## Mock Data Generation

The `utils/mock_data.py` module provides functions for generating mock data for testing:

- `generate_random_time_series()`: Generate random time series data
- `generate_model_consistent_data()`: Generate data consistent with the model
- `generate_structural_shock_data()`: Generate data with a structural shock
- `generate_test_dataset()`: Generate a complete test dataset

## Debugging Scripts

The `debug/` directory contains scripts for debugging the DSGE model:

- `debug_steady_state.py`: Debug steady state equations
- `fix_corruption.py`: Fix corrupted Python files
- `test_core_import.py`: Test core imports
- `test_import.py`: Test all imports

## Adding New Tests

To add a new test, create a new Python file in the appropriate directory. The file name should start with `test_` to be discovered by pytest. For example:

```python
# tests/unit/test_my_feature.py

def test_my_feature():
    # Test code here
    assert True
```

## Test Naming Conventions

- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`

## Best Practices

1. Each test should test a single functionality
2. Tests should be independent and not rely on the state of other tests
3. Use fixtures for common setup and teardown
4. Use assertions to verify expected behavior
5. Use parameterized tests for testing multiple inputs
6. Use mocks and stubs for isolating components
7. Include both positive and negative test cases
8. Test edge cases and boundary conditions