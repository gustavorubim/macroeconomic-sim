# Master Test Script Implementation Guide

## Overview

The master test script (`master_test.py`) is the central component of our testing framework. It handles test discovery, execution, coverage measurement, and reporting. This document provides a detailed guide for implementing this script.

## Features

The master test script includes the following key features:

1. **Test Discovery and Execution**
   - Automatic discovery of all test files
   - Configurable test selection (unit, integration, functional, performance)
   - Sequential execution of tests

2. **Code Coverage Measurement**
   - Tracks which lines of code are executed during tests
   - Reports coverage percentage by module
   - Identifies uncovered code sections

3. **Test Reporting**
   - Summary statistics (pass/fail counts, execution time)
   - Detailed failure information with stack traces
   - Coverage metrics by module
   - Performance metrics

## Implementation Details

### Script Structure

```python
#!/usr/bin/env python
"""
Master test script for running and managing all tests.

This script:
1. Discovers and runs tests
2. Measures code coverage
3. Generates reports
4. Validates expected behaviors
"""

import os
import sys
import argparse
import time
import datetime
import json
import glob
import subprocess
import shutil
from pathlib import Path
import importlib.util
import importlib.metadata
import pytest
import coverage

# Function definitions and main execution code
```

### Command-Line Arguments

The script should support the following command-line arguments:

```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests and generate reports.')
    
    parser.add_argument(
        '--unit', 
        action='store_true',
        help='Run unit tests only'
    )
    
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run integration tests only'
    )
    
    parser.add_argument(
        '--functional', 
        action='store_true',
        help='Run functional tests only'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all tests'
    )
    
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Measure code coverage'
    )
    
    parser.add_argument(
        '--report-dir', 
        type=str,
        default='test_reports',
        help='Directory for test reports'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--fail-fast', 
        action='store_true',
        help='Stop on first failure'
    )
    
    return parser.parse_args()
```

### Test Path Selection

Based on the command-line arguments, the script should determine which test directories to include:

```python
def get_test_paths(args):
    """
    Get test paths based on command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    
    Returns
    -------
    list
        List of test paths
    """
    # Default: run all tests
    if not (args.unit or args.integration or args.functional or args.performance) or args.all:
        return ['tests/unit', 'tests/integration', 'tests/functional', 'tests/performance']
    
    paths = []
    if args.unit:
        paths.append('tests/unit')
    if args.integration:
        paths.append('tests/integration')
    if args.functional:
        paths.append('tests/functional')
    if args.performance:
        paths.append('tests/performance')
    
    return paths
```

### Test Execution

The script uses pytest to run the tests and optionally measures code coverage:

```python
def run_tests(paths, report_dir, verbose=False, fail_fast=False, measure_coverage=False):
    """
    Run tests using pytest.
    
    Parameters
    ----------
    paths : list
        List of test paths
    report_dir : str
        Directory for test reports
    verbose : bool
        Verbose output
    fail_fast : bool
        Stop on first failure
    measure_coverage : bool
        Measure code coverage
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    dict
        Test results
    """
    # Create report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Build pytest arguments
    pytest_args = paths.copy()
    
    # Add options
    if verbose:
        pytest_args.append('-v')
    if fail_fast:
        pytest_args.append('-x')
    
    # Add test result options
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    junit_report = os.path.join(report_dir, f'test_results_{timestamp}.xml')
    html_report = os.path.join(report_dir, f'test_report_{timestamp}.html')
    
    pytest_args.extend([
        f'--junitxml={junit_report}',
        f'--html={html_report}',
        '--self-contained-html'
    ])
    
    # Initialize results dictionary
    results = {
        'timestamp': timestamp,
        'paths': paths,
        'junit_report': junit_report,
        'html_report': html_report
    }
    
    # Measure code coverage if requested
    if measure_coverage:
        cov = coverage.Coverage(
            source=['dsge'],
            omit=['*/__pycache__/*', '*/tests/*', '*/.tox/*', '*/.venv/*']
        )
        cov.start()
    
    # Run tests
    start_time = time.time()
    exit_code = pytest.main(pytest_args)
    execution_time = time.time() - start_time
    
    # Stop coverage measurement and generate report
    if measure_coverage:
        cov.stop()
        coverage_report = os.path.join(report_dir, f'coverage_{timestamp}')
        os.makedirs(coverage_report, exist_ok=True)
        
        # Generate reports
        cov.html_report(directory=coverage_report)
        cov.xml_report(outfile=os.path.join(coverage_report, 'coverage.xml'))
        
        # Save coverage data
        results['coverage_report'] = coverage_report
        results['coverage_percentage'] = cov.report()
    
    # Update results
    results['exit_code'] = exit_code
    results['execution_time'] = execution_time
    
    return exit_code, results
```

### Summary Generation

After running the tests, the script generates a summary of the results:

```python
def generate_summary(results):
    """
    Generate a summary of test results.
    
    Parameters
    ----------
    results : dict
        Test results
    
    Returns
    -------
    str
        Summary of test results
    """
    summary = []
    summary.append("# Test Execution Summary")
    summary.append("")
    summary.append(f"Timestamp: {results['timestamp']}")
    summary.append(f"Test paths: {', '.join(results['paths'])}")
    summary.append(f"Execution time: {results['execution_time']:.2f} seconds")
    summary.append(f"Exit code: {results['exit_code']} {'(SUCCESS)' if results['exit_code'] == 0 else '(FAILURE)'}")
    summary.append("")
    
    if 'coverage_percentage' in results:
        summary.append(f"Coverage percentage: {results['coverage_percentage']:.2f}%")
    
    summary.append("")
    summary.append("## Reports")
    summary.append("")
    summary.append(f"JUnit XML report: {results['junit_report']}")
    summary.append(f"HTML report: {results['html_report']}")
    
    if 'coverage_report' in results:
        summary.append(f"Coverage report: {results['coverage_report']}")
    
    return "\n".join(summary)
```

### Main Function

The main function ties everything together:

```python
def main():
    """Main function."""
    args = parse_args()
    
    # Get test paths
    paths = get_test_paths(args)
    
    # Run tests
    exit_code, results = run_tests(
        paths=paths,
        report_dir=args.report_dir,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        measure_coverage=args.coverage
    )
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save summary
    summary_path = os.path.join(args.report_dir, f"summary_{results['timestamp']}.md")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    # Print summary
    print("\n" + summary)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
```

## Dependencies

To implement the master test script, the following dependencies are required:

1. **pytest**: For test discovery and execution
2. **coverage**: For measuring code coverage
3. **pytest-html**: For generating HTML reports

These can be installed using pip:

```bash
pip install pytest coverage pytest-html
```

## Usage Examples

### Running All Tests

```bash
python tests/master_test.py --all
```

### Running Unit Tests Only

```bash
python tests/master_test.py --unit
```

### Measuring Code Coverage

```bash
python tests/master_test.py --all --coverage
```

### Verbose Output with Fail-Fast

```bash
python tests/master_test.py --all --verbose --fail-fast
```

### Custom Report Directory

```bash
python tests/master_test.py --all --coverage --report-dir=my_test_reports
```

## Sample Output

The master test script generates several types of output:

### Console Output

```
# Test Execution Summary

Timestamp: 20250313_192500
Test paths: tests/unit, tests/integration, tests/functional, tests/performance
Execution time: 45.23 seconds
Exit code: 0 (SUCCESS)

Coverage percentage: 87.45%

## Reports

JUnit XML report: test_reports/test_results_20250313_192500.xml
HTML report: test_reports/test_report_20250313_192500.html
Coverage report: test_reports/coverage_20250313_192500
```

### HTML Test Report

The HTML test report includes:
- Pass/fail status for each test
- Execution time for each test
- Error messages and stack traces for failed tests
- Links to source code

### Coverage Report

The coverage report includes:
- Overall coverage percentage
- Coverage by module
- Line-by-line coverage information
- Visualization of covered and uncovered lines

## Integration with Continuous Integration

The master test script can be easily integrated with CI systems:

### GitHub Actions Example

```yaml
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        python tests/master_test.py --all --coverage
    - name: Upload test reports
      uses: actions/upload-artifact@v2
      with:
        name: test-reports
        path: test_reports/
```

## Customization Options

The master test script can be extended with additional features:

### Parallel Test Execution

Add support for running tests in parallel to improve performance:

```python
parser.add_argument(
    '--parallel',
    type=int,
    default=1,
    help='Number of parallel processes to use'
)

# In run_tests function
if args.parallel > 1:
    pytest_args.extend(['-n', str(args.parallel)])
```

### Test Selection by Pattern

Add support for running specific tests matching a pattern:

```python
parser.add_argument(
    '--pattern',
    type=str,
    help='Only run tests matching this pattern'
)

# In run_tests function
if args.pattern:
    pytest_args.extend(['-k', args.pattern])
```

### Performance Trend Analysis

Add support for tracking performance trends over time:

```python
def analyze_performance_trends(report_dir):
    """Analyze performance trends from previous test runs."""
    # Implementation details...
```

## Conclusion

The master test script provides a central tool for managing all aspects of testing. It simplifies the process of running tests, measuring coverage, and generating reports. Its flexible design allows for customization and extension as the project evolves.

By implementing this script, we ensure consistent and thorough testing practices, which are essential for maintaining the quality and reliability of the codebase.