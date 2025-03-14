#!/usr/bin/env python
"""
Master test script for the DSGE model.

This script discovers and runs all tests for the DSGE model, measures code coverage,
and generates a detailed report of the test results.
"""

import os
import sys
import time
import argparse
import pytest
import coverage
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path to import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for the DSGE model.')
    
    parser.add_argument(
        '--unit', 
        action='store_true',
        help='Run unit tests only.'
    )
    
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run integration tests only.'
    )
    
    parser.add_argument(
        '--functional', 
        action='store_true',
        help='Run functional tests only.'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only.'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all tests.'
    )
    
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Measure code coverage.'
    )
    
    parser.add_argument(
        '--report', 
        type=str, 
        default='test_report.json',
        help='Path to the test report file.'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output.'
    )
    
    return parser.parse_args()


def run_tests(test_type=None, verbose=False, coverage_obj=None):
    """Run tests of the specified type."""
    start_time = time.time()
    
    # Determine which tests to run using the project root
    if test_type is None:
        test_path = os.path.join(BASE_DIR, 'tests')
    else:
        test_path = os.path.join(BASE_DIR, 'tests', test_type)
    
    # Set up pytest arguments
    pytest_args = [test_path, '-v'] if verbose else [test_path]
    
    # Add coverage if requested
    if coverage_obj is not None:
        coverage_obj.start()
    
    # Run tests
    result = pytest.main(pytest_args)
    
    # Stop coverage if requested
    if coverage_obj is not None:
        coverage_obj.stop()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'result': result,
        'execution_time': execution_time
    }


def measure_coverage(test_types=None):
    """Measure code coverage for the specified test types."""
    # Initialize coverage
    cov = coverage.Coverage(
        source=['dsge'],
        omit=['*/__pycache__/*', '*/tests/*', '*/venv/*', '*/env/*']
    )
    
    # Run tests with coverage
    if test_types is None:
        result = run_tests(coverage_obj=cov)
    else:
        result = {}
        for test_type in test_types:
            result[test_type] = run_tests(test_type, coverage_obj=cov)
    
    # Generate coverage report
    cov.save()
    cov.html_report(directory='coverage_report')
    
    return {
        'test_results': result,
        'coverage': cov.report()
    }


def generate_report(results, report_path):
    """Generate a detailed report of the test results."""
    # Create report directory if it doesn't exist
    report_dir = os.path.dirname(report_path)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Write report to file
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Determine which tests to run
    test_types = []
    if args.unit:
        test_types.append('unit')
    if args.integration:
        test_types.append('integration')
    if args.functional:
        test_types.append('functional')
    if args.performance:
        test_types.append('performance')
    if args.all or not test_types:
        test_types = ['unit', 'integration', 'functional', 'performance']
    
    # Run tests
    results = {}
    if args.coverage:
        # Run tests with coverage
        coverage_results = measure_coverage(test_types)
        results = {
            'test_results': coverage_results['test_results'],
            'coverage': coverage_results['coverage']
        }
    else:
        # Run tests without coverage
        for test_type in test_types:
            results[test_type] = run_tests(test_type, args.verbose)
    
    # Generate report
    report = generate_report(results, args.report)
    
    # Print summary
    print("\nTest Summary:")
    for test_type, result in results.items():
        if isinstance(result, dict) and 'result' in result:
            status = "PASSED" if result['result'] == 0 else "FAILED"
            print(f"  {test_type.capitalize()} Tests: {status} (Time: {result['execution_time']:.2f}s)")
        else:
            print(f"  {test_type.capitalize()} Tests: {result}")
    
    if args.coverage:
        print(f"\nCode Coverage: {results['coverage']:.2f}%")
    
    print(f"\nDetailed report saved to {args.report}")


if __name__ == '__main__':
    main()