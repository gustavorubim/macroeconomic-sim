#!/usr/bin/env python
"""
Fix corrupted Python files containing null bytes and clear cached bytecode.
"""

import os
import shutil
import sys

def is_likely_corrupted(content):
    """Check if a file's content is likely corrupted with null bytes or encoding issues."""
    if b'\x00' in content:  # Check for null bytes
        return True
    try:
        content.decode('utf-8')
        return False  # Successfully decoded, not corrupted
    except UnicodeDecodeError:
        return True  # Failed to decode, likely corrupted

def fix_file(filepath):
    """Attempt to fix a corrupted file."""
    try:
        # Try to read the file as binary
        with open(filepath, 'rb') as f:
            content = f.read()
        
        if not is_likely_corrupted(content):
            print(f"File {filepath} appears to be fine, skipping.")
            return False
        
        # Try to decode, replacing problematic characters
        try:
            # First try to strip null bytes
            content = content.replace(b'\x00', b'')
            text = content.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding {filepath}: {e}")
            return False
        
        # Fix common issues with spacing
        if filepath.endswith('__init__.py'):
            # Check if it has the spaced-out format
            if '  ' in text and text.count(' ') > len(text) // 3:
                print(f"File {filepath} has abnormal spacing, attempting to fix.")
                # This is a heuristic to detect the corrupted files with spaces between characters
                # Remove excessive spaces between characters
                text = ''.join(text.split())
                
                # Format the file as a simple init file
                module_name = os.path.basename(os.path.dirname(filepath))
                text = f'"""\n{module_name.capitalize()} module.\n"""\n\n'
                
                # For dsge/core/__init__.py, add specific imports
                if 'dsge/core' in filepath:
                    text += ('from dsge.core.base_model import SmetsWoutersModel, ModelVariables\n'
                            'from dsge.core.steady_state import (\n'
                            '    compute_steady_state,\n'
                            '    compute_base_steady_state,\n'
                            '    compute_financial_extension_steady_state,\n'
                            '    compute_open_economy_extension_steady_state,\n'
                            '    compute_fiscal_extension_steady_state,\n'
                            '    check_steady_state,\n'
                            ')\n\n'
                            '__all__ = [\n'
                            '    "SmetsWoutersModel",\n'
                            '    "ModelVariables",\n'
                            '    "compute_steady_state",\n'
                            '    "compute_base_steady_state",\n'
                            '    "compute_financial_extension_steady_state",\n'
                            '    "compute_open_economy_extension_steady_state",\n'
                            '    "compute_fiscal_extension_steady_state",\n'
                            '    "check_steady_state",\n'
                            ']\n')
                
                # For dsge/__init__.py, add specific imports
                elif 'dsge/' in filepath and len(os.path.dirname(filepath).split('/')) == 1:
                    text += ('from dsge.core import SmetsWoutersModel, ModelVariables\n'
                            'from dsge.core import compute_steady_state, check_steady_state\n'
                            'from dsge.solution import PerturbationSolver, ProjectionSolver\n'
                            'from dsge.data import DataFetcher, DataProcessor\n'
                            'from dsge.estimation import BayesianEstimator, PriorDistribution, PriorSet\n'
                            'from dsge.analysis import ImpulseResponseFunctions, ShockDecomposition, ModelDiagnostics\n'
                            'from dsge.forecasting import BaselineForecaster, ScenarioForecaster, UncertaintyQuantifier\n\n'
                            '__version__ = "0.1.0"\n\n'
                            '__all__ = [\n'
                            '    # Core\n'
                            '    "SmetsWoutersModel",\n'
                            '    "ModelVariables",\n'
                            '    "compute_steady_state",\n'
                            '    "check_steady_state",\n'
                            '    \n'
                            '    # Solution\n'
                            '    "PerturbationSolver",\n'
                            '    "ProjectionSolver",\n'
                            '    \n'
                            '    # Data\n'
                            '    "DataFetcher",\n'
                            '    "DataProcessor",\n'
                            '    \n'
                            '    # Estimation\n'
                            '    "BayesianEstimator",\n'
                            '    "PriorDistribution",\n'
                            '    "PriorSet",\n'
                            '    \n'
                            '    # Analysis\n'
                            '    "ImpulseResponseFunctions",\n'
                            '    "ShockDecomposition",\n'
                            '    "ModelDiagnostics",\n'
                            '    \n'
                            '    # Forecasting\n'
                            '    "BaselineForecaster",\n'
                            '    "ScenarioForecaster",\n'
                            '    "UncertaintyQuantifier",\n'
                            ']\n')
        
        # Backup the original file
        backup_path = filepath + '.bak'
        shutil.copy2(filepath, backup_path)
        print(f"Backed up {filepath} to {backup_path}")
        
        # Write the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Fixed {filepath}")
        return True
    
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def clear_pycache(directory):
    """Clear __pycache__ directories."""
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if d == '__pycache__':
                cache_dir = os.path.join(root, d)
                print(f"Removing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)

def main():
    """Main function."""
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files
    fixed_count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_file(filepath):
                    fixed_count += 1
    
    # Clear all __pycache__ directories
    clear_pycache(base_dir)
    
    print(f"Fixed {fixed_count} corrupted files and cleared Python cache.")
    print("Please try running your script again.")

if __name__ == "__main__":
    main()