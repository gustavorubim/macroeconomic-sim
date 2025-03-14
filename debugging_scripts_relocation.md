# Debugging Scripts Relocation Plan

## Overview

This document outlines the plan for relocating debugging scripts from the root directory to the `tests/debug/` directory. This is part of our project structure refactoring to improve organization and maintainability.

## Scripts to Relocate

The following scripts need to be moved from the root directory to `tests/debug/`:

1. `debug_steady_state.py` → `tests/debug/debug_steady_state.py`
2. `test_import.py` → `tests/debug/test_import.py`
3. `test_core_import.py` → `tests/debug/test_core_import.py`
4. `fix_corruption.py` → `tests/debug/fix_corruption.py`

## Required Modifications

### 1. debug_steady_state.py

**Original Path:** `debug_steady_state.py`
**New Path:** `tests/debug/debug_steady_state.py`

**Modifications needed:**
- Update the Python path modification from:
  ```python
  # Add the project root to Python path to import modules
  import os
  import sys
  sys.path.insert(0, os.path.abspath('.'))
  ```
  
  to:
  ```python
  # Add the project root to Python path to import modules
  import os
  import sys
  sys.path.insert(0, os.path.abspath('../..'))
  ```

**Script Purpose:**
This script identifies which equation is causing complex number issues in the steady state calculations. It performs a step-by-step evaluation of each equation at the calculated steady state values to find any discrepancies or issues.

### 2. test_import.py

**Original Path:** `test_import.py`
**New Path:** `tests/debug/test_import.py`

**Modifications needed:**
- Update the Python path modification (if present) similar to `debug_steady_state.py`
- No other changes likely needed as it appears to be a simple import test

**Script Purpose:**
This is a simple test script to verify that imports are working correctly. It attempts to import key modules like `SmetsWoutersModel`, `PerturbationSolver`, and `ImpulseResponseFunctions` to confirm they are accessible.

### 3. test_core_import.py

**Original Path:** `test_core_import.py`
**New Path:** `tests/debug/test_core_import.py`

**Modifications needed:**
- Update the Python path modification (if present) similar to `debug_steady_state.py`
- No other changes likely needed as it's a minimal import test

**Script Purpose:**
This is a minimal test script to verify that core imports are working correctly without requiring additional dependencies. It specifically tests importing `SmetsWoutersModel` directly from `base_model`.

### 4. fix_corruption.py

**Original Path:** `fix_corruption.py`
**New Path:** `tests/debug/fix_corruption.py`

**Modifications needed:**
- Update the base directory detection in the `main()` function from:
  ```python
  # Get the base directory
  base_dir = os.path.dirname(os.path.abspath(__file__))
  ```
  
  to:
  ```python
  # Get the project root directory (two levels up from this script)
  base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
  ```

**Script Purpose:**
This script fixes corrupted Python files containing null bytes and clears cached bytecode. It scans all Python files, identifies corruptions, attempts to fix them, and creates backups of the original files.

## Implementation Steps

1. **Create Directory Structure**
   ```
   tests/
   └── debug/
       └── __init__.py
   ```

2. **Copy and Modify Scripts**
   - Copy each script to its new location
   - Apply the modifications outlined above
   - Test each script to ensure it still functions correctly

3. **Create __init__.py**
   Create an `__init__.py` file in the `tests/debug/` directory with the following content:
   
   ```python
   """
   Debugging scripts for the DSGE model implementation.
   
   This package contains scripts used for debugging various aspects of the model:
   
   - debug_steady_state.py: Identifies issues in steady state calculations
   - test_import.py: Verifies basic imports are working
   - test_core_import.py: Tests core module imports specifically
   - fix_corruption.py: Fixes Python files with null byte corruption
   """
   ```

4. **Create README.md in tests/debug/**
   Create a README.md file to document the purpose of each debugging script:
   
   ```markdown
   # Debugging Scripts
   
   This directory contains scripts used for debugging various aspects of the DSGE model implementation.
   
   ## Scripts
   
   - **debug_steady_state.py**: Identifies which equation is causing complex number issues in steady state calculations
   - **test_import.py**: Simple test to verify that imports are working correctly
   - **test_core_import.py**: Minimal test for core imports without additional dependencies
   - **fix_corruption.py**: Utility to fix corrupted Python files containing null bytes and clear cached bytecode
   
   ## Usage
   
   Run scripts from the project root directory:
   
   ```bash
   python -m tests.debug.debug_steady_state
   python -m tests.debug.test_import
   python -m tests.debug.test_core_import
   python -m tests.debug.fix_corruption
   ```
   ```

5. **Update Documentation**
   Update any documentation that references these scripts to point to their new locations.

## Verification Procedure

After relocating the debugging scripts, perform the following checks:

1. Run each script from the project root directory using the module syntax:
   ```bash
   python -m tests.debug.debug_steady_state
   python -m tests.debug.test_import
   python -m tests.debug.test_core_import
   python -m tests.debug.fix_corruption
   ```

2. Verify that each script runs correctly and produces the expected output.

3. Ensure that any functionality that depends on these scripts continues to work.

## Backward Compatibility (Optional)

If backward compatibility is needed, consider creating simple wrapper scripts in the root directory that import and call the relocated scripts. For example:

```python
#!/usr/bin/env python
"""
Wrapper script to maintain backward compatibility.
"""
import os
import sys
import importlib.util

# Import the relocated script
from tests.debug.debug_steady_state import debug_equations

if __name__ == "__main__":
    debug_equations()
```

## Future Improvements

As the testing framework evolves, consider these improvements:

1. Integrate these debugging scripts with the master test script
2. Add command-line options for more flexibility
3. Improve error reporting and logging
4. Add more extensive documentation for each script

## Conclusion

Relocating these debugging scripts improves project organization by:
- Cleaning up the root directory
- Grouping related debugging tools together
- Establishing a clear separation between application code and development/testing tools
- Making the project structure more maintainable and easier to navigate