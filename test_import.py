"""
This is a simple test script to verify that the imports are working correctly.
"""

print("Attempting to import SmetsWoutersModel...")
from dsge.core import SmetsWoutersModel
print("Successfully imported SmetsWoutersModel!")

print("\nAttempting to import additional modules...")
from dsge.solution import PerturbationSolver
from dsge.analysis import ImpulseResponseFunctions
print("Successfully imported all modules!")

print("\nAll imports worked correctly. The issue has been fixed.")