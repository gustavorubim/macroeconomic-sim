"""
This is a minimal test script to verify that the core imports are working correctly
without requiring additional dependencies.
"""

print("Attempting to import SmetsWoutersModel...")
from dsge.core.base_model import SmetsWoutersModel
print("Successfully imported SmetsWoutersModel directly from base_model!")

print("\nAll core imports worked correctly. The null bytes issue has been fixed.")