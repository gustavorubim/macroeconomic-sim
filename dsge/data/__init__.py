"""
Data module for the DSGE model.

This module provides functionality for fetching and processing
macroeconomic data for use in the DSGE model.
"""

from dsge.data.fetcher import DataFetcher, fetch_fred_data
from dsge.data.processor import DataProcessor

__all__ = [
    "DataFetcher",
    "fetch_fred_data",
    "DataProcessor",
]