"""
Data fetching module for the DSGE model.

This module provides functions for fetching macroeconomic data from FRED
using pandas_datareader.
"""

import pandas as pd
import pandas_datareader.data as web
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from config.config_manager import ConfigManager

# Set up logging
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Class for fetching macroeconomic data from FRED.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigManager]] = None):
        """
        Initialize the data fetcher.
        
        Args:
            config (Optional[Union[Dict[str, Any], ConfigManager]]): Configuration.
                If a dictionary is provided, it will be used as the configuration.
                If a ConfigManager is provided, its configuration will be used.
                If None, the default configuration will be used.
        """
        # Initialize configuration
        if config is None:
            self.config = ConfigManager()
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.update(config)
        else:
            self.config = config
        
        # Extract data parameters
        self.data_params = self.config.get("data")
        
        # Initialize data storage
        self.data = {}
    
    def fetch_fred_series(
        self, 
        series_id: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch a time series from FRED.
        
        Args:
            series_id (str): FRED series ID.
            start_date (Optional[str]): Start date in ISO format (YYYY-MM-DD).
                If None, the start date from the configuration will be used.
            end_date (Optional[str]): End date in ISO format (YYYY-MM-DD).
                If None, the end date from the configuration will be used.
                
        Returns:
            pd.Series: Time series data.
            
        Raises:
            Exception: If the data could not be fetched.
        """
        # Use configuration dates if not provided
        if start_date is None:
            start_date = self.data_params["start_date"]
        if end_date is None:
            end_date = self.data_params["end_date"]
        
        # Convert dates to datetime objects
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        try:
            # Fetch data from FRED
            logger.info(f"Fetching {series_id} from FRED...")
            data = web.DataReader(series_id, "fred", start, end)
            
            # Convert to Series if DataFrame
            if isinstance(data, pd.DataFrame):
                data = data[data.columns[0]]
            
            # Set name
            data.name = series_id
            
            logger.info(f"Successfully fetched {series_id} with {len(data)} observations.")
            return data
        
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {str(e)}")
            raise
    
    def fetch_all_series(self) -> Dict[str, pd.Series]:
        """
        Fetch all series specified in the configuration.
        
        Returns:
            Dict[str, pd.Series]: Dictionary of time series data.
        """
        # Get series configuration
        series_config = self.data_params["series"]
        
        # Fetch each series
        for name, config in series_config.items():
            if config["source"] == "FRED":
                try:
                    self.data[name] = self.fetch_fred_series(config["series_id"])
                except Exception as e:
                    logger.error(f"Error fetching {name}: {str(e)}")
        
        return self.data
    
    def get_data(self, name: Optional[str] = None) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Get fetched data.
        
        Args:
            name (Optional[str]): Name of the series to get.
                If None, all fetched data will be returned.
                
        Returns:
            Union[pd.Series, Dict[str, pd.Series]]: Time series data.
            
        Raises:
            KeyError: If the requested series has not been fetched.
        """
        if name is None:
            return self.data
        
        if name not in self.data:
            raise KeyError(f"Series '{name}' has not been fetched.")
        
        return self.data[name]
    
    def save_data(self, path: str) -> None:
        """
        Save fetched data to a CSV file.
        
        Args:
            path (str): Path to save the data.
        """
        # Convert dictionary of Series to DataFrame
        df = pd.DataFrame(self.data)
        
        # Save to CSV
        df.to_csv(path)
        logger.info(f"Data saved to {path}")
    
    def load_data(self, path: str) -> Dict[str, pd.Series]:
        """
        Load data from a CSV file.
        
        Args:
            path (str): Path to load the data from.
            
        Returns:
            Dict[str, pd.Series]: Dictionary of time series data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            # Load data from CSV
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Convert DataFrame to dictionary of Series
            self.data = {col: df[col] for col in df.columns}
            
            logger.info(f"Data loaded from {path}")
            return self.data
        
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


def fetch_fred_data(
    series_ids: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch multiple time series from FRED.
    
    Args:
        series_ids (List[str]): List of FRED series IDs.
        start_date (str): Start date in ISO format (YYYY-MM-DD).
        end_date (str): End date in ISO format (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: DataFrame containing the requested time series.
    """
    # Convert dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initialize empty DataFrame
    data = pd.DataFrame()
    
    # Fetch each series
    for series_id in series_ids:
        try:
            # Fetch data from FRED
            series_data = web.DataReader(series_id, "fred", start, end)
            
            # Add to DataFrame
            data[series_id] = series_data[series_data.columns[0]]
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {str(e)}")
    
    return data