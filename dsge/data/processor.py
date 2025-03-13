"""
Data processing module for the DSGE model.

This module provides functions for processing macroeconomic data
for use in the DSGE model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
import logging
from statsmodels.tsa.filters.hp_filter import hpfilter

from config.config_manager import ConfigManager
from dsge.data.fetcher import DataFetcher

# Set up logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class for processing macroeconomic data for use in the DSGE model.
    """
    
    def __init__(
        self, 
        data: Optional[Union[Dict[str, pd.Series], pd.DataFrame]] = None,
        config: Optional[Union[Dict[str, Any], ConfigManager]] = None
    ):
        """
        Initialize the data processor.
        
        Args:
            data (Optional[Union[Dict[str, pd.Series], pd.DataFrame]]): Raw data.
                If a dictionary is provided, it should map series names to pandas Series.
                If a DataFrame is provided, columns will be treated as series.
                If None, data will need to be loaded or fetched separately.
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
        if data is None:
            self.raw_data = {}
        elif isinstance(data, pd.DataFrame):
            self.raw_data = {col: data[col] for col in data.columns}
        else:
            self.raw_data = data
        
        # Initialize processed data storage
        self.processed_data = {}
    
    def load_or_fetch_data(self) -> Dict[str, pd.Series]:
        """
        Load data from a file or fetch it from FRED if not available.
        
        Returns:
            Dict[str, pd.Series]: Raw data.
        """
        # Create data fetcher
        fetcher = DataFetcher(self.config)
        
        try:
            # Try to load data from file
            data_path = "data/raw/fred_data.csv"
            self.raw_data = fetcher.load_data(data_path)
            logger.info(f"Loaded data from {data_path}")
        
        except FileNotFoundError:
            # If file not found, fetch data from FRED
            logger.info("Data file not found. Fetching data from FRED...")
            self.raw_data = fetcher.fetch_all_series()
            
            # Save fetched data
            fetcher.save_data(data_path)
        
        return self.raw_data
    
    def transform_series(
        self, 
        series: pd.Series, 
        transformation: str
    ) -> pd.Series:
        """
        Apply a transformation to a time series.
        
        Args:
            series (pd.Series): Time series to transform.
            transformation (str): Transformation to apply.
                Options: "level", "log", "log_diff", "diff", "pct_change".
                
        Returns:
            pd.Series: Transformed time series.
            
        Raises:
            ValueError: If the transformation is not valid.
        """
        # Apply transformation
        if transformation == "level":
            # No transformation
            return series
        
        elif transformation == "log":
            # Natural logarithm
            return np.log(series)
        
        elif transformation == "log_diff":
            # Log difference (approximately percentage change)
            return np.log(series).diff()
        
        elif transformation == "diff":
            # First difference
            return series.diff()
        
        elif transformation == "pct_change":
            # Percentage change
            return series.pct_change()
        
        else:
            raise ValueError(f"Invalid transformation: {transformation}")
    
    def detrend_series(
        self, 
        series: pd.Series, 
        method: str = "hp_filter", 
        **kwargs
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detrend a time series.
        
        Args:
            series (pd.Series): Time series to detrend.
            method (str): Detrending method.
                Options: "hp_filter", "linear", "first_diff", "bandpass".
            **kwargs: Additional arguments for the detrending method.
                
        Returns:
            Tuple[pd.Series, pd.Series]: Detrended series and trend component.
            
        Raises:
            ValueError: If the method is not valid.
        """
        # Apply detrending method
        if method == "hp_filter":
            # Hodrick-Prescott filter
            lambda_param = kwargs.get("lambda", 1600)  # Default for quarterly data
            cycle, trend = hpfilter(series, lamb=lambda_param)
            return cycle, trend
        
        elif method == "linear":
            # Linear trend
            x = np.arange(len(series))
            y = series.values
            
            # Fit linear trend
            slope, intercept, _, _, _ = stats.linregress(x, y)
            trend = pd.Series(intercept + slope * x, index=series.index)
            
            # Detrended series
            detrended = series - trend
            
            return detrended, trend
        
        elif method == "first_diff":
            # First difference
            detrended = series.diff()
            
            # Trend is the cumulative sum of the differences
            trend = series - detrended
            
            return detrended, trend
        
        elif method == "bandpass":
            # Bandpass filter (not implemented here)
            # In a real implementation, this would use a bandpass filter
            # such as the Baxter-King or Christiano-Fitzgerald filter
            raise NotImplementedError("Bandpass filter not implemented")
        
        else:
            raise ValueError(f"Invalid detrending method: {method}")
    
    def handle_missing_values(
        self, 
        series: pd.Series, 
        method: str = "interpolate"
    ) -> pd.Series:
        """
        Handle missing values in a time series.
        
        Args:
            series (pd.Series): Time series with missing values.
            method (str): Method for handling missing values.
                Options: "interpolate", "forward_fill", "backward_fill", "drop".
                
        Returns:
            pd.Series: Time series with missing values handled.
            
        Raises:
            ValueError: If the method is not valid.
        """
        # Apply method for handling missing values
        if method == "interpolate":
            # Linear interpolation
            return series.interpolate(method="linear")
        
        elif method == "forward_fill":
            # Forward fill (use previous value)
            return series.ffill()
        
        elif method == "backward_fill":
            # Backward fill (use next value)
            return series.bfill()
        
        elif method == "drop":
            # Drop missing values
            return series.dropna()
        
        else:
            raise ValueError(f"Invalid method for handling missing values: {method}")
    
    def detect_outliers(
        self, 
        series: pd.Series, 
        method: str = "zscore", 
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in a time series.
        
        Args:
            series (pd.Series): Time series to check for outliers.
            method (str): Method for detecting outliers.
                Options: "zscore", "iqr".
            threshold (float): Threshold for outlier detection.
                For zscore: number of standard deviations.
                For iqr: multiple of IQR.
                
        Returns:
            pd.Series: Boolean series indicating outliers.
            
        Raises:
            ValueError: If the method is not valid.
        """
        # Apply outlier detection method
        if method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = pd.Series(False, index=series.index)
            outliers[series.dropna().index] = z_scores > threshold
            return outliers
        
        elif method == "iqr":
            # Interquartile range method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        else:
            raise ValueError(f"Invalid outlier detection method: {method}")
    
    def handle_outliers(
        self, 
        series: pd.Series, 
        outliers: pd.Series, 
        method: str = "winsorize"
    ) -> pd.Series:
        """
        Handle outliers in a time series.
        
        Args:
            series (pd.Series): Time series with outliers.
            outliers (pd.Series): Boolean series indicating outliers.
            method (str): Method for handling outliers.
                Options: "winsorize", "trim", "interpolate", "median".
                
        Returns:
            pd.Series: Time series with outliers handled.
            
        Raises:
            ValueError: If the method is not valid.
        """
        # Make a copy of the series
        result = series.copy()
        
        # Apply method for handling outliers
        if method == "winsorize":
            # Winsorize (replace with percentiles)
            q1 = series.quantile(0.05)
            q3 = series.quantile(0.95)
            result[outliers & (series < q1)] = q1
            result[outliers & (series > q3)] = q3
            return result
        
        elif method == "trim":
            # Trim (set to NaN)
            result[outliers] = np.nan
            return result
        
        elif method == "interpolate":
            # Interpolate
            result[outliers] = np.nan
            return result.interpolate(method="linear")
        
        elif method == "median":
            # Replace with median
            median = series.median()
            result[outliers] = median
            return result
        
        else:
            raise ValueError(f"Invalid method for handling outliers: {method}")
    
    def process_series(
        self, 
        name: str, 
        series: pd.Series
    ) -> pd.Series:
        """
        Process a time series according to the configuration.
        
        Args:
            name (str): Name of the series.
            series (pd.Series): Time series to process.
                
        Returns:
            pd.Series: Processed time series.
        """
        # Get series configuration
        series_config = self.data_params["series"].get(name, {})
        transformation = series_config.get("transformation", "level")
        
        # Get detrending configuration
        detrending_config = self.data_params.get("detrending", {})
        detrending_method = detrending_config.get("method", "hp_filter")
        
        # Handle missing values
        series = self.handle_missing_values(series)
        
        # Apply transformation
        transformed = self.transform_series(series, transformation)
        
        # Detect and handle outliers
        outliers = self.detect_outliers(transformed)
        transformed = self.handle_outliers(transformed, outliers)
        
        # Detrend if needed
        if detrending_method != "none":
            detrended, _ = self.detrend_series(
                transformed, 
                method=detrending_method, 
                **detrending_config
            )
            return detrended
        
        return transformed
    
    def process_all_series(self) -> Dict[str, pd.Series]:
        """
        Process all series in the raw data.
        
        Returns:
            Dict[str, pd.Series]: Processed data.
        """
        # Process each series
        for name, series in self.raw_data.items():
            try:
                self.processed_data[name] = self.process_series(name, series)
                logger.info(f"Processed series: {name}")
            
            except Exception as e:
                logger.error(f"Error processing series {name}: {str(e)}")
        
        return self.processed_data
    
    def get_processed_data(
        self, 
        name: Optional[str] = None
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Get processed data.
        
        Args:
            name (Optional[str]): Name of the series to get.
                If None, all processed data will be returned.
                
        Returns:
            Union[pd.Series, Dict[str, pd.Series]]: Processed data.
            
        Raises:
            KeyError: If the requested series has not been processed.
        """
        if name is None:
            return self.processed_data
        
        if name not in self.processed_data:
            raise KeyError(f"Series '{name}' has not been processed.")
        
        return self.processed_data[name]
    
    def save_processed_data(self, path: str) -> None:
        """
        Save processed data to a CSV file.
        
        Args:
            path (str): Path to save the data.
        """
        # Convert dictionary of Series to DataFrame
        df = pd.DataFrame(self.processed_data)
        
        # Save to CSV
        df.to_csv(path)
        logger.info(f"Processed data saved to {path}")
    
    def load_processed_data(self, path: str) -> Dict[str, pd.Series]:
        """
        Load processed data from a CSV file.
        
        Args:
            path (str): Path to load the data from.
            
        Returns:
            Dict[str, pd.Series]: Processed data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            # Load data from CSV
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Convert DataFrame to dictionary of Series
            self.processed_data = {col: df[col] for col in df.columns}
            
            logger.info(f"Processed data loaded from {path}")
            return self.processed_data
        
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def prepare_data_for_estimation(self) -> pd.DataFrame:
        """
        Prepare data for model estimation.
        
        Returns:
            pd.DataFrame: Data prepared for estimation.
        """
        # If no processed data, process all series
        if not self.processed_data:
            if not self.raw_data:
                self.load_or_fetch_data()
            self.process_all_series()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.processed_data)
        
        # Drop rows with missing values
        df = df.dropna()
        
        return df