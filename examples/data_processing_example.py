#!/usr/bin/env python
"""
Data Processing Example

This script demonstrates the complete data processing workflow for DSGE model estimation:
1. Fetching data from FRED
2. Preprocessing and transforming data
3. Handling missing values and outliers
4. Creating model-compatible datasets
5. Visualizing data

The example shows best practices for preparing macroeconomic time series for DSGE model estimation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# In a real implementation, these would be imported from the package
# For demonstration purposes, we'll mock some of these classes
class FredDataFetcher:
    """Mock class for fetching data from FRED."""
    
    def __init__(self, api_key=None):
        """
        Initialize the FRED data fetcher.
        
        Args:
            api_key (str, optional): API key for FRED. Not required for low-volume requests.
        """
        self.api_key = api_key
        print(f"Initialized FRED data fetcher{' with API key' if api_key else ''}")
    
    def fetch_series(self, series_id, start_date, end_date):
        """
        Fetch a single series from FRED.
        
        Args:
            series_id (str): FRED series identifier
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.Series: Time series data
        """
        print(f"Fetching {series_id} from {start_date} to {end_date}")
        
        # In a real implementation, this would use pandas_datareader
        # For demonstration, we'll generate synthetic data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start, end=end, freq='Q')
        
        # Generate synthetic data based on the series ID
        if series_id == "GDPC1":  # Real GDP
            data = np.linspace(10000, 20000, len(date_range))
            data = data * (1 + 0.01 * np.random.randn(len(date_range)))
        elif series_id == "PCEPILFE":  # Core PCE inflation
            data = 2 + 0.5 * np.random.randn(len(date_range))
        elif series_id == "FEDFUNDS":  # Federal funds rate
            data = 3 + 2 * np.sin(np.linspace(0, 4*np.pi, len(date_range))) + 0.5 * np.random.randn(len(date_range))
        elif series_id == "CE16OV":  # Civilian employment
            data = np.linspace(120000, 160000, len(date_range))
            data = data * (1 + 0.005 * np.random.randn(len(date_range)))
        elif series_id == "PCECC96":  # Real consumption
            data = np.linspace(8000, 15000, len(date_range))
            data = data * (1 + 0.01 * np.random.randn(len(date_range)))
        elif series_id == "GPDI":  # Gross private domestic investment
            data = np.linspace(1500, 3500, len(date_range))
            data = data * (1 + 0.03 * np.random.randn(len(date_range)))
        elif series_id == "CPIAUCSL":  # CPI
            data = np.cumsum(0.5 + 0.2 * np.random.randn(len(date_range)))
            data = 100 * np.exp(data / 100)
        else:
            # Default random data
            data = 100 + 10 * np.random.randn(len(date_range))
        
        return pd.Series(data, index=date_range, name=series_id)
    
    def fetch_multiple(self, series_ids, start_date, end_date):
        """
        Fetch multiple series from FRED.
        
        Args:
            series_ids (dict or list): FRED series identifiers with optional descriptions
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Time series data
        """
        print(f"Fetching multiple series from {start_date} to {end_date}")
        
        if isinstance(series_ids, dict):
            series_list = list(series_ids.keys())
            descriptions = series_ids
        else:
            series_list = series_ids
            descriptions = {s: s for s in series_list}
        
        data_dict = {}
        for series_id in series_list:
            data_dict[series_id] = self.fetch_series(series_id, start_date, end_date)
        
        df = pd.DataFrame(data_dict)
        
        # Add metadata
        df.attrs['descriptions'] = descriptions
        df.attrs['source'] = 'FRED'
        df.attrs['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df


class DataProcessor:
    """Class for processing and transforming macroeconomic data."""
    
    def __init__(self, data):
        """
        Initialize the data processor.
        
        Args:
            data (pd.DataFrame): Raw data to process
        """
        self.raw_data = data.copy()
        self.processed_data = data.copy()
        self.transformations = {}
        
        print(f"Initialized data processor with {len(data.columns)} series")
    
    def apply_transformation(self, series_id, transformation, **kwargs):
        """
        Apply a transformation to a data series.
        
        Args:
            series_id (str): Column name of the series to transform
            transformation (str): Type of transformation to apply
            **kwargs: Additional arguments for the transformation
        """
        if series_id not in self.processed_data.columns:
            raise ValueError(f"Series {series_id} not found in data")
        
        # Store transformation for documentation
        self.transformations[series_id] = {
            'type': transformation,
            'params': kwargs
        }
        
        # Apply the transformation
        if transformation == "log":
            self.processed_data[series_id] = np.log(self.processed_data[series_id])
            print(f"Applied log transformation to {series_id}")
            
        elif transformation == "diff":
            periods = kwargs.get('periods', 1)
            self.processed_data[series_id] = self.processed_data[series_id].diff(periods)
            print(f"Applied {periods}-period differencing to {series_id}")
            
        elif transformation == "pct_change":
            periods = kwargs.get('periods', 1)
            scale = kwargs.get('scale', 100)
            self.processed_data[series_id] = self.processed_data[series_id].pct_change(periods) * scale
            print(f"Applied {periods}-period percent change to {series_id} (scale: {scale})")
            
        elif transformation == "log_diff":
            periods = kwargs.get('periods', 1)
            scale = kwargs.get('scale', 100)
            self.processed_data[series_id] = np.log(self.processed_data[series_id]).diff(periods) * scale
            print(f"Applied {periods}-period log difference to {series_id} (scale: {scale})")
            
        elif transformation == "hp_filter":
            from statsmodels.tsa.filters.hp_filter import hpfilter
            lambda_ = kwargs.get('lambda_', 1600)
            cycle, trend = hpfilter(self.processed_data[series_id], lamb=lambda_)
            self.processed_data[series_id] = cycle
            print(f"Applied HP filter to {series_id} (lambda: {lambda_})")
            
        elif transformation == "divide":
            divisor = kwargs.get('divisor', 1)
            self.processed_data[series_id] = self.processed_data[series_id] / divisor
            print(f"Divided {series_id} by {divisor}")
            
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
    
    def impute_missing_values(self, method="linear"):
        """
        Impute missing values in the processed data.
        
        Args:
            method (str): Imputation method ('linear', 'ffill', 'bfill', 'mean')
        """
        missing_before = self.processed_data.isna().sum().sum()
        
        if method == "linear":
            self.processed_data = self.processed_data.interpolate(method='linear')
        elif method == "ffill":
            self.processed_data = self.processed_data.ffill()
        elif method == "bfill":
            self.processed_data = self.processed_data.bfill()
        elif method == "mean":
            self.processed_data = self.processed_data.fillna(self.processed_data.mean())
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        missing_after = self.processed_data.isna().sum().sum()
        print(f"Imputed {missing_before - missing_after} missing values using {method} method")
    
    def remove_outliers(self, threshold=3.0, method="winsorize"):
        """
        Remove or transform outliers in the processed data.
        
        Args:
            threshold (float): Z-score threshold for outlier detection
            method (str): Method for handling outliers ('winsorize', 'trim', 'replace')
        """
        outliers_count = 0
        
        for col in self.processed_data.columns:
            # Skip columns with non-numeric data
            if not np.issubdtype(self.processed_data[col].dtype, np.number):
                continue
                
            # Calculate z-scores
            z_scores = np.abs((self.processed_data[col] - self.processed_data[col].mean()) / self.processed_data[col].std())
            outliers = z_scores > threshold
            col_outliers = outliers.sum()
            outliers_count += col_outliers
            
            if col_outliers > 0:
                if method == "winsorize":
                    # Cap values at threshold
                    upper_bound = self.processed_data[col].mean() + threshold * self.processed_data[col].std()
                    lower_bound = self.processed_data[col].mean() - threshold * self.processed_data[col].std()
                    self.processed_data.loc[self.processed_data[col] > upper_bound, col] = upper_bound
                    self.processed_data.loc[self.processed_data[col] < lower_bound, col] = lower_bound
                
                elif method == "trim":
                    # Set outliers to NaN and then impute
                    self.processed_data.loc[outliers, col] = np.nan
                    self.processed_data[col] = self.processed_data[col].interpolate(method='linear')
                
                elif method == "replace":
                    # Replace with mean
                    self.processed_data.loc[outliers, col] = self.processed_data[col].mean()
                
                else:
                    raise ValueError(f"Unknown outlier handling method: {method}")
        
        print(f"Handled {outliers_count} outliers using {method} method (threshold: {threshold})")
    
    def get_processed_data(self):
        """
        Get the processed data.
        
        Returns:
            pd.DataFrame: Processed data
        """
        return self.processed_data
    
    def get_transformations(self):
        """
        Get the applied transformations.
        
        Returns:
            dict: Dictionary of applied transformations
        """
        return self.transformations


class ModelDataset:
    """Class for preparing data for model estimation."""
    
    def __init__(self, data):
        """
        Initialize the model dataset.
        
        Args:
            data (pd.DataFrame): Processed data
        """
        self.data = data.copy()
        self.variable_mapping = {}
        
        print(f"Initialized model dataset with {len(data.columns)} series")
    
    def map_variable(self, model_var, data_series):
        """
        Map a model variable to a data series.
        
        Args:
            model_var (str): Model variable name
            data_series (str): Data series column name
        """
        if data_series not in self.data.columns:
            raise ValueError(f"Series {data_series} not found in data")
        
        self.variable_mapping[model_var] = data_series
        print(f"Mapped model variable '{model_var}' to data series '{data_series}'")
    
    def prepare_for_estimation(self, start_date=None, end_date=None):
        """
        Prepare data for model estimation.
        
        Args:
            start_date (str, optional): Start date for estimation sample
            end_date (str, optional): End date for estimation sample
            
        Returns:
            pd.DataFrame: Estimation-ready dataset
        """
        # Filter date range if specified
        if start_date or end_date:
            mask = True
            if start_date:
                mask = mask & (self.data.index >= pd.to_datetime(start_date))
            if end_date:
                mask = mask & (self.data.index <= pd.to_datetime(end_date))
            filtered_data = self.data[mask]
            print(f"Filtered data from {filtered_data.index[0]} to {filtered_data.index[-1]}")
        else:
            filtered_data = self.data
        
        # Create estimation dataset with model variable names
        estimation_data = pd.DataFrame(index=filtered_data.index)
        
        for model_var, data_series in self.variable_mapping.items():
            estimation_data[model_var] = filtered_data[data_series]
        
        # Drop rows with missing values
        missing_before = len(estimation_data)
        estimation_data = estimation_data.dropna()
        missing_after = len(estimation_data)
        
        if missing_before > missing_after:
            print(f"Dropped {missing_before - missing_after} rows with missing values")
        
        # Add metadata
        estimation_data.attrs['variable_mapping'] = self.variable_mapping
        estimation_data.attrs['preparation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Prepared estimation dataset with {len(estimation_data.columns)} variables and {len(estimation_data)} observations")
        return estimation_data


def main():
    """Main function demonstrating the data processing workflow."""
    # Create output directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("=== DSGE Model Data Processing Example ===")
    print("\n1. Fetching Data from FRED")
    # Step 1: Fetch data from FRED
    fetcher = FredDataFetcher()
    raw_data = fetcher.fetch_multiple(
        series_ids={
            "GDPC1": "Real GDP",
            "PCEPILFE": "Core PCE Inflation",
            "FEDFUNDS": "Federal Funds Rate",
            "CPIAUCSL": "CPI",
            "CE16OV": "Civilian Employment",
            "PCECC96": "Real Consumption",
            "GPDI": "Investment"
        },
        start_date="1960-01-01",
        end_date="2019-12-31"
    )
    
    # Save raw data
    raw_data.to_csv("data/raw/fred_data.csv")
    print(f"Saved raw data to data/raw/fred_data.csv")
    
    print("\n2. Processing and Transforming Data")
    # Step 2: Process data
    processor = DataProcessor(raw_data)
    
    # Log levels for quantities
    processor.apply_transformation("GDPC1", "log")
    processor.apply_transformation("PCECC96", "log")
    processor.apply_transformation("GPDI", "log")
    processor.apply_transformation("CE16OV", "log")
    
    # Inflation rates (quarterly annualized)
    processor.apply_transformation("PCEPILFE", "pct_change", periods=1, scale=400)
    processor.apply_transformation("CPIAUCSL", "pct_change", periods=1, scale=400)
    
    # Interest rate (convert to quarterly)
    processor.apply_transformation("FEDFUNDS", "divide", divisor=4)
    
    print("\n3. Handling Missing Values and Outliers")
    # Handle missing values
    processor.impute_missing_values(method="linear")
    
    # Remove outliers
    processor.remove_outliers(threshold=3.0, method="winsorize")
    
    # Detrend with HP filter
    processor.apply_transformation("GDPC1", "hp_filter", lambda_=1600)
    processor.apply_transformation("PCECC96", "hp_filter", lambda_=1600)
    processor.apply_transformation("GPDI", "hp_filter", lambda_=1600)
    processor.apply_transformation("CE16OV", "hp_filter", lambda_=1600)
    
    # Get processed data
    processed_data = processor.get_processed_data()
    
    # Save processed data
    processed_data.to_csv("data/processed/processed_data.csv")
    print(f"Saved processed data to data/processed/processed_data.csv")
    
    print("\n4. Visualizing Processed Data")
    # Step 3: Visualize processed data
    plt.figure(figsize=(15, 10))
    
    # Plot GDP
    plt.subplot(3, 3, 1)
    plt.plot(processed_data["GDPC1"])
    plt.title("Detrended Log Real GDP")
    plt.grid(True)
    
    # Plot Inflation
    plt.subplot(3, 3, 2)
    plt.plot(processed_data["PCEPILFE"])
    plt.title("Core PCE Inflation Rate")
    plt.grid(True)
    
    # Plot Interest Rate
    plt.subplot(3, 3, 3)
    plt.plot(processed_data["FEDFUNDS"])
    plt.title("Federal Funds Rate (quarterly)")
    plt.grid(True)
    
    # Plot Consumption
    plt.subplot(3, 3, 4)
    plt.plot(processed_data["PCECC96"])
    plt.title("Detrended Log Consumption")
    plt.grid(True)
    
    # Plot Investment
    plt.subplot(3, 3, 5)
    plt.plot(processed_data["GPDI"])
    plt.title("Detrended Log Investment")
    plt.grid(True)
    
    # Plot Employment
    plt.subplot(3, 3, 6)
    plt.plot(processed_data["CE16OV"])
    plt.title("Detrended Log Employment")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("data/processed/data_visualization.png")
    print(f"Saved data visualization to data/processed/data_visualization.png")
    
    print("\n5. Preparing Data for Model Estimation")
    # Step 4: Prepare data for model estimation
    model_data = ModelDataset(processed_data)
    
    # Map processed variables to model variables
    model_data.map_variable("output", "GDPC1")
    model_data.map_variable("consumption", "PCECC96")
    model_data.map_variable("investment", "GPDI")
    model_data.map_variable("labor", "CE16OV")
    model_data.map_variable("inflation", "PCEPILFE")
    model_data.map_variable("nominal_interest", "FEDFUNDS")
    
    # Create balanced panel
    estimation_data = model_data.prepare_for_estimation(
        start_date="1966-01-01",
        end_date="2019-12-31"
    )
    
    # Save estimation-ready data
    estimation_data.to_csv("data/processed/estimation_data.csv")
    print(f"Saved estimation-ready data to data/processed/estimation_data.csv")
    
    print("\nData processing workflow completed successfully.")


if __name__ == "__main__":
    main()