#!/usr/bin/env python
"""
External Integration Example

This script demonstrates integration with external tools, data sources, and services:
1. Working with alternative data sources (BEA, World Bank, etc.)
2. Exporting results to external formats (Excel, LaTeX, etc.)
3. API integrations for data exchange
4. Integration with other statistical packages

The example shows how to extend the DSGE model's capabilities through external integrations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import StringIO, BytesIO
import csv
import xml.etree.ElementTree as ET
import zipfile
import tempfile
import re

# Optional dependencies - check if available
try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False
    print("xlsxwriter not available. Excel export examples will be skipped.")

try:
    import pylatex
    from pylatex import Document, Section, Subsection, Table, Math, TikZ, Axis, Plot
    from pylatex.utils import italic, bold
    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False
    print("pylatex not available. LaTeX export examples will be skipped.")

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not available. Statistical integration examples will be skipped.")

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    HAS_RPY2 = True
    pandas2ri.activate()
except ImportError:
    HAS_RPY2 = False
    print("rpy2 not available. R integration examples will be skipped.")


# Mock DSGE model classes for demonstration
class ConfigManager:
    """Mock configuration manager for DSGE model."""
    
    def __init__(self):
        """Initialize with default configuration."""
        self.config = {
            "base_model": {
                "beta": 0.99,
                "alpha": 0.33,
                "delta": 0.025,
                "sigma_c": 1.5,
                "h": 0.7,
                "sigma_l": 2.0,
                "xi_p": 0.75,
                "xi_w": 0.75,
                "iota_p": 0.5,
                "iota_w": 0.5,
                "rho_r": 0.8,
                "phi_pi": 1.5,
                "phi_y": 0.125,
                "phi_dy": 0.125,
                "pi_bar": 1.005,
                "r_bar": 1.0101
            }
        }
    
    def get(self, key=None, default=None):
        """Get configuration value."""
        if key is None:
            return self.config
        
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)


class SmetsWoutersModel:
    """Mock Smets-Wouters model for demonstration."""
    
    def __init__(self, config=None):
        """Initialize the model."""
        self.config = config or ConfigManager()
        self.params = {
            "beta": self.config.get("base_model.beta", 0.99),
            "alpha": self.config.get("base_model.alpha", 0.33),
            "delta": self.config.get("base_model.delta", 0.025),
            "sigma_c": self.config.get("base_model.sigma_c", 1.5),
            "h": self.config.get("base_model.h", 0.7),
            "sigma_l": self.config.get("base_model.sigma_l", 2.0),
            "xi_p": self.config.get("base_model.xi_p", 0.75),
            "xi_w": self.config.get("base_model.xi_w", 0.75),
            "iota_p": self.config.get("base_model.iota_p", 0.5),
            "iota_w": self.config.get("base_model.iota_w", 0.5),
            "rho_r": self.config.get("base_model.rho_r", 0.8),
            "phi_pi": self.config.get("base_model.phi_pi", 1.5),
            "phi_y": self.config.get("base_model.phi_y", 0.125),
            "phi_dy": self.config.get("base_model.phi_dy", 0.125),
            "pi_bar": self.config.get("base_model.pi_bar", 1.005),
            "r_bar": self.config.get("base_model.r_bar", 1.0101)
        }
        
        # Mock steady state values
        self.steady_state = {
            "output": 1.0,
            "consumption": 0.6,
            "investment": 0.2,
            "capital": 8.0,
            "labor": 1.0,
            "real_wage": 1.0,
            "inflation": self.params["pi_bar"],
            "nominal_interest": self.params["r_bar"]
        }


class ImpulseResponseFunctions:
    """Mock impulse response functions for demonstration."""
    
    def __init__(self, model):
        """Initialize with model."""
        self.model = model
    
    def compute_irfs(self, shock_names, periods=40, shock_size=1.0):
        """
        Compute impulse response functions.
        
        Args:
            shock_names (list): List of shock names
            periods (int): Number of periods
            shock_size (float): Size of the shock
            
        Returns:
            dict: Dictionary of IRFs by shock
        """
        # Generate synthetic IRFs for demonstration
        irfs = {}
        
        for shock in shock_names:
            shock_irfs = {}
            
            # Different decay rates for different variables and shocks
            if shock == "technology":
                decay_rates = {
                    "output": 20,
                    "consumption": 25,
                    "investment": 15,
                    "inflation": 10,
                    "nominal_interest": 8
                }
                signs = {
                    "output": 1,
                    "consumption": 1,
                    "investment": 1,
                    "inflation": -1,
                    "nominal_interest": -0.5
                }
            elif shock == "monetary":
                decay_rates = {
                    "output": 10,
                    "consumption": 12,
                    "investment": 8,
                    "inflation": 15,
                    "nominal_interest": 5
                }
                signs = {
                    "output": -1,
                    "consumption": -1,
                    "investment": -1,
                    "inflation": -1,
                    "nominal_interest": 1
                }
            else:  # Generic shock
                decay_rates = {
                    "output": 15,
                    "consumption": 18,
                    "investment": 12,
                    "inflation": 10,
                    "nominal_interest": 8
                }
                signs = {
                    "output": 1,
                    "consumption": 1,
                    "investment": 1,
                    "inflation": 1,
                    "nominal_interest": 1
                }
            
            # Generate IRFs
            for var, decay in decay_rates.items():
                t = np.arange(periods)
                irf = signs[var] * shock_size * np.exp(-t / decay)
                
                # Add some noise
                irf += 0.05 * np.random.randn(periods)
                
                shock_irfs[var] = irf
            
            irfs[shock] = shock_irfs
        
        return irfs


# 1. Alternative Data Sources
class AlternativeDataSources:
    """Class for working with alternative data sources."""
    
    @staticmethod
    def fetch_bea_data(api_key, table_name, frequency="A", year="X"):
        """
        Fetch data from the Bureau of Economic Analysis (BEA) API.
        
        Args:
            api_key (str): BEA API key
            table_name (str): NIPA table name (e.g., "T10101")
            frequency (str): Frequency (A=Annual, Q=Quarterly, M=Monthly)
            year (str): Year or "X" for all years
            
        Returns:
            pd.DataFrame: BEA data
        """
        print(f"Fetching BEA data for table {table_name}, frequency {frequency}, year {year}")
        
        # In a real implementation, this would use the BEA API
        # For demonstration, we'll generate synthetic data
        
        # Simulate API call
        print("Simulating BEA API call...")
        
        # Generate synthetic data
        if frequency == "A":
            years = range(2000, 2023)
            index = pd.Index(years, name="Year")
        elif frequency == "Q":
            quarters = [f"{year}Q{q}" for year in range(2000, 2023) for q in range(1, 5)]
            index = pd.Index(quarters, name="Quarter")
        else:  # Monthly
            months = [f"{year}-{month:02d}" for year in range(2000, 2023) for month in range(1, 13)]
            index = pd.Index(months, name="Month")
        
        # Generate columns based on table name
        if table_name == "T10101":  # GDP
            columns = ["GDP", "Consumption", "Investment", "Government", "Exports", "Imports"]
        elif table_name == "T20100":  # Personal Income
            columns = ["Personal_Income", "Wages", "Proprietors_Income", "Rental_Income", "Dividends", "Interest", "Transfers"]
        else:
            columns = [f"Series_{i}" for i in range(1, 6)]
        
        # Generate data
        np.random.seed(42)
        data = np.random.randn(len(index), len(columns))
        
        # Make it look like economic data with trends
        for i, col in enumerate(columns):
            # Base trend
            trend = np.linspace(100, 200, len(index))
            
            # Add cyclical component
            cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, len(index)))
            
            # Add noise
            noise = 5 * np.random.randn(len(index))
            
            # Combine
            data[:, i] = trend + cycle + noise
        
        # Create DataFrame
        df = pd.DataFrame(data, index=index, columns=columns)
        
        # Add metadata
        df.attrs["source"] = "BEA API (simulated)"
        df.attrs["table"] = table_name
        df.attrs["frequency"] = frequency
        df.attrs["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df
    
    @staticmethod
    def fetch_world_bank_data(indicator, countries, start_year=2000, end_year=2022):
        """
        Fetch data from the World Bank API.
        
        Args:
            indicator (str): World Bank indicator code (e.g., "NY.GDP.MKTP.CD")
            countries (list): List of country codes
            start_year (int): Start year
            end_year (int): End year
            
        Returns:
            pd.DataFrame: World Bank data
        """
        print(f"Fetching World Bank data for indicator {indicator}, countries {countries}, years {start_year}-{end_year}")
        
        # In a real implementation, this would use the World Bank API
        # For demonstration, we'll generate synthetic data
        
        # Simulate API call
        print("Simulating World Bank API call...")
        
        # Generate years
        years = list(range(start_year, end_year + 1))
        
        # Generate data
        np.random.seed(42)
        data = {}
        
        for country in countries:
            # Base trend
            trend = np.linspace(100, 200, len(years))
            
            # Add country-specific factor
            country_factor = hash(country) % 100 / 100.0
            trend = trend * (0.8 + 0.4 * country_factor)
            
            # Add cyclical component
            cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, len(years)))
            
            # Add noise
            noise = 5 * np.random.randn(len(years))
            
            # Combine
            data[country] = trend + cycle + noise
        
        # Create DataFrame
        df = pd.DataFrame(data, index=years)
        
        # Add metadata
        df.attrs["source"] = "World Bank API (simulated)"
        df.attrs["indicator"] = indicator
        df.attrs["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df
    
    @staticmethod
    def fetch_eurostat_data(dataset_code, dimensions=None):
        """
        Fetch data from the Eurostat API.
        
        Args:
            dataset_code (str): Eurostat dataset code
            dimensions (dict, optional): Dictionary of dimension filters
            
        Returns:
            pd.DataFrame: Eurostat data
        """
        print(f"Fetching Eurostat data for dataset {dataset_code}")
        if dimensions:
            print(f"With dimensions: {dimensions}")
        
        # In a real implementation, this would use the Eurostat API
        # For demonstration, we'll generate synthetic data
        
        # Simulate API call
        print("Simulating Eurostat API call...")
        
        # Generate time periods
        quarters = [f"{year}Q{q}" for year in range(2000, 2023) for q in range(1, 5)]
        
        # Generate countries
        countries = ["AT", "BE", "DE", "ES", "FR", "IT", "NL"]
        
        # Filter countries if specified in dimensions
        if dimensions and "geo" in dimensions:
            countries = [c for c in countries if c in dimensions["geo"]]
        
        # Generate data
        np.random.seed(42)
        data = {}
        
        for country in countries:
            # Base trend
            trend = np.linspace(100, 200, len(quarters))
            
            # Add country-specific factor
            country_factor = hash(country) % 100 / 100.0
            trend = trend * (0.8 + 0.4 * country_factor)
            
            # Add cyclical component
            cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, len(quarters)))
            
            # Add noise
            noise = 5 * np.random.randn(len(quarters))
            
            # Combine
            data[country] = trend + cycle + noise
        
        # Create DataFrame
        df = pd.DataFrame(data, index=quarters)
        
        # Add metadata
        df.attrs["source"] = "Eurostat API (simulated)"
        df.attrs["dataset"] = dataset_code
        df.attrs["dimensions"] = dimensions
        df.attrs["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df
    
    @staticmethod
    def fetch_imf_data(database_id, series_codes, countries, start_year=2000, end_year=2022):
        """
        Fetch data from the IMF API.
        
        Args:
            database_id (str): IMF database ID (e.g., "IFS")
            series_codes (list): List of series codes
            countries (list): List of country codes
            start_year (int): Start year
            end_year (int): End year
            
        Returns:
            pd.DataFrame: IMF data
        """
        print(f"Fetching IMF data for database {database_id}, series {series_codes}, countries {countries}")
        
        # In a real implementation, this would use the IMF API
        # For demonstration, we'll generate synthetic data
        
        # Simulate API call
        print("Simulating IMF API call...")
        
        # Generate years
        years = list(range(start_year, end_year + 1))
        
        # Generate data
        np.random.seed(42)
        data = {}
        
        for country in countries:
            for series in series_codes:
                # Base trend
                trend = np.linspace(100, 200, len(years))
                
                # Add country-specific factor
                country_factor = hash(country) % 100 / 100.0
                trend = trend * (0.8 + 0.4 * country_factor)
                
                # Add series-specific factor
                series_factor = hash(series) % 100 / 100.0
                trend = trend * (0.8 + 0.4 * series_factor)
                
                # Add cyclical component
                cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, len(years)))
                
                # Add noise
                noise = 5 * np.random.randn(len(years))
                
                # Combine
                data[f"{country}_{series}"] = trend + cycle + noise
        
        # Create DataFrame
        df = pd.DataFrame(data, index=years)
        
        # Add metadata
        df.attrs["source"] = "IMF API (simulated)"
        df.attrs["database"] = database_id
        df.attrs["series"] = series_codes
        df.attrs["countries"] = countries
        df.attrs["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df


# 2. Export Formatters
class ExportFormatters:
    """Class for exporting DSGE model results to various formats."""
    
    @staticmethod
    def export_to_csv(data, filepath):
        """
        Export data to CSV.
        
        Args:
            data (pd.DataFrame): Data to export
            filepath (str): Output file path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Export to CSV
        data.to_csv(filepath)
        
        print(f"Exported data to CSV: {filepath}")
    
    @staticmethod
    def export_to_excel(data, filepath, sheet_name="Data"):
        """
        Export data to Excel.
        
        Args:
            data (pd.DataFrame): Data to export
            filepath (str): Output file path
            sheet_name (str): Excel sheet name
        """
        if not HAS_XLSXWRITER:
            print("xlsxwriter not available. Excel export skipped.")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Write data
            data.to_excel(writer, sheet_name=sheet_name)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Apply formats
            for col_num, value in enumerate(data.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            
            # Add chart
            chart = workbook.add_chart({'type': 'line'})
            
            # Configure chart
            for i, col in enumerate(data.columns):
                chart.add_series({
                    'name': [sheet_name, 0, i + 1],
                    'categories': [sheet_name, 1, 0, len(data) + 1, 0],
                    'values': [sheet_name, 1, i + 1, len(data) + 1, i + 1],
                })
            
            # Set chart title and labels
            chart.set_title({'name': 'Data Visualization'})
            chart.set_x_axis({'name': 'Time'})
            chart.set_y_axis({'name': 'Value'})
            
            # Insert chart
            worksheet.insert_chart('K2', chart)
        
        print(f"Exported data to Excel: {filepath}")
    
    @staticmethod
    def export_to_latex(data, filepath, caption="DSGE Model Results", label="tab:dsge_results"):
        """
        Export data to LaTeX.
        
        Args:
            data (pd.DataFrame): Data to export
            filepath (str): Output file path
            caption (str): Table caption
            label (str): Table label
        """
        if not HAS_PYLATEX:
            print("pylatex not available. LaTeX export skipped.")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create LaTeX document
        doc = Document()
        
        # Add table
        with doc.create(Section('DSGE Model Results')):
            doc.append('The following table shows the results of the DSGE model simulation:')
            
            # Create table
            with doc.create(Table(position='htbp')) as table:
                table.add_caption(caption)
                table.append(pylatex.Command('label', label))
                
                # Add header row
                header_row = ['Time'] + list(data.columns)
                table.add_hline()
                table.add_row(header_row)
                table.add_hline()
                
                # Add data rows
                for i, (idx, row) in enumerate(data.iterrows()):
                    if i < 10:  # Limit to first 10 rows for brevity
                        table_row = [str(idx)] + [f"{val:.2f}" for val in row]
                        table.add_row(table_row)
                
                table.add_hline()
        
        # Generate PDF
        doc.generate_pdf(filepath.replace('.tex', ''), clean_tex=False)
        
        print(f"Exported data to LaTeX: {filepath}")
        print(f"Generated PDF: {filepath.replace('.tex', '.pdf')}")
    
    @staticmethod
    def export_to_json(data, filepath):
        """
        Export data to JSON.
        
        Args:
            data (pd.DataFrame): Data to export
            filepath (str): Output file path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert DataFrame to JSON
        json_data = {
            "data": data.to_dict(orient="records"),
            "index": [str(idx) for idx in data.index],
            "columns": list(data.columns),
            "metadata": {
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": len(data),
                "columns": len(data.columns)
            }
        }
        
        # Export to JSON
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Exported data to JSON: {filepath}")
    
    @staticmethod
    def export_to_xml(data, filepath, root_element="dsge_results"):
        """
        Export data to XML.
        
        Args:
            data (pd.DataFrame): Data to export
            filepath (str): Output file path
            root_element (str): Root element name
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create XML structure
        root = ET.Element(root_element)
        
        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "exported_at").text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ET.SubElement(metadata, "rows").text = str(len(data))
        ET.SubElement(metadata, "columns").text = str(len(data.columns))
        
        # Add columns
        columns = ET.SubElement(root, "columns")
        for col in data.columns:
            ET.SubElement(columns, "column").text = col
        
        # Add data
        data_element = ET.SubElement(root, "data")
        for i, (idx, row) in enumerate(data.iterrows()):
            row_element = ET.SubElement(data_element, "row")
            ET.SubElement(row_element, "index").text = str(idx)
            
            for col, val in row.items():
                col_element = ET.SubElement(row_element, col.replace(" ", "_"))
                col_element.text = str(val)
        
        # Create XML tree
        tree = ET.ElementTree(root)
        
        # Export to XML
        tree.write(filepath, encoding="utf-8", xml_declaration=True)
        
        print(f"Exported data to XML: {filepath}")


# 3. API Integrations
class APIIntegrations:
    """Class for integrating with external APIs."""
    
    @staticmethod
    def post_to_api(data, api_url, api_key=None):
        """
        Post data to an external API.
        
        Args:
            data (dict): Data to post
            api_url (str): API URL
            api_key (str, optional): API key
            
        Returns:
            dict: API response
        """
        print(f"Posting data to API: {api_url}")
        
        # In a real implementation, this would use requests to post data
        # For demonstration, we'll simulate the API call
        
        # Simulate API call
        print("Simulating API POST request...")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Simulate response
        response = {
            "status": "success",
            "message": "Data received successfully",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(str(data))
        }
        
        print(f"Received API response: {response}")
        
        return response
    
    @staticmethod
    def fetch_from_api(api_url, params=None, api_key=None):
        """
        Fetch data from an external API.
        
        Args:
            api_url (str): API URL
            params (dict, optional): Query parameters
            api_key (str, optional): API key
            
        Returns:
            dict: API response
        """
        print(f"Fetching data from API: {api_url}")
        if params:
            print(f"With parameters: {params}")
        
        # In a real implementation, this would use requests to fetch data
        # For demonstration, we'll simulate the API call
        
        # Simulate API call
        print("Simulating API GET request...")
        
        # Prepare headers
        headers = {}
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Simulate response
        np.random.seed(42)
        
        # Generate synthetic data
        data = {
            "time_series": {
                "dates": [f"2022-{month:02d}-01" for month in range(1, 13)],
                "values": list(100 + 10 * np.random.randn(12))
            },
            "metadata": {
                "source": "External API (simulated)",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params": params
            }
        }
        
        response = {
            "status": "success",
            "data": data
        }
        
        print(f"Received API response with {len(data['time_series']['dates'])} data points")
        
        return response
    
    @staticmethod
    def webhook_integration(webhook_url, event_data):
        """
        Send data to a webhook.
        
        Args:
            webhook_url (str): Webhook URL
            event_data (dict): Event data to send
            
        Returns:
            dict: Webhook response
        """
        print(f"Sending data to webhook: {webhook_url}")
        
        # In a real implementation, this would use requests to send data
        # For demonstration, we'll simulate the webhook call
        
        # Simulate webhook call
        print("Simulating webhook POST request...")
        
        # Prepare payload
        payload = {
            "event": event_data["event"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": event_data["data"]
        }
        
        # Simulate response
        response = {
            "status": "success",
            "message": "Webhook received",
            "id": f"wh_{hash(str(payload)) % 1000000:06d}"
        }
        
        print(f"Received webhook response: {response}")
        
        return response


# 4. Statistical Package Integrations
class StatisticalIntegrations:
    """Class for integrating with statistical packages."""
    
    @staticmethod
    def statsmodels_integration(data):
        """
        Integrate with statsmodels for statistical analysis.
        
        Args:
            data (pd.DataFrame): Data to analyze
            
        Returns:
            dict: Analysis results
        """
        if not HAS_STATSMODELS:
            print("statsmodels not available. Integration skipped.")
            return None
        
        print("Performing statistical analysis with statsmodels...")
        
        # Prepare data
        # For demonstration, we'll use a simple time series model
        y = data.iloc[:, 0]  # Use first column as dependent variable
        
        # Add constant
        X = sm.add_constant(np.arange(len(y)))
        
        # Fit OLS model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract results
        analysis_results = {
            "model": "OLS",
            "dependent_variable": data.columns[0],
            "coefficients": results.params.to_dict(),
            "std_errors": results.bse.to_dict(),
            "t_values": results.tvalues.to_dict(),
            "p_values": results.pvalues.to_dict(),
            "r_squared": results.rsquared,
            "adj_r_squared": results.rsquared_adj,
            "aic": results.aic,
            "bic": results.bic,
            "summary": str(results.summary())
        }
        
        print(f"Completed statsmodels analysis with R² = {analysis_results['r_squared']:.4f}")
        
        return analysis_results
    
    @staticmethod
    def r_integration(data, r_script=None):
        """
        Integrate with R for statistical analysis.
        
        Args:
            data (pd.DataFrame): Data to analyze
            r_script (str, optional): R script to execute
            
        Returns:
            dict: Analysis results
        """
        if not HAS_RPY2:
            print("rpy2 not available. R integration skipped.")
            return None
        
        print("Performing statistical analysis with R...")
        
        # Convert data to R dataframe
        r_df = pandas2ri.py2rpy(data)
        
        # Store in R environment
        ro.globalenv['data'] = r_df
        
        # If R script is provided, execute it
        if r_script:
            ro.r(r_script)
            print("Executed custom R script")
        
        # Otherwise, perform a simple analysis
        else:
            # Import R packages
            stats = importr('stats')
            base = importr('base')
            
            # Fit linear model
            formula = ro.Formula('V1 ~ V2')
            model = stats.lm(formula, data=r_df)
            summary = base.summary(model)
            
            # Extract results
            coefficients = dict(zip(summary.rx2('coefficients').names, summary.rx2('coefficients')))
            r_squared = summary.rx2('r.squared')[0]
            
            print(f"Completed R analysis with R² = {r_squared:.4f}")
            
            # Return results
            return {
                "model": "Linear Model",
                "formula": str(formula),
                "coefficients": coefficients,
                "r_squared": r_squared,
                "summary": str(summary)
            }


def main():
    """Main function demonstrating external integrations."""
    # Create output directories
    os.makedirs("results/external", exist_ok=True)
    
    print("=== DSGE Model External Integration Example ===")
    
    # Create model and generate IRFs
    print("\n1. Creating DSGE Model and Generating IRFs")
    config = ConfigManager()
    model = SmetsWoutersModel(config)
    irf = ImpulseResponseFunctions(model)
    
    # Compute IRFs
    irfs = irf.compute_irfs(
        shock_names=["technology", "monetary"],
        periods=40,
        shock_size=1.0
    )
    
    # Convert IRFs to DataFrames for easier handling
    irf_dfs = {}
    for shock in irfs:
        shock_df = pd.DataFrame(irfs[shock])
        irf_dfs[shock] = shock_df
    
    # 1. Alternative Data Sources
    print("\n2. Working with Alternative Data Sources")
    
    # BEA data
    print("\n2.1 Bureau of Economic Analysis (BEA) Data")
    bea_data = AlternativeDataSources.fetch_bea_data(
        api_key="DEMO_KEY",
        table_name="T10101",
        frequency="Q",
        year="X"
    )
    print(f"Retrieved BEA data with shape {bea_data.shape}")
    
    # World Bank data
    print("\n2.2 World Bank Data")
    wb_data = AlternativeDataSources.fetch_world_bank_data(
        indicator="NY.GDP.MKTP.CD",
        countries=["USA", "DEU", "JPN", "GBR", "FRA"],
        start_year=2000,
        end_year=2022
    )
    print(f"Retrieved World Bank data with shape {wb_data.shape}")
    
    # Eurostat data
    print("\n2.3 Eurostat Data")
    eurostat_data = AlternativeDataSources.fetch_eurostat_data(
        dataset_code="namq_10_gdp",
        dimensions={"geo": ["DE", "FR", "IT", "ES"]}
    )
    print(f"Retrieved Eurostat data with shape {eurostat_data.shape}")
    
    # IMF data
    print("\n2.4 IMF Data")
    imf_data = AlternativeDataSources.fetch_imf_data(
        database_id="IFS",
        series_codes=["NGDP_R", "PCPI"],
        countries=["US", "DE", "JP", "GB", "FR"],
        start_year=2000,
        end_year=2022
    )
    print(f"Retrieved IMF data with shape {imf_data.shape}")
    
    # 2. Export Formatters
    print("\n3. Exporting Results to Various Formats")
    
    # Export IRFs to CSV
    print("\n3.1 Exporting to CSV")
    ExportFormatters.export_to_csv(
        data=irf_dfs["technology"],
        filepath="results/external/technology_irf.csv"
    )
    
    # Export IRFs to Excel
    print("\n3.2 Exporting to Excel")
    ExportFormatters.export_to_excel(
        data=irf_dfs["monetary"],
        filepath="results/external/monetary_irf.xlsx",
        sheet_name="Monetary Shock IRF"
    )
    
    # Export IRFs to LaTeX
    print("\n3.3 Exporting to LaTeX")
    ExportFormatters.export_to_latex(
        data=irf_dfs["technology"].iloc[:10],  # First 10 periods for brevity
        filepath="results/external/technology_irf.tex",
        caption="Impulse Response Functions for Technology Shock",
        label="tab:technology_irf"
    )
    
    # Export IRFs to JSON
    print("\n3.4 Exporting to JSON")
    ExportFormatters.export_to_json(
        data=irf_dfs["monetary"],
        filepath="results/external/monetary_irf.json"
    )
    
    # Export IRFs to XML
    print("\n3.5 Exporting to XML")
    ExportFormatters.export_to_xml(
        data=irf_dfs["technology"],
        filepath="results/external/technology_irf.xml",
        root_element="impulse_response_functions"
    )
    
    # 3. API Integrations
    print("\n4. API Integrations")
    
    # Post IRFs to API
    print("\n4.1 Posting Data to API")
    api_response = APIIntegrations.post_to_api(
        data={
            "model": "Smets-Wouters",
            "shock": "technology",
            "irf": irf_dfs["technology"].to_dict()
        },
        api_url="https://api.example.com/dsge/results",
        api_key="DEMO_API_KEY"
    )
    
    # Fetch data from API
    print("\n4.2 Fetching Data from API")
    api_data = APIIntegrations.fetch_from_api(
        api_url="https://api.example.com/macroeconomic/indicators",
        params={
            "country": "USA",
            "indicators": ["GDP", "CPI", "UNEMP"],
            "start_date": "2000-01-01",
            "end_date": "2022-12-31"
        },
        api_key="DEMO_API_KEY"
    )
    
    # Webhook integration
    print("\n4.3 Webhook Integration")
    webhook_response = APIIntegrations.webhook_integration(
        webhook_url="https://webhook.example.com/dsge/events",
        event_data={
            "event": "model_estimation_completed",
            "data": {
                "model": "Smets-Wouters",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": model.params
            }
        }
    )
    
    # 4. Statistical Package Integrations
    print("\n5. Statistical Package Integrations")
    
    # Statsmodels integration
    print("\n5.1 Statsmodels Integration")
    statsmodels_results = StatisticalIntegrations.statsmodels_integration(
        data=irf_dfs["technology"]
    )
    
    if statsmodels_results:
        # Save results
        with open("results/external/statsmodels_analysis.json", "w") as f:
            # Convert numpy types to Python types for JSON serialization
            results_dict = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) 
                           for k, v in statsmodels_results.items() 
                           if k != "summary"}
            json.dump(results_dict, f, indent=2)
        
        # Save summary separately
        with open("results/external/statsmodels_summary.txt", "w") as f:
            f.write(statsmodels_results["summary"])
    
    # R integration
    print("\n5.2 R Integration")
    r_results = StatisticalIntegrations.r_integration(
        data=irf_dfs["monetary"]
    )
    
    if r_results:
        # Save results
        with open("results/external/r_analysis.json", "w") as f:
            # Convert R objects to Python types for JSON serialization
            results_dict = {k: (v if not isinstance(v, ro.vectors.FloatVector) else list(v)) 
                           for k, v in r_results.items() 
                           if k != "summary"}
            json.dump(results_dict, f, indent=2)
        
        # Save summary separately
        with open("results/external/r_summary.txt", "w") as f:
            f.write(str(r_results["summary"]))
    
    print("\nExternal integration example completed successfully.")
    print(f"Results saved to results/external/")


if __name__ == "__main__":
    main()