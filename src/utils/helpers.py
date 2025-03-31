import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


def format_currency(value):
    """
    Format a number as a currency string
    
    Parameters:
    -----------
    value : float
        The value to format
        
    Returns:
    --------
    str: Formatted currency string
    """
    return f"${value:,.2f}"


def format_number(value, decimal_places=2):
    """
    Format a number with thousands separator
    
    Parameters:
    -----------
    value : float
        The value to format
    decimal_places : int
        Number of decimal places to include
        
    Returns:
    --------
    str: Formatted number string
    """
    format_str = f"{{:,.{decimal_places}f}}"
    return format_str.format(value)


def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change between two values
    
    Parameters:
    -----------
    old_value : float
        Previous value
    new_value : float
        Current value
        
    Returns:
    --------
    float: Percentage change
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    
    return ((new_value - old_value) / abs(old_value)) * 100


def get_date_range(start_date, end_date, freq='D'):
    """
    Generate a range of dates
    
    Parameters:
    -----------
    start_date : datetime or str
        Start date
    end_date : datetime or str
        End date
    freq : str
        Frequency string (D for daily, W for weekly, etc.)
        
    Returns:
    --------
    pandas.DatetimeIndex: Range of dates
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def save_results_to_json(data, filename):
    """
    Save results to a JSON file
    
    Parameters:
    -----------
    data : dict or list
        Data to save
    filename : str
        Path to save the file
        
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Convert data to JSON-serializable format
        json_data = data
        
        # Handle special types
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient='records')
        
        # Handle datetime objects
        def serialize_datetime(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            return obj
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(json_data, f, default=serialize_datetime, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        return False


def load_results_from_json(filename):
    """
    Load results from a JSON file
    
    Parameters:
    -----------
    filename : str
        Path to the JSON file
        
    Returns:
    --------
    dict or list: Loaded data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data
    
    except Exception as e:
        print(f"Error loading results from JSON: {e}")
        return None


def interpolate_missing_values(df, column, method='linear'):
    """
    Interpolate missing values in a DataFrame column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name with missing values
    method : str
        Interpolation method (linear, time, etc.)
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with interpolated values
    """
    result_df = df.copy()
    
    if column in result_df.columns:
        result_df[column] = result_df[column].interpolate(method=method)
    
    return result_df


def calculate_moving_average(values, window_size=7):
    """
    Calculate moving average for a series of values
    
    Parameters:
    -----------
    values : list or pandas.Series
        Values to calculate moving average for
    window_size : int
        Size of the moving window
        
    Returns:
    --------
    numpy.ndarray: Moving average values
    """
    return pd.Series(values).rolling(window=window_size, min_periods=1).mean().values


def create_date_features(dates):
    """
    Create features from date values
    
    Parameters:
    -----------
    dates : list or pandas.Series
        Date values
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with date features
    """
    dates = pd.Series(pd.to_datetime(dates))
    
    features = pd.DataFrame({
        'year': dates.dt.year,
        'month': dates.dt.month,
        'day': dates.dt.day,
        'day_of_week': dates.dt.dayofweek,
        'day_of_year': dates.dt.dayofyear,
        'quarter': dates.dt.quarter,
        'is_weekend': dates.dt.dayofweek.isin([5, 6]).astype(int),
        'week_of_year': dates.dt.isocalendar().week,
        'month_sin': np.sin(2 * np.pi * dates.dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dates.dt.month / 12),
        'day_of_week_sin': np.sin(2 * np.pi * dates.dt.dayofweek / 7),
        'day_of_week_cos': np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    })
    
    return features 