import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, fill_method: str = 'interpolate') -> pd.DataFrame:
    """
    Clean the dataset by removing duplicate index values, sorting,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate index entries (keep first)
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
    
    # Remove rows with null index (NaT)
    cleaned_df = cleaned_df[cleaned_df.index.notnull()]
    
    # Handle missing values in numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            if fill_method == 'interpolate':
                cleaned_df[col] = cleaned_df[col].interpolate(method='time', limit_direction='both')
            elif fill_method == 'ffill':
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
            elif fill_method == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                
    return cleaned_df

def detect_outliers_iqr(series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Returns a boolean mask where True indicates an outlier, and the lower/upper bounds.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

def handle_outliers(df: pd.DataFrame, column: str = 'Close', method: str = 'none') -> pd.DataFrame:
    """
    Detect and handle outliers in a specific column.
    Methods:
    - 'none': Do nothing.
    - 'clip': Clip values to the IQR bounds.
    - 'remove': Remove rows that are outliers.
    """
    processed_df = df.copy()
    if method == 'none' or column not in processed_df.columns:
        return processed_df
        
    outliers, lower_bound, upper_bound = detect_outliers_iqr(processed_df[column])
    
    if method == 'clip':
        processed_df[column] = processed_df[column].clip(lower=lower_bound, upper=upper_bound)
    elif method == 'remove':
        processed_df = processed_df[~outliers]
        
    return processed_df

def prepare_time_frequency(df: pd.DataFrame, freq: str = 'B', fill_method: str = 'interpolate') -> pd.DataFrame:
    """
    Ensure the DataFrame has a consistent time series frequency.
    Stock data typically uses 'B' (Business Days).
    If there are missing dates in the index, they are filled and numeric columns interpolated/filled.
    """
    processed_df = df.copy()
    
    # Set frequency
    # We use asfreq to inject missing dates, then fill them
    processed_df = processed_df.asfreq(freq)
    
    # Fill missing values introduced by resampling
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if processed_df[col].isnull().any():
            if fill_method == 'interpolate':
                processed_df[col] = processed_df[col].interpolate(method='time', limit_direction='both')
            elif fill_method == 'ffill':
                processed_df[col] = processed_df[col].ffill().bfill()
                
    return processed_df
