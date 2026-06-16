import pandas as pd
import yfinance as yf
from datetime import datetime
import io

def fetch_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock historical data from Yahoo Finance.
    """
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}' between {start_date} and {end_date}.")
        
        # Flatten MultiIndex columns if present (can occur in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Reset index to bring Date as column for standard processing
        df = df.reset_index()
        
        # Standardize column names
        df = standardize_columns(df)
        return df
    except Exception as e:
        raise RuntimeError(f"Error fetching data from Yahoo Finance: {str(e)}")

def load_csv_data(file_like) -> pd.DataFrame:
    """
    Load stock data from a CSV file (can be a path or a file-like object).
    """
    try:
        # Load CSV
        if isinstance(file_like, str):
            df = pd.read_csv(file_like)
        else:
            # File-like object (e.g. UploadedFile from Streamlit)
            df = pd.read_csv(file_like)
            
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
            
        df = standardize_columns(df)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {str(e)}")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataset:
    - Finds date column, converts to datetime, removes timezones, sorts chronologically, and sets as index.
    - Standardizes key column names (Open, High, Low, Close, Adj Close, Volume).
    """
    # 1. Identify and standardize Date column
    date_col = None
    for col in df.columns:
        if 'date' in str(col).lower():
            date_col = col
            break
            
    if date_col is None:
        # If no column contains 'date', check if first column contains date-like strings
        try:
            pd.to_datetime(df.iloc[:, 0], errors='raise')
            date_col = df.columns[0]
        except (ValueError, TypeError):
            raise ValueError("Could not automatically identify a 'Date' column in the dataset. Please ensure your dataset has a Date column.")

    # Convert to datetime and clean timezone
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
        
    # Rename date column to 'Date'
    df = df.rename(columns={date_col: 'Date'})
    
    # Sort chronologically
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 2. Standardize price columns (case-insensitive mapping)
    col_mapping = {}
    standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for std in standard_cols:
        for col in df.columns:
            if col != 'Date' and str(col).lower() == std.lower():
                col_mapping[col] = std
                break
                
    df = df.rename(columns=col_mapping)
    
    # Check for minimum required columns (Date is handled, check for Close)
    if 'Close' not in df.columns:
        # If no close column, look for the first numeric column to treat as Close
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df = df.rename(columns={numeric_cols[0]: 'Close'})
        else:
            raise ValueError("The dataset must contain at least a 'Close' price (or a numeric column representing prices).")
            
    # Set Date as index
    df = df.set_index('Date')
    
    # Sort index to guarantee chronological order
    df = df.sort_index()
    
    return df
