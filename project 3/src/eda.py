import pandas as pd
import numpy as np

def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get dimensions, null counts, and data types of the dataframe.
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        'index_start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
        'index_end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None,
        'total_days': len(df)
    }
    return info

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get standard descriptive statistics for the numeric columns.
    """
    return df.describe()

def calculate_returns_and_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate daily returns, percentage changes, and rolling volatility.
    """
    analysis_df = df.copy()
    
    # Calculate daily returns (fractional)
    analysis_df['Daily_Return'] = analysis_df['Close'].pct_change()
    
    # Calculate daily returns percentage
    analysis_df['Daily_Return_Pct'] = analysis_df['Daily_Return'] * 100
    
    # Volatility is the rolling standard deviation of daily returns
    # Standardized annual volatility = daily volatility * sqrt(252)
    analysis_df['Daily_Volatility'] = analysis_df['Daily_Return'].rolling(window=window).std()
    analysis_df['Annual_Volatility'] = analysis_df['Daily_Volatility'] * np.sqrt(252) * 100
    
    return analysis_df

def get_key_metrics(df: pd.DataFrame) -> dict:
    """
    Get key metrics like highest, lowest, average, and latest closing price.
    """
    if 'Close' not in df.columns:
        raise ValueError("Close column must be present in the DataFrame.")
        
    metrics = {
        'total_trading_days': len(df),
        'latest_close': float(df['Close'].iloc[-1]),
        'highest_price': float(df['Close'].max()),
        'highest_date': df['Close'].idxmax().strftime('%Y-%m-%d'),
        'lowest_price': float(df['Close'].min()),
        'lowest_date': df['Close'].idxmin().strftime('%Y-%m-%d'),
        'average_close': float(df['Close'].mean()),
        'median_close': float(df['Close'].median()),
        'std_close': float(df['Close'].std()),
    }
    
    # If High and Low columns are available, compute those metrics too
    if 'High' in df.columns:
        metrics['highest_high'] = float(df['High'].max())
        metrics['highest_high_date'] = df['High'].idxmax().strftime('%Y-%m-%d')
    if 'Low' in df.columns:
        metrics['lowest_low'] = float(df['Low'].min())
        metrics['lowest_low_date'] = df['Low'].idxmin().strftime('%Y-%m-%d')
        
    return metrics
