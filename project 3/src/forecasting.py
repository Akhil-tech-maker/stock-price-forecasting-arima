import pandas as pd
import numpy as np
from datetime import datetime

def generate_test_forecast(model_fit, test_series: pd.Series) -> pd.DataFrame:
    """
    Generate forecasts for the test period and align them with the actual test data.
    """
    if model_fit is None:
        raise ValueError("Model has not been trained successfully.")
        
    steps = len(test_series)
    
    # Get forecast object
    forecast_obj = model_fit.get_forecast(steps=steps)
    
    # Extract mean prediction and confidence intervals
    forecast_mean = forecast_obj.predicted_mean
    confidence_intervals = forecast_obj.conf_int(alpha=0.05) # 95% CI
    
    # Align index with the test series
    forecast_df = pd.DataFrame(index=test_series.index)
    forecast_df['Actual'] = test_series.values
    forecast_df['Predicted'] = forecast_mean.values
    
    # Extract CI columns (columns in statsmodels are usually 'lower Close', 'upper Close' or similar)
    forecast_df['Lower_CI'] = confidence_intervals.iloc[:, 0].values
    forecast_df['Upper_CI'] = confidence_intervals.iloc[:, 1].values
    
    # Calculate errors
    forecast_df['Error'] = forecast_df['Actual'] - forecast_df['Predicted']
    forecast_df['Abs_Percent_Error'] = (forecast_df['Error'].abs() / forecast_df['Actual']) * 100
    
    return forecast_df

def generate_future_forecast(model_fit, steps: int = 30, last_date=None, freq: str = 'B') -> pd.DataFrame:
    """
    Generate out-of-sample future forecasts for the specified number of steps.
    """
    if model_fit is None:
        raise ValueError("Model has not been trained successfully.")
        
    # Get forecast object
    forecast_obj = model_fit.get_forecast(steps=steps)
    
    # Extract mean prediction and confidence intervals
    forecast_mean = forecast_obj.predicted_mean
    confidence_intervals = forecast_obj.conf_int(alpha=0.05) # 95% CI
    
    # If statsmodels index is not a DatetimeIndex or is integers, let's create a proper date index
    if not isinstance(forecast_mean.index, pd.DatetimeIndex) and last_date is not None:
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
    else:
        future_dates = forecast_mean.index
        
    future_df = pd.DataFrame(index=future_dates)
    future_df['Forecast'] = forecast_mean.values
    future_df['Lower_CI'] = confidence_intervals.iloc[:, 0].values
    future_df['Upper_CI'] = confidence_intervals.iloc[:, 1].values
    
    # Ensure index has the name Date
    future_df.index.name = 'Date'
    
    return future_df
