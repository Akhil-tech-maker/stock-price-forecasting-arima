import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(series: pd.Series) -> dict:
    """
    Perform the Augmented Dickey-Fuller (ADF) test on a pandas Series.
    """
    # Drop any NaNs
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'lags_used': 0,
            'n_obs': len(clean_series),
            'critical_values': {},
            'is_stationary': False,
            'conclusion': "Insufficient data to perform ADF test (minimum 10 observations required)."
        }
        
    try:
        result = adfuller(clean_series, autolag='AIC')
        
        test_stat = result[0]
        p_val = result[1]
        lags = result[2]
        n_obs = result[3]
        crit_vals = result[4]
        
        is_stationary = p_val < 0.05
        
        if is_stationary:
            conclusion = f"The p-value ({p_val:.4f}) is less than 0.05. We reject the null hypothesis. The series is stationary."
        else:
            conclusion = f"The p-value ({p_val:.4f}) is greater than or equal to 0.05. We fail to reject the null hypothesis. The series is non-stationary and requires differencing."
            
        return {
            'test_statistic': test_stat,
            'p_value': p_val,
            'lags_used': lags,
            'n_obs': n_obs,
            'critical_values': crit_vals,
            'is_stationary': is_stationary,
            'conclusion': conclusion
        }
    except Exception as e:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'lags_used': 0,
            'n_obs': len(clean_series),
            'critical_values': {},
            'is_stationary': False,
            'conclusion': f"ADF test failed: {str(e)}"
        }

def make_stationary(series: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int, list[dict]]:
    """
    Check stationarity and perform differencing automatically until the series
    becomes stationary or max_diff is reached.
    
    Returns:
    - Stationary/Differenced series
    - Order of differencing applied (d)
    - List of ADF results for each step (original, 1st diff, 2nd diff...)
    """
    adf_history = []
    current_series = series.copy()
    d = 0
    
    # Check original series
    results = perform_adf_test(current_series)
    adf_history.append({'d': d, 'results': results})
    
    if results['is_stationary']:
        return current_series, d, adf_history
        
    # Apply differencing if non-stationary
    while d < max_diff:
        d += 1
        current_series = current_series.diff().dropna()
        results = perform_adf_test(current_series)
        adf_history.append({'d': d, 'results': results})
        
        if results['is_stationary']:
            break
            
    return current_series, d, adf_history
