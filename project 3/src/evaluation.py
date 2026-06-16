import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_evaluation_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate regression metrics for checking model forecast accuracy.
    """
    # Filter out NaNs if present
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    act = actual[mask]
    pred = predicted[mask]
    
    if len(act) == 0:
        return {
            'MAE': np.nan,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan,
            'Accuracy_Pct': np.nan,
            'Rating': 'N/A',
            'Interpretation': 'Insufficient data for evaluation.'
        }
        
    mae = mean_absolute_error(act, pred)
    mse = mean_squared_error(act, pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE manually to prevent division by zero
    # Handle zeros in actuals just in case
    act_safe = np.where(act == 0, 1e-10, act)
    mape = np.mean(np.abs((act - pred) / act_safe)) * 100
    
    r2 = r2_score(act, pred)
    
    # Quantitative rating based on Lewis (1982) scale for MAPE:
    # < 10% : Highly accurate forecasting
    # 10% - 20% : Good forecasting
    # 20% - 50% : Reasonable forecasting
    # > 50% : Inaccurate forecasting
    if mape < 10:
        rating = "Highly Accurate"
        interpretation = "The forecast error is less than 10%, indicating a highly reliable model for this stock."
    elif mape < 20:
        rating = "Good"
        interpretation = "The forecast error is between 10% and 20%, indicating a good model fit."
    elif mape < 50:
        rating = "Reasonable"
        interpretation = "The forecast error is between 20% and 50%, which is reasonable for volatile stock markets."
    else:
        rating = "Inaccurate"
        interpretation = "The forecast error exceeds 50%. This ARIMA model is not suitable for this stock series (possibly due to high volatility, regime changes, or lack of cointegration)."
        
    accuracy_pct = max(0.0, 100.0 - mape)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Accuracy_Pct': accuracy_pct,
        'Rating': rating,
        'Interpretation': interpretation
    }

def get_residuals_metrics(model_fit) -> dict:
    """
    Extract residuals statistics (mean, variance, skewness, kurtosis).
    """
    if model_fit is None:
        return {}
        
    residuals = model_fit.resid
    
    return {
        'mean': float(residuals.mean()),
        'variance': float(residuals.var()),
        'std': float(residuals.std()),
        'skewness': float(residuals.skew()) if hasattr(residuals, 'skew') else np.nan,
        'kurtosis': float(residuals.kurtosis()) if hasattr(residuals, 'kurtosis') else np.nan,
    }
