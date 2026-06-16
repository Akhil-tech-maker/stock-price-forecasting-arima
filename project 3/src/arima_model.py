import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import warnings

# Suppress statsmodels warnings for cleaner output
warnings.filterwarnings("ignore")

def train_test_split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically into training and testing sets.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

def fit_manual_arima(series: pd.Series, p: int, d: int, q: int):
    """
    Fit a manual ARIMA(p, d, q) model using statsmodels.
    Returns the fitted model and any warning/error message.
    """
    try:
        # Define model
        model = ARIMA(series, order=(p, d, q))
        # Fit model
        model_fit = model.fit()
        return model_fit, None
    except Exception as e:
        return None, str(e)

def fit_auto_arima(series: pd.Series, max_p: int = 10, max_d: int = 3, max_q: int = 10, information_criterion: str = 'aic') -> dict:
    """
    Automatically select the best ARIMA(p, d, q) parameters using Auto-ARIMA.
    """
    try:
        # Fit Auto ARIMA
        # We set seasonal=False to perform standard ARIMA as requested, but user can configure.
        auto_model = pm.auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=max_p, max_d=max_d, max_q=max_q,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion=information_criterion
        )
        
        best_order = auto_model.order
        aic = auto_model.aic()
        bic = auto_model.bic()
        
        # Refit using statsmodels on the full training series to get standard statsmodels fit object
        refit_model, error_msg = fit_manual_arima(series, best_order[0], best_order[1], best_order[2])
        
        return {
            'best_order': best_order,
            'p': best_order[0],
            'd': best_order[1],
            'q': best_order[2],
            'aic': aic,
            'bic': bic,
            'model_fit': refit_model,
            'pm_model': auto_model,
            'error': error_msg
        }
    except Exception as e:
        return {
            'best_order': (0, 0, 0),
            'p': 0, 'd': 0, 'q': 0,
            'aic': np.nan,
            'bic': np.nan,
            'model_fit': None,
            'pm_model': None,
            'error': str(e)
        }
