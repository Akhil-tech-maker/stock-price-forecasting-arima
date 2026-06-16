import re

def clean_ticker(ticker: str) -> str:
    """
    Standardize ticker inputs (uppercase, trim spaces).
    """
    if not ticker:
        return ""
    # Strip whitespace and make uppercase
    ticker = ticker.strip().upper()
    # Remove any special characters except dot and hyphen (for tickers like BRK.B or RDS-A)
    ticker = re.sub(r'[^A-Z0-9.\-]', '', ticker)
    return ticker

def format_currency(val: float, currency_symbol: str = "₹") -> str:
    """
    Format float as currency string.
    """
    if val is None or (isinstance(val, float) and (val != val)): # Check NaN
        return "N/A"
    try:
        return f"{currency_symbol}{val:,.2f}"
    except (ValueError, TypeError):
        return str(val)

def get_percentage_change_desc(pct_change: float) -> str:
    """
    Helper to return a text description of a percentage change.
    """
    if pct_change > 0:
        return f"increased by {pct_change:.2f}%"
    elif pct_change < 0:
        return f"decreased by {abs(pct_change):.2f}%"
    else:
        return "remained unchanged (0.00%)"
