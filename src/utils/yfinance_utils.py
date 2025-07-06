import yfinance as yf
import pandas as pd
import logging
from typing import Optional, Any
from datetime import datetime, timedelta
from typing import Union

def fetch_daily_yfinance_data(
    ticker: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    auto_adjust: bool = True,
    logger: Optional[logging.Logger] = None,
    min_days_required: int = 30  # Default minimum days required for analysis
) -> pd.DataFrame:
    """
    Fetch daily stock data from Yahoo Finance with proper error handling and validation.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date in 'YYYY-MM-DD' format or datetime
        end: End date in 'YYYY-MM-DD' format or datetime
        auto_adjust: Whether to adjust prices for splits and dividends
        logger: Optional logger instance for debug logging
        min_days_required: Minimum number of days required for analysis
        
    Returns:
        pd.DataFrame: DataFrame with daily stock data
    """
    if logger:
        logger.info(f"[YF_UTILS] Fetching daily data for {ticker} with args: start={start}, end={end}")
    
    try:
        # Convert string dates to datetime objects if needed
        if isinstance(start, str):
            start = pd.to_datetime(start).date()
        if isinstance(end, str):
            end = pd.to_datetime(end).date()
            
        # Set default start date if not provided (1 year ago)
        if start is None:
            start = (datetime.now().date() - timedelta(days=365))
            
        # Set default end date if not provided (yesterday)
        if end is None:
            end = datetime.now().date() - timedelta(days=1)
        
        # Ensure dates are valid
        today = datetime.now().date()
        
        # Adjust dates to ensure they're within valid ranges
        if end >= today:
            end = today - timedelta(days=1)
            if logger:
                logger.info(f"Adjusted end date to previous trading day: {end}")
                
        if start >= end:
            start = end - timedelta(days=min_days_required)
            if logger:
                logger.warning(f"Start date adjusted to ensure minimum {min_days_required} day range: {start}")
        
        # Ensure we have at least min_days_required days of data
        date_range = (end - start).days
        if date_range < min_days_required:
            start = end - timedelta(days=min_days_required)
            if logger:
                logger.warning(f"Date range expanded to meet minimum {min_days_required} day requirement")
        
        if logger:
            logger.info(f"Fetching data from {start} to {end}")
                
        # Download the data with a buffer period to ensure we get all requested dates
        buffer_days = 5  # Add buffer to handle weekends/holidays
        data_start = start - timedelta(days=buffer_days)
        
        data = yf.download(
            ticker,
            start=data_start,
            end=end + timedelta(days=1),  # Add one day to include the end date
            interval='1d',
            auto_adjust=auto_adjust,
            progress=False,
            threads=True
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
            
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Convert all timestamps to date (remove time component)
        data.index = data.index.normalize()
        
        # Remove any duplicate dates (keep last)
        data = data[~data.index.duplicated(keep='last')]
        
        # Sort by date
        data = data.sort_index()
        
        # Filter to only include the requested date range
        data = data[(data.index.date >= start) & (data.index.date <= end)]
        
        # Ensure we have the expected columns
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
            
        # Ensure we have enough data
        if len(data) < 2:  # Need at least 2 days for calculations
            raise ValueError(f"Insufficient data points. Got {len(data)} days, need at least 2")
            
        # Verify date range is as expected
        actual_start = data.index.min().date()
        actual_end = data.index.max().date()
        
        if actual_start > start or actual_end < end:
            if logger:
                logger.warning(f"Could not fetch data for full range. Got {actual_start} to {actual_end} "
                             f"(requested {start} to {end})")
                
            # If we don't have enough data, try to fetch more
            if (actual_end - actual_start).days < min_days_required:
                raise ValueError(f"Insufficient data available. Only {len(data)} days found, need at least {min_days_required}")
            
        if logger:
            logger.info(f"Successfully fetched {len(data)} days of data for {ticker}")
            logger.debug(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
            
        return data
        
    except Exception as e:
        error_msg = f"Error fetching data for {ticker}: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)