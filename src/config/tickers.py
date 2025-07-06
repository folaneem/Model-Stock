"""
Centralized configuration for stock tickers and related settings.

This module provides dynamic stock price and statistics fetching with caching.
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Default tickers used throughout the application
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Extended list of tickers for data collection
EXTENDED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Initialize with None - will be populated on first access
CURRENT_PRICES: Dict[str, Optional[float]] = {ticker: None for ticker in EXTENDED_TICKERS}
LAST_UPDATED: Optional[datetime] = None
PRICE_CACHE_MINUTES = 5  # Cache prices for 5 minutes

def get_current_prices(force_refresh: bool = False) -> Dict[str, Optional[float]]:
    """
    Get current stock prices, using cached values if recent.
    
    Args:
        force_refresh: If True, force a refresh of prices
        
    Returns:
        dict: Dictionary of {ticker: price} with prices as float or None if unavailable
    """
    global CURRENT_PRICES, LAST_UPDATED
    
    # Return cached prices if they're recent enough
    if (not force_refresh and 
        LAST_UPDATED and 
        (datetime.now() - LAST_UPDATED) < timedelta(minutes=PRICE_CACHE_MINUTES)):
        return CURRENT_PRICES
    
    try:
        from utils.data_collector import DataCollector
        
        # Use DataCollector to get latest prices
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        dc = DataCollector(
            tickers=EXTENDED_TICKERS,
            start_date=start_date,
            end_date=end_date,
            cache_dir='./data_cache'
        )
        
        # Update prices
        updated_prices = {}
        for ticker in EXTENDED_TICKERS:
            try:
                df = dc.get_stock_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    min_days_required=1
                )
                if df is not None and not df.empty and 'Close' in df.columns:
                    updated_prices[ticker] = float(df['Close'].iloc[-1])
                else:
                    updated_prices[ticker] = CURRENT_PRICES.get(ticker)  # Keep existing if available
                    print(f"Warning: No price data available for {ticker}")
            except Exception as e:
                print(f"Error fetching price for {ticker}: {str(e)}")
                updated_prices[ticker] = CURRENT_PRICES.get(ticker)  # Keep existing if available
        
        # Update global state
        CURRENT_PRICES.update(updated_prices)
        LAST_UPDATED = datetime.now()
        
    except Exception as e:
        print(f"Error in get_current_prices: {str(e)}")
        # Return whatever we have in the cache
        
    return CURRENT_PRICES

def get_stock_stats(lookback_days: int = 252) -> Dict[str, Dict[str, float]]:
    """
    Calculate stock statistics based on historical data.
    
    Args:
        lookback_days: Number of trading days to look back (default: 252 = ~1 year)
        
    Returns:
        dict: Dictionary of stock statistics with structure:
            {
                "TICKER": {
                    "mean_return": float,  # Annualized mean return
                    "volatility": float,   # Annualized volatility
                    "last_updated": str    # Timestamp of calculation
                },
                ...
            }
    """
    from utils.data_collector import DataCollector
    
    stats = {}
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime('%Y-%m-%d')  # Extra buffer
    
    try:
        dc = DataCollector(
            tickers=EXTENDED_TICKERS,
            start_date=start_date,
            end_date=end_date,
            cache_dir='./data_cache'
        )
        
        for ticker in EXTENDED_TICKERS:
            try:
                df = dc.get_stock_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    min_days_required=min(10, lookback_days)  # Require at least 10 days of data
                )
                
                if df is not None and not df.empty and 'Close' in df.columns:
                    if len(df) >= 10:  # Need at least 10 data points for meaningful stats
                        returns = df['Close'].pct_change().dropna()
                        if len(returns) > 0:
                            stats[ticker] = {
                                "mean_return": float(returns.mean() * 252),  # Annualized
                                "volatility": float(returns.std() * np.sqrt(252)),  # Annualized
                                "last_updated": datetime.now().isoformat()
                            }
                            continue
                
                # Fallback to default values if not enough data
                stats[ticker] = {
                    "mean_return": 0.1,  # 10% default
                    "volatility": 0.2,   # 20% default
                    "last_updated": datetime.now().isoformat(),
                    "warning": "Insufficient data, using default values"
                }
                    
            except Exception as e:
                print(f"Error calculating stats for {ticker}: {str(e)}")
                stats[ticker] = {
                    "mean_return": 0.1,  # 10% default
                    "volatility": 0.2,   # 20% default
                    "last_updated": datetime.now().isoformat(),
                    "error": str(e)
                }
    
    except Exception as e:
        print(f"Error initializing DataCollector: {str(e)}")
        # Return default stats if we can't initialize DataCollector
        stats = {
            ticker: {
                "mean_return": 0.1,
                "volatility": 0.2,
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            } for ticker in EXTENDED_TICKERS
        }
    
    return stats

# Initialize with current stats
STOCK_STATS = get_stock_stats()

# Initialize prices on import
get_current_prices()
