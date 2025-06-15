import yfinance as yf
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import logging
import time
import requests
import pandas_datareader.data as web
from typing import Dict, List, Tuple, Callable, Optional, Any

class DataCollector:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, cache_dir: str = "./data_cache", default_lookback_years: int = 5):
        """
        Initialize the DataCollector
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
            cache_dir (str): Directory to store cached data
            default_lookback_years (int): Default number of years of historical data to fetch
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.default_lookback_years = default_lookback_years
        self.logger = logging.getLogger(__name__)
        self._setup_logger()
        self._setup_cache()
        
    def _setup_logger(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _setup_cache(self):
        """Set up cache directory"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _get_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate cache key for a specific ticker and date range"""
        return f"{ticker}_{start_date}_{end_date}.pkl"
    
    def _cache_data(self, ticker: str, start_date: str, end_date: str, data: pd.DataFrame):
        """Cache data to disk"""
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_path = os.path.join(self.cache_dir, cache_key)
        data.to_pickle(cache_path)
        self.logger.info(f"Cached data for {ticker} from {start_date} to {end_date}")
        
    def _load_cached_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                data = pd.read_pickle(cache_path)
                self.logger.info(f"Loaded cached data for {ticker} from {start_date} to {end_date}")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {e}")
                return None
        return None
    
    def get_stock_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical stock data for a given ticker
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.default_lookback_years * 365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Try to load from cache first
        cached_data = self._load_cached_data(ticker, start_date, end_date)
        if cached_data is not None:
            return cached_data
            
        try:
            self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
                
            # Cache the data
            self._cache_data(ticker, start_date, end_date, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
            
    def get_multiple_stocks_data(self, tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames, one per ticker
            
        Raises:
            ValueError: If no valid tickers are provided or if data cannot be fetched for any ticker
        """
        if not tickers:
            raise ValueError("No tickers provided")
            
        data_dict = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                # Validate ticker first
                if not self.validate_ticker(ticker):
                    self.logger.warning(f"Invalid or non-existent ticker: {ticker}")
                    failed_tickers.append(ticker)
                    continue
                    
                # Get data for valid ticker
                data = self.get_stock_data(ticker, start_date, end_date)
                if not data.empty:
                    data_dict[ticker] = data
                else:
                    self.logger.warning(f"No data available for ticker: {ticker}")
                    failed_tickers.append(ticker)
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}", exc_info=True)
                failed_tickers.append(ticker)
        
        # If we couldn't get data for any ticker, raise an error
        if not data_dict and tickers:
            raise ValueError(f"Failed to fetch data for any of the provided tickers: {', '.join(tickers)}")
            
        # Log which tickers failed
        if failed_tickers:
            self.logger.warning(f"Failed to fetch data for the following tickers: {', '.join(failed_tickers)}")
            
        return data_dict
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available tickers
        
        Returns:
            List[str]: List of available stock tickers
        """
        try:
            # This is a simplified version - in production you might want to use a more robust method
            # or cache the results
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # Example tickers
        except Exception as e:
            self.logger.error(f"Error getting available tickers: {str(e)}")
            return []
            
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker exists and has data
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if ticker is valid and has data, False otherwise
            
        Note:
            This makes an API call to Yahoo Finance to verify the ticker exists
            and has available data. The result is not cached.
        """
        if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
            self.logger.warning(f"Invalid ticker format: {ticker}")
            return False
            
        ticker = ticker.strip().upper()
        
        try:
            # First check if ticker is in our known list of valid tickers
            known_tickers = self.get_available_tickers()
            if ticker in known_tickers:
                return True
                
            # If not in known list, try to fetch data
            self.logger.debug(f"Ticker {ticker} not in known list, checking with Yahoo Finance...")
            
            # Try to fetch minimal data to validate ticker
            try:
                data = yf.download(
                    ticker,
                    period='1d',
                    progress=False,
                    threads=False,
                    # Remove show_errors as it's not a valid parameter
                    # show_errors is handled by the try-except block
                )
            except Exception as e:
                self.logger.debug(f"Failed to download data for {ticker}: {str(e)}")
                return False
            
            # Check if we got valid data
            is_valid = not data.empty and not data['Close'].isna().all()
            
            if is_valid:
                self.logger.info(f"Successfully validated ticker: {ticker}")
            else:
                self.logger.warning(f"No data available for ticker: {ticker}")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating ticker {ticker}: {str(e)}", exc_info=True)
            return False
            


    def stream_real_time_data(self, tickers: List[str], callback: Callable[[dict], None], interval: int = 1) -> None:
        """
        Stream real-time stock data
        
        Args:
            tickers (List[str]): List of stock ticker symbols to stream
            callback (Callable[[dict], None]): Callback function to handle incoming data
            interval (int): Update interval in seconds
        """
        try:
            while True:
                try:
                    # Get current data for all tickers
                    data_dict = {}
                    for ticker in tickers:
                        try:
                            # Get latest data
                            stock = yf.Ticker(ticker)
                            data = stock.info
                            
                            # Prepare data for callback
                            data_dict[ticker] = {
                                'price': data.get('regularMarketPrice', None),
                                'change': data.get('regularMarketChange', None),
                                'volume': data.get('regularMarketVolume', None),
                                'timestamp': datetime.now().isoformat()
                            }
                        except Exception as e:
                            self.logger.error(f"Error getting real-time data for {ticker}: {str(e)}")
                            continue
                    
                    # Call the callback function with the data
                    callback(data_dict)
                    
                    # Wait for next interval
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("Real-time streaming stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in real-time streaming loop: {str(e)}")
                    # Wait before retrying
                    time.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"Error in real-time streaming: {str(e)}")
            raise

    def calculate_risk_metrics(self, data: pd.DataFrame, risk_free_rate: float = 0.0, 
                             market_ticker: str = '^GSPC', lookback_years: int = 3) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a stock
        
        Args:
            data (pd.DataFrame): Historical stock data with 'Close' column
            risk_free_rate (float): Annual risk-free rate (default: 0.0)
            market_ticker (str): Market index ticker for beta calculation (default: '^GSPC' for S&P 500)
            lookback_years (int): Number of years of data to use for calculations
            
        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        # Initialize default return values
        metrics = {
            'volatility': None,
            'annualized_return': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'sortino_ratio': None,
            'var_95': None,
            'cvar_95': None,
            'beta': None,
            'alpha': None,
            'tracking_error': None,
            'information_ratio': None
        }
        
        try:
            if data is None or data.empty or 'Close' not in data.columns:
                raise ValueError("Invalid input data: empty or missing 'Close' column")
                
            # Calculate daily returns and remove NaN/Inf values
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 5:  # Need at least 5 observations
                raise ValueError("Insufficient data points for meaningful calculations")
            
            # 1. Basic metrics
            metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1  # Annualized return
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # 2. Risk-adjusted returns
            # Sharpe Ratio (annualized)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['volatility']
            
            # Sortino Ratio (uses only downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            if downside_volatility > 0:
                metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_volatility
            
            # 3. Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # 4. Value at Risk (VaR) and Conditional VaR (CVaR)
            metrics['var_95'] = np.percentile(returns, 5)  # 1-day 95% VaR
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()  # Average of worst 5% losses
            
            # 5. Market-related metrics (Beta, Alpha, Tracking Error, Information Ratio)
            try:
                # Get market data if not already in the input
                if market_ticker not in data.columns:
                    market_data = self.get_stock_data(
                        market_ticker, 
                        start_date=(datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    if not market_data.empty and 'Close' in market_data.columns:
                        market_returns = market_data['Close'].pct_change().dropna()
                        
                        # Align market returns with stock returns
                        common_index = returns.index.intersection(market_returns.index)
                        if len(common_index) > 5:  # Need sufficient overlapping data
                            returns_aligned = returns[common_index]
                            market_returns_aligned = market_returns[common_index]
                            
                            # Calculate covariance and variance
                            covariance = np.cov(returns_aligned, market_returns_aligned)[0, 1]
                            market_variance = np.var(market_returns_aligned, ddof=1)
                            
                            # Beta
                            if market_variance > 0:
                                metrics['beta'] = covariance / market_variance
                                
                                # Alpha (Jensen's Alpha)
                                market_return_annualized = (1 + market_returns_aligned.mean()) ** 252 - 1
                                expected_return = risk_free_rate + metrics['beta'] * (market_return_annualized - risk_free_rate)
                                metrics['alpha'] = metrics['annualized_return'] - expected_return
                            
                            # Tracking Error and Information Ratio
                            active_returns = returns_aligned - market_returns_aligned
                            metrics['tracking_error'] = np.std(active_returns, ddof=1) * np.sqrt(252)
                            if metrics['tracking_error'] > 0:
                                metrics['information_ratio'] = (metrics['annualized_return'] - market_return_annualized) / metrics['tracking_error']
            
            except Exception as e:
                self.logger.warning(f"Could not calculate market-related metrics: {str(e)}")
                # Continue with other metrics if market data is not available
            
            self.logger.info("Successfully calculated risk metrics")
            return {k: round(v, 6) if isinstance(v, (int, float)) and not pd.isna(v) else v 
                   for k, v in metrics.items()}
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}", exc_info=True)
            return metrics

    def get_portfolio_data(self, tickers: List[str], weights: Dict[str, float], 
                         start_date: str, end_date: str, 
                         risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Get comprehensive portfolio-level data for analysis and optimization
        
        Args:
            tickers (List[str]): List of stock tickers in the portfolio
            weights (Dict[str, float]): Dictionary of ticker weights (must sum to ~1.0)
            start_date (str): Start date for data in 'YYYY-MM-DD' format
            end_date (str): End date for data in 'YYYY-MM-DD' format
            risk_free_rate (float): Annual risk-free rate for calculating risk-adjusted returns
            
        Returns:
            pd.DataFrame: DataFrame with portfolio metrics and component data
            
        Raises:
            ValueError: If input validation fails or data cannot be retrieved
        """
        # Input validation
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
            
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")
            
        if abs(sum(weights.values()) - 1.0) > 0.01:  # Allow for small floating point differences
            self.logger.warning(f"Weights sum to {sum(weights.values()):.4f}, normalizing to 1.0")
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
        
        try:
            # Get data for all tickers
            self.logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
            data_dict = self.get_multiple_stocks_data(tickers, start_date, end_date)
            
            if not data_dict:
                raise ValueError("No data returned for any tickers")
                
            # Find common date index across all tickers
            common_index = None
            for ticker, df in data_dict.items():
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
            
            if len(common_index) < 5:  # Need at least 5 data points
                raise ValueError("Insufficient common data points across tickers")
            
            # Initialize portfolio data structure
            portfolio_metrics = pd.DataFrame(index=common_index)
            component_returns = pd.DataFrame(index=common_index)
            
            # Calculate weighted returns for each component
            for ticker in tickers:
                if ticker in data_dict and not data_dict[ticker].empty:
                    # Align data to common index
                    df = data_dict[ticker].loc[common_index]
                    # Calculate daily returns
                    returns = df['Close'].pct_change().fillna(0)
                    # Apply portfolio weight
                    weighted_returns = returns * weights.get(ticker, 0)
                    component_returns[ticker] = weighted_returns
            
            if component_returns.empty:
                raise ValueError("No valid return data after processing tickers")
            
            # Calculate portfolio returns (sum of weighted component returns)
            portfolio_metrics['Daily_Return'] = component_returns.sum(axis=1)
            
            # Remove any remaining NaN or infinite values
            portfolio_metrics = portfolio_metrics.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(portfolio_metrics) < 5:
                raise ValueError("Insufficient valid data points after cleaning")
            
            # Calculate key metrics
            # 1. Cumulative returns
            portfolio_metrics['Cumulative_Return'] = (1 + portfolio_metrics['Daily_Return']).cumprod()
            
            # 2. Volatility (annualized)
            portfolio_metrics['Daily_Volatility'] = component_returns.std(axis=1)
            annual_volatility = portfolio_metrics['Daily_Return'].std() * np.sqrt(252)
            
            # 3. Drawdown
            running_max = portfolio_metrics['Cumulative_Return'].cummax()
            portfolio_metrics['Drawdown'] = (portfolio_metrics['Cumulative_Return'] - running_max) / running_max
            
            # 4. Risk-adjusted returns
            annualized_return = (1 + portfolio_metrics['Daily_Return'].mean()) ** 252 - 1
            if annual_volatility > 0:
                portfolio_metrics['Sharpe_Ratio'] = (annualized_return - risk_free_rate) / annual_volatility
            
            # Calculate Sortino ratio (only downside deviation)
            downside_returns = portfolio_metrics[portfolio_metrics['Daily_Return'] < 0]['Daily_Return']
            downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            if downside_volatility > 0:
                portfolio_metrics['Sortino_Ratio'] = (annualized_return - risk_free_rate) / downside_volatility
            
            # Add component weights to the output
            for ticker, weight in weights.items():
                portfolio_metrics[f'Weight_{ticker}'] = weight
            
            # Add metadata
            portfolio_metrics['Start_Date'] = start_date
            portfolio_metrics['End_Date'] = end_date
            portfolio_metrics['Risk_Free_Rate'] = risk_free_rate
            
            # Ensure all numeric columns are float
            for col in portfolio_metrics.select_dtypes(include=['number']).columns:
                portfolio_metrics[col] = portfolio_metrics[col].astype(float)
            
            self.logger.info(f"Successfully calculated portfolio metrics for {len(portfolio_metrics)} periods")
            return portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Error in get_portfolio_data: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to calculate portfolio data: {str(e)}") from e

    def collect_sentiment_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Collect sentiment data for a stock
        
        Args:
            ticker (str): Stock ticker
            days (int): Number of days of historical data to use
            
        Returns:
            pd.DataFrame: Sentiment data with composite sentiment score and components
        """
        try:
            # Get historical data for sentiment analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get stock data for the full date range
            data = self.get_stock_data(ticker, 
                                    start_date=start_date.strftime('%Y-%m-%d'), 
                                    end_date=end_date.strftime('%Y-%m-%d'))
            
            if data.empty:
                self.logger.warning(f"No stock data found for {ticker} in the specified date range")
                return pd.DataFrame(columns=['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility'], 
                                  index=pd.DatetimeIndex([]))
            
            # Ensure we have required columns
            required_columns = ['Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns in stock data. Need: {required_columns}")
                
            # Initialize sentiment data DataFrame
            sentiment_data = pd.DataFrame(index=data.index)
            
            try:
                # 1. Price Momentum (normalized)
                price_momentum = data['Close'].pct_change()
                # Handle any potential infinite or NaN values
                price_momentum = price_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
                # Scale to -1 to 1 range using tanh
                price_momentum_scaled = np.tanh(price_momentum * 10)  # Scale factor for better sensitivity
                
                # 2. Volume Change (normalized)
                volume_change = data['Volume'].pct_change()
                # Handle division by zero, infinite, and NaN values
                volume_change = volume_change.replace([np.inf, -np.inf], np.nan).fillna(0)
                # Scale using min-max normalization
                if len(volume_change) > 1:
                    volume_change_scaled = 2 * (volume_change - volume_change.min()) / \
                                         (volume_change.max() - volume_change.min() + 1e-10) - 1
                else:
                    volume_change_scaled = pd.Series(0, index=volume_change.index)
                
                # 3. Volatility (inverse normalized)
                returns = data['Close'].pct_change()
                volatility = returns.rolling(window=min(20, len(returns)), min_periods=1).std()
                # Scale volatility to 0-1 range and invert (higher volatility = more negative sentiment)
                if len(volatility) > 1:
                    volatility_scaled = 1 - ((volatility - volatility.min()) / 
                                          (volatility.max() - volatility.min() + 1e-10))
                    volatility_scaled = 2 * volatility_scaled - 1  # Scale to -1 to 1
                else:
                    volatility_scaled = pd.Series(0, index=volatility.index)
                
                # Store components
                sentiment_data['Price_Momentum'] = price_momentum_scaled
                sentiment_data['Volume_Change'] = volume_change_scaled
                sentiment_data['Volatility'] = volatility_scaled
                
                # Calculate composite sentiment score with weights
                weights = {
                    'Price_Momentum': 0.5,
                    'Volume_Change': 0.3,
                    'Volatility': 0.2
                }
                
                sentiment_data['sentiment_score'] = (
                    weights['Price_Momentum'] * sentiment_data['Price_Momentum'] +
                    weights['Volume_Change'] * sentiment_data['Volume_Change'] -
                    weights['Volatility'] * sentiment_data['Volatility']  # Subtract volatility (inverse relationship)
                )
                
                # Clip final sentiment to -1 to 1 range
                sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'].clip(-1, 1)
                
                # Ensure all values are finite and fill any remaining NaNs
                sentiment_data = sentiment_data.replace([np.inf, -np.inf], np.nan)
                # Forward fill and then backfill any remaining NaNs
                sentiment_data = sentiment_data.ffill().bfill().fillna(0)
                # Ensure all values are within expected ranges
                sentiment_data = sentiment_data.clip(-1, 1)
                
                # Ensure we have the required columns with correct names
                sentiment_data = sentiment_data.rename(columns={
                    'Sentiment': 'sentiment_score',  # For backward compatibility
                    'sentiment': 'sentiment_score'     # For any other potential variations
                })
                
                # If sentiment_score column is missing but we have 'Sentiment', rename it
                if 'sentiment_score' not in sentiment_data.columns and 'Sentiment' in sentiment_data.columns:
                    sentiment_data['sentiment_score'] = sentiment_data['Sentiment']
                
                self.logger.info(f"Successfully generated sentiment data for {len(sentiment_data)} days")
                return sentiment_data
                
            except Exception as e:
                self.logger.error(f"Error calculating sentiment indicators: {str(e)}", exc_info=True)
                # Return a simple sentiment series if calculation fails
                sentiment_data = pd.DataFrame(index=pd.DatetimeIndex([]))
                sentiment_data['sentiment_score'] = 0
                sentiment_data['Price_Momentum'] = 0
                sentiment_data['Volume_Change'] = 0
                sentiment_data['Volatility'] = 0
                return sentiment_data
                
        except Exception as e:
            self.logger.error(f"Error in collect_sentiment_data for {ticker}: {str(e)}", exc_info=True)
            # Return empty DataFrame with the right columns to prevent crashes
            return pd.DataFrame(
                columns=['sentiment_score', 'Price_Momentum', 'Volume_Change', 'Volatility'],
                index=pd.DatetimeIndex([])
            )

    def collect_macroeconomic_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Collect macroeconomic data including interest rates, market indicators, and economic indicators
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, uses 2 years ago.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses current date.
            
        Returns:
            pd.DataFrame: DataFrame containing macroeconomic indicators
        """
        try:
            # Parse dates with validation
            end_date = pd.to_datetime(end_date) if end_date else datetime.now()
            start_date = pd.to_datetime(start_date) if start_date else (end_date - timedelta(days=365 * 5))
            
            # Validate date range (max 10 years)
            max_date_range = timedelta(days=365 * 10)
            if (end_date - start_date) > max_date_range:
                start_date = end_date - max_date_range
                self.logger.warning(f"Date range exceeds 10 years. Adjusting to: {start_date} to {end_date}")
            
            # Get market data (SPY as market proxy)
            market_data = self.get_stock_data('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if market_data.empty:
                self.logger.error("No market data available for the selected date range")
                return pd.DataFrame()
                
            # Initialize DataFrame with market data
            macro_data = pd.DataFrame(index=market_data.index)
            
            # 1. Market-based indicators
            macro_data['Market_Return'] = market_data['Close'].pct_change()
            macro_data['Market_Volatility'] = market_data['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
            
            # 2. Interest Rates (10-Year Treasury Yield)
            try:
                rates_data = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
                if not rates_data.empty:
                    macro_data['Interest_Rate'] = rates_data / 100
                else:
                    self.logger.warning("No interest rate data found")
                    macro_data['Interest_Rate'] = np.nan
            except Exception as e:
                self.logger.warning(f"Could not fetch interest rate data: {str(e)}")
                macro_data['Interest_Rate'] = np.nan
            
            # 3. VIX (Market Volatility Index)
            try:
                vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
                if not vix_data.empty:
                    macro_data['VIX'] = vix_data / 100
                else:
                    self.logger.warning("No VIX data found")
                    macro_data['VIX'] = np.nan
            except Exception as e:
                self.logger.warning(f"Could not fetch VIX data: {str(e)}")
                macro_data['VIX'] = np.nan
            
            # Calculate derived metrics
            macro_data['Risk_Free_Rate'] = macro_data['Interest_Rate']
            macro_data['Equity_Risk_Premium'] = macro_data['Market_Return'] - macro_data['Risk_Free_Rate'].shift(1)
            
            # Replace infinite values with NaN first
            macro_data = macro_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with forward fill, then backward fill
            macro_data = macro_data.ffill().bfill()
            
            # For any remaining NaNs, fill with column means or 0 if all values are NaN
            for col in macro_data.columns:
                if macro_data[col].isna().any():
                    if macro_data[col].count() > 0:  # If there are some non-NaN values
                        macro_data[col] = macro_data[col].fillna(macro_data[col].mean())
                    else:  # If all values are NaN
                        macro_data[col] = 0
                        
            # Log any remaining NaN values (should be none at this point)
            if macro_data.isna().any().any():
                self.logger.warning(f"Macro data still contains NaN values after all filling attempts: {macro_data.isna().sum()}")
            else:
                self.logger.info("Successfully cleaned all NaN and infinite values from macro data")
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Error collecting macroeconomic data: {str(e)}")
            raise

    def update_data(self, rebalance_frequency: str) -> None:
        """
        Update data based on rebalancing frequency
        
        Args:
            rebalance_frequency (str): Frequency of rebalancing ('Daily', 'Weekly', 'Monthly')
        """
        try:
            if rebalance_frequency == 'Daily':
                period = '1d'
            elif rebalance_frequency == 'Weekly':
                period = '1wk'
            elif rebalance_frequency == 'Monthly':
                period = '1mo'
            else:
                raise ValueError(f"Invalid rebalance frequency: {rebalance_frequency}")
                
            # Update stock data
            for ticker in self.tickers:
                try:
                    data = yf.download(ticker, period=period)
                    if not data.empty:
                        self._cache_data(ticker, self.start_date, self.end_date, data)
                except Exception as e:
                    self.logger.warning(f"Failed to update data for {ticker}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            raise

    def get_rebalance_dates(self, rebalance_frequency: str) -> pd.DatetimeIndex:
        """
        Get rebalancing dates based on frequency
        
        Args:
            rebalance_frequency (str): Frequency of rebalancing ('Daily', 'Weekly', 'Monthly')
            
        Returns:
            pd.DatetimeIndex: Dates for rebalancing
        """
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            if rebalance_frequency == 'Daily':
                dates = pd.date_range(start, end, freq='B')  # Business days
            elif rebalance_frequency == 'Weekly':
                dates = pd.date_range(start, end, freq='W-FRI')  # Fridays
            elif rebalance_frequency == 'Monthly':
                dates = pd.date_range(start, end, freq='BM')  # Business month-end
            else:
                raise ValueError(f"Invalid rebalance frequency: {rebalance_frequency}")
                
            return dates
            
        except Exception as e:
            self.logger.error(f"Error getting rebalance dates: {str(e)}")
            raise

    def collect_company_data(self, ticker: str) -> Dict[str, Any]:
        """
        Collect comprehensive company data
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Dict[str, Any]: Company information
        """
        try:
            # Get basic company info
            info = self.get_company_info(ticker)
            
            # Get financial data
            stock = yf.Ticker(ticker)
            financials = {
                'revenue': stock.info.get('revenue', 0),
                'profit_margin': stock.info.get('profitMargins', 0),
                'debt_to_equity': stock.info.get('debtToEquity', 0),
                'return_on_assets': stock.info.get('returnOnAssets', 0)
            }
            
            # Combine all data
            return {**info, **financials}
        except Exception as e:
            self.logger.error(f"Error collecting company data: {str(e)}")
            raise

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information for a given ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict[str, Any]: Company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'country': info.get('country', 'N/A'),
                'website': info.get('website', 'N/A')
            }
        except Exception as e:
            self.logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {
                'name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'country': 'N/A',
                'website': 'N/A'
            }