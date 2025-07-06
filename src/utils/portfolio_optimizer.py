import logging
from typing import Dict, List, Tuple, Union, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Optimizes portfolio allocation using modern portfolio theory.
    
    Features:
    - Minimum variance and maximum Sharpe ratio optimization
    - Diversification constraints
    - Robust covariance estimation
    - Position sizing and risk management
    - Support for custom constraints
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = np.clip(risk_free_rate, 0, 1)
        self.portfolio_weights = {}
        
        # Optimization parameters
        self.min_weight = 0.05  # 5% minimum weight per asset
        self.max_weight = 0.30  # 30% maximum weight per asset
        self.diversification_factor = 0.5  # 0-1, higher = more diversification
        self.max_assets = 20  # Maximum assets in portfolio
        
        # Winsorization parameters for outlier handling
        self.winsorize_lower = 0.05  # 5th percentile for winsorization
        self.winsorize_upper = 0.95  # 95th percentile for winsorization
        
        # Numerical stability
        self.epsilon = 1e-10
        self.n_restarts = 5
        
        # Data validation settings
        self.min_data_points = 60  # Minimum data points required for optimization

    def optimize_portfolio(self, tickers: List[str]) -> dict:
        """
        Optimize portfolio allocation with proper error handling.
        
        Args:
            tickers: List of ticker symbols to include in the portfolio
            
        Returns:
            dict: Dictionary containing optimization results or error information
        """
        from ..utils.data_processor import prepare_data_for_prediction
        from ..utils.yfinance_utils import calculate_returns
        
        if not tickers:
            logger.warning("No tickers provided for optimization")
            return {
                "error": "No tickers provided",
                "tickers": [],
                "status": "error"
            }
        
        # Get historical data
        returns_data = []
        valid_tickers = []
        
        for ticker in tickers:
            try:
                data = prepare_data_for_prediction(ticker)
                if data is not None and not data.empty:
                    returns = calculate_returns(data['Close'])
                    if len(returns) > 0:
                        returns_data.append(returns)
                        valid_tickers.append(ticker)
                        logger.debug(f"Added {ticker} to optimization")
            except Exception as e:
                logger.warning(f"Skipping {ticker} due to error: {str(e)}")
        
        if not valid_tickers:
            logger.warning("No valid tickers for optimization")
            return {
                "error": "No valid tickers for optimization",
                "tickers": [],
                "status": "error"
            }
        
        try:
            # Create returns DataFrame
            returns_df = pd.concat(returns_data, axis=1)
            returns_df.columns = valid_tickers
            
            # Handle any remaining null values
            returns_df = returns_df.fillna(0)
            
            # Run optimization
            if len(valid_tickers) == 1:
                # Special case for single asset
                weights = np.array([1.0])
                expected_return = returns_df.mean().iloc[0]
                volatility = returns_df.std().iloc[0]
                sharpe_ratio = (expected_return - self.risk_free_rate/252) / (volatility + self.epsilon)
            else:
                # Multiple assets
                self.returns = returns_df
                weights = self.optimize()
                
                # Calculate metrics
                expected_return = np.sum(returns_df.mean() * weights) * 252
                cov_matrix = returns_df.cov() * 252
                volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (expected_return - self.risk_free_rate) / (volatility + self.epsilon)
            
            # Prepare results
            weights_dict = {ticker: float(w) for ticker, w in zip(valid_tickers, weights)}
            
            return {
                "weights": weights_dict,
                "tickers": valid_tickers,
                "expected_return": float(expected_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "tickers": valid_tickers,
                "status": "error"
            }
    
    def calculate_volatility(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Portfolio volatility (standard deviation)
        """
        try:
            # Ensure inputs are numpy arrays
            if isinstance(weights, list):
                weights = np.array(weights)
                
            # Calculate portfolio returns
            if isinstance(returns, pd.DataFrame):
                portfolio_returns = returns.values @ weights
            else:
                portfolio_returns = returns @ weights
                
            # Calculate and return annualized volatility
            return np.std(portfolio_returns) * np.sqrt(252)  # 252 trading days in a year
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
            
    def calculate_sharpe_ratio(self, returns: pd.DataFrame, weights: np.ndarray = None) -> float:
        """
        Calculate the Sharpe ratio for the given portfolio weights.
        
        Args:
            returns: DataFrame of asset returns
            weights: Optional array of asset weights. If None, equal weights are used.
            
        Returns:
            float: Annualized Sharpe ratio
        """
        try:
            if weights is None:
                # Use equal weights if none provided
                n_assets = returns.shape[1] if hasattr(returns, 'shape') else len(returns.columns)
                weights = np.ones(n_assets) / n_assets
            elif isinstance(weights, list):
                weights = np.array(weights)
                
            # Calculate portfolio returns
            if isinstance(returns, pd.DataFrame):
                portfolio_returns = returns.values @ weights
            else:
                portfolio_returns = returns @ weights
                
            # Calculate annualized return and volatility
            annual_return = np.mean(portfolio_returns) * 252  # 252 trading days in a year
            annual_vol = np.std(portfolio_returns) * np.sqrt(252)
            
            # Avoid division by zero
            if annual_vol < 1e-10:
                return 0.0
                
            # Calculate Sharpe ratio (annualized)
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
            
    def calculate_max_drawdown(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculate maximum drawdown of the portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Maximum drawdown as a decimal (e.g., 0.15 for 15%)
        """
        try:
            # Ensure inputs are numpy arrays
            if isinstance(weights, list):
                weights = np.array(weights)
                
            # Calculate portfolio returns
            if isinstance(returns, pd.DataFrame):
                portfolio_returns = returns.values @ weights
            else:
                portfolio_returns = returns @ weights
                
            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + portfolio_returns) - 1
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate drawdowns
            drawdowns = (1 + cum_returns) / (1 + running_max) - 1
            
            # Return maximum drawdown (as positive number)
            return abs(np.min(drawdowns))
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
            
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns with robust error handling.
        
        Args:
            data: DataFrame with price data (columns = assets, index = dates)
            
        Returns:
            DataFrame of percentage returns
            
        Raises:
            ValueError: If data cannot be processed
        """
        try:
            # Convert to numeric and handle missing values
            numeric_data = data.apply(pd.to_numeric, errors='coerce')
            
            # Check for invalid columns
            bad_cols = numeric_data.columns[numeric_data.isnull().all()].tolist()
            if bad_cols:
                logger.error(f"Invalid data in columns: {bad_cols}")
                numeric_data = numeric_data.drop(columns=bad_cols)
                if numeric_data.empty:
                    raise ValueError("No valid data columns after processing")
            
            # Calculate returns and clean
            returns = numeric_data.pct_change().dropna()
            
            if returns.empty:
                logger.warning("No valid returns calculated")
                return pd.DataFrame(0.0, index=data.index[1:], columns=data.columns)
                
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise
        
    def _preprocess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess returns data by handling missing values and outliers.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            pd.DataFrame: Preprocessed returns data
        """
        if returns.empty:
            return returns
            
        # Make a copy to avoid modifying the original
        returns = returns.copy()
            
        # Handle missing values using ffill and bfill
        returns = returns.ffill().bfill().fillna(0)
        
        # Handle extreme outliers using winsorization
        def winsorize_series(s, lower=None, upper=None):
            lower = self.winsorize_lower if lower is None else lower
            upper = self.winsorize_upper if upper is None else upper
            q = s.quantile([lower, upper])
            return s.clip(q.iloc[0], q.iloc[1])
            
        # Only apply winsorization if we have enough data
        if len(returns) > 10:  # Require at least 10 data points
            try:
                return returns.apply(winsorize_series)
            except Exception as e:
                logger.warning(f"Winsorization failed: {str(e)}")
                return returns
        return returns

    def _calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regularized covariance matrix using Ledoit-Wolf shrinkage.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Regularized covariance matrix as DataFrame
        """
        if returns.empty or len(returns) < 2:
            n = len(returns.columns)
            return pd.DataFrame(np.eye(n) * 1e-4, 
                             columns=returns.columns, 
                             index=returns.columns)
        
        try:
            # Add tiny noise to break perfect collinearity
            returns = returns + np.random.normal(0, 1e-6, returns.shape)
            
            # Use Ledoit-Wolf shrinkage
            lw = LedoitWolf(assume_centered=True)
            cov = lw.fit(returns).covariance_
            
            # Ensure positive semi-definite
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            if min_eig < 0:
                cov -= 1.1 * min_eig * np.eye(*cov.shape)
                
            return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
            
        except Exception as e:
            logger.warning(f"Covariance estimation failed: {str(e)}")
            # Fall back to simple covariance with minimum regularization
            cov = returns.cov()
            cov += np.eye(len(cov)) * 1e-4
            return cov
        
    def _validate_inputs(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for optimization
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame")
            return False
            
        if data.empty or len(data) < 2:
            logger.warning("Insufficient data points for optimization")
            return False
            
        if (data == 0).all().all():
            logger.warning("All input data is zero")
            return False
            
        # Check for NaN or infinite values
        if data.isnull().any().any() or np.isinf(data).any().any():
            logger.warning("Input data contains NaN or infinite values")
            return False
            
        return True
        
    def _calculate_portfolio_volatility(self, weights: np.ndarray, cov_matrix: Union[pd.DataFrame, np.ndarray]) -> float:
        """
        Calculate portfolio volatility (annualized standard deviation of returns)
        with robust handling of edge cases.
        
        Args:
            weights: Portfolio weights as numpy array
            cov_matrix: Covariance matrix of returns (DataFrame or numpy array)
            
        Returns:
            float: Annualized portfolio volatility. Returns small positive value if calculation fails.
        """
        if cov_matrix is None or (hasattr(cov_matrix, 'empty') and cov_matrix.empty):
            logger.warning("Empty or None covariance matrix provided for volatility calculation.")
            return 1e-6  # Small positive value instead of zero
            
        try:
            # Convert to numpy array if needed
            cov_np = cov_matrix.values if hasattr(cov_matrix, 'values') else np.asarray(cov_matrix)
            weights_1d = np.asarray(weights).squeeze()
            
            # Handle scalar weight case
            if weights_1d.ndim == 0:
                weights_1d = np.array([weights_1d.item()])
                
            # Handle different covariance matrix cases
            if cov_np.ndim == 0:  # Scalar variance
                portfolio_variance_daily = cov_np * (weights_1d[0] ** 2)
            elif cov_np.shape == (1, 1) and len(weights_1d) == 1:  # 1x1 matrix
                portfolio_variance_daily = cov_np[0, 0] * (weights_1d[0] ** 2)
            elif cov_np.shape[0] != len(weights_1d) or cov_np.shape[1] != len(weights_1d):
                logger.error(f"Shape mismatch in volatility calculation: "
                           f"cov_matrix {cov_np.shape}, weights {weights_1d.shape}")
                return 1e-6
            elif cov_np.ndim == 2:  # Standard case
                portfolio_variance_daily = np.dot(weights_1d.T, np.dot(cov_np, weights_1d))
            else:
                logger.error(f"Unhandled covariance matrix dimension: {cov_np.ndim}")
                return 1e-6

            # Ensure non-negative variance and handle numerical stability
            portfolio_variance_daily = max(1e-10, float(portfolio_variance_daily))
            
            # Annualize the volatility (252 trading days)
            volatility_daily = np.sqrt(portfolio_variance_daily)
            volatility_annual = volatility_daily * np.sqrt(252)
            
            # Log the calculation for debugging
            logger.debug(f"Portfolio volatility - "
                       f"Daily Var: {portfolio_variance_daily:.6f}, "
                       f"Daily Vol: {volatility_daily:.6f}, "
                       f"Annual Vol: {volatility_annual:.6f}")
            
            # Ensure we don't return zero or negative volatility
            return max(1e-6, float(volatility_annual))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {str(e)}", exc_info=True)
            return 1e-6  # Return small positive value on error

    def _get_close_column_name(self, df: pd.DataFrame) -> str:
        """
        Get the name of the close price column from a DataFrame.
        
        Args:
            df: Input DataFrame with price data
            
        Returns:
            str: Name of the close price column
        """
        # List of possible column names for close prices (case-insensitive)
        possible_names = ['close', 'Close', 'CLOSE', 'price', 'Price', 'PRICE', 'adj close', 'Adj Close']
        
        # Check if any of the possible names exist in the DataFrame columns
        for name in possible_names:
            if name in df.columns:
                return name
                
        # If no match found, return the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
            
        # If no numeric columns, return the first column
        return df.columns[0] if len(df.columns) > 0 else 'Close'
        
    def _calculate_portfolio_return(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate portfolio expected return with robust error handling and proper scaling.
        
        Args:
            weights: Portfolio weights
            returns: DataFrame of asset returns (not prices)
            
        Returns:
            float: Annualized portfolio return (bounded between -99% and +100%)
        """
        try:
            if returns.empty:
                logger.warning("Empty returns data for portfolio return calculation")
                return 0.0
                
            # Ensure returns is a DataFrame
            if not isinstance(returns, pd.DataFrame):
                returns = pd.DataFrame(returns)
            
            # Check for potential percentage returns (values > 1)
            max_return = returns.abs().max().max()
            if max_return > 1.0:  # Likely percentage returns (e.g., 1.5 for 150%)
                logger.warning(f"Detected potential percentage returns (max={max_return:.2%}). Converting to decimal.")
                returns = returns / 100.0
            
            # Handle any remaining NaN or infinite values
            if returns.isnull().any().any() or np.isinf(returns).any().any():
                logger.warning("Cleaning returns data...")
                returns = returns.replace([np.inf, -np.inf], np.nan)
                returns = returns.ffill().bfill().fillna(0)
            
            # Ensure weights are in correct format
            weights_1d = np.asarray(weights).squeeze()
            if weights_1d.ndim == 0:
                weights_1d = np.array([weights_1d.item()])
            
            # Ensure weights sum to 1 (with small tolerance)
            weight_sum = np.sum(weights_1d)
            if not np.isclose(weight_sum, 1.0, rtol=1e-3):
                logger.warning(f"Weights sum to {weight_sum:.6f}, normalizing to 1.0")
                weights_1d = weights_1d / (weight_sum + 1e-10)
            
            # Ensure dimensions match
            if len(weights_1d) != returns.shape[1]:
                min_len = min(len(weights_1d), returns.shape[1])
                if min_len == 0:
                    logger.error("Zero-length arrays encountered")
                    return 0.0
                weights_1d = weights_1d[:min_len]
                returns = returns.iloc[:, :min_len]
            
            # Calculate daily portfolio returns
            daily_returns = (returns * weights_1d).sum(axis=1)
            
            # Calculate mean daily return
            mean_daily_return = daily_returns.mean()
            
            # Annualize returns using compounding
            # (1 + daily_return)^252 - 1
            annualized_return = (1 + mean_daily_return) ** 252 - 1
            
            # Cap returns at reasonable values
            annualized_return = np.clip(annualized_return, -0.99, 1.0)  # -99% to +100%
            
            # Log the calculation for debugging
            logger.debug(
                f"Portfolio return - "
                f"Mean daily return: {mean_daily_return:.6f}, "
                f"Annualized return: {annualized_return:.6f}, "
                f"Weights sum: {np.sum(weights_1d):.6f}"
            )
            
            return float(annualized_return)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio return: {str(e)}", exc_info=True)
            return 0.0
        
    def _calculate_sharpe_ratio(self, weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate Sharpe ratio with risk-free rate
        
        Args:
            weights: Portfolio weights as numpy array
            returns: DataFrame of asset returns
            cov_matrix: Covariance matrix of returns
            
        Returns:
            float: Annualized Sharpe ratio. Returns 0.0 if volatility is zero or undefined.
        """
        if returns.empty or cov_matrix.empty:
            logger.warning("Empty returns or covariance matrix provided for Sharpe ratio calculation.")
            return 0.0
            
        try:
            portfolio_return = self._calculate_portfolio_return(weights, returns)
            portfolio_volatility = self._calculate_portfolio_volatility(weights, cov_matrix)
            
            # Add small epsilon to avoid division by zero
            adjusted_volatility = portfolio_volatility + self.epsilon
            
            # Calculate Sharpe ratio with risk-free rate adjustment
            excess_return = portfolio_return - self.risk_free_rate
            sharpe_ratio = excess_return / adjusted_volatility
            
            # Log calculation details for debugging
            logger.debug(f"Sharpe ratio calculation - "
                       f"Return: {portfolio_return:.4f}, "
                       f"Volatility: {portfolio_volatility:.6f}, "
                       f"Risk-free: {self.risk_free_rate:.4f}, "
                       f"Sharpe: {sharpe_ratio:.4f}")
            
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}", exc_info=True)
            return 0.0

    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio"""
        portfolio_volatility = self._calculate_portfolio_volatility(weights, cov_matrix)
        if portfolio_volatility is None or portfolio_volatility == 0 or np.isnan(portfolio_volatility):
            logger.warning("Portfolio volatility is zero for diversification ratio. Returning 1.0.")
            return 1.0

        # Ensure diag input is 2D if it comes from DataFrame, or 1D if from numpy array
        asset_variances_np = np.diag(cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix)
        asset_volatilities_np = np.sqrt(np.maximum(0, asset_variances_np))  # ensure non-negative before sqrt

        weights_1d = np.asarray(weights).squeeze()
        if weights_1d.ndim == 0:
            weights_1d = np.array([weights_1d.item()])

        weighted_sum_asset_volatilities = np.sum(weights_1d * asset_volatilities_np)
        
        if portfolio_volatility == 0:  # Should be caught above, but defensive
            return 1.0
        return weighted_sum_asset_volatilities / portfolio_volatility
        
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return with bounds checking"""
        if returns.empty:
            return 0.0
            
        # Calculate mean return per period (daily)
        mean_return = returns.mean()
        
        # Annualize (252 trading days) with bounds checking
        annualized = (1 + mean_return) ** 252 - 1
        
        # Cap at reasonable values to prevent extreme numbers
        return float(np.clip(annualized, -0.99, 10.0))  # -99% to +1000%
    
    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility with bounds checking"""
        if returns.empty or len(returns) < 2:
            return 0.0
            
        # Calculate standard deviation with a small epsilon to avoid division by zero
        std_dev = returns.std()
        
        # If standard deviation is very small, return a small positive value
        if std_dev < 1e-9:
            return 1e-6
            
        # Annualize (sqrt of time)
        return float(std_dev * np.sqrt(252))
    
    def _calculate_sharpe_ratio_metric(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio with robust error handling"""
        if returns.empty or len(returns) < 2:
            return 0.0
            
        # Calculate excess return over risk-free rate (daily)
        excess_return = returns - (risk_free_rate / 252)
        
        # Calculate mean and std of excess returns
        mean_excess = excess_return.mean()
        std_excess = excess_return.std()
        
        # Handle edge cases
        if std_excess < 1e-9:  # Avoid division by zero
            return 0.0
            
        # Annualize
        sharpe = (mean_excess / std_excess) * np.sqrt(252)
        
        # Cap at reasonable values
        return float(np.clip(sharpe, -10.0, 10.0))
    
    def _calculate_diversification_ratio_metric(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio with error handling"""
        try:
            if cov_matrix.empty or len(weights) == 0:
                return 1.0
                
            # Calculate weighted average of individual volatilities
            asset_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.sum(weights * asset_volatilities)
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Handle edge cases
            if portfolio_vol < 1e-9:
                return 1.0
                
            diversification_ratio = weighted_avg_vol / portfolio_vol
            
            # Ensure the ratio is reasonable
            return float(np.clip(diversification_ratio, 1.0, 100.0))
            
        except Exception as e:
            logger.warning(f"Error calculating diversification ratio: {str(e)}")
            return 1.0

    def _validate_returns(self, returns: pd.DataFrame) -> bool:
        """Validate returns data before processing.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            bool: True if returns data appears valid
        """
        if returns.empty:
            logger.warning("Empty returns data provided")
            return False
            
        # Check for extreme values that suggest incorrect scaling
        max_return = returns.abs().max().max()
        if max_return > 1.0:  # If any return is > 100%
            logger.warning(f"Extreme return values detected (max={max_return:.2%}). Ensure returns are in decimal format (e.g., 0.01 for 1%).")
            return False
            
        # Check for NaN or infinite values
        if returns.isnull().any().any() or np.isinf(returns).any().any():
            logger.warning("Returns data contains NaN or infinite values")
            return False
            
        return True
        
    def _is_price_data(
        self,
        data: pd.DataFrame
    ) -> bool:
        """Check if the data appears to be price data (all positive values)."""
        if data.empty:
            return False
        return (data > 0).all().all()

    def _calculate_metrics_from_returns(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio metrics from return data with proper validation and scaling.
        
        Args:
            returns: DataFrame of asset returns (columns = assets, index = dates)
            
        Returns:
            Dictionary of portfolio metrics with consistent calculations as in _log_optimization_results
        """
        # Initialize default metrics
        metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'diversification_ratio': 1.0,
            'warning': None
        }
        
        try:
            # Validate returns data
            if not self._validate_returns(returns):
                # If validation fails, try to fix scaling issues
                if not returns.empty and np.any(returns.abs() > 1.0):
                    logger.warning("Attempting to fix returns scaling (dividing by 100)")
                    returns = returns / 100.0
                    if not self._validate_returns(returns):
                        raise ValueError("Returns data failed validation after scaling attempt")
                else:
                    raise ValueError("Invalid returns data provided")
            
            if not hasattr(self, 'portfolio_weights') or not self.portfolio_weights:
                raise ValueError("Portfolio weights not set")
                
            # Filter returns to only include assets with weights
            valid_assets = [a for a in returns.columns if a in self.portfolio_weights]
            if not valid_assets:
                raise ValueError("No valid assets with weights found in returns data")
                
            returns = returns[valid_assets]
            weights = np.array([self.portfolio_weights[a] for a in valid_assets])
            
            # Ensure weights sum to 1 (with small tolerance)
            weight_sum = weights.sum()
            if not np.isclose(weight_sum, 1.0, rtol=1e-6):
                logger.warning(f"Weights sum to {weight_sum:.6f}, normalizing to 1.0")
                weights = weights / (weight_sum + 1e-10)
            
            # Calculate daily metrics first
            daily_returns = returns @ weights
            
            # Calculate max drawdown from daily returns
            cum_returns = (1 + daily_returns).cumprod()
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = abs(min(0, drawdowns.min()))  # As positive number
            
            # Calculate covariance matrix (daily)
            cov_matrix_daily = returns.cov()
            
            # Calculate portfolio metrics
            port_return_daily = daily_returns.mean()
            port_vol_daily = np.sqrt(weights @ cov_matrix_daily.values @ weights.T)
            
            # Annualize metrics
            port_return_annual = (1 + port_return_daily) ** 252 - 1
            port_vol_annual = port_vol_daily * np.sqrt(252)
            
            # Calculate Sharpe ratio with risk-free rate
            excess_return = port_return_annual - self.risk_free_rate
            sharpe = excess_return / (port_vol_annual + 1e-10)  # Avoid division by zero
            
            # Calculate diversification ratio
            div_ratio = self._calculate_diversification_ratio(weights, cov_matrix_daily)
            
            # Log calculated values for debugging
            logger.debug(
                f"Portfolio metrics - "
                f"Daily Return: {port_return_daily:.6f}, "
                f"Annual Return: {port_return_annual:.6f}, "
                f"Daily Vol: {port_vol_daily:.6f}, "
                f"Annual Vol: {port_vol_annual:.6f}, "
                f"Sharpe: {sharpe:.6f}, "
                f"MaxDD: {max_drawdown:.6f}, "
                f"DivRatio: {div_ratio:.6f}"
            )
            
            # Update metrics with calculated values
            metrics.update({
                'expected_return': float(port_return_annual * 100),  # As percentage
                'volatility': float(port_vol_annual * 100),         # As percentage
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown * 100),          # As percentage
                'diversification_ratio': float(div_ratio)
            })
            
        except Exception as e:
            error_msg = f"Error calculating metrics from returns: {str(e)}"
            logger.error(error_msg, exc_info=True)
            metrics['warning'] = error_msg
            
        return metrics

    def _calculate_metrics_from_prices(
        self,
        prices: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio metrics from price data."""
        metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'diversification_ratio': 1.0
        }
        
        if prices.empty:
            logger.warning("Empty price data provided")
            return metrics
            
        try:
            # Convert prices to returns
            returns = prices.pct_change().dropna()
            if returns.empty:
                logger.warning("Empty returns after price conversion")
                return metrics
                
            # Calculate metrics from returns
            metrics = self._calculate_metrics_from_returns(returns)
            
            # Add current price
            if len(prices.columns) == 1:  # Single asset case
                metrics['ticker'] = prices.columns[0]
                metrics['current_price'] = float(prices.iloc[-1].values[0])
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics from prices: {str(e)}", exc_info=True)
            metrics['error'] = str(e)
            return metrics
            
    def _robust_optimization(
        self, 
        returns: pd.DataFrame, 
        cov_matrix: pd.DataFrame, 
        method: str = 'min_variance',
        min_weight: float = 0.05, 
        max_weight: float = 0.30,
        diversification_factor: float = 0.5,
        n_restarts: int = 10  # Increased from 5 to 10 for better robustness
    ) -> np.ndarray:
        """
        Robust portfolio optimization with multiple restarts and fallback strategies.
        
        Args:
            returns: DataFrame of asset returns
            cov_matrix: Covariance matrix
            method: 'min_variance' or 'max_sharpe'
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            diversification_factor: 0-1, higher = more diversification
            n_restarts: Number of optimization restarts
            
        Returns:
            Optimized weights as numpy array
        """
        if returns.empty or cov_matrix.empty:
            logger.warning("Empty returns or covariance matrix provided")
            return np.array([])
            
        n_assets = len(returns.columns)
        
        # Ensure we have valid data
        if n_assets == 0:
            logger.warning("No assets to optimize")
            return np.array([])
            
        # If only one asset, return 100% weight
        if n_assets == 1:
            return np.array([1.0])
        
        # Limit number of assets if needed
        if n_assets > self.max_assets:
            logger.warning(f"Limiting to top {self.max_assets} assets by Sharpe ratio")
            try:
                # Use more robust Sharpe ratio calculation
                sharpe_ratios = returns.mean() / (returns.std() + 1e-10)
                top_assets = sharpe_ratios.nlargest(self.max_assets).index
                returns = returns[top_assets]
                cov_matrix = cov_matrix.loc[top_assets, top_assets]
                n_assets = len(top_assets)
            except Exception as e:
                logger.error(f"Error limiting assets: {str(e)}")
                return np.ones(n_assets) / n_assets
        
        best_weights = np.ones(n_assets) / n_assets
        best_obj = -np.inf if method == 'max_sharpe' else np.inf
        
        # Try different optimization approaches
        for attempt in range(n_restarts):
            try:
                # Vary the diversification factor
                current_df = max(0.1, min(1.0, diversification_factor * (1.0 + 0.2 * (attempt % 3 - 1))))
                
                # Different initialization strategies
                weights = self._initialize_weights(attempt, n_assets, cov_matrix, current_df)
                
                # Run optimization with relaxed constraints if needed
                current_min_weight = max(0.01, min_weight * (1.0 - 0.1 * (attempt % 3)))
                current_max_weight = min(1.0, max_weight * (1.0 + 0.1 * (attempt % 3)))
                
                result = self._solve_optimization(
                    returns, cov_matrix, weights, 
                    method, current_min_weight, current_max_weight, current_df
                )
                
                if result is not None:
                    current_weights, current_obj = result
                    
                    # Update best solution
                    if ((method == 'max_sharpe' and current_obj > best_obj) or
                        (method == 'min_variance' and current_obj < best_obj)):
                        best_weights = current_weights
                        best_obj = current_obj
                        
            except Exception as e:
                logger.debug(f"Optimization attempt {attempt + 1} failed: {str(e)}")
        
        # Final validation and fallback
        if (np.any(np.isnan(best_weights)) or 
            abs(np.sum(best_weights) - 1.0) > 1e-6 or
            np.any(best_weights < -1e-6) or
            np.any(best_weights > 1.0 + 1e-6)):
            
            logger.warning("Optimization failed validation checks, using inverse volatility weights")
            try:
                # Fallback to inverse volatility weighting
                vol = np.sqrt(np.diag(cov_matrix))
                best_weights = 1 / (vol + 1e-10)
                best_weights = best_weights / best_weights.sum()
                
                # Apply min/max constraints
                best_weights = np.clip(best_weights, min_weight, max_weight)
                best_weights = best_weights / best_weights.sum()
                
                if np.any(np.isnan(best_weights)) or abs(np.sum(best_weights) - 1.0) > 1e-6:
                    raise ValueError("Fallback weights invalid")
                    
            except Exception as e:
                logger.warning("Fallback to inverse volatility failed, using equal weights")
                best_weights = np.ones(n_assets) / n_assets
            
        return best_weights
        
    def _initialize_weights(
        self, 
        attempt: int, 
        n_assets: int, 
        cov_matrix: pd.DataFrame,
        diversification_factor: float
    ) -> np.ndarray:
        """Initialize weights using different strategies."""
        if attempt == 0:
            # Equal weights
            return np.ones(n_assets) / n_assets
        elif attempt == 1:
            # Inverse volatility
            vol = np.sqrt(np.diag(cov_matrix))
            weights = 1 / (vol + 1e-10)
            return weights / weights.sum()
        else:
            # Random but diversified
            alpha = 1 + diversification_factor * 9  # Scale alpha based on diversification factor
            return np.random.dirichlet(np.ones(n_assets) * alpha)
    
    def _solve_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        init_weights: np.ndarray,
        method: str,
        min_weight: float,
        max_weight: float,
        diversification_factor: float
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Solve the convex optimization problem with enhanced stability.
        
        Args:
            returns: DataFrame of asset returns
            cov_matrix: Covariance matrix
            init_weights: Initial weights for warm start
            method: Optimization method ('min_variance' or 'max_sharpe')
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            diversification_factor: 0-1, higher = more diversification
            
        Returns:
            Tuple of (optimized_weights, objective_value) or None if optimization fails
        """
        try:
            n_assets = len(returns.columns)
            if n_assets == 0:
                return None

            # Add small regularization to covariance matrix
            cov_matrix = cov_matrix.copy()
            cov_matrix.values[np.diag_indices_from(cov_matrix)] += 1e-6
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Basic constraints with slight relaxation for numerical stability
            constraints = [
                cp.sum(weights) == 1,
                weights >= min_weight - 1e-8,
                weights <= max_weight + 1e-8
            ]
            
            # Calculate expected returns and risk
            returns_expected = returns.mean().values * 252  # Annualized
            risk = cp.quad_form(weights, cov_matrix.values)
            
            # Define objective based on method
            if method == 'min_variance':
                obj = risk
                if diversification_factor > 0:
                    # Use L1 norm for diversification with bounds
                    diversification = cp.norm(weights - 1/n_assets, 1)
                    obj += diversification_factor * diversification
                objective = cp.Minimize(obj)
            else:  # max_sharpe
                excess_return = returns_expected @ weights - self.risk_free_rate
                # Use variance instead of std for DCP compliance
                sharpe = excess_return / cp.sqrt(risk + 1e-10)
                
                if diversification_factor > 0:
                    # Use negative entropy for DCP compliance
                    entropy = -cp.sum(cp.entr(weights + 1e-10)) / np.log(n_assets)
                    objective = cp.Maximize(
                        (1 - diversification_factor) * sharpe + 
                        diversification_factor * entropy
                    )
                else:
                    objective = cp.Maximize(sharpe)
            
            # Define problem
            problem = cp.Problem(objective, constraints)
            
            # Try different solvers with increased iterations
            solvers = [
                ('ECOS', {'max_iters': 2000, 'abstol': 1e-8, 'reltol': 1e-8}),
                ('SCS', {'max_iters': 5000, 'eps': 1e-6})
            ]
            
            # Only try CVXOPT if it's available
            try:
                import cvxopt
                solvers.append(('CVXOPT', {'max_iter': 200, 'abstol': 1e-7, 'reltol': 1e-6}))
            except ImportError:
                logger.debug("CVXOPT solver not available")
            
            result = None
            for solver_name, params in solvers:
                try:
                    if solver_name == 'ECOS':
                        problem.solve(solver=cp.ECOS, max_iters=params['max_iters'],
                                    abstol=params['abstol'], reltol=params['reltol'],
                                    verbose=False)
                    elif solver_name == 'SCS':
                        problem.solve(solver=cp.SCS, max_iters=params['max_iters'],
                                    eps=params.get('eps', 1e-6), verbose=False)
                    elif solver_name == 'CVXOPT':
                        problem.solve(solver=cp.CVXOPT, max_iter=params['max_iter'],
                                    abstol=params['abstol'], reltol=params['reltol'],
                                    verbose=False)
                    
                    if (weights.value is not None and 
                        not np.any(np.isnan(weights.value)) and
                        np.all(weights.value >= -1e-6) and
                        abs(np.sum(weights.value) - 1.0) < 1e-4):
                        result = weights.value.copy()
                        break
                        
                except Exception as e:
                    logger.debug(f"Solver {solver_name} failed: {str(e)}")
            
            if result is None:
                logger.warning("All available solvers failed")
                return None
                
            # Process results
            result = np.maximum(result, 0)
            result_sum = np.sum(result)
            if result_sum > 0:
                result = result / result_sum
            else:
                result = np.ones_like(result) / len(result)
                
            # Calculate objective value
            if method == 'min_variance':
                obj_value = np.sqrt(result @ cov_matrix.values @ result.T)
            else:
                port_return = returns_expected @ result
                port_vol = np.sqrt(result @ cov_matrix.values @ result.T)
                obj_value = (port_return - self.risk_free_rate) / (port_vol + 1e-10)
            
            return result, float(obj_value)
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            return None
        
    def optimize_portfolio(
        self,
        data: pd.DataFrame,
        method: str = 'min_variance',
        min_weight: float = None,
        max_weight: float = None,
        target_volatility: float = None,
        target_return: float = None,
        diversification_factor: float = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using specified method and constraints.
        
        Args:
            data: DataFrame with price/return data (columns = assets, index = dates)
            method: 'min_variance' or 'max_sharpe'
            min_weight: Minimum weight per asset (None for default)
            max_weight: Maximum weight per asset (None for default)
            target_volatility: Optional target annualized volatility
            target_return: Optional target annualized return
            diversification_factor: 0-1, higher = more diversification
                
        Returns:
            Dict of {ticker: weight} with optimized weights
            
        Example:
            >>> optimizer = PortfolioOptimizer()
            >>> weights = optimizer.optimize_portfolio(prices, method='max_sharpe')
        """
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Input data must be a non-empty DataFrame")
                
            # Handle single asset case
            if len(data.columns) == 1:
                return {data.columns[0]: 1.0}
            
            # Set parameters
            min_w = min_weight if min_weight is not None else self.min_weight
            max_w = max_weight if max_weight is not None else self.max_weight
            div_factor = (diversification_factor if diversification_factor is not None 
                         else self.diversification_factor)
            
            # Calculate returns and covariance
            returns = self._preprocess_returns(self._calculate_returns(data))
            if len(returns) < 2:
                raise ValueError("Insufficient data points for optimization")
                
            cov_matrix = self._calculate_covariance(returns)
            
            # Run optimization
            optimized_weights = self._robust_optimization(
                returns=returns,
                cov_matrix=cov_matrix,
                method=method,
                min_weight=min_w,
                max_weight=max_w,
                diversification_factor=div_factor
            )
            
            # Apply target constraints if specified
            if target_volatility is not None or target_return is not None:
                optimized_weights = self._apply_target_constraints(
                    optimized_weights, returns, cov_matrix, 
                    target_volatility, target_return
                )
            
            # Final weight processing
            optimized_weights = np.clip(optimized_weights, min_w, max_w)
            optimized_weights /= optimized_weights.sum()
            
            # Store and return results
            self.portfolio_weights = {
                ticker: float(wt)
                for ticker, wt in zip(data.columns, optimized_weights)
                if wt > 1e-6  # Filter negligible weights
            }
            
            # Log optimization results
            self._log_optimization_results(optimized_weights, returns, cov_matrix)
            
            return self.portfolio_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            # Fallback to equal weights
            n_assets = len(data.columns)
            return {ticker: 1.0/n_assets for ticker in data.columns}
        
    def calculate_metrics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics with robust error handling.
        
        Args:
            data: DataFrame with price/return data
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Initialize default metrics
        default_metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'diversification_ratio': 1.0,
            'warning': None
        }
        
        try:
            # Check for empty data
            if data is None or data.empty:
                raise ValueError("No data provided for metrics calculation")
                
            # Check if portfolio weights are available
            if not hasattr(self, 'portfolio_weights') or not self.portfolio_weights:
                raise ValueError("No portfolio weights available for metrics calculation")
            
            # Ensure we only use tickers that exist in both data and portfolio_weights
            valid_tickers = [t for t in data.columns if t in self.portfolio_weights]
            if not valid_tickers:
                raise ValueError("No matching tickers between data and portfolio weights")
                
            # Filter data to only include valid tickers
            data = data[valid_tickers]
            
            # Calculate returns with error handling
            try:
                returns = self._calculate_returns(data)
                if returns.empty or len(returns) < 2:
                    raise ValueError("Insufficient data points for returns calculation")
                
                # Clean returns data
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                if returns.empty:
                    raise ValueError("No valid returns data after cleaning")
            except Exception as e:
                logger.error(f"Returns calculation failed: {str(e)}")
                raise
                
            # Calculate covariance with error handling
            try:
                cov_matrix = self._calculate_covariance(returns)
                if cov_matrix.isnull().any().any():
                    raise ValueError("Covariance matrix contains NaN values")
            except Exception as e:
                logger.warning(f"Covariance calculation warning: {str(e)}")
                # Fallback to diagonal matrix if covariance fails
                n_assets = len(returns.columns)
                cov_matrix = pd.DataFrame(
                    np.eye(n_assets) * returns.var().mean(),
                    index=returns.columns,
                    columns=returns.columns
                )
            
            # Get weights for valid tickers and ensure they sum to 1
            try:
                weights = np.array([self.portfolio_weights[ticker] for ticker in valid_tickers])
                weights_sum = weights.sum()
                if np.isclose(weights_sum, 0):
                    raise ValueError("Portfolio weights sum to zero")
                weights = weights / weights_sum  # Normalize to sum to 1
            except Exception as e:
                logger.error(f"Error processing portfolio weights: {str(e)}")
                # Fall back to equal weights if weight processing fails
                weights = np.ones(len(valid_tickers)) / len(valid_tickers)
            
            # Calculate each metric individually with error handling
            metrics = {}
            try:
                metrics['expected_return'] = self._calculate_portfolio_return(weights, returns)
                metrics['volatility'] = self._calculate_portfolio_volatility(weights, cov_matrix)
                metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(weights, returns, cov_matrix)
                metrics['diversification_ratio'] = self._calculate_diversification_ratio(weights, cov_matrix)
                
                # Validate metrics
                for key, value in metrics.items():
                    if not np.isfinite(value):
                        raise ValueError(f"Invalid {key} value: {value}")
                
                return metrics
                
            except Exception as metric_error:
                logger.error(f"Error calculating {key}: {str(metric_error)}")
                # If any metric fails, use default for that metric
                return {
                    **default_metrics,
                    **metrics,  # Include any successfully calculated metrics
                    'warning': f'Some metrics could not be calculated: {str(metric_error)}'
                }
            
        except Exception as e:
            logger.error(f"Error in calculate_metrics: {str(e)}")
            # Return default metrics with warning
            return {
                **default_metrics,
                'warning': f'Using default metrics due to: {str(e)}'
            }
        
    def _apply_target_constraints(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        target_vol: Optional[float],
        target_ret: Optional[float]
    ) -> np.ndarray:
        """
        Adjust weights to meet target volatility or return constraints.
        
        Args:
            weights: Current portfolio weights
            returns: DataFrame of asset returns
            cov_matrix: Covariance matrix
            target_vol: Target annualized volatility
            target_ret: Target annualized return
            
        Returns:
            Adjusted weights
        """
        if target_vol is None and target_ret is None:
            return weights
            
        n_assets = len(returns.columns)
        current_vol = np.sqrt(weights @ cov_matrix.values @ weights.T) * np.sqrt(252)
        current_ret = (returns.mean() @ weights) * 252
        
        # If no valid target is specified, return current weights
        if (target_vol is not None and target_vol <= 0) and \
           (target_ret is not None and target_ret <= 0):
            return weights
            
        # Calculate scaling factor for volatility
        if target_vol is not None and current_vol > 0:
            scale = target_vol / current_vol
            # Don't scale up too much to avoid extreme allocations
            scale = min(scale, 2.0)
            weights = weights * scale
            weights = weights / weights.sum()  # Renormalize
            
        # Adjust for target return if needed
        if target_ret is not None and current_ret < target_ret * 0.9:
            # Increase allocation to higher return assets
            returns_annual = returns.mean() * 252
            excess_returns = returns_annual - returns_annual.min()
            tilt = excess_returns / (excess_returns.sum() + 1e-10)
            # Gradually tilt towards higher return assets
            weights = 0.7 * weights + 0.3 * tilt
            weights = np.maximum(weights, 0)  # Ensure no negative weights
            weights = weights / weights.sum()
            
        return weights
        
    def _log_optimization_results(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame
    ) -> None:
        """
        Log key portfolio metrics after optimization.
        
        Args:
            weights: Optimized portfolio weights
            returns: DataFrame of asset returns
            cov_matrix: Covariance matrix (already annualized)
        """
        # Log input data for debugging
        logger.debug("\n" + "="*80)
        logger.debug("LOGGING OPTIMIZATION RESULTS")
        logger.debug("="*80)
        logger.debug(f"Number of assets: {len(weights)}")
        logger.debug(f"Number of return periods: {len(returns)}")
        logger.debug(f"Mean returns: {returns.mean().to_dict()}")
        
        # Calculate portfolio metrics
        port_return = (returns.mean() @ weights) * 252  # Annualized return
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights.T)  # Annualized volatility (cov_matrix is already annualized)
        sharpe = port_return / (port_vol + 1e-10)  # Avoid division by zero
        div_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        
        # Log detailed calculations
        logger.debug("\nMetric Calculations:")
        logger.debug(f"- Portfolio return: {returns.mean() @ weights:.6f} (daily) * 252 = {port_return:.6f} (annualized)")
        logger.debug(f"- Portfolio volatility: sqrt({weights} @ {cov_matrix.values.diagonal()} @ {weights.T}) = {port_vol:.6f}")
        logger.debug(f"- Sharpe ratio: {port_return:.6f} / {port_vol:.6f} = {sharpe:.6f}")
        logger.debug(f"- Diversification ratio: {div_ratio:.6f}")
        logger.debug("="*80 + "\n")
        
        # Calculate concentration (Herfindahl-Hirschman Index)
        hhi = np.sum(weights ** 2)
        
        # Calculate turnover (L1 norm of weight changes from previous weights)
        turnover = 0.0
        if hasattr(self, 'previous_weights') and len(self.previous_weights) == len(weights):
            turnover = np.sum(np.abs(weights - self.previous_weights))
        self.previous_weights = weights.copy()
        
        # Log the results
        logger.info(
            "Optimization results - "
            f"Return: {port_return:.2%}, "
            f"Volatility: {port_vol:.2%}, "
            f"Sharpe: {sharpe:.2f}, "
            f"Diversification: {div_ratio:.2f}, "
            f"Concentration (HHI): {hhi:.3f}, "
            f"Turnover: {turnover:.1%}"
        )
        
        # Log top 5 holdings if we have many assets
        if len(weights) > 5:
            top_holdings = sorted(
                zip(returns.columns, weights),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            top_str = ", ".join(f"{t}: {w:.1%}" for t, w in top_holdings)
            logger.info(f"Top holdings: {top_str}")
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate diversification ratio with robust error handling
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix of returns
            
        Returns:
            float: Diversification ratio (>= 1.0)
        """
        try:
            if cov_matrix.empty or len(weights) == 0:
                return 1.0
                
            # Ensure weights are in correct format
            weights_1d = np.asarray(weights).squeeze()
            if weights_1d.ndim == 0:
                weights_1d = np.array([weights_1d.item()])
            
            # Calculate portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(weights_1d, cov_matrix)
            
            # Handle edge cases
            if portfolio_vol < 1e-10:  # Avoid division by zero
                return 1.0
                
            # Calculate weighted average of individual volatilities
            asset_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.sum(weights_1d * asset_volatilities)
            
            # Calculate diversification ratio
            diversification_ratio = weighted_avg_vol / portfolio_vol
            
            # Ensure the ratio is reasonable
            return float(np.clip(diversification_ratio, 1.0, 100.0))
            
        except Exception as e:
            logger.warning(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
        
    def get_rebalancing_schedule(
        self,
        weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate rebalancing schedule
        """
        rebalancing = {}
        
        # Calculate current portfolio value
        total_value = sum(current_prices[ticker] * weights[ticker] for ticker in weights)
        
        # Calculate target values
        for ticker in weights:
            target_value = total_value * weights[ticker]
            current_value = current_prices[ticker] * weights[ticker]
            
            if abs(target_value - current_value) > 0.01 * current_value:  # 1% threshold
                rebalancing[ticker] = target_value - current_value
                
        return rebalancing
        
    def optimize_with_constraints(
        self,
        data: pd.DataFrame,
        max_weight: float = 0.3,
        min_weight: float = 0.05,
        target_return: float = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio with constraints
        """
        returns = self._calculate_returns(data)
        cov_matrix = self._calculate_covariance(returns)
        
        n_assets = len(data.columns)
        weights = cp.Variable(n_assets)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights <= max_weight,
            weights >= min_weight,
            weights >= 0
        ]
        
        if target_return:
            constraints.append(cp.sum(weights @ returns.mean()) >= target_return)
            
        # Objective: minimize portfolio volatility
        objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Get optimized weights
        optimized_weights = weights.value
        
        # Create weights dictionary
        self.portfolio_weights = {
            ticker: weight
            for ticker, weight in zip(data.columns, optimized_weights)
        }
        
        return self.portfolio_weights

# Example usage:
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Example data
    data = pd.DataFrame({
        'AAPL': np.random.normal(0.01, 0.02, 100),
        'GOOGL': np.random.normal(0.01, 0.02, 100),
        'MSFT': np.random.normal(0.01, 0.02, 100)
    })
    
    # Optimize portfolio
    weights = optimizer.optimize_portfolio(data)
    print("\nOptimized Weights:", weights)
    
    # Calculate metrics
    metrics = optimizer.calculate_metrics(data)
    print("\nPortfolio Metrics:", metrics)
    
    # Get rebalancing schedule
    current_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0}
    rebalancing = optimizer.get_rebalancing_schedule(weights, current_prices)
    print("\nRebalancing Schedule:", rebalancing)
