import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats
import ta
from .yfinance_utils import fetch_daily_yfinance_data
import warnings
from datetime import datetime
from typing import Optional
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scalers = {}
        self.features = None
        self.feature_scalers = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame using the ta library.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure required OHLCV columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain these columns: {required_columns}")
        
        try:
            # Add RSI
            df['momentum_rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
            
            # Add MACD
            macd = ta.trend.MACD(close=df['Close'])
            df['trend_macd'] = macd.macd()
            df['trend_macd_signal'] = macd.macd_signal()
            df['trend_macd_diff'] = macd.macd_diff()
            
            # Add Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )
            df['momentum_stoch'] = stoch.stoch()
            df['momentum_stoch_signal'] = stoch.stoch_signal()
            
            # Add Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=df['Close'])
            df['volatility_bb_high'] = bollinger.bollinger_hband()
            df['volatility_bb_low'] = bollinger.bollinger_lband()
            
            # Add Moving Averages
            df['trend_sma_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['trend_sma_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['trend_sma_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
            
            # Add Volume indicators
            df['volume_obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['Close'], 
                volume=df['Volume']
            ).on_balance_volume()
            
            # Add ATR (Average True Range)
            df['volatility_atr'] = ta.volatility.AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ).average_true_range()
            
            return df
            
        except Exception as e:
            error_msg = f"Error calculating technical indicators: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def load_data(self, ticker: str, start_date: datetime, end_date: datetime) -> tuple:
        """
        Load and preprocess stock data with enhanced validation and error handling.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Tuple of (X, y, data) where:
            - X: Processed feature matrix
            - y: Target variable
            - data: Raw data with all features
            
        Raises:
            ValueError: If data loading or preprocessing fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Input validation
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Ticker must be a non-empty string")
                
            if not isinstance(start_date, (datetime, pd.Timestamp)) or not isinstance(end_date, (datetime, pd.Timestamp)):
                raise ValueError("start_date and end_date must be datetime objects")
                
            if start_date >= end_date:
                raise ValueError("start_date must be before end_date")
                
            logger.info(f"[DataProcessor] Loading data for {ticker} from {start_date} to {end_date}")
            
            # Download data from Yahoo Finance with retry logic
            max_retries = 3
            data = None
            
            for attempt in range(max_retries):
                try:
                    data = fetch_daily_yfinance_data(
                        ticker,
                        start=start_date.date(),
                        end=end_date.date(),
                        logger=logger
                    )
                    if data is not None and not data.empty:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(1)
            if data is None or data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            logger.info(f"ROWCOUNT: After download: {len(data)} rows for {ticker}")
            self._validate_price_data(data)
            logger.info(f"ROWCOUNT: After cleaning: {len(data)} rows for {ticker}")

            # Create comprehensive features with error handling
            data = self._create_features(data)
            if data is None or data.empty:
                raise ValueError("Feature creation failed or returned empty data")
            logger.info(f"ROWCOUNT: After feature engineering: {len(data)} rows for {ticker}")

            # Clean data
            data = self._clean_data(data)
            if data is None or data.empty:
                raise ValueError("Data cleaning failed or returned empty data")

            # Detect and handle outliers
            data = self._detect_outliers(data)
            if data is None or data.empty:
                raise ValueError("Outlier detection failed or returned empty data")

            # Scale features
            X, y = self._scale_data(data)
            if X is None or y is None or X.empty or y.empty:
                raise ValueError("Feature scaling failed")

            # Create additional features based on scaled data
            X = self._create_additional_features(X)
            if X is None or X.empty:
                raise ValueError("Additional feature creation failed")
            
            # Final validation before returning
            self._validate_processed_data(X, y)
            
            logger.info(f"Successfully processed data with shape: X={X.shape}, y={y.shape}")
            return X, y, data
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}", exc_info=True)
            raise
            
    def _validate_price_data(self, data: pd.DataFrame) -> None:
        """
        Validate raw price data for required columns and basic quality checks.
        
        Args:
            data: DataFrame with OHLCV data
            
        Raises:
            ValueError: If data fails validation
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Check for NaN values
        if data[required_columns].isnull().any().any():
            nan_counts = data[required_columns].isnull().sum()
            raise ValueError(f"Data contains NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            
        # Check for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (data[price_cols] <= 0).any().any():
            zero_price_dates = data.index[data[price_cols].le(0).any(axis=1)].tolist()
            raise ValueError(f"Zero or negative prices found on dates: {zero_price_dates}")
            
        # Check for zero volume
        if (data['Volume'] < 0).any():
            negative_volume_dates = data.index[data['Volume'] < 0].tolist()
            raise ValueError(f"Negative volume found on dates: {negative_volume_dates}")
            
        # Check for data gaps
        date_diff = pd.Series(data.index).diff().dropna()
        if len(date_diff.unique()) > 2:  # More than one unique difference (e.g., weekends + holidays)
            gaps = date_diff[date_diff > pd.Timedelta(days=1)]
            if not gaps.empty:
                raise ValueError(f"Significant data gaps found: {gaps.value_counts().to_dict()}")
    
    def check_recent_contiguous_days(self, data: pd.DataFrame, n_days: int, logger=None) -> bool:
        """
        Check if the last n_days in the DataFrame index are contiguous trading days (no missing business days or NYSE holidays).
        Logs any missing days or gaps, distinguishes between expected (holidays/weekends) and unexpected gaps.
        Returns True if contiguous and present, False otherwise.
        """
        import pandas as pd
        try:
            import pandas_market_calendars as mcal
        except ImportError:
            raise ImportError("pandas_market_calendars must be installed for holiday-aware contiguity checks.")
        if logger is None:
            logger = logging.getLogger(__name__)
        if data is None or data.empty or len(data) < n_days:
            logger.warning(f"Not enough data to check contiguity: have {len(data) if data is not None else 0}, need {n_days}")
            return False
        recent = data.iloc[-n_days:]
        idx = pd.DatetimeIndex(recent.index)
        nyse = mcal.get_calendar('NYSE')
        expected = nyse.valid_days(start_date=idx[0], end_date=idx[-1])
        missing = set(expected) - set(idx)
        if missing:
            logger.warning(f"Missing trading days in last {n_days} (excluding weekends/holidays): {sorted([d.strftime('%Y-%m-%d') for d in missing])}")
            return False
        logger.info(f"Last {n_days} trading days are contiguous and present (NYSE calendar aware).")
        return True

    def _validate_processed_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate processed feature and target data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Raises:
            ValueError: If validation fails
        """
        if len(X) != len(y):
            raise ValueError(f"Feature and target length mismatch: X={len(X)}, y={len(y)}")
            
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"NaN values found in features: {nan_cols}")
            
        if y.isnull().any():
            raise ValueError("NaN values found in target variable")
            
        # Check for constant features
        constant_cols = X.columns[X.nunique() == 1]
        if not constant_cols.empty:
            logger.warning(f"Constant features detected: {constant_cols.tolist()}")

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive set of technical indicators and features with safe handling of Series operations.
        
        Args:
            data: Input stock data with OHLCV columns
            
        Returns:
            DataFrame with additional technical indicators and features
            
        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        try:
            logger = logging.getLogger(__name__)
            logger.info("Creating features...")
            
            # Make a copy to avoid SettingWithCopyWarning
            result = data.copy()
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
            # Ensure we have enough data points
            min_required_rows = 60  # Minimum rows needed for some indicators
            if len(result) < min_required_rows:
                raise ValueError(f"Insufficient data points: {len(result)}. Need at least {min_required_rows}.")
            
            # Ensure we have pandas Series for calculations
            close_series = pd.to_numeric(result['Close'], errors='coerce')
            high_series = pd.to_numeric(result['High'], errors='coerce')
            low_series = pd.to_numeric(result['Low'], errors='coerce')
            volume_series = pd.to_numeric(result['Volume'], errors='coerce')
            
            # Log basic statistics before feature creation
            logger.info(f"Creating features for {len(result)} data points")
            
            # Safely extract min/max values with proper error handling
            try:
                price_min = close_series.min()
                price_max = close_series.max()
                vol_min = volume_series.min()
                vol_max = volume_series.max()
                
                # Convert to native Python types
                price_min = float(price_min) if pd.notna(price_min) else 0.0
                price_max = float(price_max) if pd.notna(price_max) else 0.0
                vol_min = int(vol_min) if pd.notna(vol_min) else 0
                vol_max = int(vol_max) if pd.notna(vol_max) else 0
                
                logger.info(f"Price range: {price_min:.2f} - {price_max:.2f}")
                logger.info(f"Volume range: {vol_min:,} - {vol_max:,}")
                
                # Verify we have valid price data
                if price_max <= 0 or not np.isfinite(price_max):
                    raise ValueError("Invalid price data detected in the series")
                    
            except Exception as e:
                logger.error(f"Error calculating price/volume statistics: {str(e)}")
                # Provide default values that won't break the application
                price_min, price_max, vol_min, vol_max = 0.0, 1.0, 0, 1
                logger.warning("Using fallback price/volume statistics")
            
            # Dictionary to store feature creation results
            features = {}
            
            # 1. Price-based features
            try:
                # Simple Moving Averages
                for window in [5, 10, 20, 50, 200]:
                    if len(close_series) >= window:
                        features[f'SMA_{window}'] = close_series.rolling(window=window).mean()
                        features[f'EMA_{window}'] = close_series.ewm(span=window, adjust=False).mean()
                
                # Bollinger Bands
                window_bb = 20
                if len(close_series) >= window_bb:
                    sma = close_series.rolling(window=window_bb).mean()
                    std = close_series.rolling(window=window_bb).std()
                    features['BB_upper'] = sma + (std * 2)
                    features['BB_lower'] = sma - (std * 2)
                    features['BB_width'] = (features['BB_upper'] - features['BB_lower']) / sma
                
                # Donchian Channels
                window_dc = 20
                if len(high_series) >= window_dc and len(low_series) >= window_dc:
                    features['DC_upper'] = high_series.rolling(window=window_dc).max()
                    features['DC_lower'] = low_series.rolling(window=window_dc).min()
                    features['DC_mid'] = (features['DC_upper'] + features['DC_lower']) / 2
                
                # Price Rate of Change
                for period in [1, 3, 5, 10, 20]:
                    if len(close_series) > period:
                        features[f'ROC_{period}'] = close_series.pct_change(periods=period) * 100
                
                logger.info("Added price-based features")
                
            except Exception as e:
                logger.error(f"Error creating price-based features: {str(e)}", exc_info=True)
                # Continue with other features even if some fail
            
            # 2. Volume-based features
            try:
                # Volume Moving Average
                for window in [5, 10, 20]:
                    if len(volume_series) >= window:
                        features[f'Volume_MA_{window}'] = volume_series.rolling(window=window).mean()
                
                # Volume Rate of Change
                for period in [1, 5, 20]:
                    if len(volume_series) > period:
                        features[f'Volume_ROC_{period}'] = volume_series.pct_change(periods=period) * 100
                
                # Volume Weighted Average Price
                if 'Volume' in result.columns and 'Close' in result.columns:
                    features['VWAP'] = (result['Volume'] * result['Close']).cumsum() / result['Volume'].cumsum()
                
                logger.info("Added volume-based features")
                
            except Exception as e:
                logger.error(f"Error creating volume-based features: {str(e)}", exc_info=True)
            
            # 3. Technical Indicators
            try:
                # RSI (Relative Strength Index)
                delta = close_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD and EMAs
                ema12 = close_series.ewm(span=12, adjust=False).mean()
                ema26 = close_series.ewm(span=26, adjust=False).mean()
                features['EMA_12'] = ema12  # Add EMA_12 to features
                features['EMA_26'] = ema26  # Add EMA_26 to features
                features['MACD'] = ema12 - ema26
                features['MACD_signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
                features['MACD_hist'] = features['MACD'] - features['MACD_signal']
                
                # Stochastic Oscillator
                low_min = low_series.rolling(window=14).min()
                high_max = high_series.rolling(window=14).max()
                features['%K'] = 100 * ((close_series - low_min) / (high_max - low_min))
                features['%D'] = features['%K'].rolling(window=3).mean()
                
                # ATR (Average True Range)
                tr1 = high_series - low_series
                tr2 = (high_series - close_series.shift()).abs()
                tr3 = (low_series - close_series.shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                features['ATR'] = tr.rolling(window=14).mean()
                
                logger.info("Added technical indicators")
                
            except Exception as e:
                logger.error(f"Error creating technical indicators: {str(e)}", exc_info=True)
            
            # 4. Volatility features
            try:
                # Historical Volatility
                for window in [5, 10, 20, 60]:
                    if len(close_series) > window:
                        log_returns = np.log(close_series / close_series.shift(1))
                        features[f'Volatility_{window}'] = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                
                # Average True Range (ATR) based volatility
                if 'ATR' in features:
                    features['ATR_Volatility'] = features['ATR'] / close_series * 100  # ATR as % of price
                
                logger.info("Added volatility features")
                
            except Exception as e:
                logger.error(f"Error creating volatility features: {str(e)}", exc_info=True)
            
            # 5. Combine all features into a single DataFrame
            try:
                # Convert features dictionary to DataFrame
                features_df = pd.DataFrame(features, index=data.index)
                
                # Add original data (except Volume which is already included in features)
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col not in features_df.columns:
                        features_df[col] = data[col]
                
                # Add day of week and month as features
                features_df['day_of_week'] = features_df.index.dayofweek
                features_df['month'] = features_df.index.month
                features_df['quarter'] = features_df.index.quarter
                
                # Add price change features
                features_df['daily_return'] = close_series.pct_change()
                features_df['log_return'] = np.log(close_series / close_series.shift(1))
                
                # Add moving average crossovers
                if 'SMA_50' in features_df.columns and 'SMA_200' in features_df.columns:
                    features_df['SMA_50_200_cross'] = np.where(
                        features_df['SMA_50'] > features_df['SMA_200'], 1, -1)
                
                # Add target variable (next day's return)
                features_df['target'] = features_df['Close'].pct_change().shift(-1)
                
                # Drop rows with NaN values that resulted from feature calculations
                features_df = features_df.dropna()
                
                logger.info(f"Created {len(features_df.columns)} features")
                
                return features_df
                
            except Exception as e:
                logger.error(f"Error combining features: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            logger.error(f"Error in _create_features: {str(e)}", exc_info=True)
            raise
            # Log warnings for features that cannot be computed
            # Initialize ta indicators
            try:
                ta_indicators = ta.add_all_ta_features(
                    result, 
                    open="Open", 
                    high="High", 
                    low="Low", 
                    close="Close", 
                    volume="Volume",
                    fillna=True
                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"ta.add_all_ta_features failed: {e}. Proceeding with basic features only.")
                ta_indicators = pd.DataFrame(index=result.index)
            # Add only indicators that are possible given the data length
            selected_indicators = [
                'volume_obv', 'volume_obvm', 'volume_fi', 'volume_mfi',
                'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbbli',
                'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
                'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',
                'momentum_rsi', 'momentum_mfi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
                'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind',
                'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff',
                'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',
                'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator'
            ]
            present_indicators = [col for col in selected_indicators if col in ta_indicators.columns and not ta_indicators[col].isna().all()]
            missing_indicators = [col for col in selected_indicators if col not in ta_indicators.columns or ta_indicators[col].isna().all()]
            logging.getLogger(__name__).info(f"Present technical indicators: {present_indicators}")
            if missing_indicators:
                logging.getLogger(__name__).warning(f"Skipped technical indicators (insufficient data or not computed): {missing_indicators}")
            # Add present indicators as new columns to result (preserve OHLCV)
            for col in present_indicators:
                result[col] = ta_indicators[col]
            # If no indicators could be computed, proceed with OHLCV and custom features only
            
            # Add basic moving averages
            result['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
            result['SMA_50'] = ta.trend.sma_indicator(close_series, window=50)
            result['SMA_200'] = ta.trend.sma_indicator(close_series, window=200)
            result['EMA_12'] = ta.trend.ema_indicator(close_series, window=12)
            result['EMA_26'] = ta.trend.ema_indicator(close_series, window=26)
            
            # Add any additional custom indicators not covered by add_all_ta_features
            result['HL_PCT'] = (high_series - low_series) / close_series * 100.0
            result['PCT_change'] = close_series.pct_change() * 100.0
            result['VWAP'] = (close_series * volume_series).cumsum() / volume_series.cumsum()
            # Compute Daily_Return after all NaN filling and cleaning to avoid loss of valid returns
            # Remove any old 'Daily_Return' column to avoid confusion
            if 'Daily_Return' in result.columns:
                result = result.drop(columns=['Daily_Return'])
            # After all filling/interpolation, recompute returns
            result['Daily_Return'] = result['Close'].pct_change()
            
            # Fill the first row's NaN in Daily_Return with 0 (no change)
            result['Daily_Return'] = result['Daily_Return'].fillna(0)
            
            # Calculate volatility, ensuring we have enough data points
            min_periods = min(5, len(result) - 1)  # Use at least 5 periods or available data - 1
            result['Volatility'] = result['Daily_Return'].rolling(
                window=21, min_periods=min_periods
            ).std() * np.sqrt(252)  # Annualized volatility
            
            # Forward fill any remaining NaN values in Volatility
            result['Volatility'] = result['Volatility'].fillna(method='ffill').fillna(0)
            
            # Log first few returns and count of non-NaN returns
            logging.getLogger(__name__).info(f"First 5 Daily_Return values after cleaning:\n{result[['Close', 'Daily_Return', 'Volatility']].head()}")
            logging.getLogger(__name__).info(f"Non-NaN Daily_Return count after cleaning: {result['Daily_Return'].notna().sum()}")
            logging.getLogger(__name__).info(f"NaN counts after return calculation: {result[['Close', 'Daily_Return', 'Volatility']].isna().sum()}")
            
            # Price to Moving Average Ratios
            result['Price_SMA20_Ratio'] = close_series / result['SMA_20']
            result['Price_SMA50_Ratio'] = close_series / result['SMA_50']
            result['Price_SMA200_Ratio'] = close_series / result['SMA_200']
            
            # Log shape before NaN handling
            logging.getLogger(__name__).info(f"Shape before NaN handling: {result.shape}")
            
            # Define essential columns that must have values (less aggressive: only 'Close' is truly essential)
            essential_cols = ['Close']
            
            # Count NaNs in each column for logging
            nan_counts = result.isna().sum()
            logging.getLogger(__name__).info(f"NaN counts before handling: {nan_counts}")
            logging.getLogger(__name__).info(f"Shape before cleaning: {result.shape}")
            logging.getLogger(__name__).info(f"First 5 rows before cleaning:\n{result.head()}")
            
            # First try to fill NaNs with forward/backward fill for time series continuity
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            # For any remaining NaNs in non-essential columns, use interpolation
            non_essential_cols = [col for col in result.columns if col not in essential_cols]
            if non_essential_cols:
                result[non_essential_cols] = result[non_essential_cols].interpolate(method='linear', limit_direction='both')
            
            # Only drop rows where essential columns still have NaNs
            initial_rows = len(result)
            result = result.dropna(subset=essential_cols)
            rows_dropped = initial_rows - len(result)

            if rows_dropped > 0:
                logging.getLogger(__name__).warning(f"Dropped {rows_dropped} rows with NaNs in essential columns (now only 'Close')")

            # Log final shape and NaN counts after cleaning
            logging.getLogger(__name__).info(f"Shape after NaN handling: {result.shape}")
            logging.getLogger(__name__).info(f"NaN counts after cleaning: {result.isna().sum()}")
            logging.getLogger(__name__).info(f"First 5 rows after cleaning:\n{result.head()}")

            if result.empty:
                raise ValueError("No valid data remaining after feature creation")
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error in _create_features: {str(e)}")

    def _clean_data(self, data):
        """
        Clean the data by handling missing values and outliers. This is the ONLY place where NA filling and dropping should occur.
        """
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data is None or empty")
        try:
            cleaned_data = data.copy()
            logging.getLogger(__name__).info(f"Shape before cleaning: {cleaned_data.shape}")
            nan_counts = cleaned_data.isna().sum()
            logging.getLogger(__name__).info(f"NaN counts before cleaning: {nan_counts}")
            essential_cols = ['Close'] if 'Close' in cleaned_data.columns else []
            if not essential_cols and not cleaned_data.empty:
                essential_cols = [cleaned_data.columns[0]]
            # Forward fill, then backward fill, then interpolate, then drop rows with NA in essential cols
            cleaned_data = cleaned_data.fillna(method='ffill')
            cleaned_data = cleaned_data.fillna(method='bfill')
            cleaned_data = cleaned_data.interpolate(method='linear', limit_direction='both')
            if not cleaned_data.empty and essential_cols:
                initial_rows = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=essential_cols)
                rows_dropped = initial_rows - len(cleaned_data)
                if rows_dropped > 0:
                    logging.getLogger(__name__).warning(f"Dropped {rows_dropped} rows with NaNs in essential columns: {essential_cols}")
            logging.getLogger(__name__).info(f"Shape after cleaning: {cleaned_data.shape}")
            final_nan_counts = cleaned_data.isna().sum()
            logging.getLogger(__name__).info(f"Final NaN counts: {final_nan_counts}")
            return cleaned_data
        except Exception as e:
            raise RuntimeError(f"Error cleaning data: {str(e)}")

    def _detect_outliers(self, data, threshold=3.0):
        """
        Detect and handle outliers using Z-score method. Only mark outliers as NaN; do not fill or interpolate here.
        NA handling is done in the final cleaning step.
        """
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data is None or empty")
        try:
            result = data.copy()
            if isinstance(result, pd.Series):
                result = result.to_frame()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return result
            for col in numeric_cols:
                col_data = result[col].values
                if len(col_data) < 2:
                    continue
                z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
                outlier_mask = np.isnan(z_scores) | (z_scores > threshold)
                if np.any(outlier_mask):
                    mask_array = np.array(outlier_mask)
                    result.loc[mask_array, col] = np.nan
            nan_counts = result.isna().sum()
            logging.getLogger(__name__).info(f"NaNs after outlier detection: {nan_counts}")
            return result
        except Exception as e:
            raise RuntimeError(f"Error detecting outliers: {str(e)}")

    def _scale_data(self, df):
        """
        Scale the data using multiple scalers and store the scalers
        
        Args:
            df: Input DataFrame containing the features to scale
            
        Returns:
            Tuple of (X, y) where:
                X: Numpy array of shape (n_sequences, sequence_length, n_features)
                y: Numpy array of shape (n_sequences,) containing target values
            
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If scaling fails
        """
        logger = logging.getLogger(__name__)
        
        # Early guard clause for robust input validation
        if df is None:
            error_msg = "Error in _scale_data: received None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not isinstance(df, pd.DataFrame) or df.empty:
            error_msg = "Error in _scale_data: received empty or invalid DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check for required columns
        required_columns = {'Close'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log basic info about the input data
        logger.info(f"Starting _scale_data with DataFrame of shape {df.shape}")
        logger.info(f"Columns in data: {df.columns.tolist()}")
        logger.info(f"First 5 rows of data:\n{df[list(required_columns)].head() if required_columns.issubset(df.columns) else 'Missing required columns'}")
        
        # Check for NaN values in required columns
        nan_counts = df[list(required_columns)].isna().sum()
        logger.info(f"NaN counts in required columns: {nan_counts.to_dict()}")
        
        # Check if we have enough data after accounting for NaN values
        min_required_samples = self.sequence_length + 10  # buffer for sequence creation
        if len(df) - nan_counts.max() < min_required_samples:
            error_msg = (
                f"Insufficient valid data points. Need at least {min_required_samples} non-NaN values, "
                f"but only have {len(df) - nan_counts.max()} after accounting for {nan_counts.max()} NaNs"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Handle NaN values before scaling
            data = df.copy()
            
            # Log initial data info
            logger.info(f"Data shape before cleaning: {data.shape}")
            logger.info(f"NaN counts before cleaning:\n{data.isna().sum()}")
            
            # Track original number of rows
            initial_row_count = len(data)
            
            # Forward fill then backfill to handle NaNs
            data = data.ffill().bfill()
            
            # If we still have NaNs after filling, drop those rows
            nan_rows = data.isnull().any(axis=1)
            if nan_rows.any():
                dropped_count = nan_rows.sum()
                logger.warning(f"Dropping {dropped_count} rows with remaining NaN values after filling")
                data = data.dropna()
                
                # Check if we still have enough data after dropping NaNs
                remaining_samples = len(data)
                if remaining_samples < min_required_samples:
                    error_msg = (
                        f"Insufficient data after dropping NaNs. Need at least {min_required_samples} samples, "
                        f"but only have {remaining_samples} after dropping {dropped_count} rows with NaNs"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Log data after cleaning
            logger.info(f"Data shape after cleaning: {data.shape} (dropped {initial_row_count - len(data)} rows)")
            logger.info(f"NaN counts after cleaning:\n{data.isna().sum()}")
            
            # Ensure we have numeric data
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            if not non_numeric_cols.empty:
                logger.warning(f"Converting non-numeric columns to numeric: {non_numeric_cols.tolist()}")
                error_msg = (
                    f"Insufficient data after dropping NaNs. Need at least {min_required_samples} samples, "
                    f"but only have {remaining_samples} after dropping {dropped_count} rows with NaNs"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Define default features if not set
            if not hasattr(self, 'features') or not self.features:
                # Start with basic price and volume features
                self.features = [
                    'Close', 'Volume', 'Daily_Return', 'Volatility',
                    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                    'MACD', 'MACD_Signal', 'RSI', 'ATR', 'ADX', 'CCI',
                    'Day_of_Week', 'Month', 'Quarter'
                ]
                logger.info(f"Using default features: {self.features}")
            
            # Ensure all features exist in data and are numeric
            available_features = []
            for feature in self.features:
                if feature not in data.columns:
                    logger.warning(f"Feature '{feature}' not found in data")
                    continue
                    
                # Skip non-numeric columns
                if not np.issubdtype(data[feature].dtype, np.number):
                    logger.warning(f"Skipping non-numeric feature: {feature}")
                    continue
                    
                available_features.append(feature)
            
            if not available_features:
                error_msg = "No valid numeric features available for scaling"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if len(available_features) < len(self.features):
                missing = set(self.features) - set(available_features)
                logger.warning(f"Dropping {len(missing)} features that are missing or non-numeric")
                self.features = available_features
                
            logger.info(f"[DIAGNOSTIC] Final features for scaling: {self.features}")
            logger.info(f"[DIAGNOSTIC] Data types of selected features:\n{data[self.features].dtypes}")
            
            # Prepare data for scaling
            try:
                scaled_data = np.zeros((len(data), len(self.features)), dtype=np.float32)
                
                # Initialize feature scalers if not already done
                if not hasattr(self, 'feature_scalers'):
                    self.feature_scalers = {}
                
                # Scale each feature
                for i, feature in enumerate(self.features):
                    if feature not in data.columns:
                        logging.getLogger(__name__).warning(f"Feature '{feature}' not found in data, skipping")
                        continue
                        
                    # Get the feature values
                    feature_values = data[feature].values.reshape(-1, 1)
                    
                    # Initialize scaler for this feature if it doesn't exist
                    if feature not in self.feature_scalers:
                        if feature in ['RSI', 'MACD', 'MACD_Signal', 'CCI']:
                            # Use MinMaxScaler for bounded indicators
                            self.feature_scalers[feature] = MinMaxScaler(feature_range=(-1, 1))
                        else:
                            # Use RobustScaler for most features to handle outliers
                            self.feature_scalers[feature] = RobustScaler()
                    
                    try:
                        # Scale the feature values
                        scaled_values = self.feature_scalers[feature].fit_transform(feature_values)
                        scaled_data[:, i] = scaled_values.flatten()
                    except Exception as e:
                        logging.getLogger(__name__).error(f"Error scaling feature '{feature}': {str(e)}")
                        # Fill with zeros if scaling fails
                        scaled_data[:, i] = 0
                
                # Log shape and NaN counts before sequence creation
                logger.info(f"[DIAGNOSTIC] Data shape before sequence creation: {scaled_data.shape}")
                logger.info(f"[DIAGNOSTIC] Sample values before sequence creation:\n{scaled_data[:5]} ...")
                # Create sequences
                X, y = self._create_sequences(scaled_data)
                # Log shape of created sequences
                logger.info(f"[DIAGNOSTIC] Created X shape: {X.shape}, y shape: {y.shape}")
                
                if X is None or len(X) == 0:
                    logging.getLogger(__name__).error("Failed to create sequences from scaled data")
                    return np.zeros((0, self.sequence_length, len(self.features))), np.array([])
                
                return X, y
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in _scale_data: {str(e)}")
                return np.zeros((0, self.sequence_length, len(self.features))), np.array([])
        
        except Exception as e:
            logging.getLogger(__name__).error(f"Unexpected error in _scale_data: {str(e)}")
            return np.zeros((0, self.sequence_length, len(self.features) if hasattr(self, 'features') and self.features else 1)), np.array([])
            try:
                X, y = self._create_sequences(scaled_data)
                
                # Validate array shapes
                if X is None or not isinstance(X, np.ndarray) or X.size == 0 or X.shape[0] == 0:
                    logging.getLogger(__name__).warning(
                        f"No valid sequences created. Required sequence length: {self.sequence_length}, "
                        f"Data length: {len(data)}"
                    )
                    # Return empty arrays with proper shape instead of raising an error
                    empty_X = np.zeros((0, self.sequence_length, len(self.features)))
                    empty_y = np.array([])
                    return empty_X, empty_y
                
                return X, y
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in sequence creation: {str(e)}")
                empty_X = np.zeros((0, self.sequence_length, len(self.features)))
                empty_y = np.array([])
                return empty_X, empty_y
                
        except Exception as e:
            # Log the error
            logging.getLogger(__name__).error(f"Error in _scale_data: {str(e)}")

            # Store error in Streamlit session state if available
            try:
                import streamlit as st
                st.session_state['pipeline_error'] = f"Error in _scale_data: {str(e)}"
            except Exception:
                pass

            # Return empty arrays instead of raising an exception
            # Define features if not already defined
            if not hasattr(self, 'features') or not self.features:
                self.features = ['Close', 'Volume']
            empty_X = np.zeros((0, self.sequence_length, len(self.features)))
            empty_y = np.array([])
            return empty_X, empty_y
            # This return is already handled in the except block above
            return empty_X, empty_y
            
        # This code was unreachable and has been moved to _create_additional_features
        return X, y
        
    def _create_additional_features(self, X):
        """
        Create additional features from the input sequences
        
        Args:
            X: Input sequences of shape (n_sequences, sequence_length, n_features)
            
        Returns:
            X with additional features
        """
        if X is None or X.size == 0 or X.shape[0] == 0:
            logging.getLogger(__name__).warning("Cannot create additional features for empty array")
            return X
            
        try:
            # Calculate feature combinations
            X_combined = np.zeros_like(X)
            
            # Calculate rolling statistics
            for i in range(X.shape[0]):
                # Calculate rolling mean and std for each feature
                for j in range(X.shape[2]):
                    X_combined[i, :, j] = np.mean(X[i, :, j])
                    # Make sure we don't go out of bounds
                    if j+1 < X.shape[2]:
                        X_combined[i, :, j+1] = np.std(X[i, :, j])
                    
                # Calculate correlations between features if we have enough features
                if X.shape[2] > 1 and hasattr(self, 'features') and self.features:
                    try:
                        corr_matrix = np.corrcoef(X[i, :, :].T)
                        # Make sure we don't go out of bounds
                        feature_len = min(len(self.features), X_combined.shape[2])
                        if corr_matrix.size > 0:
                            flat_corr = corr_matrix.reshape(-1)
                            # Only copy as many elements as will fit
                            copy_len = min(flat_corr.size, feature_len)
                            X_combined[i, :, -copy_len:] = flat_corr[:copy_len]
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Error calculating correlation matrix: {str(e)}")
            
            # Combine original features with new features
            X = np.concatenate([X, X_combined], axis=2)
            
            return X
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in _create_additional_features: {str(e)}")
            # Return the original X if there's an error
            return X

    def inverse_transform(self, predictions):
        """
        Inverse transform predictions back to original scale
        """
        if 'Close' not in self.scalers:
            raise ValueError("Scaler not found for Close price")
            
        return self.scalers['Close'].inverse_transform(predictions.reshape(-1, 1)).reshape(-1)

    def _create_time_features(self, n_samples):
        """
        Create time-based features for each sequence
        """
        # Validate input
        if n_samples <= 0:
            logging.getLogger(__name__).warning(f"Cannot create time features for {n_samples} samples")
            return np.array([])
            
        # Create time features
        time_features = np.zeros((n_samples, self.sequence_length, 2))
        
        # Add time index
        time_features[:, :, 0] = np.arange(self.sequence_length)
        
        # Add cyclical time features
        time_features[:, :, 1] = np.sin(2 * np.pi * np.arange(self.sequence_length) / self.sequence_length)
        
        return time_features

    def _create_sequences(self, scaled_data):
        """
        Create sequences for LSTM input with robust error handling
        
        Args:
            scaled_data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (X, y) where:
                X: Array of shape (n_sequences, sequence_length, n_features + time_features)
                y: Array of shape (n_sequences,) with target values
        """
        logger = logging.getLogger(__name__)
        
        # Validate input data
        if scaled_data is None or not isinstance(scaled_data, np.ndarray) or scaled_data.size == 0:
            msg = "Cannot create sequences: input data is None, not a numpy array, or empty"
            logger.warning(msg)
            # Try to provide more context for debugging
            if scaled_data is not None:
                logger.warning(f"Input data type: {type(scaled_data)}, size: {getattr(scaled_data, 'size', 'unknown')}")
            return np.array([]), np.array([])
            
        # Ensure we have a 2D array
        if len(scaled_data.shape) != 2:
            msg = f"Expected 2D input data, got shape {scaled_data.shape}"
            logger.warning(msg)
            return np.array([]), np.array([])
            
        n_samples, n_features = scaled_data.shape
        logger.info(f"Creating sequences from {n_samples} samples with {n_features} features")
        
        # Provide more detailed information about data requirements
        logger.info(f"Current sequence length: {self.sequence_length}, minimum required samples: {self.sequence_length + 1}")
        
        # Dynamically adjust sequence length if we don't have enough data
        min_required = self.sequence_length + 1
        if n_samples < min_required:
            # Try to reduce sequence length to fit available data
            new_seq_length = max(5, n_samples - 1)  # Minimum sequence length of 5
            if new_seq_length < 5:
                msg = f"Insufficient data points. Need at least 6, got {n_samples}. Cannot create valid sequences."
                logger.error(msg)
                try:
                    import streamlit as st
                    st.session_state['pipeline_error'] = msg
                except ImportError:
                    pass
                return np.array([]), np.array([])
                
            msg = (
                f"Reducing sequence length from {self.sequence_length} to {new_seq_length} "
                f"to fit available data points ({n_samples})"
            )
            logger.warning(msg)
            try:
                import streamlit as st
                st.session_state['pipeline_warning'] = msg
            except ImportError:
                pass
            
            # Store original sequence length to restore later
            original_seq_length = self.sequence_length
            self.sequence_length = new_seq_length
            
            try:
                X, y = self._create_sequences_with_length(scaled_data)
                # Log the outcome
                if X.size > 0 and y.size > 0:
                    logger.info(f"Successfully created {len(X)} sequences with reduced length {new_seq_length}")
                else:
                    logger.error(f"Failed to create sequences even with reduced length {new_seq_length}")
                return X, y
            finally:
                # Restore original sequence length
                self.sequence_length = original_seq_length
        else:
            X, y = self._create_sequences_with_length(scaled_data)
            # Log the outcome
            if X.size > 0 and y.size > 0:
                logger.info(f"Successfully created {len(X)} sequences with standard length {self.sequence_length}")
            else:
                logger.error(f"Failed to create sequences with standard length {self.sequence_length}")
            return X, y
            
    def _create_sequences_with_length(self, scaled_data):
        """Helper method to create sequences with the current sequence length"""
        logger = logging.getLogger(__name__)
        n_samples, n_features = scaled_data.shape
        
        # Log detailed information about sequence creation attempt
        logger.info(f"Attempting to create sequences with length {self.sequence_length} from {n_samples} samples")
        expected_sequences = max(0, n_samples - self.sequence_length)
        logger.info(f"Expected to create {expected_sequences} sequences if successful")
        
        try:
            # Check if we can create any sequences
            if n_samples <= self.sequence_length:
                msg = f"Cannot create sequences: need more than {self.sequence_length} samples, got {n_samples}"
                logger.error(msg)
                try:
                    import streamlit as st
                    st.session_state['pipeline_error'] = msg
                except ImportError:
                    pass
                return np.array([]), np.array([])
            
            # Create sequences
            X = np.zeros((n_samples - self.sequence_length, self.sequence_length, n_features))
            y = np.zeros(n_samples - self.sequence_length)
            
            for i in range(self.sequence_length, n_samples):
                X[i - self.sequence_length] = scaled_data[i-self.sequence_length:i]
                y[i - self.sequence_length] = scaled_data[i, 0]  # Assuming first column is target
            
            # Validate created sequences
            if X.size == 0 or y.size == 0:
                msg = "Created empty sequence arrays despite sufficient data points"
                logger.error(msg)
                try:
                    import streamlit as st
                    st.session_state['pipeline_error'] = msg
                except ImportError:
                    pass
                return np.array([]), np.array([])
                
            # Add time features if we have valid sequences
            try:
                time_features = self._create_time_features(X.shape[0])
                
                # Only concatenate if we have valid time features
                if (time_features is not None and 
                    time_features.size > 0 and 
                    time_features.shape[0] == X.shape[0] and 
                    time_features.shape[1] == X.shape[1]):
                    
                    X = np.concatenate([X, time_features], axis=2)
                else:
                    msg = (
                        f"Skipping time features concatenation due to shape mismatch or empty features. "
                        f"X shape: {X.shape}, time_features shape: {time_features.shape if time_features is not None else 'None'}"
                    )
                    logger.debug(msg)
                    
            except Exception as e:
                msg = f"Error creating or adding time features: {str(e)}"
                logger.error(msg, exc_info=True)
                # Continue without time features
                
            logger.info(f"Successfully created {len(X)} sequences of length {self.sequence_length}")
            return X, y
            
        except Exception as e:
            msg = f"Error in _create_sequences: {str(e)}"
            logger.error(msg, exc_info=True)
            return np.array([]), np.array([])
            
        try:
            # Create sequences
            X = np.zeros((n_samples - self.sequence_length, self.sequence_length, n_features))
            y = np.zeros(n_samples - self.sequence_length)
            
            for i in range(self.sequence_length, n_samples):
                X[i - self.sequence_length] = scaled_data[i-self.sequence_length:i]
                y[i - self.sequence_length] = scaled_data[i, 0]  # Assuming first column is target
            
            # Validate created sequences
            if X.size == 0 or y.size == 0:
                msg = "Created empty sequence arrays"
                logging.getLogger(__name__).warning(msg)
                return np.array([]), np.array([])
                
            # Add time features if we have valid sequences
            try:
                time_features = self._create_time_features(X.shape[0])
                
                # Only concatenate if we have valid time features
                if (time_features is not None and 
                    time_features.size > 0 and 
                    time_features.shape[0] == X.shape[0] and 
                    time_features.shape[1] == X.shape[1]):
                    
                    X = np.concatenate([X, time_features], axis=2)
                else:
                    msg = (
                        f"Skipping time features concatenation due to shape mismatch or empty features. "
                        f"X shape: {X.shape}, time_features shape: {time_features.shape if time_features is not None else 'None'}"
                    )
                    logging.getLogger(__name__).debug(msg)
                    
            except Exception as e:
                msg = f"Error creating or adding time features: {str(e)}"
                logging.getLogger(__name__).error(msg, exc_info=True)
                # Continue without time features
                
            return X, y
            
        except Exception as e:
            msg = f"Error in _create_sequences: {str(e)}"
            logging.getLogger(__name__).error(msg, exc_info=True)
            return np.array([]), np.array([])

    def preprocess(self, data):
        """
        Full preprocessing pipeline with robust error propagation and Streamlit error state setting.
        
        Args:
            data: Input DataFrame containing stock data
            
        Returns:
            dict: Dictionary containing processed data with keys:
                - 'stock_data': Processed DataFrame
                - 'features': Numpy array of features for model input
                - 'returns': Numpy array of target values
            
        Raises:
            ValueError: If input data is invalid or processing fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            import streamlit as st
            has_streamlit = True
        except ImportError:
            has_streamlit = False
            
        def set_error(msg, exc_info=None):
            """Helper to set error state consistently"""
            logger.error(msg, exc_info=exc_info)
            if has_streamlit:
                st.session_state['pipeline_error'] = msg
            return ValueError(msg)
            
        def set_warning(msg):
            """Helper to set warning state consistently"""
            logger.warning(msg)
            if has_streamlit:
                st.session_state['pipeline_warning'] = msg
            return None
            
        # Initialize return values
        result = {
            'stock_data': None,
            'features': np.zeros((0, self.sequence_length, 1)),  # Empty array with proper shape
            'returns': np.array([])
        }
        
        try:
            # 1. Input validation
            if data is None or (hasattr(data, 'empty') and data.empty):
                raise set_error("Input data is None or empty at start of preprocessing.")
                
            logger.info(f"Starting preprocessing pipeline. Initial data shape: {data.shape}")
            logger.debug(f"Initial columns: {list(data.columns)}")
            
            # 2. Feature creation
            try:
                logger.info("Creating features...")
                data = self._create_features(data)
                if data is None or (hasattr(data, 'empty') and data.empty):
                    raise set_error("No data returned after feature creation.")
                logger.info(f"After feature creation: {data.shape} rows")
                
            except Exception as e:
                raise set_error(f"Error in feature creation: {str(e)}", exc_info=True)
            
            # 3. Data cleaning
            try:
                logger.info("Cleaning data...")
                data = self._clean_data(data)
                if data is None or (hasattr(data, 'empty') and data.empty):
                    raise set_error("No data returned after cleaning.")
                logger.info(f"After cleaning: {data.shape} rows")
                
            except Exception as e:
                raise set_error(f"Error in data cleaning: {str(e)}", exc_info=True)
            
            # 4. Outlier detection
            try:
                logger.info("Detecting outliers...")
                data = self._detect_outliers(data)
                if data is None or (hasattr(data, 'empty') and data.empty):
                    raise set_error("No data returned after outlier detection.")
                logger.info(f"After outlier detection: {data.shape} rows")
                
            except Exception as e:
                set_warning(f"Warning during outlier detection: {str(e)}. Continuing with original data.")
                # Continue with the data we have
                
            # 5. Feature scaling and sequence creation
            try:
                logger.info("Scaling data and creating sequences...")
                X, y = self._scale_data(data)
                
                # Validate scaled data
                if X is None or y is None or X.size == 0 or y.size == 0:
                    set_warning("No valid sequences could be created from the data. This may be due to insufficient data points or data quality issues.")
                else:
                    logger.info(f"Created {len(X)} sequences of length {X.shape[1]} with {X.shape[2]} features")
                    
                    # 6. Add additional features if we have valid sequences
                    try:
                        X = self._create_additional_features(X)
                        if X is None or X.size == 0:
                            set_warning("No data after creating additional features.")
                        else:
                            logger.info(f"After additional features: {X.shape}")
                    except Exception as e:
                        set_warning(f"Warning during additional feature creation: {str(e)}. Using base features only.")
                        # Continue with the features we have
                
                # Create x_stock and x_additional components for TwoStagePredictor
                if X is not None and X.shape[2] >= 1:
                    # Extract stock data (first 3-4 features)
                    feature_count = min(4, X.shape[2])
                    x_stock = X[:, :, :feature_count]
                    
                    # Extract additional features (remaining features)
                    if X.shape[2] > feature_count:
                        x_additional = X[:, :, feature_count:]
                    else:
                        # Create empty additional features if none exist
                        x_additional = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.float32)
                else:
                    # Default empty arrays if X is None
                    empty_shape = (0, self.sequence_length, 1)
                    x_stock = np.zeros(empty_shape)
                    x_additional = np.zeros(empty_shape)

                # Update result with processed data
                result.update({
                    'stock_data': data,
                    'features': X if X is not None else np.zeros((0, self.sequence_length, len(self.features) if hasattr(self, 'features') and self.features else 1)),
                    'returns': y if y is not None else np.array([]),
                    # Add X_train as a tuple of (x_stock, x_additional) for TwoStagePredictor
                    'X_train': (x_stock, x_additional),
                    'y_train': y if y is not None else np.array([]),
                    # Add X_test as a tuple of (x_stock, x_additional) for TwoStagePredictor
                    # Using the last 20% of the data as test data if available
                    'X_test': (x_stock[-int(x_stock.shape[0]*0.2):] if x_stock.shape[0] > 0 else x_stock,
                              x_additional[-int(x_additional.shape[0]*0.2):] if x_additional.shape[0] > 0 else x_additional),
                    'y_test': y[-int(y.shape[0]*0.2):] if y is not None and y.shape[0] > 0 else np.array([])
                })
                
            except Exception as e:
                raise set_error(f"Error during scaling and sequence creation: {str(e)}", exc_info=True)
            
            # 7. Final validation
            if result['features'].size == 0 or result['returns'].size == 0:
                set_warning("Warning: No valid sequences could be created. This may be due to insufficient data points or data quality issues.")
            
            logger.info("Preprocessing completed successfully")
            return result
            
        except ValueError as ve:
            # Already handled by set_error
            logger.error("Preprocessing failed with validation error")
            if has_streamlit:
                st.error(f"Data processing error: {str(ve)}")
            return result
            
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error during preprocessing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if has_streamlit:
                st.session_state['pipeline_error'] = error_msg
                st.error("An unexpected error occurred during data processing. Please check the logs for details.")
            return result
    def _handle_temporal_alignment(self, data):
        """
        Handle temporal alignment by accounting for non-trading days
        """
        # Fill gaps in trading days
        all_dates = pd.date_range(start=data.index.min(), end=data.index.max())
        data = data.reindex(all_dates)
        
        # Handle weekends and holidays
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')
        
        return data



    def _create_technical_indicators(self, data):
        """
        Add advanced technical indicators to the data with robust error handling
        and dynamic window sizing based on available data.
        
        Args:
            data: DataFrame containing stock price data with at least 'Close' column
            
        Returns:
            DataFrame: Input data with added technical indicators
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            logger.error("Cannot create technical indicators: No data provided")
            return data
            
        if 'Close' not in data.columns:
            logger.error("Cannot create technical indicators: 'Close' column missing")
            return data
            
        data = data.copy()
        created_indicators = []
        
        # Calculate available data points
        n_points = len(data)
        logger.info(f"Generating technical indicators for {n_points} data points")
        
        def safe_rolling(series, window, min_window=5, **kwargs):
            """Helper function to safely calculate rolling statistics"""
            if len(series) < min_window:
                return pd.Series(np.nan, index=series.index)
            # Use the smaller of the requested window or available data points
            window = min(window, len(series) - 1)  # -1 to ensure at least 2 points for calculations
            if window < min_window:
                return pd.Series(np.nan, index=series.index)
            return series.rolling(window=window, **kwargs)
        
        # Calculate Moving Averages with dynamic window sizing
        try:
            windows = [20, 50, 200]
            for window in windows:
                if n_points >= window:  # Only calculate if we have enough data
                    col_name = f'SMA_{window}'
                    data[col_name] = safe_rolling(data['Close'], window).mean()
                    created_indicators.append(col_name)
                    logger.debug(f"Created {col_name} indicator")
                else:
                    logger.warning(f"Skipping SMA_{window}: Not enough data points ({n_points} < {window})")
        except Exception as e:
            logger.error(f"Error creating SMA indicators: {str(e)}")
            logger.exception("SMA calculation error")
        
        # Calculate RSI with multiple periods
        try:
            for period in [7, 14, 21]:
                if n_points > period:  # Need at least period+1 points for RSI
                    delta = data['Close'].diff()
                    gain = safe_rolling(delta.where(delta > 0, 0), period).mean()
                    loss = safe_rolling(-delta.where(delta < 0, 0), period).mean()
                    
                    # Handle division by zero
                    rs = gain / loss.replace(0, np.finfo(float).eps)
                    data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                    created_indicators.append(f'RSI_{period}')
                    logger.debug(f"Created RSI_{period} indicator")
                else:
                    logger.warning(f"Skipping RSI_{period}: Not enough data points ({n_points} < {period + 1})")
        except Exception as e:
            logger.error(f"Error creating RSI indicators: {str(e)}")
            logger.exception("RSI calculation error")
        
        # Calculate MACD with multiple configurations
        try:
            for fast, slow, signal in [(12, 26, 9), (8, 17, 5), (15, 30, 12)]:
                if n_points >= max(fast, slow, signal):
                    exp1 = data['Close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
                    exp2 = data['Close'].ewm(span=slow, adjust=False, min_periods=slow).mean()
                    macd = exp1 - exp2
                    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
                    
                    data[f'MACD_{fast}_{slow}_{signal}'] = macd
                    created_indicators.append(f'MACD_{fast}_{slow}_{signal}')
                    data[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
                    created_indicators.append(f'MACD_Signal_{fast}_{slow}_{signal}')
                    data[f'MACD_Hist_{fast}_{slow}_{signal}'] = macd - signal_line
                    created_indicators.append(f'MACD_Hist_{fast}_{slow}_{signal}')
                    logger.debug(f"Created MACD {fast}_{slow}_{signal} indicators")
                else:
                    logger.warning(f"Skipping MACD {fast}_{slow}_{signal}: Not enough data points")
        except Exception as e:
            logger.error(f"Error creating MACD indicators: {str(e)}")
            logger.exception("MACD calculation error")
        
        # Calculate Bollinger Bands
        try:
            if n_points >= 20:  # Minimum window for BB
                bb_window = min(20, n_points - 1)
                data['BB_Middle_20'] = safe_rolling(data['Close'], bb_window).mean()
                std = safe_rolling(data['Close'], bb_window).std()
                data['BB_Upper_20'] = data['BB_Middle_20'] + 2 * std
                data['BB_Lower_20'] = data['BB_Middle_20'] - 2 * std
                created_indicators.extend(['BB_Middle_20', 'BB_Upper_20', 'BB_Lower_20'])
                logger.debug("Created Bollinger Bands indicators")
            else:
                logger.warning(f"Skipping Bollinger Bands: Not enough data points ({n_points} < 20)")
        except Exception as e:
            logger.error(f"Error creating Bollinger Bands: {str(e)}")
            logger.exception("Bollinger Bands calculation error")
        
        # Calculate Volume Indicators if Volume data is available
        try:
            if 'Volume' in data.columns and n_points > 1:
                volume_diff = data['Volume'].diff().fillna(0)
                price_diff = data['Close'].diff().fillna(0)
                obv = (np.sign(price_diff) * volume_diff).cumsum()
                data['OBV'] = obv
                created_indicators.append('OBV')
                logger.debug("Created OBV indicator")
                
                # Add Volume MA
                if n_points >= 20:
                    data['Volume_MA_20'] = safe_rolling(data['Volume'], 20).mean()
                    created_indicators.append('Volume_MA_20')
            else:
                logger.warning("Volume data not available or insufficient, skipping volume indicators")
        except Exception as e:
            logger.error(f"Error creating volume indicators: {str(e)}")
            logger.exception("Volume indicators calculation error")
        
        # Calculate Volatility
        try:
            if n_points >= 10:
                returns = data['Close'].pct_change()
                data['Volatility_10'] = safe_rolling(returns, 10).std() * np.sqrt(252)  # Annualized
                created_indicators.append('Volatility_10')
                
                if n_points >= 20:
                    data['Volatility_20'] = safe_rolling(returns, 20).std() * np.sqrt(252)  # Annualized
                    created_indicators.append('Volatility_20')
                logger.debug("Created Volatility indicators")
            else:
                logger.warning(f"Skipping Volatility: Not enough data points ({n_points} < 10)")
        except Exception as e:
            logger.error(f"Error creating Volatility indicators: {str(e)}")
            logger.exception("Volatility calculation error")
        
        # Calculate Momentum
        try:
            for period in [5, 10, 20]:
                if n_points > period:
                    data[f'Momentum_{period}'] = data['Close'].pct_change(periods=period)
                    created_indicators.append(f'Momentum_{period}')
                    logger.debug(f"Created Momentum_{period} indicator")
                else:
                    logger.warning(f"Skipping Momentum_{period}: Not enough data points")
        except Exception as e:
            logger.error(f"Error creating Momentum indicators: {str(e)}")
            logger.exception("Momentum calculation error")
        
        # Update features list with only successfully created indicators that exist in the DataFrame
        self.features = ['Close']  # Always include Close price as a feature
        
        # Define feature groups and their prefixes
        feature_groups = [
            ('SMA_', ['SMA_20', 'SMA_50', 'SMA_200']),  # Moving Averages
            ('RSI_', [f'RSI_{p}' for p in [7, 14, 21]]),  # RSI indicators
            ('MACD_', [f'MACD_{f}_{s}_{sig}' for f, s, sig in [(12, 26, 9), (8, 17, 5), (15, 30, 12)]]),
            ('MACD_Signal_', [f'MACD_Signal_{f}_{s}_{sig}' for f, s, sig in [(12, 26, 9), (8, 17, 5), (15, 30, 12)]]),
            ('MACD_Hist_', [f'MACD_Hist_{f}_{s}_{sig}' for f, s, sig in [(12, 26, 9), (8, 17, 5), (15, 30, 12)]]),
            ('BB_', ['BB_Middle_20', 'BB_Upper_20', 'BB_Lower_20']),  # Bollinger Bands
            ('OBV', ['OBV']),  # On-Balance Volume
            ('Volume_MA_', ['Volume_MA_20']),  # Volume Moving Average
            ('Volatility_', ['Volatility_10', 'Volatility_20']),  # Volatility
            ('Momentum_', [f'Momentum_{p}' for p in [5, 10, 20]])  # Momentum
        ]
        
        # Add features that exist in the DataFrame
        for prefix, features in feature_groups:
            for feature in features:
                if feature in data.columns and feature not in self.features:
                    self.features.append(feature)
        
        # Log the final feature set
        logger.info(f"Created {len(self.features)} technical indicators: {', '.join(self.features)}")
        
        # Return the data with all indicators
        return data
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data

    def _handle_temporal_alignment(self, data):
        """
        Handle temporal alignment by accounting for non-trading days
        """
        # Fill gaps in trading days
        all_dates = pd.date_range(start=data.index.min(), end=data.index.max())
        data = data.reindex(all_dates)
        
        # Handle weekends and holidays
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')
        
        return data

    def _create_sequences(self, scaled_data):
        """
        Create sequences for LSTM with advanced handling
        """
        X = []
        y = []
        
        # Create sequences with overlapping windows
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predicting Close price
        
        X = np.array(X)
        y = np.array(y)
        
        # Add time-based features
        time_features = self._create_time_features(X.shape[0])
        
        # Combine time features with existing sequences
        X = np.concatenate([X, time_features], axis=2)
        
        return X, y

    def _create_time_features(self, n_samples):
        """
        Create time-based features for each sequence
        """        # Create time features
        time_features = np.zeros((n_samples, self.sequence_length, 2))
        
        # Add time index
        time_features[:, :, 0] = np.arange(self.sequence_length)
        
        # Add cyclical time features
        time_features[:, :, 1] = np.sin(2 * np.pi * np.arange(self.sequence_length) / self.sequence_length)
        
        return time_features

    def _scale_data(self, df, sequence_length=60, features=None):
        """
        Scale the data using multiple scalers and store the scalers with robust error handling.
        
        Args:
            df: Input DataFrame containing the features to scale
            sequence_length: Length of the sequence to create
            features: List of feature columns to use (defaults to ['Close'])
            
        Returns:
            Tuple of (X, y) where:
                X: Numpy array of shape (n_sequences, sequence_length, n_features)
                y: Numpy array of shape (n_sequences,) containing target values
                
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If scaling fails
        """
        logger = logging.getLogger(__name__)
        
        # 1. Input Validation
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            error_msg = "Input data is None or empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 2. Set default features if not provided
        if features is None:
            features = ['Close']
            logger.info("No features specified, using default: ['Close']")
        
        # 3. Ensure Close is always included if not already in features
        if 'Close' not in features:
            features = ['Close'] + [f for f in features if f != 'Close']
            logger.info(f"Added 'Close' to features: {features}")
            
        # 4. Check for required columns
        missing_columns = [f for f in features if f not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 5. Handle missing values
        data = df[features].copy()
        initial_row_count = len(data)
        
        # Forward fill then backfill to handle NaNs
        data = data.ffill().bfill()
        
        # If we still have NaNs after filling, drop those rows
        nan_rows = data.isnull().any(axis=1)
        if nan_rows.any():
            dropped_count = nan_rows.sum()
            logger.warning(f"Dropping {dropped_count} rows with NaN values after filling")
            data = data.dropna()
            if data.empty:
                error_msg = "No valid data remaining after dropping NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        # 6. Check for sufficient data after cleaning
        min_required_points = sequence_length + 1
        if len(data) < min_required_points:
            error_msg = (
                f"Insufficient data points. Need at least {min_required_points} points "
                f"for sequence length {sequence_length}, but only have {len(data)} after cleaning"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 7. Initialize feature scalers if not already done
        if not hasattr(self, 'feature_scalers'):
            self.feature_scalers = {}
            
        # 8. Scale each feature with proper error handling
        scaled_features = []
        valid_features = []
        
        for feature in features:
            try:
                # Get feature values
                values = data[feature].values.reshape(-1, 1)
                
                # Skip if all values are the same (causes issues with scaling)
                if np.all(values == values[0]):
                    logger.warning(f"Feature '{feature}' has constant values, skipping")
                    continue
                    
                # Initialize scaler for this feature if it doesn't exist
                if feature not in self.feature_scalers:
                    # Choose scaler based on feature characteristics
                    if feature in ['RSI', 'Momentum', 'ATR', 'ADX', 'CCI', 'MACD', 'MACD_Signal']:
                        self.feature_scalers[feature] = MinMaxScaler(feature_range=(-1, 1))
                    elif feature in ['Volume', 'Daily_Return']:
                        self.feature_scalers[feature] = RobustScaler()
                    else:
                        self.feature_scalers[feature] = StandardScaler()
                
                # Fit or transform the scaler
                if not hasattr(self.feature_scalers[feature], 'n_features_in_'):
                    scaled_values = self.feature_scalers[feature].fit_transform(values)
                else:
                    scaled_values = self.feature_scalers[feature].transform(values)
                
                scaled_features.append(scaled_values)
                valid_features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error scaling feature '{feature}': {str(e)}")
                continue
        
        # 9. Check if we have any valid features left
        if not scaled_features:
            error_msg = "No valid features could be scaled"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 10. Combine all scaled features
        try:
            scaled_data = np.hstack(scaled_features)
            logger.info(f"Successfully scaled {len(valid_features)} features: {valid_features}")
            
            # 11. Create sequences
            X, y = self._create_sequences(scaled_data)
            
            if X is None or X.size == 0 or y is None or y.size == 0:
                raise ValueError("No valid sequences could be created from the scaled data")
                
            return X, y
            
        except Exception as e:
            error_msg = f"Error creating sequences: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
def prepare_features(data, sequence_length=60):
    """
    Prepare the features for the model
    
    Args:
        data: Input DataFrame containing the stock data
        sequence_length: Length of the sequences to create
        
    Returns:
        Tuple of (X, y) where:
            X: Numpy array of shape (n_sequences, sequence_length, n_features)
            y: Numpy array of shape (n_sequences,) containing target values
    
    Raises:
        ValueError: If input data is invalid or processing fails
    """
    processor = DataProcessor(sequence_length=sequence_length)
    
    # Preprocess the data
    processed = processor.preprocess(data)
    if processed is None:
        raise ValueError("Error in preprocessing pipeline")
    
    # Extract features and labels
    X = processed['features']
    y = processed['returns']
    
    return X, y

def validate_technical_indicators(df: pd.DataFrame, required_columns: list, logger=None, fill_value=np.nan):
    """
    Validate presence of required technical indicator columns in the DataFrame.
    Logs and optionally fills missing columns with a default value (np.nan by default).
    Returns the DataFrame (with missing columns added if needed) and a list of missing columns.
    """
    import numpy as np
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.warning(f"Missing technical indicator columns: {missing}")
        for col in missing:
            df[col] = fill_value
    return df, missing

def validate_data_for_prediction(data: pd.DataFrame, min_required_days: int = 60) -> bool:
    """
    Validate data quality for prediction with enhanced checks for contiguity.
    
    Args:
        data: DataFrame with stock data (must have a DatetimeIndex)
        min_required_days: Minimum number of days of data required
        
    Returns:
        bool: True if data passes validation, False otherwise
    """
    if data is None or data.empty:
        logger.warning("No data provided for validation")
        return False
        
    # Check for minimum data points
    if len(data) < min_required_days:
        logger.warning(f"Insufficient data points: {len(data)} < {min_required_days}")
        return False
        
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return False
        
    # Check for null values in required columns
    if data[required_columns].isnull().any().any():
        logger.warning("Data contains null values in required columns")
        return False
        
    # Check for zero or negative prices
    price_columns = ['Open', 'High', 'Low', 'Close']
    if (data[price_columns] <= 0).any().any():
        logger.warning("Data contains zero or negative prices")
        return False
        
    # Check if index is DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Data index is not a DatetimeIndex")
        return False    
        
    # Check for sufficient price movement
    price_std = data['Close'].pct_change().std()
    if price_std < 0.001:  # Minimum standard deviation of returns
        logger.warning(f"Insufficient price movement (std={price_std:.6f})")
        return False
        
    # Check if we have the most recent days contiguously
    latest_date = data.index.max()
    expected_dates = pd.date_range(end=latest_date, periods=min_required_days, freq='B')  # Business days
    missing_dates = expected_dates.difference(data.index)
    
    if not missing_dates.empty:
        logger.warning(f"Missing {len(missing_dates)} trading days in the last {min_required_days} days")
        logger.warning(f"Earliest missing date: {missing_dates.min()}, Latest missing date: {missing_dates.max()}")
        return False
        
    return True

def prepare_data_for_prediction(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Prepare and validate data for prediction.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data needed
        
    Returns:
        Optional[pd.DataFrame]: Prepared and validated data, or None if validation fails
    """
    from .yfinance_utils import fetch_daily_data
    from .data_collector import load_cached_data
    
    try:
        # Calculate date range with buffer for weekends/holidays
        buffer_days = int(days * 1.5)
        
        # Try to get cached data first
        data = load_cached_data(ticker)
        
        # If no data or insufficient data, fetch fresh
        if data is None or len(data) < buffer_days:
            logger.info(f"Fetching fresh data for {ticker} (last {buffer_days} days)")
            data = fetch_daily_data(ticker, period=f"{buffer_days}d")
            
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Get only the most recent data
        latest_date = data.index.max()
        start_date = latest_date - pd.Timedelta(days=buffer_days)
        data = data[data.index >= start_date]
        
        # Validate data
        if not validate_data_for_prediction(data, days):
            logger.warning(f"Data validation failed for {ticker}")
            return None
            
        # Ensure we have exactly the number of days requested
        if len(data) > days:
            data = data.iloc[-days:]
            
        return data
            
    except Exception as e:
        logger.error(f"Error preparing data for {ticker}: {str(e)}", exc_info=True)
        return None
        return None

__all__ = ["prepare_features", "DataProcessor", "validate_technical_indicators",
           "validate_data_for_prediction", "prepare_data_for_prediction"]
