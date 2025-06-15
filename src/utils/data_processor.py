import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats
import yfinance as yf
import ta
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scalers = {}
        self.features = None
        self.feature_scalers = {}
        
    def load_data(self, ticker, start_date, end_date):
        """
        Load and preprocess stock data
        """
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        # Create comprehensive features
        data = self._create_features(data)
        
        # Clean data
        data = self._clean_data(data)
        
        # Detect and handle outliers
        data = self._detect_outliers(data)
        
        # Scale features
        X, y = self._scale_data(data)
        
        # Create additional features based on scaled data
        X = self._create_additional_features(X)
        
        return X, y, data

    def _create_features(self, data):
        """
        Create comprehensive set of technical indicators and features with safe handling of Series operations.
        
        Args:
            data (pd.DataFrame): Input stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators and features
            
        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        try:
            # Make a copy to avoid SettingWithCopyWarning
            result = data.copy()
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Ensure we have numpy arrays for calculations
            close_prices = np.asarray(result['Close'], dtype=np.float64)
            high_prices = np.asarray(result['High'], dtype=np.float64)
            low_prices = np.asarray(result['Low'], dtype=np.float64)
            volume_values = np.asarray(result['Volume'], dtype=np.float64)
            
            # Calculate minimum required data points for all indicators
            min_required = max(200, 252)  # Based on the largest lookback period (SMA_200)
            if len(close_prices) < min_required:
                raise ValueError(f"Insufficient data points. Need at least {min_required} days of data, got {len(close_prices)}")
            
            # Initialize ta indicators
            ta_indicators = ta.add_all_ta_features(
                result, 
                open="Open", 
                high="High", 
                low="Low", 
                close="Close", 
                volume="Volume",
                fillna=True
            )
            
            # Add selected indicators to result
            result = ta_indicators[[
                'volume_obv', 'volume_obvm', 'volume_fi', 'volume_mfi',
                'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
                'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
                'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',
                'momentum_rsi', 'momentum_mfi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
                'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind',
                'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff',
                'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',
                'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator'
            ]].copy()
            
            # Add basic moving averages
            result['SMA_20'] = ta.trend.sma_indicator(close_prices, window=20)
            result['SMA_50'] = ta.trend.sma_indicator(close_prices, window=50)
            result['SMA_200'] = ta.trend.sma_indicator(close_prices, window=200)
            result['EMA_12'] = ta.trend.ema_indicator(close_prices, window=12)
            result['EMA_26'] = ta.trend.ema_indicator(close_prices, window=26)
            
            # Add any additional custom indicators not covered by add_all_ta_features
            result['HL_PCT'] = (result['High'] - result['Low']) / result['Close'] * 100.0
            result['PCT_change'] = result['Close'].pct_change() * 100.0
            result['VWAP'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
            result['Daily_Return'] = result['Close'].pct_change()
            result['Volatility'] = result['Daily_Return'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
            
            # Price to Moving Average Ratios
            result['Price_SMA20_Ratio'] = result['Close'] / result['SMA_20']
            result['Price_SMA50_Ratio'] = result['Close'] / result['SMA_50']
            result['Price_SMA200_Ratio'] = result['Close'] / result['SMA_200']
            
            # Drop rows with NaN values that might have been introduced by indicators
            result = result.dropna()
            
            if result.empty:
                raise ValueError("No valid data remaining after feature creation")
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error in _create_features: {str(e)}")

    def _clean_data(self, data):
        """
        Clean the data by handling missing values and outliers
        """
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data is None or empty")
            
        try:
            # Make a copy to avoid SettingWithCopyWarning
            cleaned_data = data.copy()
            
            # Forward fill then backfill any remaining NaNs
            cleaned_data = cleaned_data.ffill().bfill()
            
            # Remove any remaining rows with NaN values
            if not cleaned_data.empty:
                cleaned_data = cleaned_data.dropna()
            
            if cleaned_data.empty:
                raise ValueError("No valid data remaining after cleaning")
                
            return cleaned_data
            
        except Exception as e:
            raise RuntimeError(f"Error cleaning data: {str(e)}")

    def _detect_outliers(self, data, threshold=3.0):
        """
        Detect and handle outliers using Z-score method
        
        Args:
            data: Input data (DataFrame or Series) to process
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
            
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If outlier detection fails
        """
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data is None or empty")
            
        try:
            # Make a copy to avoid modifying the original data
            result = data.copy()
            
            # Convert to DataFrame if it's a Series
            if isinstance(result, pd.Series):
                result = result.to_frame()
                
            # Calculate Z-scores for numerical columns
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return result
                
            # Calculate Z-scores and handle outliers for each numeric column
            for col in numeric_cols:
                col_data = result[col].values
                if len(col_data) < 2:  # Need at least 2 points for z-score
                    continue
                    
                # Calculate Z-scores
                z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
                
                # Find outliers (using numpy array operations instead of pandas Series)
                outlier_mask = np.isnan(z_scores) | (z_scores > threshold)
                
                # Replace outliers with NaN
                if np.any(outlier_mask):
                    # Convert to numpy array for boolean indexing
                    mask_array = np.array(outlier_mask)
                    result.loc[mask_array, col] = np.nan
            
            # Forward fill then backfill any NaNs created by outlier removal
            result = result.ffill().bfill()
            
            # If input was a Series, return as Series
            if isinstance(data, pd.Series):
                return result[result.columns[0]]
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error detecting outliers: {str(e)}")

    def _scale_data(self, data):
        """
        Scale the data using multiple scalers and store the scalers
        
        Args:
            data: Input DataFrame containing the features to scale
            
        Returns:
            Tuple of (X, y) where:
                X: Numpy array of shape (n_sequences, sequence_length, n_features)
                y: Numpy array of shape (n_sequences,) containing target values
            
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If scaling fails
        """
        # Validate input
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data cannot be None or empty")
        
        try:
            # Convert to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Make a copy to avoid modifying the original data
            data = data.copy()
            
            # Validate minimum required data points
            min_required_points = self.sequence_length + 1
            if len(data) < min_required_points:
                raise ValueError(
                    f"Not enough data points. Need at least {min_required_points} points, got {len(data)}"
                )
            
            # Select features for prediction
            potential_features = [
                'Close', 'Volume', 'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
                'Volatility', 'ATR', 'ADX', 'CCI', 'Daily_Return',
                'Day_of_Week', 'Month', 'Quarter'
            ]
            
            # Only use features that exist in the data
            self.features = [f for f in potential_features if f in data.columns]
            
            if not self.features:
                self.features = ['Close', 'Volume']
                warnings.warn("Falling back to default features: Close, Volume")

            
            # Initialize feature_scalers if it doesn't exist
            if not hasattr(self, 'feature_scalers'):
                self.feature_scalers = {}
            
            # Scale each feature
            scaled_data = np.zeros((len(data), len(self.features)), dtype=np.float32)
            
            for i, feature in enumerate(self.features):
                # Get the feature values as numpy array
                feature_values = data[feature].values.reshape(-1, 1)
                
                # Create appropriate scaler for this feature if it doesn't exist
                if feature not in self.feature_scalers:
                    if feature.startswith('RSI_') or feature.startswith('Momentum_'):
                        # Scale RSI and Momentum between 0 and 1
                        self.feature_scalers[feature] = MinMaxScaler()
                    elif feature.startswith('Volatility_'):
                        # Standardize volatility measures
                        self.feature_scalers[feature] = StandardScaler()
                    else:
                        # Default MinMax scaling
                        self.feature_scalers[feature] = MinMaxScaler()
                
                # Scale the feature values
                scaled_data[:, i] = self.feature_scalers[feature].fit_transform(feature_values).flatten()
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Validate array shapes
            if X.shape[0] == 0:
                raise ValueError(
                    f"No sequences created. Required sequence length: {self.sequence_length}, "
                    f"Data length: {len(data)}"
                )
            
            return X, y
            
        except Exception as e:
            raise RuntimeError(f"Error in _scale_data: {str(e)}")

    def _create_additional_features(self, X):
        """
        Create additional features from scaled data
        """
        # Calculate feature combinations
        X_combined = np.zeros_like(X)
        
        # Calculate rolling statistics
        for i in range(X.shape[0]):
            # Calculate rolling mean and std for each feature
            for j in range(X.shape[2]):
                X_combined[i, :, j] = np.mean(X[i, :, j])
                X_combined[i, :, j+1] = np.std(X[i, :, j])
                
            # Calculate correlations between features
            corr_matrix = np.corrcoef(X[i, :, :].T)
            X_combined[i, :, -len(self.features):] = corr_matrix.reshape(-1)
        
        # Combine original features with new features
        X = np.concatenate([X, X_combined], axis=2)
        
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
        # Create time features
        time_features = np.zeros((n_samples, self.sequence_length, 2))
        
        # Add time index
        time_features[:, :, 0] = np.arange(self.sequence_length)
        
        # Add cyclical time features
        time_features[:, :, 1] = np.sin(2 * np.pi * np.arange(self.sequence_length) / self.sequence_length)
        
        return time_features

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

    def preprocess(self, data):
        """
        Preprocess the stock market data with advanced techniques
        """
        # Clean data
        data = self._clean_data(data)
        
        # Detect and handle outliers
        data = self._detect_outliers(data)
        
        # Create technical indicators
        data = self._create_technical_indicators(data)
        
        # Handle temporal alignment
        data = self._handle_temporal_alignment(data)
        
        # Scale the data
        scaled_data = self._scale_data(data)
        
        # Create sequences
        sequences = self._create_sequences(scaled_data)
        
        return sequences

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

    def _scale_data(self, data):
        """
        Scale the data using multiple scalers
        """
        # Select relevant features
        features = self.features
        
        # Scale each feature differently
        scaled_data = np.zeros((len(data), len(features)))
        
        for i, feature in enumerate(features):
            if feature.startswith('RSI_') or feature.startswith('Momentum_'):
                # Scale RSI and Momentum between 0 and 1
                scaled_data[:, i] = self.scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
            elif feature.startswith('Volatility_'):
                # Standardize volatility measures
                scaled_data[:, i] = StandardScaler().fit_transform(data[feature].values.reshape(-1, 1)).flatten()
            else:
                # Default MinMax scaling
                scaled_data[:, i] = self.scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        
        return scaled_data

    def _create_technical_indicators(self, data):
        """
        Add advanced technical indicators to the data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if we have enough data for technical indicators
        if len(data) < 200:
            logger.warning(f"Insufficient data for technical indicators: {len(data)} rows, need at least 200")
            logger.warning(f"Date range: {data.index.min()} to {data.index.max()}")
        
        logger.info(f"Generating technical indicators for {len(data)} data points")
        
        # Track successfully created indicators
        created_indicators = []
        
        try:
            # Calculate Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            created_indicators.append('SMA_20')
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            created_indicators.append('SMA_50')
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            created_indicators.append('SMA_200')
            logger.debug(f"Created SMA indicators: {created_indicators}")
        except Exception as e:
            logger.error(f"Error creating SMA indicators: {str(e)}")
        
        # Calculate RSI with multiple periods
        try:
            for period in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                # Handle division by zero
                rs = gain / loss.replace(0, np.finfo(float).eps)
                data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                created_indicators.append(f'RSI_{period}')
            logger.debug(f"Created RSI indicators: {[f'RSI_{p}' for p in [7, 14, 21]]}")
        except Exception as e:
            logger.error(f"Error creating RSI indicators: {str(e)}")
        
        # Calculate MACD with multiple configurations
        try:
            for fast, slow, signal in [(12, 26, 9), (8, 17, 5), (15, 30, 12)]:
                exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
                exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                data[f'MACD_{fast}_{slow}_{signal}'] = macd
                created_indicators.append(f'MACD_{fast}_{slow}_{signal}')
                data[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
                created_indicators.append(f'MACD_Signal_{fast}_{slow}_{signal}')
                data[f'MACD_Hist_{fast}_{slow}_{signal}'] = macd - signal_line
                created_indicators.append(f'MACD_Hist_{fast}_{slow}_{signal}')
            logger.debug(f"Created MACD indicators")
        except Exception as e:
            logger.error(f"Error creating MACD indicators: {str(e)}")
        
        # Calculate Bollinger Bands
        try:
            data['BB_Middle_20'] = data['Close'].rolling(window=20).mean()
            data['BB_Upper_20'] = data['BB_Middle_20'] + 2 * data['Close'].rolling(window=20).std()
            data['BB_Lower_20'] = data['BB_Middle_20'] - 2 * data['Close'].rolling(window=20).std()
            created_indicators.extend(['BB_Middle_20', 'BB_Upper_20', 'BB_Lower_20'])
            logger.debug(f"Created Bollinger Bands indicators")
        except Exception as e:
            logger.error(f"Error creating Bollinger Bands: {str(e)}")
        
        # Calculate Volume Indicators
        try:
            if 'Volume' in data.columns:
                data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
                created_indicators.append('OBV')
                logger.debug(f"Created OBV indicator")
            else:
                logger.warning("Volume data not available, skipping OBV calculation")
        except Exception as e:
            logger.error(f"Error creating OBV: {str(e)}")
        
        # Calculate Volatility
        try:
            data['Volatility_10'] = data['Close'].pct_change().rolling(window=10).std()
            data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
            created_indicators.extend(['Volatility_10', 'Volatility_20'])
            logger.debug(f"Created Volatility indicators")
        except Exception as e:
            logger.error(f"Error creating Volatility indicators: {str(e)}")
        
        # Calculate Momentum
        try:
            for period in [5, 10, 20]:
                data[f'Momentum_{period}'] = data['Close'].pct_change(periods=period)
                created_indicators.append(f'Momentum_{period}')
            logger.debug(f"Created Momentum indicators")
        except Exception as e:
            logger.error(f"Error creating Momentum indicators: {str(e)}")
        
        # Update features list with only successfully created indicators
        self.features = ['Close']
        if 'SMA_20' in created_indicators: self.features.append('SMA_20')
        if 'SMA_50' in created_indicators: self.features.append('SMA_50')
        if 'SMA_200' in created_indicators: self.features.append('SMA_200')
        
        # Add RSI features if available
        rsi_features = [f for f in created_indicators if f.startswith('RSI_')]
        if rsi_features:
            self.features.extend(rsi_features)
        
        # Add MACD features if available
        macd_features = [f for f in created_indicators if f.startswith('MACD_')]
        if macd_features:
            self.features.extend(macd_features)
        
        # Add other indicators if available
        if 'OBV' in created_indicators: self.features.append('OBV')
        if 'Volatility_10' in created_indicators: self.features.append('Volatility_10')
        if 'Volatility_20' in created_indicators: self.features.append('Volatility_20')
        
        # Add momentum features if available
        momentum_features = [f for f in created_indicators if f.startswith('Momentum_')]
        if momentum_features:
            self.features.extend(momentum_features)
            
        # Log summary of created indicators
        logger.info(f"Created {len(created_indicators)} technical indicators")
        logger.info(f"Using {len(self.features)} features for model: {self.features}")
        
        # Fill NaN values that may have been created by rolling windows
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

    def _scale_data(self, data):
        """
        Scale the data using multiple scalers
        """
        # Select relevant features
        features = self.features
        
        # Scale each feature differently
        scaled_data = np.zeros((len(data), len(features)))
        
        for i, feature in enumerate(features):
            if feature.startswith('RSI_') or feature.startswith('Momentum_'):
                # Scale RSI and Momentum between 0 and 1
                scaled_data[:, i] = self.scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
            elif feature.startswith('Volatility_'):
                # Standardize volatility measures
                scaled_data[:, i] = StandardScaler().fit_transform(data[feature].values.reshape(-1, 1)).flatten()
            else:
                # Default MinMax scaling
                scaled_data[:, i] = self.scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        
        return scaled_data

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
        """
        # Create time features
        time_features = np.zeros((n_samples, self.sequence_length, 2))
        
        # Add time index
        time_features[:, :, 0] = np.arange(self.sequence_length)
        
        # Add cyclical time features
        time_features[:, :, 1] = np.sin(2 * np.pi * np.arange(self.sequence_length) / self.sequence_length)
        
        return time_features
