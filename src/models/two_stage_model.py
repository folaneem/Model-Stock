import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, MultiHeadAttention, 
    LayerNormalization, BatchNormalization, Concatenate, Add
)
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List, Optional, Union, Any
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys
import io

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure root logger
def configure_logging():
    """Configure root logger with UTF-8 safe handlers"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter with ASCII-safe encoding
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler with UTF-8 support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    log_file = os.path.join(log_dir, 'model.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

# Initialize logging
configure_logging()

# Module logger
def get_logger(name):
    """Get or create a logger with the given name"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Use root logger's level and handlers
        logger.setLevel(logging.INFO)
    return logger

class TwoStagePredictor:
    """Two-stage stock price prediction model with LSTM and attention mechanism"""
    
    def __init__(
        self,
        lstm_units: int = 64, 
        attention_units: int = 32, 
        dropout_rate: float = 0.3, 
        learning_rate: float = 0.0005,  # Reduced from 0.001
        l2_lambda: float = 0.001, 
        num_heads: int = 4, 
        use_learning_rate_scheduler: bool = True,
        prediction_days: int = 60,
        clipnorm: float = 1.0,  # Gradient clipping
        clipvalue: float = 0.5,  # Gradient clipping
        kernel_initializer: str = 'he_normal'  # Weight initialization
    ):
        """
        Initialize the TwoStagePredictor with the given parameters.
        
        Args:
            lstm_units: Number of units in LSTM layers
            attention_units: Number of units in attention layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Initial learning rate
            l2_lambda: L2 regularization factor
            num_heads: Number of attention heads
            use_learning_rate_scheduler: Whether to use learning rate scheduling
            prediction_days: Number of days to use for prediction window
        """
        # Model architecture parameters
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.num_heads = num_heads
        self.use_learning_rate_scheduler = use_learning_rate_scheduler
        self.prediction_days = prediction_days
        self.kernel_initializer = kernel_initializer
        
        # Gradient clipping parameters
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        
        # Model components
        self.model = None
        self.encoder = None
        self.decoder = None
        
        # Data scalers
        self.stock_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()  # For inverse transforming predictions
        
        # Training history
        self.history = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        # Create output directories
        self.model_dir = Path('models')
        self.log_dir = Path('logs')
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger.info(
            f"Initialized TwoStagePredictor with {lstm_units} LSTM units and "
            f"{attention_units} attention units, learning rate: {learning_rate}"
        )

    def _create_attention_layer(self, query_value, attention_units):
        """Create multi-head attention layer with residual connection"""
        # Store input for residual connection
        input_tensor = query_value
        input_units = input_tensor.shape[-1]
        
        # Project input to match attention units if needed
        if input_units != attention_units:
            res = Dense(attention_units)(input_tensor)
        else:
            res = input_tensor
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=attention_units // self.num_heads,
            dropout=self.dropout_rate
        )(query_value, query_value)
        
        # Add & Norm (first residual connection)
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(res + attention_output)
        
        # Store for second residual connection
        res2 = attention_output
        
        # Feed Forward
        ffn_output = Dense(attention_units * 4, activation='relu')(attention_output)
        ffn_output = Dense(attention_units)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        
        # Add & Norm (second residual connection)
        return LayerNormalization(epsilon=1e-6)(res2 + ffn_output)

    def build_model(self, 
                   stock_shape: Tuple[int, int], 
                   features_shape: Tuple[int, int] = None,
                   lstm_units: int = None,
                   attention_units: int = None,
                   dropout_rate: float = None) -> Model:
        """
        Build the two-stage prediction model with improved architecture
        
        Args:
            stock_shape: Shape of stock input data (timesteps, features)
            features_shape: Shape of additional features (timesteps, num_features)
            lstm_units: Number of LSTM units
            attention_units: Number of attention units
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        # Use instance variables if parameters are not provided
        lstm_units = lstm_units or self.lstm_units
        attention_units = attention_units or self.attention_units
        dropout_rate = dropout_rate or self.dropout_rate
        
        self.logger.info(f"Building model with stock_shape: {stock_shape}, features_shape: {features_shape}")
        
        # Input layers
        stock_input = Input(shape=stock_shape, name='stock_input')
        
        # First LSTM layer with more units and better initialization
        lstm1 = LSTM(
            attention_units * 2,  # Double the units
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
            kernel_initializer=self.kernel_initializer,
            recurrent_initializer='orthogonal',
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5
        )(stock_input)
        lstm1 = BatchNormalization()(lstm1)
        
        # Second LSTM layer with better initialization
        lstm2 = LSTM(
            attention_units,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
            kernel_initializer=self.kernel_initializer,
            recurrent_initializer='orthogonal',
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5
        )(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention layer
        attention_out = self._create_attention_layer(lstm2, attention_units)
        
        # If additional features are provided, concatenate them
        if features_shape is not None:
            features_input = Input(shape=features_shape, name='features_input')
            
            # Process features with a smaller network
            features_dense = Dense(attention_units // 2, activation='relu')(features_input)
            features_dense = BatchNormalization()(features_dense)
            
            # Ensure shapes match before concatenation
            if features_dense.shape[-1] != attention_out.shape[-1]:
                features_dense = Dense(attention_units)(features_dense)
                
            # Concatenate LSTM output with features
            concat = Concatenate(axis=-1)([attention_out, features_dense])
            
            # Final dense layers
            x = Dense(attention_units * 2, activation='relu')(concat)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Output layer
            output = Dense(1, activation='linear', name='output')(x)
            
            # Create model with both inputs
            model = Model(
                inputs=[stock_input, features_input],
                outputs=output,
                name='TwoStagePredictor'
            )
        else:
            # If no additional features, just use LSTM + attention output
            x = Dense(attention_units, activation='relu')(attention_out)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Output layer
            output = Dense(1, activation='linear', name='output')(x)
            
            # Create model with only stock input
            model = Model(
                inputs=stock_input,
                outputs=output,
                name='TwoStagePredictor'
            )
        
        # Compile model with gradient clipping and better optimizer settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=self.clipnorm,  # Using only clipnorm for gradient clipping
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            name='Adam',
        )
        model.compile(
            optimizer=optimizer, 
            loss='huber',  # More robust to outliers than MSE
            metrics=[
                'mae', 
                'mse',
                tf.keras.metrics.MeanAbsolutePercentageError(),
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )
        
        self.model = model
        self.logger.info("Model built successfully")
        return model
        
        x = Dense(attention_units, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        output = Dense(1, activation='linear', name='output')(x)
        
        # Create and compile model
        model = Model(
            inputs=[stock_input, features_input],
            outputs=output,
            name='two_stage_predictor'
        )
        
        # Enhanced optimizer with learning rate scheduling and gradient clipping
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        # Log initial learning rate
        self.logger.info(f"Using initial learning rate: {self.learning_rate}")
        
        # Compile model with multiple loss functions and metrics
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=[
                'mae',
                'mse',
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsolutePercentageError()
            ]
        )
        
        # Log model summary
        model.summary(print_fn=self.logger.info)
        
        self.model = model
        self.logger.info("Model built successfully")
        return model
        
    def _validate_input_data(self, data: pd.DataFrame, name: str) -> None:
        """
        Validate input data for NaN, inf, and other issues.
        
        Args:
            data: Input DataFrame to validate
            name: Name of the data for logging
            
        Raises:
            ValueError: If data contains invalid values
        """
        if data is None or data.empty:
            raise ValueError(f"{name} data cannot be None or empty")
            
        # Check for NaN or infinite values
        if data.isnull().any().any():
            nan_cols = data.columns[data.isnull().any()].tolist()
            self.logger.warning(f"{name} contains NaN values in columns: {nan_cols}")
            # Fill NaN values with forward and backward fill
            data = data.ffill().bfill()
            if data.isnull().any().any():
                raise ValueError(f"{name} contains NaN values that could not be filled")
                
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            inf_cols = data.columns[np.isinf(data.select_dtypes(include=[np.number])).any()].tolist()
            self.logger.warning(f"{name} contains infinite values in columns: {inf_cols}")
            # Replace inf with max/min values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                max_val = data[col].max()
                min_val = data[col].min()
                data[col] = data[col].fillna((max_val + min_val) / 2)
                
        return data
        
    def preprocess_data(self, stock_data: pd.DataFrame, 
                       sentiment_data: pd.DataFrame = None, 
                       macro_data: pd.DataFrame = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Preprocess the input data for the model with improved error handling and logging.
            
        Args:
            stock_data: DataFrame containing stock price data with datetime index.
                      Expected columns: 'Open', 'High', 'Low', 'Close', 'Volume',
                      and optionally technical indicators like 'RSI', 'MACD', etc.
            sentiment_data: Optional DataFrame containing sentiment data with datetime index.
                          Expected column: 'sentiment_score'
            macro_data: Optional DataFrame containing macroeconomic data with datetime index.
                      Expected columns: Any relevant macroeconomic indicators
                      
        Returns:
            Tuple containing ((X_stock, X_features), y) where:
            - X_stock: Sequence of stock prices (samples, timesteps, n_stock_features)
            - X_features: Additional features (samples, n_additional_features)
            - y: Target values (samples,)
            
        Raises:
            ValueError: If input data is invalid or preprocessing fails
        """
        self.logger.info("Starting data preprocessing...")
        
        # Log input data information
        if stock_data is not None and not stock_data.empty:
            self.logger.debug(f"Stock data columns: {list(stock_data.columns)}")
            self.logger.debug(f"Stock data index type: {type(stock_data.index)}")
            self.logger.debug(f"Stock data shape: {stock_data.shape}")
            if hasattr(stock_data.index, 'dtype'):
                self.logger.debug(f"Stock data index dtype: {stock_data.index.dtype}")
        
        if sentiment_data is not None and not sentiment_data.empty:
            self.logger.debug(f"Sentiment data columns: {list(sentiment_data.columns)}")
            self.logger.debug(f"Sentiment data shape: {sentiment_data.shape}")
            
        if macro_data is not None and not macro_data.empty:
            self.logger.debug(f"Macro data columns: {list(macro_data.columns)}")
            self.logger.debug(f"Macro data shape: {macro_data.shape}")
            
        try:
            # Input validation and cleaning
            stock_data = self._validate_input_data(stock_data, "Stock")
            if sentiment_data is not None:
                sentiment_data = self._validate_input_data(sentiment_data, "Sentiment")
            if macro_data is not None:
                macro_data = self._validate_input_data(macro_data, "Macro")
                    
            # Ensure we have a datetime index
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                try:
                    stock_data.index = pd.to_datetime(stock_data.index)
                except Exception as e:
                    raise ValueError(f"Could not convert index to datetime: {str(e)}")
            
            # Sort by date to ensure chronological order
            stock_data = stock_data.sort_index()
            
            # Handle missing data with forward and backward filling
            stock_data = stock_data.ffill().bfill()
            
            # Check if we have enough data
            if len(stock_data) < self.prediction_days + 1:  # +1 for target
                raise ValueError(f"Insufficient data points. Need at least {self.prediction_days + 1} days, "
                              f"got {len(stock_data)}")
            
            # Get close prices and scale them (handle various naming patterns)
            close_patterns = ['close', 'close_', '_close', 'price', 'adj close']
            close_col = None
            
            # Try different patterns to find the close price column
            for pattern in close_patterns:
                matches = [col for col in stock_data.columns if pattern.lower() in col.lower()]
                if matches:
                    close_col = matches[0]
                    break
                    
            if close_col is None and 'Close_AAPL' in stock_data.columns:
                close_col = 'Close_AAPL'
                
            if close_col is None:
                # If still not found, try to find any column with 'close' in any position
                close_col = next((col for col in stock_data.columns 
                               if 'close' in col.lower()), None)
                
            if close_col is None and len(stock_data.columns) > 0:
                # If still not found, take the first column that looks like a price column
                price_like_cols = [col for col in stock_data.columns 
                                 if any(price_term in col.lower() 
                                      for price_term in ['price', 'close', 'last', 'adj'])]
                if price_like_cols:
                    close_col = price_like_cols[0]
            
            if close_col is None:
                raise ValueError("Could not find 'Close' price column in stock data. "
                               f"Available columns: {list(stock_data.columns)}")
                                
            self.logger.info(f"Using column '{close_col}' for close prices")
                
            close_prices = stock_data[close_col].values.reshape(-1, 1)
            scaled_prices = self.stock_scaler.fit_transform(close_prices)
            # Also fit the target scaler with the same data
            self.target_scaler.fit(close_prices)
            
            # Create sequences for stock data
            X_stock = []
            y = []
            
            for i in range(self.prediction_days, len(scaled_prices)):
                X_stock.append(scaled_prices[i-self.prediction_days:i])
                y.append(scaled_prices[i])
            
            X_stock = np.array(X_stock)
            y = np.array(y).reshape(-1, 1)
            
            # Prepare additional features
            feature_data = []
            
            # 1. Add volume if available (handle various volume column names)
            volume_col = None
            volume_patterns = ['volume', 'vol', 'traded_volume', 'trading_volume']
            
            for pattern in volume_patterns:
                matches = [col for col in stock_data.columns if pattern.lower() in col.lower()]
                if matches:
                    volume_col = matches[0]
                    break
                    
            if volume_col is not None:
                try:
                    volume_scaler = MinMaxScaler()
                    volume_scaled = volume_scaler.fit_transform(stock_data[volume_col].values.reshape(-1, 1))
                    feature_data.append(volume_scaled)
                    self.logger.info(f"Using column '{volume_col}' for volume data")
                except Exception as e:
                    self.logger.warning(f"Could not process volume data: {str(e)}")
            else:
                self.logger.debug("No volume column found in the data")
                
            # 2. Add technical indicators if they exist (handle various naming patterns)
            tech_indicators = {
                'rsi': ['rsi', 'relative_strength'],
                'macd': ['macd'],
                'bb_upper': ['bb_upper', 'bollinger_upper', 'upper_band'],
                'bb_lower': ['bb_lower', 'bollinger_lower', 'lower_band'],
                'sma_20': ['sma_20', 'sma20', 'sma 20', '20_sma', '20 sma', '20ma', 'ma20', '20_ma', '20 ma'],
                'ema_50': ['ema_50', 'ema50', 'ema 50', '50_ema', '50 ema', '50ema']
            }
            
            available_indicators = []
            
            for indicator_name, patterns in tech_indicators.items():
                # Skip if we've already found this indicator
                if any(ind in available_indicators for ind in patterns):
                    continue
                    
                # Try each possible pattern for this indicator
                for pattern in patterns:
                    # Look for exact matches or columns containing the pattern
                    matching_cols = [col for col in stock_data.columns 
                                   if pattern.lower() in col.lower() or \
                                      col.lower().endswith(f'_{pattern.lower()}') or \
                                      col.lower().startswith(f'{pattern.lower()}_')]
                    
                    if matching_cols:
                        col = matching_cols[0]
                        try:
                            # Skip if we already have this column
                            if col in available_indicators:
                                continue
                                
                            # Scale the indicator data
                            indicator_scaler = MinMaxScaler()
                            indicator_scaled = indicator_scaler.fit_transform(
                                stock_data[col].values.reshape(-1, 1)
                            )
                            feature_data.append(indicator_scaled)
                            available_indicators.append(col)
                            self.logger.debug(f"Using column '{col}' for {indicator_name}")
                            break  # Found a match, move to next indicator
                        except Exception as e:
                            self.logger.warning(f"Could not process technical indicator {col}: {str(e)}")
            
            self.logger.info(f"Processed {len(available_indicators)} technical indicators: {available_indicators}")
            if not available_indicators:
                self.logger.warning("No technical indicators found in the data")
            
            # 3. Add sentiment data if available
            if sentiment_data is not None and not sentiment_data.empty:
                try:
                    # Ensure sentiment data has datetime index
                    if not isinstance(sentiment_data.index, pd.DatetimeIndex):
                        sentiment_data.index = pd.to_datetime(sentiment_data.index)
                    
                    # Resample and forward fill sentiment data to match stock data frequency
                    sentiment_data = sentiment_data.reindex(stock_data.index, method='ffill')
                    
                    if 'sentiment_score' in sentiment_data.columns:
                        # Scale sentiment scores to [0,1]
                        sentiment_scaler = MinMaxScaler()
                        sentiment_scaled = sentiment_scaler.fit_transform(
                            sentiment_data['sentiment_score'].values.reshape(-1, 1)
                        )
                        feature_data.append(sentiment_scaled)
                except Exception as e:
                    self.logger.warning(f"Could not process sentiment data: {str(e)}")
            
            # 4. Add macro data if available
            if macro_data is not None and not macro_data.empty:
                try:
                    # Ensure macro data has datetime index
                    if not isinstance(macro_data.index, pd.DatetimeIndex):
                        macro_data.index = pd.to_datetime(macro_data.index)
                    
                    # Resample and forward fill macro data to match stock data frequency
                    macro_data = macro_data.reindex(stock_data.index, method='ffill')
                    
                    # Scale each macro indicator
                    for col in macro_data.columns:
                        try:
                            macro_scaler = MinMaxScaler()
                            macro_scaled = macro_scaler.fit_transform(
                                macro_data[col].values.reshape(-1, 1)
                            )
                            feature_data.append(macro_scaled)
                        except Exception as e:
                            self.logger.warning(f"Could not process macro indicator {col}: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Could not process macro data: {str(e)}")
            
            # Create sequences for additional features to match stock data sequences
            X_additional = []
            
            if feature_data:
                # Stack all additional features
                all_features = np.hstack(feature_data)
                
                # Clean stacked features before building sequences
                all_features = np.nan_to_num(
                    all_features,
                    nan=0.0,
                    posinf=np.finfo(np.float32).max,
                    neginf=np.finfo(np.float32).min
                )
                
                # Create sequences with the same length as stock data sequences
                for i in range(self.prediction_days, len(scaled_prices)):
                    X_additional.append(all_features[i-self.prediction_days:i])
                
                X_additional = np.array(X_additional)
                
                # Validate shapes after cleaning
                if X_stock.shape[0] != X_additional.shape[0]:
                    self.logger.warning(
                        f"Mismatch in sequence counts after cleaning: "
                        f"X_stock has {X_stock.shape[0]} sequences, "
                        f"X_additional has {X_additional.shape[0]} sequences"
                    )
                self.logger.info(f"Processed {X_additional.shape[2]} additional features with sequence length {X_additional.shape[1]}")
            else:
                # If no additional features, create a sequence of zeros with the same length as stock data
                X_additional = np.zeros((len(X_stock), self.prediction_days, 1))
                self.logger.warning("No additional features were processed, using zero padding")
            
            # Log shapes for debugging
            self.logger.info(
                f"Preprocessed data shapes - "
                f"X_stock: {X_stock.shape}, "
                f"X_additional: {X_additional.shape}, "
                f"y: {y.shape}"
            )
            
            return (X_stock, X_additional), y
                
        except Exception as e:
            error_msg = f"Error in preprocess_data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def _get_callbacks(self):
        """
        Create and return a list of callbacks for model training
        """
        # Create necessary directories
        Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Terminate on NaN to prevent wasted computation
            tf.keras.callbacks.TerminateOnNaN(),
            
            # Early stopping with validation monitoring
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Reduced patience to stop earlier if not improving
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4  # Minimum change to qualify as improvement
            ),
            
            # Model checkpointing with explicit path
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/checkpoints/best_model.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_weights_only=False,
                save_freq='epoch'
            ),
            
            # Learning rate reducer with more aggressive settings
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction
                patience=3,   # Fewer epochs before reducing LR
                min_lr=1e-6,
                verbose=1,
                min_delta=1e-4,
                cooldown=2,  # Number of epochs to wait before resuming normal operation
                mode='min'
            ),
            
            # CSV logger for tracking training progress
            tf.keras.callbacks.CSVLogger(
                'logs/training.log',
                separator=',',
                append=False
            )
        ]
        
        # Add learning rate scheduler if enabled
        if hasattr(self, 'use_learning_rate_scheduler') and self.use_learning_rate_scheduler:
            def lr_schedule(epoch, lr):
                """Learning rate schedule with warmup and cosine decay"""
                # Warmup for first 5 epochs
                if epoch < 5:
                    return float(self.learning_rate * (epoch + 1) / 5)
                # Cosine decay after warmup
                decay = 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (100 - 5)))
                return float(self.learning_rate * decay)
                
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lr_schedule,
                verbose=1
            )
            callbacks.append(lr_scheduler)
            
        # Add TensorBoard if needed
        if self.log_dir.exists():
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.log_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,  # Disabled to save space
                    update_freq='epoch',
                    profile_batch=0  # Disable profiling for performance
                )
            )
            
        return callbacks
        
    def _validate_and_reshape_inputs(self, x_stock: np.ndarray, x_features: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and reshape input data for training/prediction.
        
        Args:
            x_stock: Stock price sequences (n_samples, seq_len, n_stock_features)
            x_features: Additional feature sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples, 1)
            
        Returns:
            Tuple of validated and reshaped (x_stock, x_features, y)
            
        Raises:
            ValueError: If input shapes are invalid or inconsistent
        """
        # Convert inputs to numpy arrays if they aren't already
        x_stock = np.asarray(x_stock, dtype=np.float32)
        x_features = np.asarray(x_features, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Basic shape validation
        if len(x_stock.shape) != 3:
            raise ValueError(f"x_stock must be 3D (samples, timesteps, features), got shape {x_stock.shape}")
            
        if len(x_features.shape) != 3:
            raise ValueError(f"x_features must be 3D (samples, timesteps, features), got shape {x_features.shape}")
            
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Ensure consistent number of samples
        n_samples = x_stock.shape[0]
        if x_features.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of samples: x_stock has {n_samples}, "
                f"x_features has {x_features.shape[0]}"
            )
            
        if y.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of samples: x_stock has {n_samples}, "
                f"y has {y.shape[0]}"
            )
            
        # Ensure sequence lengths match
        seq_len = x_stock.shape[1]
        if x_features.shape[1] != seq_len:
            self.logger.warning(
                f"Sequence length mismatch: x_stock has {seq_len} timesteps, "
                f"x_features has {x_features.shape[1]}. Truncating to match."
            )
            min_seq_len = min(seq_len, x_features.shape[1])
            x_stock = x_stock[:, -min_seq_len:, :]
            x_features = x_features[:, -min_seq_len:, :]
            
        # Log final shapes
        self.logger.info(
            f"Input shapes - x_stock: {x_stock.shape}, "
            f"x_features: {x_features.shape}, y: {y.shape}"
        )
        
        return x_stock, x_features, y
        
    def train(self, 
             x_train: Tuple[np.ndarray, np.ndarray], 
             y_train: np.ndarray, 
             epochs: int = 20,  # Reduced from 200
             batch_size: int = 32,  # Reduced from 64
             validation_split: float = 0.2,
             class_weight: dict = None):
        """
        Train the model with proper data handling and shape validation.
        
        Args:
            x_train: Tuple of (stock_data, additional_features) arrays
            y_train: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            class_weight: Optional class weights for imbalanced data
            
        Returns:
            Training history
            
        Raises:
            ValueError: If input data is invalid or training fails
        """
        try:
            # Validate input shapes
            if not isinstance(x_train, (tuple, list)) or len(x_train) != 2:
                raise ValueError("x_train must be a tuple of (stock_data, additional_features)")
                
            x_stock, x_features = x_train
            
            # Convert to numpy arrays if they aren't already
            x_stock = np.asarray(x_stock, dtype=np.float32)
            x_features = np.asarray(x_features, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)
            
            # Log data statistics before any processing
            self.logger.info(f"Raw data shapes - x_stock: {x_stock.shape}, x_features: {x_features.shape}, y_train: {y_train.shape}")
            self.logger.info(f"Data ranges - x_stock: [{np.min(x_stock):.4f}, {np.max(x_stock):.4f}], "
                            f"x_features: [{np.min(x_features):.4f}, {np.max(x_features):.4f}], "
                            f"y_train: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
            
            # Check for NaN or infinite values with detailed reporting
            nan_count_stock = np.isnan(x_stock).sum()
            inf_count_stock = np.isinf(x_stock).sum()
            if nan_count_stock > 0 or inf_count_stock > 0:
                self.logger.error(f"x_stock contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
                # Replace NaN/inf with interpolated values instead of failing
                self.logger.warning("Attempting to clean x_stock data by replacing NaN/inf values with interpolation")
                x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_features = np.isnan(x_features).sum()
            inf_count_features = np.isinf(x_features).sum()
            if nan_count_features > 0 or inf_count_features > 0:
                self.logger.error(f"x_features contains {nan_count_features} NaN and {inf_count_features} infinite values")
                # Replace NaN/inf with interpolated values
                self.logger.warning("Attempting to clean x_features data by replacing NaN/inf values with interpolation")
                x_features = np.nan_to_num(x_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_y = np.isnan(y_train).sum()
            inf_count_y = np.isinf(y_train).sum()
            if nan_count_y > 0 or inf_count_y > 0:
                self.logger.error(f"y_train contains {nan_count_y} NaN and {inf_count_y} infinite values")
                # Replace NaN/inf with interpolated values
                self.logger.warning("Attempting to clean y_train data by replacing NaN/inf values with interpolation")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
                
            # Check for extreme values that might cause training instability
            max_abs_stock = np.max(np.abs(x_stock))
            max_abs_features = np.max(np.abs(x_features))
            max_abs_y = np.max(np.abs(y_train))
            
            if max_abs_stock > 1000:
                self.logger.warning(f"x_stock contains extreme values (max abs: {max_abs_stock:.2f}). Consider rescaling.")
            if max_abs_features > 1000:
                self.logger.warning(f"x_features contains extreme values (max abs: {max_abs_features:.2f}). Consider rescaling.")
            if max_abs_y > 1000:
                self.logger.warning(f"y_train contains extreme values (max abs: {max_abs_y:.2f}). Consider rescaling.")
                
            # Validate and reshape inputs
            x_stock, x_features, y_train = self._validate_and_reshape_inputs(
                x_stock, x_features, y_train
            )
            
            # Log data statistics after processing
            self.logger.info(f"Training on {len(x_stock)} samples")
            self.logger.info(f"Processed input shapes - x_stock: {x_stock.shape}, x_features: {x_features.shape}, y: {y_train.shape}")
            
            # Build model if not already built
            if self.model is None:
                self.build_model(
                    stock_shape=x_stock.shape[1:],
                    features_shape=x_features.shape[1:]
                )
            
            # Add early stopping with more aggressive settings to prevent instability
            callbacks = self._get_callbacks()
            
            # Add a custom callback to monitor for extreme loss values
            class ExtremeValueMonitor(tf.keras.callbacks.Callback):
                def __init__(self, logger, threshold=1000):
                    super().__init__()
                    self.logger = logger
                    self.threshold = threshold
                    
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    for metric, value in logs.items():
                        if np.isnan(value) or np.isinf(value):
                            self.logger.error(f"Epoch {epoch}: {metric} is {value} (NaN or Inf). Training may be unstable.")
                        elif abs(value) > self.threshold:
                            self.logger.warning(f"Epoch {epoch}: {metric} is {value:.2f}, which exceeds threshold {self.threshold}. Training may be unstable.")
                            
            callbacks.append(ExtremeValueMonitor(self.logger, threshold=1000))
            
            # Train the model with validation and more robust error handling
            self.logger.info("Starting model training...")
            try:
                history = self.model.fit(
                    x=[x_stock, x_features],
                    y=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1
                )
                
                self.history = history.history
                
                # Check for NaN or extreme values in training history
                for metric, values in history.history.items():
                    if any(np.isnan(v) for v in values) or any(np.isinf(v) for v in values):
                        self.logger.warning(f"Training metric {metric} contains NaN or Inf values: {values}")
                    elif any(abs(v) > 1000 for v in values):
                        self.logger.warning(f"Training metric {metric} contains extreme values: {values}")
                        
                self.logger.info("Training completed successfully")
                return history
                
            except Exception as train_error:
                self.logger.error(f"Error during model.fit(): {str(train_error)}")
                # Try to recover by rebuilding the model with simpler architecture
                self.logger.warning("Attempting to recover by rebuilding model with simpler architecture")
                try:
                    # Backup the original configuration
                    original_lstm_units = self.lstm_units
                    original_dense_units = self.dense_units
                    
                    # Simplify the model architecture
                    self.lstm_units = [32, 16]  # Simpler LSTM layers
                    self.dense_units = [16]     # Simpler dense layers
                    
                    # Rebuild the model
                    self.build_model(
                        stock_shape=x_stock.shape[1:],
                        features_shape=x_features.shape[1:]
                    )
                    
                    # Try training with the simpler model
                    self.logger.info("Retraining with simplified model architecture...")
                    history = self.model.fit(
                        x=[x_stock, x_features],
                        y=y_train,
                        epochs=min(10, epochs),  # Fewer epochs for the recovery attempt
                        batch_size=batch_size * 2,  # Larger batch size for stability
                        validation_split=validation_split,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=1
                    )
                    
                    self.history = history.history
                    self.logger.info("Recovery training completed successfully")
                    return history
                    
                except Exception as recovery_error:
                    self.logger.error(f"Recovery training also failed: {str(recovery_error)}")
                    raise RuntimeError("Both initial and recovery training attempts failed") from recovery_error
                finally:
                    # Restore original configuration for future attempts
                    self.lstm_units = original_lstm_units
                    self.dense_units = original_dense_units
            
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
    def evaluate(self, x_test: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data with enhanced error handling and diagnostics
        
        Args:
            x_test: Tuple of (stock_data, additional_features) arrays
            y_test: Target values
            
        Returns:
            Dict of evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            # Ensure x_test is a tuple of (stock_data, additional_features)
            if not isinstance(x_test, tuple) or len(x_test) != 2:
                raise ValueError("x_test must be a tuple of (stock_data, additional_features)")
                
            stock_data, additional_features = x_test
            
            # Convert to numpy arrays if they aren't already
            stock_data = np.asarray(stock_data, dtype=np.float32)
            additional_features = np.asarray(additional_features, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
            
            # Log data statistics before processing
            self.logger.info(f"Evaluation data shapes - stock_data: {stock_data.shape}, "
                            f"additional_features: {additional_features.shape}, y_test: {y_test.shape}")
            
            # Check for NaN or infinite values with detailed reporting
            nan_count_stock = np.isnan(stock_data).sum()
            inf_count_stock = np.isinf(stock_data).sum()
            if nan_count_stock > 0 or inf_count_stock > 0:
                self.logger.warning(f"Evaluation stock_data contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
                # Replace NaN/inf with interpolated values
                stock_data = np.nan_to_num(stock_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_features = np.isnan(additional_features).sum()
            inf_count_features = np.isinf(additional_features).sum()
            if nan_count_features > 0 or inf_count_features > 0:
                self.logger.warning(f"Evaluation additional_features contains {nan_count_features} NaN and {inf_count_features} infinite values")
                # Replace NaN/inf with interpolated values
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_y = np.isnan(y_test).sum()
            inf_count_y = np.isinf(y_test).sum()
            if nan_count_y > 0 or inf_count_y > 0:
                self.logger.warning(f"Evaluation y_test contains {nan_count_y} NaN and {inf_count_y} infinite values")
                # Replace NaN/inf with interpolated values
                y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Validate and reshape inputs
            stock_data, additional_features, y_test = self._validate_and_reshape_inputs(
                stock_data, additional_features, y_test
            )
            
            # Evaluate the model with error handling
            self.logger.info(f"Evaluating model on {len(stock_data)} samples")
            try:
                metrics = self.model.evaluate(
                    x=[stock_data, additional_features],
                    y=y_test,
                    verbose=1
                )
                
                # Create a dictionary of metric names and values
                metric_names = self.model.metrics_names
                metrics_dict = dict(zip(metric_names, metrics))
                
                # Check for extreme metric values that might indicate issues
                for name, value in metrics_dict.items():
                    if np.isnan(value) or np.isinf(value):
                        self.logger.error(f"Evaluation metric {name} is {value} (NaN or Inf). Model may be unstable.")
                    elif abs(value) > 1000:
                        self.logger.warning(f"Evaluation metric {name} is {value:.2f}, which is unusually high. Model may be unstable.")
                    else:
                        self.logger.info(f"{name}: {value:.4f}")
                        
                return metrics_dict
                
            except Exception as eval_error:
                self.logger.error(f"Error during model evaluation: {str(eval_error)}")
                # Return a dictionary with error information
                return {"error": str(eval_error), "status": "failed"}
                
        except Exception as e:
            error_msg = f"Error during model evaluation setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Ensure x_test is a tuple of (stock_data, additional_features)
        if not isinstance(x_test, tuple) or len(x_test) != 2:
            raise ValueError("x_test must be a tuple of (stock_data, additional_features)")

        stock_data, additional_features = x_test

        # Convert to numpy arrays if they aren't already
        stock_data = np.asarray(stock_data, dtype=np.float32)
        additional_features = np.asarray(additional_features, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        # Log data statistics before processing
        self.logger.info(f"Evaluation data shapes - stock_data: {stock_data.shape}, "
                         f"additional_features: {additional_features.shape}, y_test: {y_test.shape}")

        # Check for NaN or infinite values with detailed reporting
        nan_count_stock = np.isnan(stock_data).sum()
        inf_count_stock = np.isinf(stock_data).sum()
        if nan_count_stock > 0 or inf_count_stock > 0:
            self.logger.warning(f"Evaluation stock_data contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
            # Replace NaN/inf with interpolated values
        try:
            # Log input shapes for debugging
            self.logger.info(f"Prediction input - stock_data: {np.array(stock_data).shape}, "
                      f"additional_features: {np.array(additional_features).shape}")
            
            # Ensure inputs are numpy arrays with correct shape and type
            stock_data = np.array(stock_data, dtype=np.float32)
            additional_features = np.array(additional_features, dtype=np.float32)
            
            # Check for NaN or infinite values with detailed reporting
            nan_count_stock = np.isnan(stock_data).sum()
            inf_count_stock = np.isinf(stock_data).sum()
            if nan_count_stock > 0 or inf_count_stock > 0:
                self.logger.warning(f"Prediction stock_data contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
                # Replace NaN/inf with interpolated values
                self.logger.info("Cleaning stock_data by replacing NaN/inf values with interpolation")
                stock_data = np.nan_to_num(stock_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_features = np.isnan(additional_features).sum()
            inf_count_features = np.isinf(additional_features).sum()
            if nan_count_features > 0 or inf_count_features > 0:
                self.logger.warning(f"Prediction additional_features contains {nan_count_features} NaN and {inf_count_features} infinite values")
                # Replace NaN/inf with interpolated values
                self.logger.info("Cleaning additional_features by replacing NaN/inf values with interpolation")
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            # Ensure stock_data is 3D (samples, timesteps, features)
            if len(stock_data.shape) == 2:
                self.logger.info(f"Expanding stock_data from 2D shape {stock_data.shape} to 3D")
                stock_data = np.expand_dims(stock_data, axis=-1)
                
            # Ensure additional_features is 3D (samples, timesteps, features)
            if len(additional_features.shape) == 2:
                self.logger.info(f"Expanding additional_features from 2D shape {additional_features.shape} to 3D")
                additional_features = np.expand_dims(additional_features, axis=-1)
            
            # Log data ranges for debugging
            self.logger.info(f"Data ranges - stock_data: [{np.min(stock_data):.4f}, {np.max(stock_data):.4f}], "
                           f"additional_features: [{np.min(additional_features):.4f}, {np.max(additional_features):.4f}]")
                
            # Make predictions with error handling
            try:
                self.logger.info("Making predictions...")
                predictions_scaled = self.model.predict([stock_data, additional_features], verbose=0)
                
                # Check for NaN or infinite values in predictions
                nan_count_pred = np.isnan(predictions_scaled).sum()
                inf_count_pred = np.isinf(predictions_scaled).sum()
                if nan_count_pred > 0 or inf_count_pred > 0:
                    self.logger.warning(f"Predictions contain {nan_count_pred} NaN and {inf_count_pred} infinite values")
                    # Replace NaN/inf with interpolated values
                    self.logger.info("Cleaning predictions by replacing NaN/inf values")
                    predictions_scaled = np.nan_to_num(predictions_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Get the number of samples in the input
                n_samples = stock_data.shape[0]
                
                # Ensure predictions is 2D for inverse transform
                if predictions_scaled.ndim > 2:
                    # If we have multiple predictions per sample, take the last one
                    if predictions_scaled.shape[1] > 1:
                        self.logger.info(f"Taking last prediction from shape {predictions_scaled.shape}")
                        predictions_scaled = predictions_scaled[:, -1, :]  # Take last prediction for each sample
                    else:
                        predictions_scaled = predictions_scaled.reshape(-1, 1)
                elif predictions_scaled.ndim == 1:
                    predictions_scaled = predictions_scaled.reshape(-1, 1)
                
                # Ensure we have the correct number of predictions
                if len(predictions_scaled) != n_samples:
                    self.logger.warning(f"Number of predictions ({len(predictions_scaled)}) doesn't match number of samples ({n_samples}). Truncating/expanding.")
                    if len(predictions_scaled) > n_samples:
                        predictions_scaled = predictions_scaled[:n_samples]
                    else:
                        padding = np.zeros((n_samples - len(predictions_scaled), 1))
                        predictions_scaled = np.vstack([predictions_scaled, padding])
                
                # Log reshaped predictions
                self.logger.info(f"Reshaped predictions for inverse transform: {predictions_scaled.shape}, "
                              f"range: [{np.min(predictions_scaled):.4f}, {np.max(predictions_scaled):.4f}]")
                    
                # Inverse transform predictions to original scale
                try:
                    # Check if target_scaler exists and is fitted
                    if not hasattr(self, 'target_scaler') or self.target_scaler is None:
                        self.logger.error("target_scaler is not initialized")
                        self.target_scaler = MinMaxScaler()
                        self.logger.warning("Initialized a new target_scaler, but it's not fitted yet")
                        
                    if not hasattr(self.target_scaler, 'scale_'):
                        self.logger.warning("target_scaler is not fitted. Using raw predictions.")
                        predictions = predictions_scaled.flatten()
                    else:
                        predictions = self.target_scaler.inverse_transform(predictions_scaled)
                        # Ensure output is 1D with length matching input samples
                        predictions = np.asarray(predictions).flatten()
                        
                    self.logger.info(f"Final predictions shape: {predictions.shape}, "
                                  f"range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
                    return predictions
                    
                except Exception as inverse_error:
                    self.logger.error(f"Error in inverse transform: {str(inverse_error)}")
                    # Return the raw predictions if inverse transform fails
                    self.logger.warning("Returning raw predictions without inverse transform")
                    return predictions_scaled.flatten()
                    
            except Exception as pred_error:
                error_msg = f"Error during model.predict(): {str(pred_error)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Try with a smaller batch size as a fallback
                try:
                    self.logger.warning("Attempting prediction with smaller batch size as fallback")
                    n_samples = stock_data.shape[0]
                    predictions_scaled = np.zeros((n_samples, 1), dtype=np.float32)
                    
                    # Process in smaller batches
                    batch_size = 1
                    for i in range(0, n_samples, batch_size):
                        batch_stock = stock_data[i:i+batch_size]
                        batch_features = additional_features[i:i+batch_size]
                        batch_pred = self.model.predict([batch_stock, batch_features], verbose=0)
                        predictions_scaled[i:i+batch_size] = batch_pred
                    
                    self.logger.info("Fallback prediction completed successfully")
                    
                    # Inverse transform if possible
                    if hasattr(self, 'target_scaler') and hasattr(self.target_scaler, 'scale_'):
                        predictions = self.target_scaler.inverse_transform(predictions_scaled)
                        return np.asarray(predictions).flatten()
                    else:
                        return predictions_scaled.flatten()
                        
                except Exception as fallback_error:
                    error_msg = f"Fallback prediction also failed: {str(fallback_error)}"
                    self.logger.error(error_msg, exc_info=True)
                    # Return zeros as last resort
                    self.logger.warning(f"Returning zeros array of shape ({stock_data.shape[0]},)")
                    return np.zeros(stock_data.shape[0])
            
        except Exception as e:
            error_msg = f"Error during prediction setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Return empty array as last resort
            if 'stock_data' in locals() and hasattr(stock_data, 'shape'):
                return np.zeros(stock_data.shape[0])
            else:
                return np.array([])
        
    def evaluate(self, x_test: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data with enhanced error handling and diagnostics
        
        Args:
            x_test: Tuple of (stock_data, additional_features) arrays
            y_test: Target values
            
        Returns:
            Dict of evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            # Ensure x_test is a tuple of (stock_data, additional_features)
            if not isinstance(x_test, tuple) or len(x_test) != 2:
                raise ValueError("x_test must be a tuple of (stock_data, additional_features)")
                
            stock_data, additional_features = x_test
            
            # Convert to numpy arrays if they aren't already
            stock_data = np.asarray(stock_data, dtype=np.float32)
            additional_features = np.asarray(additional_features, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
            
            # Log data statistics before processing
            self.logger.info(f"Evaluation data shapes - stock_data: {stock_data.shape}, "
                            f"additional_features: {additional_features.shape}, y_test: {y_test.shape}")
            
            # Check for NaN or infinite values with detailed reporting
            nan_count_stock = np.isnan(stock_data).sum()
            inf_count_stock = np.isinf(stock_data).sum()
            if nan_count_stock > 0 or inf_count_stock > 0:
                self.logger.warning(f"Evaluation stock_data contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
                # Replace NaN/inf with interpolated values
                stock_data = np.nan_to_num(stock_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_features = np.isnan(additional_features).sum()
            inf_count_features = np.isinf(additional_features).sum()
            if nan_count_features > 0 or inf_count_features > 0:
                self.logger.warning(f"Evaluation additional_features contains {nan_count_features} NaN and {inf_count_features} infinite values")
                # Replace NaN/inf with interpolated values
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_y = np.isnan(y_test).sum()
            inf_count_y = np.isinf(y_test).sum()
            if nan_count_y > 0 or inf_count_y > 0:
                self.logger.warning(f"Evaluation y_test contains {nan_count_y} NaN and {inf_count_y} infinite values")
                # Replace NaN/inf with interpolated values
                y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Validate and reshape inputs
            stock_data, additional_features, y_test = self._validate_and_reshape_inputs(
                stock_data, additional_features, y_test
            )
            
            # Evaluate the model with error handling
            self.logger.info(f"Evaluating model on {len(stock_data)} samples")
            try:
                metrics = self.model.evaluate(
                    x=[stock_data, additional_features],
                    y=y_test,
                    verbose=1
                )
                
                # Create a dictionary of metric names and values
                metric_names = self.model.metrics_names
                metrics_dict = dict(zip(metric_names, metrics))
                
                # Check for extreme metric values that might indicate issues
                for name, value in metrics_dict.items():
                    if np.isnan(value) or np.isinf(value):
                        self.logger.error(f"Evaluation metric {name} is {value} (NaN or Inf). Model may be unstable.")
                    elif abs(value) > 1000:
                        self.logger.warning(f"Evaluation metric {name} is {value:.2f}, which is unusually high. Model may be unstable.")
                    else:
                        self.logger.info(f"{name}: {value:.4f}")
                        
                return metrics_dict
                
            except Exception as eval_error:
                self.logger.error(f"Error during model evaluation: {str(eval_error)}")
                # Return a dictionary with error information
                return {"error": str(eval_error), "status": "failed"}
                
        except Exception as e:
            error_msg = f"Error during model evaluation setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Return a dictionary with error information
            return {"error": error_msg, "status": "failed"}
            
        if not hasattr(self.target_scaler, 'scale_'):
            self.logger.warning("target_scaler is not fitted. Will attempt to fit with provided y_test")
            try:
                self.target_scaler.fit(y_test.reshape(-1, 1))
                self.logger.info("Successfully fitted target_scaler with y_test")
            except Exception as e:
                self.logger.error(f"Failed to fit target_scaler: {str(e)}")
                raise ValueError("target_scaler is not properly initialized and could not be fitted")
            
        try:
            # Unpack and validate inputs
            stock_data, additional_features = x_test
            
            # Log input shapes for debugging
            self.logger.debug(f"Evaluation input shapes - stock_data: {np.array(stock_data).shape}, "
                      f"additional_features: {np.array(additional_features).shape}, "
                      f"y_test: {np.array(y_test).shape}")
            
            # Ensure inputs are numpy arrays with correct shape and type
            stock_data = np.array(stock_data, dtype=np.float32)
            additional_features = np.array(additional_features, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)
                
            # Ensure stock_data is 3D (samples, timesteps, features)
            if len(stock_data.shape) == 2:
                stock_data = np.expand_dims(stock_data, axis=-1)
                
            # Ensure additional_features is 3D (samples, timesteps, features)
            if len(additional_features.shape) == 2:
                additional_features = np.expand_dims(additional_features, axis=-1)
            
            # Log shapes before prediction
            self.logger.debug(f"Shapes before prediction - stock_data: {stock_data.shape}, "
                       f"additional_features: {additional_features.shape}")
            
            # Make predictions
            predictions_scaled = self.model.predict([stock_data, additional_features], verbose=0)
            
            # Ensure predictions is 2D for inverse transform
            if predictions_scaled.ndim > 2:
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            elif predictions_scaled.ndim == 1:
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            
            # Ensure y_test is 2D for inverse transform
            y_test_2d = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
            
            # Inverse transform predictions and y_test to original scale
            predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
            y_test_original = self.target_scaler.inverse_transform(y_test_2d).flatten()
            
            # Ensure lengths match
            min_len = min(len(predictions), len(y_test_original))
            predictions = predictions[:min_len]
            y_test_original = y_test_original[:min_len]
            
            # Calculate metrics
            metrics = {}
            
            # Calculate MAPE (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.nanmean(np.abs((y_test_original - predictions) / y_test_original)) * 100
            metrics['mape'] = mape
            
            # Calculate directional accuracy (skip if only one data point)
            if len(y_test_original) > 1:
                direction_correct = np.sign(predictions[1:] - predictions[:-1]) == \
                                    np.sign(y_test_original[1:] - y_test_original[:-1])
                directional_accuracy = np.mean(direction_correct) * 100
                metrics['directional_accuracy'] = directional_accuracy
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((predictions - y_test_original) ** 2))
            metrics['rmse'] = rmse
            
            # Log metrics for debugging
            self.logger.info(
                "Evaluation Metrics - "
                f"MAPE: {mape:.2f}% "
                f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}% "
                f"RMSE: {rmse:.4f}"
            )
            
            # Ensure all metrics are stored in the metrics dictionary
            metrics.update({
                'mape': mape,
                'rmse': rmse,
                'directional_accuracy': metrics.get('directional_accuracy', 0)
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            self.logger.error(f"Input shapes - stock_data: {stock_data.shape if 'stock_data' in locals() else 'N/A'}, "
                       f"additional_features: {additional_features.shape if 'additional_features' in locals() else 'N/A'}")
            self.logger.error(f"Predictions shape: {predictions_scaled.shape if 'predictions_scaled' in locals() else 'N/A'}")
            self.logger.error(f"Scaler fitted: {hasattr(self.target_scaler, 'scale_')}")
            raise
        
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate sample weights for regression based on target values.
        Weights are higher for more recent samples to prioritize recent trends.
        
        Args:
            y: Target values (n_samples, 1)
            
        Returns:
            Dictionary of sample weights (all 1.0 for equal weighting)
        """
        # For regression, we can return equal weights
        # or implement time-based weighting if needed
        n_samples = len(y)
        return {i: 1.0 for i in range(n_samples)}
        
    def evaluate(self, x_test: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data with enhanced error handling and diagnostics
        
        Args:
            x_test: Tuple of (stock_data, additional_features) arrays
            y_test: Target values
            
        Returns:
            Dict of evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            # Ensure x_test is a tuple of (stock_data, additional_features)
            if not isinstance(x_test, tuple) or len(x_test) != 2:
                raise ValueError("x_test must be a tuple of (stock_data, additional_features)")
                
            stock_data, additional_features = x_test
            
            # Convert to numpy arrays if they aren't already
            stock_data = np.asarray(stock_data, dtype=np.float32)
            additional_features = np.asarray(additional_features, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
            
            # Log data statistics before processing
            self.logger.info(f"Evaluation data shapes - stock_data: {stock_data.shape}, "
                            f"additional_features: {additional_features.shape}, y_test: {y_test.shape}")
            
            # Check for NaN or infinite values with detailed reporting
            nan_count_stock = np.isnan(stock_data).sum()
            inf_count_stock = np.isinf(stock_data).sum()
            if nan_count_stock > 0 or inf_count_stock > 0:
                self.logger.warning(f"Evaluation stock_data contains {nan_count_stock} NaN and {inf_count_stock} infinite values")
                # Replace NaN/inf with interpolated values
                stock_data = np.nan_to_num(stock_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_features = np.isnan(additional_features).sum()
            inf_count_features = np.isinf(additional_features).sum()
            if nan_count_features > 0 or inf_count_features > 0:
                self.logger.warning(f"Evaluation additional_features contains {nan_count_features} NaN and {inf_count_features} infinite values")
                # Replace NaN/inf with interpolated values
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_y = np.isnan(y_test).sum()
            inf_count_y = np.isinf(y_test).sum()
            if nan_count_y > 0 or inf_count_y > 0:
                self.logger.warning(f"Evaluation y_test contains {nan_count_y} NaN and {inf_count_y} infinite values")
                # Replace NaN/inf with interpolated values
                y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Validate and reshape inputs
            stock_data, additional_features, y_test = self._validate_and_reshape_inputs(
                stock_data, additional_features, y_test
            )
            
            # Evaluate the model with error handling
            self.logger.info(f"Evaluating model on {len(stock_data)} samples")
            try:
                metrics = self.model.evaluate(
                    x=[stock_data, additional_features],
                    y=y_test,
                    verbose=1
                )
                
                # Create a dictionary of metric names and values
                metric_names = self.model.metrics_names
                metrics_dict = dict(zip(metric_names, metrics))
                
                # Check for extreme metric values that might indicate issues
                for name, value in metrics_dict.items():
                    if np.isnan(value) or np.isinf(value):
                        self.logger.error(f"Evaluation metric {name} is {value} (NaN or Inf). Model may be unstable.")
                    elif abs(value) > 1000:
                        self.logger.warning(f"Evaluation metric {name} is {value:.2f}, which is unusually high. Model may be unstable.")
                    else:
                        self.logger.info(f"{name}: {value:.4f}")
                        
                return metrics_dict
                
            except Exception as eval_error:
                self.logger.error(f"Error during model evaluation: {str(eval_error)}")
                # Return a dictionary with error information
                return {"error": str(eval_error), "status": "failed"}
                
        except Exception as e:
            error_msg = f"Error during model evaluation setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Return a dictionary with error information
            return {"error": error_msg, "status": "failed"}
            
        if not hasattr(self.target_scaler, 'scale_'):
            self.logger.warning("target_scaler is not fitted. Will attempt to fit with provided y_test")
            try:
                self.target_scaler.fit(y_test.reshape(-1, 1))
                self.logger.info("Successfully fitted target_scaler with y_test")
            except Exception as e:
                self.logger.error(f"Failed to fit target_scaler: {str(e)}")
                raise ValueError("target_scaler is not properly initialized and could not be fitted")
            
        try:
            # Unpack and validate inputs
            stock_data, additional_features = x_test
            
            # Log input shapes for debugging
            self.logger.debug(f"Evaluation input shapes - stock_data: {np.array(stock_data).shape}, "
                      f"additional_features: {np.array(additional_features).shape}, "
                      f"y_test: {np.array(y_test).shape}")
            
            # Ensure inputs are numpy arrays with correct shape and type
            stock_data = np.array(stock_data, dtype=np.float32)
            additional_features = np.array(additional_features, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)
                
            # Ensure stock_data is 3D (samples, timesteps, features)
            if len(stock_data.shape) == 2:
                stock_data = np.expand_dims(stock_data, axis=-1)
                
            # Ensure additional_features is 3D (samples, timesteps, features)
            if len(additional_features.shape) == 2:
                additional_features = np.expand_dims(additional_features, axis=-1)
            
            # Log shapes before prediction
            self.logger.debug(f"Shapes before prediction - stock_data: {stock_data.shape}, "
                       f"additional_features: {additional_features.shape}")
            
            # Make predictions
            predictions_scaled = self.model.predict([stock_data, additional_features], verbose=0)
            
            # Ensure predictions is 2D for inverse transform
            if predictions_scaled.ndim > 2:
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            elif predictions_scaled.ndim == 1:
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            
            # Ensure y_test is 2D for inverse transform
            y_test_2d = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
            
            # Inverse transform predictions and y_test to original scale
            predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
            y_test_original = self.target_scaler.inverse_transform(y_test_2d).flatten()
            
            # Ensure lengths match
            min_len = min(len(predictions), len(y_test_original))
            predictions = predictions[:min_len]
            y_test_original = y_test_original[:min_len]
            
            # Calculate metrics
            metrics = {}
            
            # Calculate MAPE (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.nanmean(np.abs((y_test_original - predictions) / y_test_original)) * 100
            metrics['mape'] = mape
            
            # Calculate directional accuracy (skip if only one data point)
            if len(y_test_original) > 1:
                direction_correct = np.sign(predictions[1:] - predictions[:-1]) == \
                                    np.sign(y_test_original[1:] - y_test_original[:-1])
                directional_accuracy = np.mean(direction_correct) * 100
                metrics['directional_accuracy'] = directional_accuracy
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((predictions - y_test_original) ** 2))
            metrics['rmse'] = rmse
            
            # Log metrics for debugging
            self.logger.info(
                "Evaluation Metrics - "
                f"MAPE: {mape:.2f}% "
                f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}% "
                f"RMSE: {rmse:.4f}"
            )
            
            # Ensure all metrics are stored in the metrics dictionary
            metrics.update({
                'mape': mape,
                'rmse': rmse,
                'directional_accuracy': metrics.get('directional_accuracy', 0)
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            self.logger.error(f"Input shapes - stock_data: {stock_data.shape if 'stock_data' in locals() else 'N/A'}, "
                           f"additional_features: {additional_features.shape if 'additional_features' in locals() else 'N/A'}")
            self.logger.error(f"y_test shape: {y_test.shape if 'y_test' in locals() else 'N/A'}")
            raise
            
    def predict(self, stock_data: np.ndarray, additional_features: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using the trained model with enhanced error handling and debugging.
        
        Args:
            stock_data: Either stock price sequences (n_samples, seq_len, n_stock_features) or
                       a tuple of (stock_data, additional_features) if called from Keras
            additional_features: Additional feature sequences (n_samples, seq_len, n_features).
                               Can be None if stock_data contains all necessary data or is a tuple.
            
        Returns:
            Predictions array (n_samples, 1)
            
        Raises:
            ValueError: If model is not trained or inputs have invalid shapes
        """
        # Ensure model is trained
        if self.model is None:
            error_msg = "Model not trained yet. Cannot make predictions."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Handle different input patterns
            # Case 1: stock_data is actually a tuple of (stock_data, additional_features)
            if isinstance(stock_data, tuple) and len(stock_data) == 2 and additional_features is None:
                self.logger.info("Detected tuple input pattern, unpacking...")
                stock_data, additional_features = stock_data
            # Case 2: stock_data is a single array and additional_features is None
            elif additional_features is None:
                self.logger.info("No additional features provided, using empty array")
                # Create empty additional features with same batch size and sequence length
                if len(stock_data.shape) == 3:  # (batch, seq_len, features)
                    additional_features = np.zeros((stock_data.shape[0], stock_data.shape[1], 0), dtype=np.float32)
                elif len(stock_data.shape) == 2:  # (seq_len, features)
                    additional_features = np.zeros((stock_data.shape[0], 0), dtype=np.float32)
            
            # Convert inputs to numpy arrays with error handling
            try:
                stock_data = np.asarray(stock_data, dtype=np.float32)
                additional_features = np.asarray(additional_features, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Error converting inputs to numpy arrays: {str(e)}")
                raise ValueError(f"Invalid input format: {str(e)}")
                
            # Log input shapes and check for NaN/inf values
            self.logger.info(f"Input shapes - stock_data: {stock_data.shape}, additional_features: {additional_features.shape}")
            
            # Check for NaN or infinite values
            if np.isnan(stock_data).any() or np.isinf(stock_data).any():
                self.logger.warning("Input stock_data contains NaN or infinite values. Cleaning...")
                stock_data = np.nan_to_num(stock_data, nan=0.0, posinf=1.0, neginf=0.0)
                
            if np.isnan(additional_features).any() or np.isinf(additional_features).any():
                self.logger.warning("Input additional_features contains NaN or infinite values. Cleaning...")
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure inputs have the right shape for the model
            if len(stock_data.shape) == 2:
                stock_data = np.expand_dims(stock_data, axis=0)
                self.logger.info(f"Expanded stock_data dimensions to {stock_data.shape}")
                
            if len(additional_features.shape) == 2:
                additional_features = np.expand_dims(additional_features, axis=0)
                self.logger.info(f"Expanded additional_features dimensions to {additional_features.shape}")
            
            # Make prediction - our model always expects two inputs
            try:
                self.logger.info("Using [stock_data, additional_features] as input")
                predictions = self.model.predict([stock_data, additional_features], verbose=0)
                    
                self.logger.info(f"Prediction shape: {predictions.shape}, range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
            except Exception as pred_error:
                self.logger.error(f"Error during model prediction: {str(pred_error)}")
                
                # Try predicting in smaller batches if batch size might be the issue
                if len(stock_data) > 1:
                    self.logger.info("Attempting to predict in smaller batches...")
                    predictions = []
                    for i in range(len(stock_data)):
                        try:
                            # Always use both inputs for our model
                            batch_pred = self.model.predict(
                                [stock_data[i:i+1], additional_features[i:i+1]], 
                                verbose=0
                            )
                                
                            predictions.append(batch_pred[0])
                        except Exception as batch_error:
                            self.logger.error(f"Error predicting batch {i}: {str(batch_error)}")
                            # Use zero as fallback
                            predictions.append(np.array([0.0]))
                    
                    predictions = np.array(predictions)
                    self.logger.info(f"Batch prediction complete. Shape: {predictions.shape}")
                else:
                    # Last resort with proper input format
                    self.logger.info("Trying last resort prediction...")
                    predictions = self.model.predict([stock_data, additional_features], verbose=0)
                    self.logger.info("Last resort prediction succeeded")
            
            # Check for NaN or infinite values in predictions
            if np.isnan(predictions).any() or np.isinf(predictions).any():
                self.logger.warning("Predictions contain NaN or infinite values. Cleaning...")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Reshape predictions if necessary for inverse transformation
            original_shape = predictions.shape
            self.logger.info(f"Original prediction shape: {original_shape}")
            
            # Check for unusually large predictions array that might indicate a reshaping issue
            total_elements = np.prod(original_shape)
            expected_max_elements = 10000  # A reasonable upper limit for most stock prediction tasks
            
            if total_elements > expected_max_elements:
                self.logger.warning(f"Prediction array unusually large with {total_elements} elements. Attempting to fix shape.")
                # Try to determine the correct shape based on the first dimension (samples)
                if len(original_shape) >= 2 and original_shape[0] > 0:
                    # Reshape to have the correct number of samples with a single feature
                    predictions = predictions.reshape(original_shape[0], -1)
                    # If still too large, take only the first feature
                    if predictions.shape[1] > 10:  # Arbitrary threshold for reasonable feature count
                        self.logger.warning(f"Still too many features ({predictions.shape[1]}). Taking only first feature.")
                        predictions = predictions[:, 0:1]
                    self.logger.info(f"Fixed prediction shape: {predictions.shape}")
                    original_shape = predictions.shape
            
            # Reshape based on dimensionality
            if len(original_shape) == 3:  # (samples, timesteps, features)
                self.logger.info("Reshaping 3D predictions for inverse transformation")
                # Extract the last timestep for each sample if predicting sequences
                if original_shape[1] > 1:  # Multiple timesteps
                    predictions_2d = predictions[:, -1, :]
                    self.logger.info(f"Extracted last timestep, new shape: {predictions_2d.shape}")
                else:  # Single timestep
                    predictions_2d = predictions.reshape(original_shape[0], original_shape[2])
                    self.logger.info(f"Reshaped to 2D, new shape: {predictions_2d.shape}")
            else:
                predictions_2d = predictions
            
            # Inverse transform if scaler is available
            if self.target_scaler is not None and hasattr(self.target_scaler, 'inverse_transform'):
                try:
                    # Ensure predictions are 2D for inverse_transform
                    if len(predictions_2d.shape) == 1:
                        predictions_2d = predictions_2d.reshape(-1, 1)
                    
                    # Check if predictions_2d has the right shape for the scaler
                    if predictions_2d.shape[1] != self.target_scaler.n_features_in_:
                        self.logger.warning(f"Prediction features ({predictions_2d.shape[1]}) don't match scaler features ({self.target_scaler.n_features_in_}). Reshaping...")
                        # Adjust to match the expected number of features
                        if predictions_2d.shape[1] > self.target_scaler.n_features_in_:
                            # Take only the needed features
                            predictions_2d = predictions_2d[:, :self.target_scaler.n_features_in_]
                        else:
                            # Pad with zeros if we have too few features
                            pad_width = ((0, 0), (0, self.target_scaler.n_features_in_ - predictions_2d.shape[1]))
                            predictions_2d = np.pad(predictions_2d, pad_width, mode='constant')
                    
                    inverse_predictions = self.target_scaler.inverse_transform(predictions_2d)
                    self.logger.info(f"Inverse transformed predictions range: [{np.min(inverse_predictions):.4f}, {np.max(inverse_predictions):.4f}]")
                    
                    # If we need to restore the original shape (for sequence outputs)
                    if len(original_shape) == 3 and original_shape[1] > 1:
                        # Create a copy of the original predictions
                        restored_shape = np.copy(predictions)
                        # Replace only the last timestep with inverse transformed values
                        restored_shape[:, -1, :] = inverse_predictions
                        return restored_shape
                    else:
                        return inverse_predictions
                        
                except Exception as scaler_error:
                    self.logger.error(f"Error during inverse transformation: {str(scaler_error)}")
                    self.logger.warning("Returning raw predictions without inverse transformation")
                    return predictions
            else:
                self.logger.info("No target scaler available for inverse transformation")
                return predictions
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise
    
    def predict_future(self, last_sequence: np.ndarray,
                     additional_features: np.ndarray,
                     days_ahead: int,
                     date_index: pd.DatetimeIndex = None) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Generate future predictions for specified number of days ahead with enhanced error handling and debugging.
        
        Args:
            last_sequence: Last known sequence of stock prices (seq_len, n_features)
            additional_features: Additional features for the sequence (seq_len, n_features)
            days_ahead: Number of days to predict ahead
            date_index: DatetimeIndex for the last sequence (for generating future dates)
            
        Returns:
            Tuple of (predictions array, dates array)
        """
        # Validate days_ahead parameter with detailed logging
        self.logger.info(f"Predicting {days_ahead} days ahead")
        if days_ahead <= 0:
            error_msg = f"days_ahead must be greater than 0, got {days_ahead}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Ensure model is trained
        if self.model is None:
            error_msg = "Model not trained yet. Cannot make future predictions."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
                
            # Convert inputs to numpy arrays with error handling
            try:
                last_sequence = np.array(last_sequence, dtype=np.float32)
                additional_features = np.array(additional_features, dtype=np.float32)
            except Exception as convert_error:
                self.logger.error(f"Error converting inputs to numpy arrays: {str(convert_error)}")
                raise ValueError(f"Invalid input data format: {str(convert_error)}") from convert_error
            
            # Log input shapes and check for NaN/inf values
            self.logger.info(f"Input shapes - last_sequence: {last_sequence.shape}, additional_features: {additional_features.shape}")
            
            # Check for NaN or infinite values
            nan_count_seq = np.isnan(last_sequence).sum()
            inf_count_seq = np.isinf(last_sequence).sum()
            if nan_count_seq > 0 or inf_count_seq > 0:
                self.logger.warning(f"Input sequence contains {nan_count_seq} NaN and {inf_count_seq} infinite values")
                self.logger.info("Cleaning input sequence by replacing NaN/inf values")
                last_sequence = np.nan_to_num(last_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
                
            nan_count_feat = np.isnan(additional_features).sum()
            inf_count_feat = np.isinf(additional_features).sum()
            if nan_count_feat > 0 or inf_count_feat > 0:
                self.logger.warning(f"Additional features contain {nan_count_feat} NaN and {inf_count_feat} infinite values")
                self.logger.info("Cleaning additional features by replacing NaN/inf values")
                additional_features = np.nan_to_num(additional_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Reshape inputs if needed with detailed logging
            if len(last_sequence.shape) == 2:
                self.logger.info(f"Reshaping last_sequence from 2D shape {last_sequence.shape} to 3D")
                # Add batch dimension (1, seq_len, n_features)
                last_sequence = np.expand_dims(last_sequence, axis=0)
                
            if len(additional_features.shape) == 2:
                self.logger.info(f"Reshaping additional_features from 2D shape {additional_features.shape} to 3D")
                # Add batch dimension (1, seq_len, n_features)
                additional_features = np.expand_dims(additional_features, axis=0)
                
            # Log reshaped input dimensions
            self.logger.info(f"Reshaped inputs - last_sequence: {last_sequence.shape}, additional_features: {additional_features.shape}")
            
            # Initialize arrays to store predictions and dates
            future_preds = np.zeros(days_ahead)
            
            # Generate future dates if date_index is provided
            if date_index is not None:
                try:
                    last_date = date_index[-1]
                    self.logger.info(f"Generating future dates starting from {last_date}")
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=days_ahead,
                        freq='B'  # Business day frequency
                    )
                    self.logger.info(f"Generated {len(future_dates)} future business dates")
                except Exception as date_error:
                    self.logger.error(f"Error generating future dates: {str(date_error)}")
                    self.logger.warning("Continuing without date information")
                    future_dates = None
            else:
                self.logger.info("No date_index provided, future dates will not be generated")
                future_dates = None
                
            # Make a copy of the input sequences to avoid modifying the originals
            seq = last_sequence.copy()
            feat = additional_features.copy()
            
            # Iteratively predict future values with error handling
            self.logger.info(f"Starting iterative prediction for {days_ahead} days ahead")
            for i in range(days_ahead):
                try:
                    # Make prediction for next day
                    self.logger.debug(f"Predicting day {i+1}/{days_ahead}")
                    next_pred = self.predict((seq, feat))
                    
                    # Check for NaN or infinite values in prediction
                    if np.isnan(next_pred).any() or np.isinf(next_pred).any():
                        self.logger.warning(f"Prediction for day {i+1} contains NaN or infinite values")
                        # Replace with last valid prediction or zero
                        if i > 0:
                            next_pred = np.array([future_preds[i-1]])
                        else:
                            next_pred = np.array([seq[0, -1, 0]])  # Use last known value
                        self.logger.info(f"Replaced invalid prediction with value: {next_pred[0]}")
                    
                    # Store the prediction
                    future_preds[i] = next_pred[0]  # Take first (and only) value
                    self.logger.debug(f"Day {i+1} prediction: {future_preds[i]:.4f}")
                    
                    # Update sequence for next prediction by removing oldest value and adding new prediction
                    seq = np.roll(seq, -1, axis=1)
                    seq[0, -1, 0] = next_pred[0]  # Set last value to prediction
                    
                    # For additional features, we just shift (this is a simplification)
                    # In a real implementation, you might want to predict these features as well
                    feat = np.roll(feat, -1, axis=1)
                    
                except Exception as pred_error:
                    self.logger.error(f"Error predicting day {i+1}: {str(pred_error)}")
                    # Use last valid prediction or last known value
                    if i > 0:
                        future_preds[i] = future_preds[i-1]
                    else:
                        future_preds[i] = seq[0, -1, 0]  # Use last known value
                    self.logger.info(f"Using fallback value for day {i+1}: {future_preds[i]:.4f}")
            
            # Log prediction statistics
            self.logger.info(f"Future predictions complete. Shape: {future_preds.shape}, "
                          f"Range: [{np.min(future_preds):.4f}, {np.max(future_preds):.4f}]")
            
            # Check for extreme values in predictions
            if np.max(np.abs(future_preds)) > 1000:
                self.logger.warning("Future predictions contain extreme values. Model may be unstable.")
                
            return future_preds, future_dates
            
        except Exception as e:
            error_msg = f"Error during future prediction: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Return empty arrays as fallback
            self.logger.warning("Returning empty arrays due to prediction failure")
            return np.array([]), None

# Example usage:
if __name__ == "__main__":
    # Initialize model
    predictor = TwoStagePredictor()
    
    # Example data
    stock_data = pd.DataFrame({
        'Close': np.random.normal(100, 10, 100)
    })
    
    sentiment_data = pd.DataFrame({
        'Sentiment': np.random.normal(0, 1, 100)
    })
    
    macro_data = pd.DataFrame({
        'Interest_Rate': np.random.normal(2, 0.5, 100)
    })
    
    # Preprocess data
    x_train, y_train = predictor.preprocess_data(stock_data, sentiment_data, macro_data)
    
    # Train model
    history = predictor.train(x_train, y_train)
    
    # Make predictions
    predictions = predictor.predict(x_train)
    
    # Evaluate performance
    metrics = predictor.evaluate(x_train, y_train)
    print("\nPerformance Metrics:", metrics)
