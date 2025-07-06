import os
import sys
import logging
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import io
import json
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, BatchNormalization, Concatenate, Activation, Multiply, Lambda, Bidirectional
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import constraints

@tf.keras.utils.register_keras_serializable(package='two_stage_model')
class GradientCentralizedMaxNorm(Constraint):
    """Max norm constraint with gradient centralization.
    
    This constraint enforces a maximum norm on the weights of incoming connections
    to each hidden unit, after applying gradient centralization which helps improve
    optimization and generalization in deep neural networks.
    
    Args:
        max_value: The maximum norm for the incoming weights.
    """
    
    def __init__(self, max_value=1.5):
        self.max_value = max_value

    def __call__(self, w):
        # Gradient centralization: center the gradients to have zero mean
        w_mean = tf.math.reduce_mean(w, axis=[0, 1], keepdims=True)
        w_centered = w - w_mean
        
        # Apply max norm constraint
        norms = tf.sqrt(tf.reduce_sum(tf.square(w_centered), axis=[0, 1], keepdims=True))
        desired = tf.clip_by_value(norms, 0, self.max_value)
        w_normed = w_centered * (desired / (tf.maximum(norms, 1e-7)))
        return w_normed

    def get_config(self):
        return {'max_value': self.max_value}

# Custom constraint class combining MaxNorm and gradient centralization
@tf.keras.utils.register_keras_serializable(package='two_stage_model')
class GradientCentralizedMaxNorm(tf.keras.constraints.Constraint):
    def __init__(self, max_value=1.5):
        self.max_value = max_value
        self.max_norm = tf.keras.constraints.MaxNorm(max_value)
    
    def __call__(self, w):
        # Apply gradient centralization
        if len(w.shape) > 1:
            w = w - tf.reduce_mean(w, axis=list(range(len(w.shape)-1)), keepdims=True)
        # Apply max norm constraint
        return self.max_norm(w)
    
    def get_config(self):
        return {'max_value': self.max_value}

# Register the custom constraint
setattr(constraints, 'GradientCentralizedMaxNorm', GradientCentralizedMaxNorm)

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform, HeUniform, GlorotNormal, HeNormal
from tensorflow.keras.constraints import MaxNorm, UnitNorm, MinMaxNorm


@tf.keras.utils.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a warmup cosine decay schedule.
    
    This schedule applies a linear warmup phase followed by a cosine decay phase.
    During the warmup phase, the learning rate increases linearly from 0 to the
    initial learning rate. Then it follows a cosine decay schedule.
    
    Args:
        initial_learning_rate: The initial learning rate at the end of warmup.
        warmup_steps: Number of warmup steps.
        decay_steps: Number of steps to decay over.
        alpha: Minimum learning rate as a fraction of initial_learning_rate.
        name: Optional name for the operation.
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
        name: str = None
    ) -> None:
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            # Convert to float32 to avoid mixed precision issues
            initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            step = tf.cast(step, tf.float32)
            
            # Linear warmup phase
            warmup_percent_done = step / warmup_steps
            warmup_learning_rate = initial_learning_rate * warmup_percent_done
            
            # Cosine decay phase
            # Make sure decay_steps is positive to avoid division by zero
            decay_steps = tf.maximum(decay_steps, 1.0)
            decayed = (step - warmup_steps) / decay_steps
            decayed = tf.minimum(decay_steps, tf.maximum(0.0, decayed))
            cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * decayed))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            decayed_learning_rate = initial_learning_rate * decayed
            
            # Choose between warmup and decayed learning rate
            is_warmup = tf.cast(step < warmup_steps, tf.float32)
            learning_rate = (is_warmup * warmup_learning_rate + 
                           (1 - is_warmup) * decayed_learning_rate)
            
            return learning_rate
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha,
            'name': self.name
        }

# Custom TrainingMonitor callback for better training control
class TrainingMonitor(tf.keras.callbacks.Callback):
    """Custom callback for monitoring training progress and handling early stopping.
    
    This callback tracks the best model weights based on validation loss,
    implements early stopping, and provides detailed logging.
    """
    def __init__(self, logger=None, patience=10):
        """Initialize the TrainingMonitor.
        
        Args:
            logger: Logger instance for logging messages (defaults to root logger if None)
            patience: Number of epochs to wait before early stopping
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.best_weights = None
        self.best_val_loss = float('inf')
        self.patience = patience
        self.wait = 0
        self.initial_weights = None
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            raise AttributeError("Model has not been set yet. This callback must be used with a model.")
        return self._model
        
    @model.setter
    def model(self, model):
        self._model = model
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        import copy
        self.initial_weights = [w.copy() for w in self.model.get_weights()]
        self.best_weights = [w.copy() for w in self.initial_weights]
        self.best_val_loss = float('inf')
        self.wait = 0
        self.logger.info("TrainingMonitor: Initialized with model weights")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Safely get validation loss, defaulting to infinity if not available
        try:
            current_val_loss = float(logs.get('val_loss', float('inf')))
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Could not parse validation loss: {e}")
            current_val_loss = float('inf')
        
        # Safely get learning rate
        try:
            if hasattr(self.model.optimizer, 'learning_rate'):
                if hasattr(self.model.optimizer.learning_rate, 'numpy'):
                    lr = float(self.model.optimizer.learning_rate.numpy())
                else:
                    lr = float(self.model.optimizer.learning_rate)
            else:
                lr = float('nan')
        except Exception as e:
            self.logger.warning(f"Could not get learning rate: {e}")
            lr = float('nan')
        
        # Log training progress
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 'N/A'):.4f}, "
            f"val_loss={current_val_loss:.4f}, "
            f"lr={lr:.6f}"
        )
        
        # Monitor for extreme loss values
        for metric, value in logs.items():
            try:
                if value is not None and (np.isnan(float(value)) or np.isinf(float(value))):
                    self.logger.error(f"Epoch {epoch}: {metric} is {value}, stopping training")
                    self.model.stop_training = True
                    return
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Could not check value for metric {metric}: {e}")
        
        # Skip early stopping logic if validation loss is not available
        if 'val_loss' not in logs:
            self.logger.warning("Validation loss not available, skipping early stopping check")
            return
            
        # Early stopping with patience
        if current_val_loss < self.best_val_loss:
            improvement = self.best_val_loss - current_val_loss
            self.logger.info(f"Validation loss improved by {improvement:.6f} to {current_val_loss:.6f}")
            self.best_val_loss = current_val_loss
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.logger.info(f"Early stopping: validation loss did not improve for {self.patience} epochs")
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.logger.info("Restoring best weights from epoch with best validation loss")
                    try:
                        self.model.set_weights(self.best_weights)
                    except Exception as e:
                        self.logger.error(f"Error restoring best weights: {e}")
                else:
                    self.logger.warning("No best weights available to restore, using final weights")

import numpy as np
import pandas as pd
import tensorflow as tf

# Enable eager execution for TensorFlow 2.x
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Suppress TensorFlow info and warning messages
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import os
import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, MultiHeadAttention, 
    LayerNormalization, BatchNormalization, Concatenate, Add,
    GlobalAveragePooling1D, Activation, Multiply, Bidirectional
)

# Custom R² metric for regression evaluation
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Reshape if needed
        if len(tf.shape(y_true)) > 1:
            y_true = tf.reshape(y_true, [-1])
        if len(tf.shape(y_pred)) > 1:
            y_pred = tf.reshape(y_pred, [-1])
            
        # Calculate residual sum of squares
        residuals = tf.math.subtract(y_true, y_pred)
        ss_res = tf.reduce_sum(tf.square(residuals))
        
        # Calculate total sum of squares
        mean_y_true = tf.reduce_mean(y_true)
        total_error = tf.math.subtract(y_true, mean_y_true)
        ss_tot = tf.reduce_sum(tf.square(total_error))
        
        # Update state
        self.ss_res.assign_add(ss_res)
        self.ss_tot.assign_add(ss_tot)
    
    def result(self):
        # Calculate R²
        # Use tf.equal for tensor comparison and tf.cond for conditional logic
        return tf.cond(
            tf.equal(self.ss_tot, 0.0),
            lambda: tf.constant(0.0),
            lambda: 1 - (self.ss_res / (self.ss_tot + tf.keras.backend.epsilon()))
        )
    
    def reset_state(self):
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)
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
import json

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
        lstm_units: int = 256, 
        attention_units: int = 64, 
        dropout_rate: float = 0.3, 
        learning_rate: float = 0.0001,  # Further reduced for better convergence
        l2_lambda: float = 0.001, 
        num_heads: int = 8,  # Increased number of attention heads
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
            clipnorm: Maximum norm for gradient clipping
            clipvalue: Maximum absolute value for gradient clipping
            kernel_initializer: Initializer for the kernel weights matrix
        """
        # Initialize logger first
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
        
        # Model architecture parameters
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = [64, 32]  # Default dense layer sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.num_heads = num_heads
        self.use_learning_rate_scheduler = use_learning_rate_scheduler
        self.prediction_days = prediction_days
        self.kernel_initializer = kernel_initializer
        
        # Gradient clipping parameters - only one should be used at a time
        # Prefer clipnorm over clipvalue if both are provided
        if clipnorm is not None and clipvalue is not None:
            if clipnorm != 1.0:  # Only warn if clipnorm is not default
                self.logger.warning("Both clipnorm and clipvalue are set. Using clipnorm and ignoring clipvalue.")
            self.clipnorm = clipnorm
            self.clipvalue = None
        else:
            self.clipnorm = clipnorm
            self.clipvalue = clipvalue
        
        # Model components
        self.model = None
        self.encoder = None
        self.decoder = None
        
        # Data scalers with improved ranges
        self.stock_scaler = MinMaxScaler(feature_range=(-1, 1))  # Wider range for better gradient flow
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))  # For inverse transforming predictions
        
        # Training history
        self.history = None
        
        self.logger.info("TwoStagePredictor initialized with %s LSTM units and %s attention units", 
                         lstm_units, attention_units)

    def _create_attention_layer(self, query_value, attention_units):
        """Create multi-head attention layer with residual connection"""
        # Store input for residual connection and get static shape
        input_tensor = query_value
        input_shape = tf.keras.backend.int_shape(input_tensor)
        
        # Ensure we have valid shapes for the residual connection
        if len(input_shape) != 3:
            raise ValueError(f"Expected query_value to be 3D, got shape {input_shape}")
            
        input_units = input_shape[-1]  # Get the last dimension size
        
        # Project input to match attention units if needed
        if input_units != attention_units:
            res = Dense(attention_units)(input_tensor)
        else:
            res = input_tensor
        
        # Multi-head attention with safe dimension calculation
        key_dim = max(1, attention_units // max(1, self.num_heads))  # Ensure at least 1 dimension
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate
        )(query_value, query_value)
        
        # Add & Norm (first residual connection)
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(res + attention_output)
        
        # Store for second residual connection
        res2 = attention_output
        
        # Feed Forward with safe dimension calculation
        ffn_units = max(1, attention_units * 4)  # Ensure at least 1 unit
        ffn_output = Dense(
            ffn_units,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)
        )(attention_output)
        ffn_output = Dense(attention_units)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        
        # Add & Norm (second residual connection)
        return LayerNormalization(epsilon=1e-6)(res2 + ffn_output)

    def build_model(self, stock_shape: Tuple[int, int], features_shape: Tuple[int, int] = None,
                   lstm_units: int = None,
                   attention_units: int = None,
                   dropout_rate: float = None,
                   warmup_steps: int = 1000,
                   decay_steps: int = 9000) -> tf.keras.Model:
        """
        Build the two-stage prediction model with improved architecture
        
        This implementation includes:
        - Multi-head attention mechanism
        - Layer normalization and residual connections
        - Better weight initialization and regularization
        - Gradient clipping and learning rate scheduling
        - Comprehensive error handling and logging
        
        Args:
            stock_shape: Shape of stock input data (timesteps, features)
            features_shape: Shape of additional features (timesteps, num_features)
            lstm_units: Number of LSTM units
            attention_units: Number of attention units
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
            
        Raises:
            ValueError: If model building fails due to invalid input shapes or other issues
        """
        try:
            # Use instance variables if parameters are not provided
            lstm_units = lstm_units or self.lstm_units
            attention_units = attention_units or self.attention_units
            # Ensure attention_units is divisible by num_heads using integer division
            if attention_units % self.num_heads != 0:
                attention_units = (attention_units // self.num_heads + 1) * self.num_heads
                self.logger.info(f"Adjusted attention_units to be divisible by num_heads: {attention_units}")
            dropout_rate = dropout_rate or self.dropout_rate
            
            self.logger.info(f"Building model with stock_shape: {stock_shape}, features_shape: {features_shape}")
            
            # Input layers with explicit batch size for better debugging
            stock_input = Input(shape=stock_shape, name='stock_input', batch_size=None)
            
            # Use the module-level GradientCentralizedMaxNorm constraint

            # Stochastic depth for better regularization
            def stochastic_depth(x, p_survival=0.8):
                def training():
                    batch_size = tf.shape(x)[0]
                    random_tensor = p_survival + tf.random.uniform([batch_size, 1, 1], dtype=x.dtype)
                    binary_tensor = tf.floor(random_tensor)
                    return tf.math.divide(x, p_survival) * binary_tensor
                
                def inference():
                    return x
                
                # Use tf.cond with learning_phase as a boolean tensor
                return tf.cond(tf.cast(tf.keras.backend.learning_phase(), tf.bool), training, inference)

            # Create constraint instance
            max_norm_constraint = GradientCentralizedMaxNorm(max_value=1.5)
            
            lstm_1 = Bidirectional(LSTM(
                12,  # Reduced units
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-3, l2=1e-2),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-2),
                bias_regularizer=tf.keras.regularizers.l2(1e-2),
                dropout=0.6,
                recurrent_dropout=0.6,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                name='lstm_1',
                kernel_constraint=max_norm_constraint
            ))(stock_input)
            
            # Apply stochastic depth - using a custom layer instead of Lambda
            lstm_1 = tf.keras.layers.Dropout(rate=0.2)(lstm_1)  # Simplified approach with standard dropout
            
            # Layer normalization for better training stability
            layer_norm_1 = LayerNormalization(epsilon=1e-6)(lstm_1)
            
            # Multi-head attention mechanism with explicit integer division for key_dim
            # Ensure key_dim is an integer by converting dimensions to Python int
            key_dim = int(64) // int(self.num_heads)
            attention = MultiHeadAttention(
                num_heads=int(self.num_heads),  # Ensure num_heads is an int
                key_dim=key_dim,
                dropout=0.2
            )(layer_norm_1, layer_norm_1, return_attention_scores=False)
            
            # Residual connection with layer normalization
            add_layer = Add()([layer_norm_1, attention])
            norm_layer = LayerNormalization(epsilon=1e-6)(add_layer)
            
            # Global average pooling instead of flattening to reduce parameters
            pooled = GlobalAveragePooling1D()(norm_layer)
            
            # Process additional features if provided
            if features_shape is not None:
                features_input = Input(shape=features_shape, name='features_input', batch_size=None)
                
                # Bidirectional LSTM for processing additional features with even stronger regularization
                features_lstm = Bidirectional(LSTM(
                    16,  # Further reduced number of units to prevent overfitting
                    name='features_lstm',
                    kernel_regularizer=tf.keras.regularizers.l2(0.02),  # Even stronger L2 regularization
                    recurrent_regularizer=tf.keras.regularizers.l2(0.02),  # Even stronger L2 regularization
                    bias_regularizer=tf.keras.regularizers.l2(0.02),  # Stronger bias regularization
                    dropout=0.6,  # Further increased dropout
                    recurrent_dropout=0.6,  # Further increased recurrent dropout
                    dtype=tf.float32  # Ensure consistent precision
                ))(features_input)
                
                # Batch normalization for better training stability
                batch_norm_1 = BatchNormalization()(features_lstm)
                
                # Increased dropout for better regularization
                dropout_1 = Dropout(0.3)(batch_norm_1)
                
                # Feature attention
                attention_weights = Dense(1, activation='tanh')(dropout_1)
                attention_weights = Activation('softmax')(attention_weights)
                y = Multiply()([dropout_1, attention_weights])
                
                # Concatenate the main and feature branches
                concat = Concatenate()([pooled, y])
                model_inputs = [stock_input, features_input]
            else:
                model_inputs = [stock_input]
            
            # Dense layers with gradient centralization and improved regularization
            dense = Dense(
                6,  # Further reduced units
                activation='swish',  # Better gradient properties than elu
                kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-3, l2=5e-2),
                kernel_constraint=GradientCentralizedMaxNorm(max_value=1.0),
                activity_regularizer=tf.keras.regularizers.l1(5e-4)
            )(concat if features_shape is not None else pooled)
            dense = LayerNormalization(epsilon=1e-5)(dense)
            dense = Dropout(0.5)(dense)  # Reduced dropout for better training stability
            
            # Second dense layer with even stronger regularization - simplified
            dense_2 = Dense(
                16,  # Further reduced units
                activation='relu',  # Simple activation
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=2e-2)  # Combined L1 and L2 regularization
            )(dense)
            dense_2 = BatchNormalization()(dense_2)
            dense_2 = Dropout(0.7)(dense_2)  # Further increased dropout
            
            # Add a residual connection if possible (if dimensions match)
            if dense.shape[-1] == dense_2.shape[-1]:
                dense_2 = Add()([dense, dense_2])  # Residual connection
            
            # Output layer with proper regression configuration
            output = Dense(
                1,  # Single output for regression
                activation='linear',  # Linear activation for regression
                name='output',
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),  # Increased regularization
                kernel_initializer='glorot_normal',  # Better initialization for final layer
                bias_initializer='zeros'  # Initialize bias to zero for regression
            )(dense_2)
            
            # Create model
            model = tf.keras.Model(
                inputs=model_inputs,
                outputs=output,
                name='TwoStageStockPredictor'
            )
            
            # Configure optimizer with gradient clipping and fixed learning rate
            self.logger.info(f"Using fixed learning rate: {self.learning_rate}")
            
            # Initialize callbacks list if it doesn't exist
            if not hasattr(self, 'callbacks'):
                self.callbacks = []
                
            # Create the learning rate schedule if using learning rate scheduling
            if self.use_learning_rate_scheduler:
                # Create the combined warmup and cosine decay schedule
                lr_schedule = WarmupCosineDecay(
                    initial_learning_rate=float(self.learning_rate),
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    alpha=0.0,  # Minimum learning rate multiplier
                    name="WarmupCosineDecay"
                )
                
                # Create learning rate scheduler callback
                lr_callback = tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch, lr: lr_schedule(epoch * steps_per_epoch),
                    verbose=1
                )
                self.callbacks.append(lr_callback)
            
            # Create optimizer with fixed learning rate (scheduling is handled by callback)
            optimizer_kwargs = {
                'learning_rate': float(self.learning_rate),
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'amsgrad': True,
                'weight_decay': 0.0005
            }
            
            # Add gradient clipping if specified
            if hasattr(self, 'clipnorm') and self.clipnorm is not None:
                if hasattr(self, 'clipvalue') and self.clipvalue is not None:
                    self.logger.warning("Both clipnorm and clipvalue are set. Using clipnorm and ignoring clipvalue.")
                optimizer_kwargs['clipnorm'] = float(self.clipnorm)
            elif hasattr(self, 'clipvalue') and self.clipvalue is not None:
                optimizer_kwargs['clipvalue'] = float(self.clipvalue)
            
            # Create optimizer with configured parameters
            self.logger.info(f"Creating Adam optimizer with kwargs: {optimizer_kwargs}")
            optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
            
            # Compile the model with combined loss function for better regularization
            # Use appropriate metrics for regression tasks
            
            # Enhanced custom loss function that combines MSE with L1 regularization and Huber loss
            def combined_loss(y_true, y_pred):
                # MSE component (manual implementation for maximum compatibility)
                mse = tf.reduce_mean(tf.square(y_true - y_pred))
                
                # L1 component for sparsity
                l1 = tf.reduce_mean(tf.abs(y_pred))
                
                # Huber loss component for robustness to outliers
                delta = 1.0
                error = y_true - y_pred
                abs_error = tf.abs(error)
                quadratic = tf.minimum(abs_error, delta)
                linear = abs_error - quadratic
                huber_loss = 0.5 * tf.square(quadratic) + delta * linear
                huber_loss = tf.reduce_mean(huber_loss)
                
                # Heavily weighted Huber loss for better outlier handling
                total_loss = 0.3 * mse + 0.6 * huber_loss + 0.1 * l1  # More weight on Huber loss
                # More aggressive gradient clipping
                return tf.clip_by_value(total_loss, -1e2, 1e2)
            
            model.compile(
                optimizer=optimizer,
                loss=combined_loss,  # Custom combined loss
                metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]  # Add RMSE metric
            )
            
            # Initialize the model by running a forward pass with dummy data
            try:
                # Create dummy inputs to build the model
                dummy_stock = tf.zeros((1, *stock_shape))
                dummy_features = tf.zeros((1, *features_shape)) if features_shape is not None else None
                dummy_y = tf.zeros((1, 1))
                
                # Build the model by running a forward pass
                if dummy_features is not None:
                    # Run a single training step to initialize metrics
                    model.train_on_batch([dummy_stock, dummy_features], dummy_y)
                else:
                    model.train_on_batch(dummy_stock, dummy_y)
                    
                self.logger.info("Model and metrics initialized successfully")
                    
            except Exception as e:
                self.logger.warning(f"Could not initialize model with dummy data: {str(e)}")
                # Continue anyway as the model might still work
            
            # Force initialization of optimizer weights
            self.logger.info("Ensuring optimizer weights are initialized")
            optimizer.build(model.trainable_variables)
            
            # Save model architecture
            model_arch_path = os.path.join(self.model_dir, 'model_architecture.txt')
            try:
                # First try to get the summary as a string
                with open(model_arch_path, 'w', encoding='utf-8') as fh:
                    # Create a string buffer to capture the summary
                    summary_list = []
                    model.summary(print_fn=lambda x: summary_list.append(x))
                    # Write the captured summary to file
                    fh.write('\n'.join(summary_list) + '\n')
                    fh.write('\nModel Configuration:\n')
                    fh.write('-------------------\n')
                    fh.write(f'Stock shape: {stock_shape}\n')
                    fh.write(f'Features shape: {features_shape}\n')
                    fh.write(f'LSTM units: {lstm_units}\n')
                    fh.write(f'Attention units: {attention_units}\n')
                    fh.write(f'Dropout rate: {dropout_rate}\n')
                    fh.write(f'Learning rate: {self.learning_rate}\n')
                    fh.write(f'L2 lambda: {self.l2_lambda}\n')
                    fh.write(f'Number of attention heads: {self.num_heads}\n')
                    fh.write(f'Use learning rate scheduler: {self.use_learning_rate_scheduler}\n')
                    
                self.logger.info(f"Model architecture saved to {model_arch_path}")
                
                # Safely log optimizer configuration without converting symbolic tensors
                try:
                    # Use get_config() which is safe in graph mode
                    if hasattr(optimizer, 'get_config'):
                        opt_config = optimizer.get_config()
                        # Convert any tensor values to their string representation
                        safe_config = {k: str(v) for k, v in opt_config.items()}
                        self.logger.info("Model compiled with optimizer config: %s", safe_config)
                    else:
                        self.logger.info("Using optimizer: %s", optimizer.__class__.__name__)
                except Exception as e:
                    self.logger.warning("Could not log optimizer config: %s", str(e))
                    self.logger.info("Using optimizer: %s", optimizer.__class__.__name__)
                
                self.logger.info("Model built successfully")
                return model
                
            except Exception as e:
                self.logger.error(f"Error saving model architecture: {str(e)}")
                # Continue and return the model even if saving architecture fails
                self.logger.info("Model built successfully (architecture not saved)")
                return model
                
        except Exception as e:
            error_msg = f"Error building model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    def _validate_input_data(self, data, data_type=""):
        """
        Validate input data and handle common issues.
        
        Args:
            data: DataFrame to validate
            data_type: String describing the data type (for logging)
            
        Returns:
            Validated and cleaned DataFrame
        """
        try:
            if data is None:
                self.logger.error(f"{data_type} data is None")
                return pd.DataFrame()
                
            if not isinstance(data, pd.DataFrame):
                self.logger.error(f"{data_type} data is not a pandas DataFrame")
                return pd.DataFrame()
                
            if data.empty:
                self.logger.error(f"{data_type} data is empty")
                return pd.DataFrame()
                
            # Check for datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning(f"{data_type} data does not have DatetimeIndex, converting...")
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    self.logger.error(f"Failed to convert index to datetime: {str(e)}")
                    return pd.DataFrame()
            
            # Handle missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                self.logger.warning(f"{data_type} data has {missing_values} missing values, filling...")
                data = data.ffill().bfill()  # Forward fill then backward fill
            
            # Handle infinite values
            inf_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_values > 0:
                self.logger.warning(f"{data_type} data has {inf_values} infinite values, replacing...")
                data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            
            # Ensure lowercase column names for consistency
            data.columns = [col.lower() if isinstance(col, str) else col for col in data.columns]
            
            self.logger.info(f"Validated {data_type} data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"Error validating {data_type} data: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, stock_data):
        """
        Add technical indicators to stock data with enhanced trend-specific features.
        
        Args:
            stock_data: DataFrame containing stock price data
            
        Returns:
            DataFrame with technical indicators and trend-specific features
            
        Raises:
            ValueError: If input data is invalid or processing fails
        """
        try:
            # Make a copy to avoid modifying the original data
            df = stock_data.copy()
            
            # Ensure column names are lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate moving averages with different timeframes
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_10'] = df['close'].rolling(window=10).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # Calculate relative strength index (RSI)
            delta = df['close'].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            roll_up = up.rolling(window=14).mean()
            roll_down = down.rolling(window=14).mean().abs()
            RS = roll_up / roll_down
            df['rsi'] = 100.0 - (100.0 / (1.0 + RS))
            
            # Calculate MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['ma_20']
            df['bb_upper'] = df['ma_20'] + 2*df['close'].rolling(window=20).std()
            df['bb_lower'] = df['ma_20'] - 2*df['close'].rolling(window=20).std()
            
            # Calculate momentum indicators
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['rate_of_change_5'] = df['close'].pct_change(periods=5) * 100
            df['rate_of_change_10'] = df['close'].pct_change(periods=10) * 100
            
            # Calculate trend direction indicators
            df['trend_5_10'] = np.where(df['ma_5'] > df['ma_10'], 1, -1)
            df['trend_10_20'] = np.where(df['ma_10'] > df['ma_20'], 1, -1)
            df['trend_20_50'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
            
            # Calculate price acceleration (change in momentum)
            df['acceleration_5'] = df['momentum_5'].diff()
            df['acceleration_10'] = df['momentum_10'].diff()
            
            # Calculate volatility indicators
            df['volatility_10'] = df['close'].rolling(window=10).std() / df['close']
            df['volatility_20'] = df['close'].rolling(window=20).std() / df['close']
            
            # Calculate volume indicators
            if 'volume' in df.columns:
                df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
                df['volume_change'] = df['volume'].pct_change()
                df['volume_ratio'] = df['volume'] / df['volume_ma_10']
                
                # On-Balance Volume (OBV)
                df['obv'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                                     np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()
            
            # Calculate support and resistance levels
            df['support_level'] = df['low'].rolling(window=20).min()
            df['resistance_level'] = df['high'].rolling(window=20).max()
            df['price_to_support'] = (df['close'] - df['support_level']) / df['close']
            df['price_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
            
            # Fill NaN values with forward and backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # If any NaNs remain, fill with column mean
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
            
            # Log basic statistics before cleaning
            self.logger.info(f"Stock data shape before cleaning: {df.shape}")
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Log columns with missing values
            nan_cols = df.columns[df.isnull().any()].tolist()
            if nan_cols:
                self.logger.warning(f"Found {len(nan_cols)} columns with missing values in stock data")
                self.logger.debug(f"Columns with missing values: {nan_cols}")
            
            self.logger.info(f"Added {len(df.columns) - len(stock_data.columns)} technical indicators and trend features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            # Return original data if there's an error
            return stock_data

    def _augment_sequences(self, X_stock, X_additional, y, noise_level=0.02, num_augmentations=1):
        """
        Augment training data with random noise and shifts to prevent overfitting
        
        Args:
            X_stock: Stock data sequences
            X_additional: Additional feature sequences
            y: Target values
            noise_level: Level of Gaussian noise to add (as fraction of data std)
            num_augmentations: Number of augmented copies to create
            
        Returns:
            Augmented data sequences and targets
        """
        if X_stock.shape[0] <= 2:  # Don't augment tiny datasets
            return X_stock, X_additional, y
            
        try:
            # Calculate standard deviation for noise scaling
            stock_std = np.std(X_stock, axis=0, keepdims=True).clip(min=1e-8)
            additional_std = np.std(X_additional, axis=0, keepdims=True).clip(min=1e-8) if X_additional is not None else None
            
            # Lists to hold augmented data
            aug_X_stock = [X_stock]
            aug_X_additional = [X_additional] if X_additional is not None else None
            aug_y = [y]
            
            for i in range(num_augmentations):
                # Add random Gaussian noise scaled by data std
                noisy_X_stock = X_stock + np.random.normal(0, noise_level, X_stock.shape) * stock_std
                aug_X_stock.append(noisy_X_stock)
                
                # Add noise to additional features if they exist
                if X_additional is not None:
                    noisy_X_additional = X_additional + np.random.normal(0, noise_level, X_additional.shape) * additional_std
                    aug_X_additional.append(noisy_X_additional)
                
                # Keep the same targets
                aug_y.append(y)
            
            # Concatenate all augmented data
            X_stock_aug = np.vstack(aug_X_stock)
            X_additional_aug = np.vstack(aug_X_additional) if X_additional is not None else None
            y_aug = np.vstack(aug_y) if len(y.shape) > 1 else np.concatenate(aug_y)
            
            self.logger.info(f"Data augmentation: Original size={X_stock.shape[0]}, Augmented size={X_stock_aug.shape[0]}")
            return X_stock_aug, X_additional_aug, y_aug
            
        except Exception as e:
            self.logger.warning(f"Data augmentation failed: {str(e)}. Using original data.")
            return X_stock, X_additional, y
    
    def preprocess_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame = None, macro_data: pd.DataFrame = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Preprocess the input data for the model.
        
        Args:
            stock_data: DataFrame containing stock price data with datetime index
                     Expected columns: 'open', 'high', 'low', 'close', 'volume',
                     and optionally technical indicators like 'RSI', 'MACD', etc.
            sentiment_data: Optional DataFrame containing sentiment data with datetime index.
                         Expected column: 'sentiment_score'
            macro_data: Optional DataFrame containing macroeconomic data with datetime index.
                     Expected columns: Any relevant macroeconomic indicators
                          
        Returns:
            Tuple containing (X_stock, X_additional, y) where:
            - X_stock: Sequence of stock prices (samples, timesteps, n_stock_features)
            - X_additional: Additional features (samples, timesteps, n_additional_features)
            - y: Target values (samples,)
            
        Raises:
            ValueError: If input data is invalid or preprocessing fails
        """
        self.logger.info("Starting data preprocessing...")
        
        try:
            # Input validation and cleaning
            stock_data = self._validate_input_data(stock_data, "Stock")
            
            # Ensure we have a datetime index
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
            
            # Sort by date to ensure chronological order
            stock_data = stock_data.sort_index()
            
            # Handle missing data with forward and backward filling
            stock_data = stock_data.ffill().bfill()
            
            # Add technical indicators
            stock_data = self._validate_input_data(self._add_technical_indicators(stock_data), "Stock with indicators")
            
            # Initialize additional features list
            additional_features = []
            
            # Process sentiment data if available:
            if sentiment_data is not None:
                try:
                    sentiment_data = self._validate_input_data(sentiment_data, "Sentiment")
                    sentiment_data = sentiment_data.reindex(stock_data.index, method='ffill')
                    sentiment_cols = ['sentiment_score', 'Price_Momentum', 'Volume_Change', 'Volatility']
                    sentiment_data = sentiment_data[list(set(sentiment_cols) & set(sentiment_data.columns))]
                    self.logger.info(f"Processed {len(sentiment_cols)} sentiment features")
                    additional_features.append(sentiment_data)
                except Exception as e:
                    self.logger.warning(f"Error processing sentiment data: {str(e)}")
            
            # Process macro data if available
            if macro_data is not None:
                try:
                    macro_data = self._validate_input_data(macro_data, "Macro")
                    macro_data = macro_data.reindex(stock_data.index, method='ffill')
                    
                    # Process macro columns if data is valid
                    if not macro_data.empty:
                        macro_cols = ['Market_Return', 'Market_Volatility', 'VIX', 'Equity_Risk_Premium']
                        available_cols = list(set(macro_cols) & set(macro_data.columns))
                        if available_cols:
                            macro_data = macro_data[available_cols]
                            self.logger.info(f"Processed {len(available_cols)} macro features: {available_cols}")
                            additional_features.append(macro_data)
                        else:
                            self.logger.warning("No valid macro features found in the provided data")
                except Exception as e:
                    self.logger.warning(f"Error processing macro data: {str(e)}")
            
            # Combine all features
            if additional_features:
                additional_features = pd.concat(additional_features, axis=1)
                # Ensure no duplicate columns
                additional_features = additional_features.loc[:,~additional_features.columns.duplicated()]
                
                # Handle any remaining missing values in additional features
                additional_features = additional_features.ffill().bfill()
                
                # Ensure we have the same number of samples
                if len(stock_data) != len(additional_features):
                    raise ValueError(
                        f"Mismatch in number of samples between stock data ({len(stock_data)}) "
                        f"and additional features ({len(additional_features)})"
                    )
            else:
                additional_features = pd.DataFrame(index=stock_data.index)
            
            # Get close prices (handle various naming patterns)
            close_col = next((col for col in stock_data.columns 
                            if any(term in col.lower() for term in ['close', 'price', 'last', 'adj'])),
                            stock_data.columns[0] if len(stock_data.columns) > 0 else None)
            
            if close_col is None:
                raise ValueError("Could not determine price column in stock data. "
                               f"Available columns: {list(stock_data.columns)}")
            
            self.logger.info(f"Using column '{close_col}' for close prices")
            close_prices = stock_data[close_col].values.reshape(-1, 1)
            
            # Scale the prices
            scaled_prices = self.stock_scaler.fit_transform(close_prices)
            self.target_scaler.fit(close_prices)  # Also fit target scaler
            
            # Create sequences for stock data
            X_stock = []
            X_additional = []
            y = []
            
            for i in range(self.prediction_days, len(scaled_prices)):
                X_stock.append(scaled_prices[i-self.prediction_days:i])
                y.append(scaled_prices[i])
            
            # Convert lists to numpy arrays outside the loop
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
                    
                    # Get all numeric columns from sentiment data
                    numeric_cols = sentiment_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        self.logger.warning("No numeric columns found in sentiment data")
                    else:
                        self.logger.info(f"Processing {len(numeric_cols)} sentiment features: {numeric_cols}")
                        
                        # Scale each sentiment feature
                        for col in numeric_cols:
                            try:
                                sentiment_scaler = MinMaxScaler()
                                sentiment_scaled = sentiment_scaler.fit_transform(
                                    sentiment_data[col].values.reshape(-1, 1)
                                )
                                feature_data.append(sentiment_scaled)
                                self.logger.debug(f"Added sentiment feature: {col}")
                            except Exception as e:
                                self.logger.warning(f"Could not process sentiment feature {col}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error processing sentiment data: {str(e)}", exc_info=True)
            else:
                self.logger.warning("No sentiment data provided")
            
            # 4. Add macro data if available
            if macro_data is not None and not macro_data.empty:
                try:
                    # Ensure macro data has datetime index
                    if not isinstance(macro_data.index, pd.DatetimeIndex):
                        macro_data.index = pd.to_datetime(macro_data.index)
                    
                    # Resample and forward fill macro data to match stock data frequency
                    macro_data = macro_data.reindex(stock_data.index, method='ffill')
                    
                    # Get all numeric columns from macro data
                    numeric_cols = macro_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        self.logger.warning("No numeric columns found in macro data")
                    else:
                        self.logger.info(f"Processing {len(numeric_cols)} macro features")
                        
                        # Scale each macro indicator
                        for col in numeric_cols:
                            try:
                                macro_scaler = MinMaxScaler()
                                macro_scaled = macro_scaler.fit_transform(
                                    macro_data[col].values.reshape(-1, 1)
                                )
                                feature_data.append(macro_scaled)
                                
                                # Update feature scaler
                                if not hasattr(self, 'feature_scaler') or not hasattr(self.feature_scaler, 'scale_'):
                                    self.feature_scaler = MinMaxScaler()
                                    self.feature_scaler.scale_ = macro_scaler.scale_
                                    self.feature_scaler.min_ = macro_scaler.min_
                                else:
                                    self.feature_scaler.scale_ = np.concatenate([
                                        self.feature_scaler.scale_, macro_scaler.scale_
                                    ])
                                    self.feature_scaler.min_ = np.concatenate([
                                        self.feature_scaler.min_, macro_scaler.min_
                                    ])
                                self.logger.debug(f"Scaled macro feature: {col}")
                                
                            except Exception as e:
                                self.logger.warning(f"Error processing macro feature {col}: {str(e)}")
                                continue
                except Exception as e:
                    self.logger.error(f"Error processing macro data: {str(e)}", exc_info=True)
                    return (None, None), None
            else:
                self.logger.warning("No macro data provided")           
            
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
                
                # Initialize X_additional list
                X_additional = []
                
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
            
            # Ensure we have the same number of samples in all arrays
            min_samples = min(len(X_stock), len(X_additional), len(y))
            if min_samples == 0:
                raise ValueError("No valid samples found after preprocessing")
                
            X_stock = X_stock[:min_samples]
            X_additional = X_additional[:min_samples]
            y = y[:min_samples]
            
            # Apply data augmentation for training (not for prediction)
            if not getattr(self, 'is_predicting', False):
                self.logger.info("Applying data augmentation to prevent overfitting")
                X_stock, X_additional, y = self._augment_sequences(
                    X_stock, X_additional, y, 
                    noise_level=0.03,  # 3% noise
                    num_augmentations=2  # Create 2 augmented copies
                )
            
            # Return the processed data
            self.logger.info(f"Final preprocessed data shapes: X_stock={X_stock.shape}, X_additional={X_additional.shape if X_additional is not None else None}, y={y.shape}")
            return X_stock, X_additional, y
                
        except Exception as e:
            error_msg = f"Error during data preprocessing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _log_training_metrics(self, history: dict):
        """Log training metrics from history object."""
        if not history:
            self.logger.warning("No training history to log")
            return
            
        self.logger.info("Training metrics:")
        for metric, values in history.items():
            if len(values) > 0:
                if metric == 'lr':
                    self.logger.info(f"  {metric}: {values[-1]:.6f}")
                else:
                    self.logger.info(f"  {metric}: {values[-1]:.6f}")

    def _get_training_callbacks(self, log_dir: str):
        """Get list of callbacks for model training."""
        callbacks = []
        
        # TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Custom callback for logging learning rate
        class LRSchedulerCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self._logger = logger
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                try:
                    # Get the learning rate from the optimizer
                    lr = self.model.optimizer.learning_rate
                    
                    # Handle both float and LearningRateSchedule cases
                    if hasattr(lr, 'numpy'):  # For tensor-like objects
                        lr = float(lr.numpy())
                    elif callable(lr):  # For LearningRateSchedule
                        lr = float(lr(self.model.optimizer.iterations))
                    else:  # For plain float
                        lr = float(lr)
                        
                    logs['lr'] = lr
                    self._logger.info(f"Epoch {epoch + 1}: Learning rate = {lr:.6f}")
                except Exception as e:
                    self._logger.warning(f"Could not log learning rate: {str(e)}")
        
        # Add learning rate logger with logger reference
        lr_logger = LRSchedulerCallback(logger=self.logger)
        callbacks.append(lr_logger)
        
        return callbacks
        
    def _clean_and_validate_inputs(self, x_stock: np.ndarray, x_features: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clean and validate input data by handling missing values, infinite values, and shape issues.
        
        Args:
            x_stock: Stock price sequences
            x_features: Additional feature sequences
            y: Target values
            
        Returns:
            Tuple of cleaned (x_stock, x_features, y)
        """
        # Convert inputs to numpy arrays if they aren't already
        x_stock = np.asarray(x_stock, dtype=np.float32)
        x_features = np.asarray(x_features, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Handle NaN and infinite values
        for name, arr in [('x_stock', x_stock), ('x_features', x_features), ('y', y)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                self.logger.warning(f"{name} contains NaN or infinite values, replacing with zeros")
                if name == 'x_stock':
                    x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=0.0, neginf=0.0)
                elif name == 'x_features':
                    x_features = np.nan_to_num(x_features, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate and reshape
        return self._validate_and_reshape_inputs(x_stock, x_features, y)
        
    def _log_data_statistics(self, x_stock, x_features, y, title: str):
        """Log statistics about the input data. Handles both numpy arrays and lists robustly."""
        import numpy as np
        def ensure_array(arr, name):
            if not isinstance(arr, np.ndarray):
                try:
                    arr = np.asarray(arr)
                except Exception as e:
                    self.logger.error(f"{title}: Could not convert {name} to numpy array. Type: {type(arr)}, Error: {e}, Value: {arr}")
                    return None
            return arr

        x_stock = ensure_array(x_stock, 'x_stock')
        x_features = ensure_array(x_features, 'x_features')
        y = ensure_array(y, 'y')

        if x_stock is None or x_features is None or y is None:
            self.logger.error(f"{title}: One or more inputs could not be converted to numpy arrays. Skipping statistics log.")
            return
        if len(x_stock) == 0 or len(x_features) == 0 or len(y) == 0:
            self.logger.warning(f"{title}: One or more input arrays are empty")
            return
        try:
            self.logger.info(f"{title}:")
            self.logger.info(f"  x_stock: shape={x_stock.shape}, "
                            f"range=[{np.min(x_stock):.4f}, {np.max(x_stock):.4f}], "
                            f"mean={np.mean(x_stock):.4f}±{np.std(x_stock):.4f}")
            self.logger.info(f"  x_features: shape={x_features.shape}, "
                            f"range=[{np.min(x_features):.4f}, {np.max(x_features):.4f}], "
                            f"mean={np.mean(x_features):.4f}±{np.std(x_features):.4f}")
            self.logger.info(f"  y: shape={y.shape}, "
                            f"range=[{np.min(y):.4f}, {np.max(y):.4f}], "
                            f"mean={np.mean(y):.4f}±{np.std(y):.4f}")
        except Exception as e:
            self.logger.error(f"{title}: Error logging data statistics: {e}")
            self.logger.error(f"Types: x_stock={type(x_stock)}, x_features={type(x_features)}, y={type(y)}")
            self.logger.error(f"Shapes: x_stock={getattr(x_stock, 'shape', None)}, x_features={getattr(x_features, 'shape', None)}, y={getattr(y, 'shape', None)}")


    def _validate_and_reshape_inputs(self, x_stock: np.ndarray, x_features: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and reshape input data for training/prediction with enhanced error handling.
        
        Args:
            x_stock: Stock price sequences (n_samples, seq_len, n_stock_features)
            x_features: Additional feature sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples, 1)
            
        Returns:
            Tuple of validated and reshaped (x_stock, x_features, y)
            
        Raises:
            ValueError: If input shapes are invalid or data contains invalid values
        """
        try:
            # Ensure inputs are numpy arrays
            x_stock = np.asarray(x_stock, dtype=np.float32)
            x_features = np.asarray(x_features, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            
            # Replace NaN and infinite values with zeros
            x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=0.0, neginf=0.0)
            x_features = np.nan_to_num(x_features, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            elif len(y.shape) > 2:
                raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
                
            # Ensure consistent number of samples
            n_samples = x_stock.shape[0]
            if x_features.shape[0] != n_samples:
                self.logger.warning(
                    f"Inconsistent number of samples: x_stock has {n_samples}, "
                    f"x_features has {x_features.shape[0]}. Using minimum samples."
                )
                n_samples = min(n_samples, x_features.shape[0])
                x_stock = x_stock[:n_samples]
                x_features = x_features[:n_samples]
                y = y[:n_samples]
                
            if y.shape[0] != n_samples:
                self.logger.warning(
                    f"Inconsistent number of samples: x_stock has {n_samples}, "
                    f"y has {y.shape[0]}. Using minimum samples."
                )
                n_samples = min(n_samples, y.shape[0])
                x_stock = x_stock[:n_samples]
                x_features = x_features[:n_samples]
                y = y[:n_samples]
                
            # Ensure sequence lengths match
            seq_len = x_stock.shape[1]
            if x_features.shape[1] != seq_len:
                self.logger.warning(
                    f"Sequence length mismatch: x_stock has {seq_len} timesteps, "
                    f"x_features has {x_features.shape[1]}. Padding shorter sequence to match."
                )
                
                # Instead of truncating, pad the shorter sequence to match the longer one
                if x_features.shape[1] < seq_len:
                    # Pad x_features
                    padding_length = seq_len - x_features.shape[1]
                    padding = np.zeros((x_features.shape[0], padding_length, x_features.shape[2]), dtype=x_features.dtype)
                    x_features = np.concatenate([padding, x_features], axis=1)
                    self.logger.info(f"Padded x_features to shape {x_features.shape}")
                else:
                    # Pad x_stock
                    padding_length = x_features.shape[1] - seq_len
                    padding = np.zeros((x_stock.shape[0], padding_length, x_stock.shape[2]), dtype=x_stock.dtype)
                    x_stock = np.concatenate([padding, x_stock], axis=1)
                    self.logger.info(f"Padded x_stock to shape {x_stock.shape}")
                    seq_len = x_features.shape[1]  # Update seq_len to the new length
            
            # Ensure we have at least one sample
            if n_samples == 0:
                raise ValueError("No valid samples found after input validation")
                
            # Log final shapes and statistics
            self.logger.info(
                f"Final input shapes - x_stock: {x_stock.shape}, "
                f"x_features: {x_features.shape}, y: {y.shape}"
            )
            self.logger.debug(
                f"Data statistics - x_stock: mean={np.mean(x_stock):.4f}±{np.std(x_stock):.4f}, "
                f"x_features: mean={np.mean(x_features):.4f}±{np.std(x_features):.4f}, "
                f"y: mean={np.mean(y):.4f}±{np.std(y):.4f}"
            )
            
            return x_stock, x_features, y
            
        except Exception as e:
            error_msg = f"Error in input validation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def train(self, x_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray,
             epochs: int = 50, batch_size: int = 64, validation_split: float = 0.2,
             class_weight: dict = None, verbose: int = 1, validation_dataset = None,
             callbacks = None):
        """
        Train the model with the given data.
        
        Args:
            x_train: Tuple of (stock_data, additional_features) arrays
            y_train: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation (ignored if validation_dataset is provided)
            class_weight: Optional dictionary mapping class indices to weight values
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            validation_dataset: Optional pre-created validation dataset (tf.data.Dataset)
                               If provided, validation_split is ignored
            
        Returns:
            Training history
        """
        try:
            # Reset is_predicting flag to ensure data augmentation is applied during training
            self.is_predicting = False
            
            # Unpack inputs
            x_stock, x_features = x_train
            
            # Convert inputs to numpy arrays
            x_stock = np.asarray(x_stock, dtype=np.float32)
            x_features = np.asarray(x_features, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)
            
            # Ensure 3D shape for both inputs
            if len(x_stock.shape) == 2:
                x_stock = np.expand_dims(x_stock, axis=-1)
            if len(x_features.shape) == 2:
                x_features = np.expand_dims(x_features, axis=-1)
                
            # Ensure y_train is 2D (samples, 1)
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            
            # Log initial data statistics
            self.logger.info(f"Input shapes - x_stock: {x_stock.shape}, x_features: {x_features.shape}, y_train: {y_train.shape}")
            
            # Clean and validate input data with additional checks
            try:
                # Ensure inputs are float32 for consistency
                x_stock = x_stock.astype(np.float32)
                x_features = x_features.astype(np.float32)
                y_train = y_train.astype(np.float32)
                
                # Clip extreme values to prevent numerical instability
                x_stock = np.clip(x_stock, -1e6, 1e6)
                x_features = np.clip(x_features, -1e6, 1e6)
                y_train = np.clip(y_train, 1e-7, 1 - 1e-7)  # Keep within valid probability range
                
                x_stock, x_features, y_train = self._clean_and_validate_inputs(
                    x_stock, x_features, y_train
                )
            except Exception as e:
                self.logger.error(f"Error in input validation: {str(e)}")
                raise ValueError(f"Input validation failed: {str(e)}") from e
                
            # Log final shapes
            self.logger.info(f"Final shapes - x_stock: {x_stock.shape}, x_features: {x_features.shape}, y_train: {y_train.shape}")
            
            # Calculate warmup and decay steps for learning rate schedule
            total_steps = int(epochs * (len(x_stock) * (1 - validation_split)) / batch_size)
            warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
            decay_steps = total_steps - warmup_steps  # Remaining 90% for decay
            
            self.logger.info(f"Learning rate schedule: {warmup_steps} warmup steps, {decay_steps} decay steps")
            
            # Build model if not already built
            if self.model is None:
                self.logger.info("Building model...")
                try:
                    self.model = self.build_model(
                        stock_shape=x_stock.shape[1:],
                        features_shape=x_features.shape[1:],
                        lstm_units=self.lstm_units,
                        attention_units=self.attention_units,
                        dropout_rate=self.dropout_rate,
                        warmup_steps=warmup_steps,
                        decay_steps=decay_steps
                    )
                    self.logger.info("Model built successfully")
                    
                    # Log model summary
                    try:
                        import io
                        from contextlib import redirect_stdout
                        
                        f = io.StringIO()
                        with redirect_stdout(f):
                            self.model.summary()
                        summary_str = f.getvalue()
                        self.logger.info("\nModel Summary:" + "\n" + "-"*50)
                        self.logger.info(summary_str)
                        self.logger.info("-"*50)
                    except Exception as e:
                        self.logger.warning(f"Could not generate model summary: {str(e)}")
                        
                except Exception as e:
                    error_msg = f"Error building model: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
            
            # Ensure model is compiled with proper optimizer
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                self.logger.info("Compiling model...")
                optimizer_kwargs = {
                    'learning_rate': float(self.learning_rate),
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-7,
                    'amsgrad': False
                }
                if self.clipnorm is not None:
                    optimizer_kwargs['clipnorm'] = float(self.clipnorm)
                elif self.clipvalue is not None:
                    optimizer_kwargs['clipvalue'] = float(self.clipvalue)
                
                optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
                
                # Use Huber loss for regression - better for stock price prediction
                loss_fn = tf.keras.losses.Huber(
                    delta=1.0,  # Controls the threshold where the loss becomes linear
                    reduction=tf.keras.losses.Reduction.AUTO,
                    name='huber_loss'
                )
                
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss_fn,
                    metrics=[
                        tf.keras.metrics.MeanAbsoluteError(name='mae'),
                        tf.keras.metrics.MeanSquaredError(name='mse'),
                        tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
                        R2Score(name='r2_score')  # Custom R² metric
                    ],
                    run_eagerly=False
                )
                self.logger.info("Model compiled successfully")
            
            # Ensure target is 2D
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            elif len(y_train.shape) > 2:
                raise ValueError(f"y must be 1D or 2D, got shape {y_train.shape}")
            
            # Verify optimizer configuration
            optimizer = self.model.optimizer
            try:
                lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
                self.logger.info(f"Using {optimizer.__class__.__name__} optimizer with learning rate: {lr:.6f}")
                
                # Initialize optimizer weights if needed
                if not hasattr(optimizer, 'iterations') or optimizer.iterations is None:
                    # Initialize optimizer variables
                    _ = optimizer.iterations
                    self.logger.debug("Initialized optimizer iterations")
            except Exception as e:
                self.logger.warning(f"Error initializing optimizer: {str(e)}")
            
            # Create TensorFlow dataset
            dataset = tf.data.Dataset.from_tensor_slices((
                {'stock_input': x_stock, 'features_input': x_features},
                y_train
            ))
            
            # Shuffle and batch the dataset
            dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
            
            # Get dataset size and split into train/validation
            dataset_size = len(x_stock)
            
            # Always enforce a minimum validation set size, even for small datasets
            min_val_samples = 1  # Minimum number of validation samples
            min_train_samples = 1  # Minimum number of training samples (warn if less)
            
            # If dataset is extremely small, use aggressive augmentation as before
            if dataset_size <= 3:
                self.logger.warning(f"Dataset critically small ({dataset_size} samples), applying aggressive data augmentation")
                augmentation_factor = 10
                augmented_x_stock, augmented_x_features, augmented_y = [], [], []
                for i in range(dataset_size):
                    for j in range(augmentation_factor):
                        sample_x_stock = x_stock[i:i+1]
                        sample_x_features = x_features[i:i+1]
                        sample_y = y_train[i:i+1]
                        noise_level = 0.01 + (j * 0.01)
                        noise = np.random.normal(0, noise_level, sample_x_stock.shape)
                        augmented_sample_x_stock = sample_x_stock + noise
                        noise = np.random.normal(0, noise_level, sample_x_features.shape)
                        augmented_sample_x_features = sample_x_features + noise
                        target_noise = np.random.normal(0, noise_level * 0.1, sample_y.shape)
                        augmented_sample_y = sample_y + target_noise
                        augmented_x_stock.append(augmented_sample_x_stock)
                        augmented_x_features.append(augmented_sample_x_features)
                        augmented_y.append(augmented_sample_y)
                x_stock = np.vstack([x_stock] + augmented_x_stock)
                x_features = np.vstack([x_features] + augmented_x_features)
                y_train = np.vstack([y_train] + augmented_y)
                dataset = tf.data.Dataset.from_tensor_slices((
                    {'stock_input': x_stock, 'features_input': x_features},
                    y_train
                ))
                dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
                dataset_size = len(x_stock)
                self.logger.info(f"Augmented dataset size: {dataset_size} samples")
            
            # Calculate validation and training sizes (always at least 1 validation sample)
            val_size = max(min_val_samples, int(validation_split * dataset_size))
            if val_size >= dataset_size:
                val_size = min_val_samples
            train_size = dataset_size - val_size
            if train_size < min_train_samples:
                self.logger.warning(f"Training set is very small ({train_size} samples). Results may be unreliable.")
            
            # Split dataset
            train_dataset = dataset.skip(val_size)
            val_dataset = dataset.take(val_size)
            self.logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
            
            # Initialize callbacks
            callbacks = []
            
            # Add TensorBoard callback if log_dir is specified
            if hasattr(self, 'log_dir') and self.log_dir:
                tensorboard_cb = tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
                callbacks.append(tensorboard_cb)
                monitor = 'val_loss'
                checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best_model.h5'),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    verbose=1
                )
                callbacks.append(checkpoint_cb)
            # Always monitor validation metrics
            monitor = 'val_loss'
            if dataset_size < 20:
                patience = 3
                min_delta = 0.005
            else:
                patience = 5
                min_delta = 0.001
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                min_delta=min_delta,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Add learning rate scheduler if requested
            if self.use_learning_rate_scheduler:
                # Improved learning rate scheduler with faster decay to prevent overfitting
                def step_decay_with_warmup(epoch, lr):
                    # Parameters - more aggressive decay
                    initial_lr = 1e-3  # Start with higher learning rate
                    min_lr = 1e-6
                    warmup_epochs = 2  # Shorter warmup
                    
                    # Warmup phase
                    if epoch < warmup_epochs:
                        return initial_lr * (epoch + 1) / warmup_epochs
                    
                    # Decay phase - more aggressive decay
                    decay_rate = 0.75  # Faster decay
                    decay_steps = 2  # Decay every 2 epochs
                    decayed_lr = initial_lr * (decay_rate ** ((epoch - warmup_epochs) // decay_steps))
                    
                    # Ensure we don't go below minimum learning rate
                    return max(decayed_lr, min_lr)
                
                # Create learning rate scheduler callback
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                    step_decay_with_warmup,
                    verbose=1
                )
                callbacks.append(lr_scheduler)
                
                # Also add ReduceLROnPlateau for adaptive reduction based on metrics
                # Use the same monitor as early stopping for consistency
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,  # Use the same monitor as early stopping
                    factor=0.5,
                    patience=max(1, patience - 1),  # Slightly more aggressive than early stopping
                    min_lr=1e-6,
                    verbose=1
                )
                callbacks.append(reduce_lr)
            
            # Train the model
            try:
                # Run a single batch through the model to ensure metrics are built
                try:
                    for batch in train_dataset.take(1):
                        x_batch, y_batch = batch
                        self.model.train_on_batch(x_batch, y_batch)
                        self.logger.info("Successfully ran initial batch to build metrics")
                        break
                except Exception as e:
                    self.logger.warning(f"Could not run initial batch: {str(e)}")
                
                # Calculate steps per epoch
                steps_per_epoch = max(1, train_size // batch_size)
                validation_steps = max(1, val_size // batch_size)
                
                self.logger.info(f"Training with {steps_per_epoch} steps per epoch")
                self.logger.info(f"Validating with {validation_steps} steps per epoch")
                
                # Train the model
                history = self.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0,  # We use our own logging
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps
                )
                
                self.history = history.history
                self._log_training_metrics(self.history)
                return self.history
                
            except Exception as e:
                error_msg = f"Error during model training: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in training process: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def train_model(self, x_train, y_train, validation_data=None, epochs=10, batch_size=32, verbose=1, callbacks=None, validation_split=0.2, use_mixed_precision=True):
        """
        Enhanced wrapper for the train method with improved validation data handling and mixed precision support.
        
        Args:
            x_train: Tuple of (stock_data, additional_features) arrays
            y_train: Target values
            validation_data: Optional tuple of (X_val, y_val) where X_val is a tuple of (stock_data_val, additional_features_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            callbacks: List of Keras callbacks
            validation_split: Fraction of training data to use for validation if validation_data is not provided
            use_mixed_precision: Whether to use mixed precision training (faster training with minimal accuracy impact)
            
        Returns:
            Dictionary of training metrics and history
        """
        # Enable mixed precision training if supported and requested
        if use_mixed_precision:
            try:
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_policy(policy)
                self.logger.info(f"Mixed precision enabled. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
            except Exception as e:
                self.logger.warning(f"Failed to enable mixed precision: {str(e)}. Falling back to full precision.")
        else:
            self.logger.info("Using full precision (FP32) training")
        
        # Initialize callbacks list
        combined_callbacks = []
        
        # Add model checkpointing
        checkpoint_path = os.path.join(self.model_dir, 'best_model.keras')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=verbose
        )
        combined_callbacks.append(model_checkpoint)
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        combined_callbacks.append(early_stopping)
        
        # Add learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )
        combined_callbacks.append(reduce_lr)
        
        # Add any user-provided callbacks
        if callbacks:
            if isinstance(callbacks, list):
                combined_callbacks.extend(callbacks)
            else:
                combined_callbacks.append(callbacks)
        
        # Prepare validation data if provided
        validation_dataset = None
        if validation_data is not None and len(validation_data) == 2:
            X_val, y_val = validation_data
            if isinstance(X_val, (list, tuple)) and len(X_val) == 2:
                stock_val, features_val = X_val
                # Convert to numpy arrays if they're not already
                stock_val = np.array(stock_val, dtype=np.float32)
                features_val = np.array(features_val, dtype=np.float32)
                y_val = np.array(y_val, dtype=np.float32)
                
                # Clean and validate validation inputs
                stock_val, features_val, y_val = self._clean_and_validate_inputs(
                    stock_val, features_val, y_val
                )
                
                # Create validation dataset
                validation_dataset = tf.data.Dataset.from_tensor_slices((
                    {'stock_input': stock_val, 'features_input': features_val}, 
                    y_val
                )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                self.logger.info(f"Using provided validation data with {len(stock_val)} samples")
        
        # Convert training data to numpy arrays
        stock_train, features_train = x_train
        stock_train = np.array(stock_train, dtype=np.float32)
        features_train = np.array(features_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        # Clean and validate training inputs
        stock_train, features_train, y_train = self._clean_and_validate_inputs(
            stock_train, features_train, y_train
        )
        
        # Log data shapes
        self.logger.info(f"Training data shapes - stock: {stock_train.shape}, features: {features_train.shape}, targets: {y_train.shape}")
        if validation_dataset is not None:
            for x_val, _ in validation_dataset.take(1):
                self.logger.info(f"Validation batch shapes - stock: {x_val['stock_input'].shape}, features: {x_val['features_input'].shape}")
        
        # Call the original train method with proper validation data
        history = self.train(
            x_train=(stock_train, features_train),
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if validation_dataset is None else 0.0,
            validation_dataset=validation_dataset,
            verbose=verbose,
            callbacks=combined_callbacks
        )
        
        # Log final metrics
        if history and hasattr(history, 'history') and isinstance(history.history, dict):
            self.logger.info("Training completed with final metrics:")
            for metric, values in history.history.items():
                if values:  # Only log if there are values
                    # Get the last value for each metric
                    last_value = values[-1] if isinstance(values, list) else values
                    self.logger.info(f"  {metric}: {last_value:.6f}")
        elif history:  # Fallback for non-standard history objects
            self.logger.info("Training completed, but could not parse training metrics")
        
        # Return the history object if it exists, otherwise return an empty dict
        if hasattr(history, 'history') and isinstance(history.history, dict):
            return history.history
        return history if history else {}
                    
    def predict(self, stock_data, additional_features=None, batch_size=32):
        """
        Make predictions using the trained model with enhanced error handling and logging.
        
        Args:
            stock_data: Input stock data (numpy array, list, or pandas DataFrame)
            additional_features: Additional features (numpy array, list, or pandas DataFrame)
            batch_size: Batch size for prediction
        
        Returns:
            numpy.ndarray: Predictions or None if prediction fails
        """
        try:
            if self.model is None:
                error_msg = "Model has not been trained yet. Please train the model before making predictions."
                self.logger.error(error_msg)
                return None
                
            # Log input data statistics
            self.logger.info("Starting prediction with input data:")
            self.logger.info(f"Stock data type: {type(stock_data).__name__}, shape: {np.array(stock_data).shape if hasattr(stock_data, '__array__') else 'N/A'}")
            self.logger.info(f"Additional features type: {type(additional_features).__name__ if additional_features is not None else 'None'}, "
                          f"shape: {np.array(additional_features).shape if hasattr(additional_features, '__array__') else 'N/A'}")
            
            # Convert pandas DataFrames to numpy arrays
            if hasattr(stock_data, 'values'):
                stock_data = stock_data.values
            if additional_features is not None and hasattr(additional_features, 'values'):
                additional_features = additional_features.values
                
            # Convert inputs to numpy arrays if they're not already
            stock_data = np.asarray(stock_data, dtype=np.float32)
            
            # Create default additional features if none provided
            if additional_features is None:
                additional_features = np.zeros((stock_data.shape[0], stock_data.shape[1], 6), dtype=np.float32)
            else:
                additional_features = np.asarray(additional_features, dtype=np.float32)

            # Shape validation with automatic padding/reshaping
            # Ensure we have 3D arrays
            if stock_data.ndim < 3:
                if stock_data.ndim == 2:
                    stock_data = np.expand_dims(stock_data, axis=-1)
                    self.logger.warning(f"Expanded stock_data to shape {stock_data.shape}")
                else:
                    self.logger.error(f"stock_data has invalid dimensions: {stock_data.ndim}, expected 2 or 3")
                    raise ValueError(f"stock_data must have 2 or 3 dimensions, got {stock_data.ndim}")
                    
            if additional_features.ndim < 3:
                if additional_features.ndim == 2:
                    additional_features = np.expand_dims(additional_features, axis=-1)
                    self.logger.warning(f"Expanded additional_features to shape {additional_features.shape}")
                else:
                    self.logger.error(f"additional_features has invalid dimensions: {additional_features.ndim}, expected 2 or 3")
                    raise ValueError(f"additional_features must have 2 or 3 dimensions, got {additional_features.ndim}")
            
            # Handle sequence length mismatches by padding
            if stock_data.shape[1] != additional_features.shape[1]:
                self.logger.warning(f"Sequence length mismatch: stock_data has {stock_data.shape[1]} timesteps, "
                                   f"additional_features has {additional_features.shape[1]} timesteps")
                
                # Pad the shorter sequence
                if stock_data.shape[1] < additional_features.shape[1]:
                    padding_length = additional_features.shape[1] - stock_data.shape[1]
                    padding = np.zeros((stock_data.shape[0], padding_length, stock_data.shape[2]), dtype=stock_data.dtype)
                    stock_data = np.concatenate([padding, stock_data], axis=1)
                    self.logger.info(f"Padded stock_data to shape {stock_data.shape}")
                else:
                    padding_length = stock_data.shape[1] - additional_features.shape[1]
                    padding = np.zeros((additional_features.shape[0], padding_length, additional_features.shape[2]), dtype=additional_features.dtype)
                    additional_features = np.concatenate([padding, additional_features], axis=1)
                    self.logger.info(f"Padded additional_features to shape {additional_features.shape}")
                    
            # Ensure feature dimensions are correct
            try:
                if stock_data.shape[2] != 1:
                    self.logger.warning(f"stock_data has {stock_data.shape[2]} features, expected 1. Reshaping...")
                    # If we have multiple features, take the first one
                    if stock_data.shape[2] > 1:
                        stock_data = stock_data[:, :, :1]
                    # If we have no features, add a dimension
                    else:
                        stock_data = np.expand_dims(stock_data, axis=-1)
                        
                if additional_features.shape[2] != 6:
                    self.logger.warning(f"additional_features has {additional_features.shape[2]} features, expected 6. Adjusting...")
                    # If we have too many features, truncate
                    if additional_features.shape[2] > 6:
                        additional_features = additional_features[:, :, :6]
                    # If we have too few features, pad with zeros
                    else:
                        padding = np.zeros((additional_features.shape[0], additional_features.shape[1], 
                                         6 - additional_features.shape[2]), dtype=additional_features.dtype)
                        additional_features = np.concatenate([additional_features, padding], axis=2)

                self.logger.info(f"Predicting with stock_data shape: {stock_data.shape}, "
                              f"additional_features shape: {additional_features.shape}")

                # Use a try-except block specifically for the model.predict call
                predictions_scaled = self.model.predict(
                    [stock_data, additional_features], 
                    batch_size=batch_size,
                    verbose=0  # Disable progress bar for cleaner output
                )
                
                if predictions_scaled is None:
                    raise ValueError("Model returned None predictions")
                
                # Ensure we have the correct number of predictions
                n_samples = stock_data.shape[0]
                if len(predictions_scaled) != n_samples:
                    self.logger.warning(f"Number of predictions ({len(predictions_scaled)}) doesn't match "
                                     f"number of samples ({n_samples}). Adjusting...")
                    if len(predictions_scaled) > n_samples:
                        predictions_scaled = predictions_scaled[:n_samples]
                        self.logger.info(f"Truncated predictions to {n_samples} samples")
                    else:
                        padding = np.zeros((n_samples - len(predictions_scaled), 1))
                        predictions_scaled = np.vstack([predictions_scaled, padding])
                        self.logger.info(f"Padded predictions to {n_samples} samples")
                
                # Handle multiple features in output if needed
                if predictions_scaled.ndim > 1 and predictions_scaled.shape[1] > 1:
                    self.logger.warning(f"Multiple features in output ({predictions_scaled.shape[1]}). "
                                     f"Taking only first feature.")
                    predictions_scaled = predictions_scaled[:, :1]
                
                # Ensure predictions are 2D for inverse transform
                if predictions_scaled.ndim == 1:
                    predictions_scaled = predictions_scaled.reshape(-1, 1)
                
                # Check for NaN or infinite values
                if not np.all(np.isfinite(predictions_scaled)):
                    nan_count = np.sum(~np.isfinite(predictions_scaled))
                    self.logger.warning(f"Found {nan_count} non-finite values in predictions. Replacing with zeros.")
                    predictions_scaled = np.nan_to_num(predictions_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Log prediction statistics
                self.logger.info(f"Prediction shape: {predictions_scaled.shape}, "
                              f"range: [{np.min(predictions_scaled):.4f}, {np.max(predictions_scaled):.4f}]")

                # Inverse transform predictions to original scale if target_scaler is available
                if hasattr(self, 'target_scaler') and hasattr(self.target_scaler, 'scale_'):
                    try:
                        # Ensure predictions_scaled has the right shape for inverse_transform
                        if predictions_scaled.ndim == 1:
                            predictions_scaled = predictions_scaled.reshape(-1, 1)
                            
                        predictions = self.target_scaler.inverse_transform(predictions_scaled)
                        self.logger.info(f"Inverse transformed predictions range: "
                                      f"[{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
                        return predictions.flatten()
                    except Exception as e:
                        self.logger.error(f"Error in inverse transform: {str(e)}")
                        self.logger.warning("Returning raw predictions due to inverse transform error")
                        return predictions_scaled.flatten()
                else:
                    self.logger.warning("target_scaler not available, returning raw predictions")
                    return predictions_scaled.flatten()
                    
            except tf.errors.InvalidArgumentError as e:
                self.logger.error(f"TensorFlow invalid argument error during prediction: {str(e)}")
                self.logger.error("This may be due to incompatible shapes. Check model input requirements.")
                return np.zeros(stock_data.shape[0] if hasattr(stock_data, 'shape') else 1)
                
            except Exception as e:
                self.logger.error(f"Error during model prediction: {str(e)}", exc_info=True)
                return np.zeros(stock_data.shape[0] if hasattr(stock_data, 'shape') else 1)
                
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}", exc_info=True)
            # Return zeros as fallback with shape matching input samples
            return np.zeros(stock_data.shape[0] if hasattr(stock_data, 'shape') and len(stock_data.shape) > 0 else 1)

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

    def predict(self, stock_data, additional_features=None, batch_size=32):
        """
        Make predictions using the trained model with robust input validation and correction.

        Args:
            stock_data: Stock price sequences (n_samples, seq_len, n_stock_features) or
                        (seq_len, n_stock_features) for single sample, or list/tuple/array/DataFrame
            additional_features: Additional feature sequences (n_samples, seq_len, n_features) or None.
            batch_size: Batch size for prediction

        Returns:
            numpy.ndarray: Predictions array (n_samples, 1) or None if prediction fails
        """
        try:
            if self.model is None:
                self.logger.error("Model has not been trained. Call train() first.")
                return None
                
            # Log input data statistics
            self.logger.info("Starting prediction with input data:")
            self.logger.info(f"Stock data type: {type(stock_data).__name__}, shape: {np.array(stock_data).shape if hasattr(stock_data, '__array__') else 'N/A'}")
            self.logger.info(f"Additional features type: {type(additional_features).__name__ if additional_features is not None else 'None'}, "
                          f"shape: {np.array(additional_features).shape if hasattr(additional_features, '__array__') else 'N/A'}")
            
            # Convert pandas DataFrames to numpy arrays
            if hasattr(stock_data, 'values'):
                stock_data = stock_data.values
            if additional_features is not None and hasattr(additional_features, 'values'):
                additional_features = additional_features.values
                
            # Convert inputs to numpy arrays if they're not already
            stock_data = np.asarray(stock_data, dtype=np.float32)
            
            # Handle single sample case
            if stock_data.ndim == 2:
                stock_data = np.expand_dims(stock_data, axis=0)
                
            if additional_features is not None:
                additional_features = np.asarray(additional_features, dtype=np.float32)
                if additional_features.ndim == 2:
                    additional_features = np.expand_dims(additional_features, axis=0)
                    
                # Ensure sequence lengths match
                if stock_data.shape[1] != additional_features.shape[1]:
                    min_seq_len = min(stock_data.shape[1], additional_features.shape[1])
                    stock_data = stock_data[:, -min_seq_len:, :]
                    additional_features = additional_features[:, -min_seq_len:, :]
                    self.logger.warning(f"Adjusted sequence lengths to {min_seq_len} for both stock and feature data")
            
            # Make predictions with error handling
            try:
                if additional_features is not None:
                    predictions = self.model.predict(
                        [stock_data, additional_features],
                        batch_size=batch_size,
                        verbose=0
                    )
                else:
                    predictions = self.model.predict(
                        stock_data,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                # Ensure we have the correct shape (n_samples, 1)
                if predictions.ndim > 2:
                    predictions = predictions.reshape(predictions.shape[0], -1)[:, -1:]
                elif predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                    
                self.logger.info(f"Successfully generated predictions with shape: {predictions.shape}")
                
                # Check for NaN or infinite values in predictions
                if np.any(~np.isfinite(predictions)):
                    self.logger.warning("Predictions contain NaN or infinite values")
                    return None
                    
                return predictions
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {str(e)}", exc_info_info=True)
                return None
                
        except Exception as e:
            self.logger.error(f"Error during prediction preparation: {str(e)}", exc_info=True)
            return None
            
    def ensure_array(self, arr, name="input"):
        """Convert input to numpy array with proper shape handling.
        
        Args:
            arr: Input array (numpy array, list, or pandas DataFrame)
            name: Name of the input for error messages
            
        Returns:
            Properly shaped numpy array
            
        Raises:
            ValueError: If the input cannot be converted to a valid shape
        """
        try:
            if arr is None:
                return None
                
            # Convert pandas DataFrame to numpy array if needed
            if hasattr(arr, 'values'):
                arr = arr.values
                
            # Convert to numpy array if not already
            arr = np.array(arr, dtype=np.float32)
            
            # Handle different input shapes
            if arr.ndim == 3:
                if arr.shape[0] > 1:
                    self.logger.warning(f"{name} has batch size > 1: {arr.shape}. Using only the last sequence.")
                    arr = arr[-1:]
                return arr
            elif arr.ndim == 2:
                return np.expand_dims(arr, axis=0)
            elif arr.ndim == 1:
                return arr.reshape(1, -1, 1)
            else:
                raise ValueError(f"{name} must be 1D, 2D, or 3D, got shape {arr.shape}")
                
        except Exception as e:
            self.logger.error(f"Error in ensure_array: {str(e)}")
            raise

            
            # Check for NaN values
            if np.isnan(stock_data).any():
                raise ValueError("stock_data contains NaN values which are not allowed")
                
            # Handle single sample case (2D input)
            if len(stock_data.shape) == 2:
                stock_data = np.expand_dims(stock_data, axis=0)
            elif len(stock_data.shape) != 3:
                raise ValueError(f"stock_data must be 2D or 3D, got shape {stock_data.shape}")
                
            # Ensure stock data has the expected shape (samples, 60, 1)
            if stock_data.shape[-1] != 1:
                # If we have more than 1 feature, use only the first one
                self.logger.warning(f"stock_data has {stock_data.shape[-1]} features, using only the first feature")
                stock_data = stock_data[..., :1]
                
            # Ensure we have the right number of timesteps (60)
            if stock_data.shape[1] != 60:
                self.logger.warning(f"Adjusting stock data timesteps from {stock_data.shape[1]} to 60")
                # If we have more timesteps, take the most recent 60
                if stock_data.shape[1] > 60:
                    stock_data = stock_data[:, -60:, :]
                # If we have fewer timesteps, pad with zeros at the beginning
                else:
                    pad_width = ((0, 0), (60 - stock_data.shape[1], 0), (0, 0))
                    stock_data = np.pad(stock_data, pad_width=pad_width, mode='constant')
            
            # Handle additional_features
            if additional_features is None:
                # Create zero features if not provided
                additional_features = np.zeros((stock_data.shape[0], 60, 6))  # Match the expected timesteps (60) and features (6)
                self.logger.warning("No additional features provided, using zeros with shape (samples, 60, 6)")
            else:
                # Handle list of arrays with different lengths
                if isinstance(additional_features, (list, tuple)):
                    # Convert each element to numpy array if it isn't already
                    additional_features = [np.asarray(x, dtype=np.float32) for x in additional_features]

                    # Get max sequence length
                    maxlen = max(x.shape[0] for x in additional_features)

                    # Pad sequences
                    padded_data = []
                    for i, seq in enumerate(additional_features):
                        if seq.ndim == 1:
                            seq = seq.reshape(-1, 1)  # Ensure 2D (timesteps, features)
                        # Pad sequences to max length
                        padded = np.pad(
                            seq,
                            ((0, maxlen - seq.shape[0]), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )
                        padded_data.append(padded)
                    additional_features = np.stack(padded_data)
                else:
                    # Handle regular numpy array input
                    additional_features = np.asarray(additional_features, dtype=np.float32)

                # Handle single sample case (2D input)
                if len(additional_features.shape) == 2:
                    additional_features = np.expand_dims(additional_features, axis=0)
                elif len(additional_features.shape) != 3:
                    raise ValueError(f"additional_features must be 2D or 3D, got shape {additional_features.shape}")

                # Ensure we have the right number of timesteps (60)
                if additional_features.shape[1] != 60:
                    self.logger.warning(f"Adjusting additional features timesteps from {additional_features.shape[1]} to 60")
                    # If we have more timesteps, take the most recent 60
                    if additional_features.shape[1] > 60:
                        additional_features = additional_features[:, -60:, :]
                    # If we have fewer timesteps, pad with zeros at the beginning
                    else:
                        pad_width = ((0, 0), (60 - additional_features.shape[1], 0), (0, 0))
                        additional_features = np.pad(additional_features, pad_width=pad_width, mode='constant')
                
                # Get expected feature dimension from the model
                expected_feature_dim = 10  # Default value
                
                # Try to get the actual expected dimension from the model input
                if self.model is not None:
                    try:
                        # Get the expected shape from the model's input
                        input_shapes = self.model.input_shape
                        if isinstance(input_shapes, list) and len(input_shapes) > 1:
                            expected_feature_dim = input_shapes[1][-1]
                            self.logger.info(f"Model expects additional features with dimension: {expected_feature_dim}")
                    except Exception as e:
                        self.logger.warning(f"Could not determine expected feature dimension from model: {str(e)}")
                
                # Ensure correct feature dimension
                if additional_features.shape[-1] > expected_feature_dim:
                    self.logger.warning(f"additional_features has {additional_features.shape[-1]} features, using first {expected_feature_dim} features")
                    additional_features = additional_features[..., :expected_feature_dim]
                elif additional_features.shape[-1] < expected_feature_dim:
                    # If fewer than expected features, pad with zeros
                    self.logger.warning(f"additional_features has only {additional_features.shape[-1]} features, padding to {expected_feature_dim} features")
                    padding = np.zeros((*additional_features.shape[:-1], expected_feature_dim - additional_features.shape[-1]))
                    additional_features = np.concatenate([additional_features, padding], axis=-1)
            
            # Log final shapes
            self.logger.info(f"Final shapes - stock_data: {stock_data.shape}, additional_features: {additional_features.shape}")
            
            # Make predictions with proper tensor handling
            import tensorflow as tf
            
            try:
                # Ensure we're in eager execution mode for prediction
                if not tf.executing_eagerly():
                    tf.config.run_functions_eagerly(True)
                
                # Convert inputs to tensors if they aren't already
                stock_tensor = tf.convert_to_tensor(stock_data, dtype=tf.float32)
                features_tensor = tf.convert_to_tensor(additional_features, dtype=tf.float32)
                
                # Make predictions with proper batching
                batch_size = 32  # Adjust based on your GPU memory
                num_samples = stock_data.shape[0]
                predictions = []
                
                for i in range(0, num_samples, batch_size):
                    batch_stock = stock_tensor[i:i+batch_size]
                    batch_features = features_tensor[i:i+batch_size]
                    
                    # Get expected feature dimension from the model for this batch
                    if self.model is not None:
                        try:
                            # Get the expected shape from the model's input
                            input_shapes = self.model.input_shape
                            if isinstance(input_shapes, list) and len(input_shapes) > 1:
                                expected_feature_dim = input_shapes[1][-1]
                                # Check if batch_features has the correct feature dimension
                                current_dim = batch_features.shape[-1]
                                if current_dim != expected_feature_dim:
                                    self.logger.warning(f"Batch features dimension mismatch: got {current_dim}, expected {expected_feature_dim}")
                                    # Fix the dimension for this batch
                                    if current_dim < expected_feature_dim:
                                        # Pad with zeros
                                        padding = tf.zeros((*batch_features.shape[:-1], expected_feature_dim - current_dim))
                                        batch_features = tf.concat([batch_features, padding], axis=-1)
                                    else:
                                        # Truncate
                                        batch_features = batch_features[..., :expected_feature_dim]
                        except Exception as e:
                            self.logger.warning(f"Could not adjust batch feature dimensions: {str(e)}")
                    
                    # Make predictions for this batch
                    batch_pred = self.model([batch_stock, batch_features], training=False)
                    
                    # Convert to numpy if needed
                    if hasattr(batch_pred, 'numpy'):
                        batch_pred = batch_pred.numpy()
                    predictions.append(batch_pred)
                
                # Combine all batch predictions
                if predictions:
                    predictions = np.vstack(predictions)
                else:
                    predictions = np.array([])
                
                # Ensure predictions is 2D (samples, 1)
                predictions = np.asarray(predictions, dtype=np.float32)
                if predictions.ndim > 2:
                    predictions = np.reshape(predictions, (predictions.shape[0], -1))
                if predictions.ndim == 1:
                    predictions = np.reshape(predictions, (-1, 1))
                    
            except Exception as e:
                self.logger.error(f"Tensor conversion/prediction error: {str(e)}")
                raise ValueError(f"Failed to make predictions: {str(e)}") from e
                
            # Ensure we have at least one sample dimension
            if predictions.ndim == 0:
                predictions = np.array([[predictions]])
                
            self.logger.info(f"Prediction successful. Output shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Enhanced error information
            if hasattr(self, 'model') and hasattr(self.model, 'inputs'):
                self.logger.error(f"Model expects inputs: {[inp.shape for inp in self.model.inputs]}")
            
            # Log model architecture details if available
            if hasattr(self.model, 'layers'):
                self.logger.error("Model architecture:")
                for i, layer in enumerate(self.model.layers):
                    try:
                        self.logger.error(f"  Layer {i}: {layer.name} - Input: {layer.input_shape}, Output: {layer.output_shape}")
                    except:
                        self.logger.error(f"  Layer {i}: {layer.name} - Input/Output shape not available")
            
            # Log input data statistics
            try:
                self.logger.error(f"Stock data stats - shape: {stock_data.shape if 'stock_data' in locals() else 'N/A'}, "
                               f"contains NaN: {np.isnan(stock_data).any() if 'stock_data' in locals() else 'N/A'}")
                if 'additional_features' in locals() and additional_features is not None:
                    self.logger.error(f"Additional features stats - shape: {additional_features.shape}, "
                                   f"contains NaN: {np.isnan(additional_features).any()}")
            except Exception as stats_err:
                self.logger.error(f"Error logging data statistics: {str(stats_err)}")
            
            raise ValueError(f"Prediction failed: {error_msg}") from e