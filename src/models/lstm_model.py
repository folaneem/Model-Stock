import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, Attention, Bidirectional, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Normalization

class LSTMModel:
    def __init__(self, sequence_length=60, features=6):
        self.model = None
        self.sequence_length = sequence_length
        self.features = features
        self.history = None
        self.metrics = {}
        self.model_path = "models/lstm_model.keras"
        self.checkpoint_path = "models/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.keras"
        
    def build_model(self, units=64, dropout=0.3, learning_rate=0.001, conv_filters=32, kernel_size=3, attention_units=32):
        """
        Build advanced LSTM model with Conv1D, batch normalization, and improved attention
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Initial learning rate
            conv_filters: Number of filters in Conv1D layer
            kernel_size: Size of the convolution window
            attention_units: Number of units in attention layer
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.features))
        
        # Feature normalization
        x = Normalization(axis=-1)(inputs)  # Normalize each feature independently
        
        # 1D Convolutional layers for local pattern extraction
        x = Conv1D(filters=conv_filters, kernel_size=kernel_size, 
                  activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        # Bidirectional LSTM layers with batch normalization
        x = Bidirectional(LSTM(units=units, return_sequences=True, 
                            kernel_regularizer=l2(0.01),
                            recurrent_regularizer=l2(0.01)))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        # Second LSTM layer
        x = Bidirectional(LSTM(units=units, return_sequences=True, 
                            kernel_regularizer=l2(0.01),
                            recurrent_regularizer=l2(0.01)))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        # Improved attention mechanism
        attention_units = min(units * 2, 128)  # Ensure attention_units is reasonable
        attention = TimeDistributed(Dense(attention_units, activation='tanh'))(x)
        attention = TimeDistributed(Dense(1, activation='relu'))(attention)
        attention = tf.squeeze(attention, axis=-1)
        attention = tf.nn.softmax(attention, axis=1)
        context = tf.matmul(tf.expand_dims(attention, axis=1), x)
        context = tf.squeeze(context, axis=1)
        
        # Output layer
        outputs = Dense(1, activation='linear')(context)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Enhanced learning rate schedule with warmup
        def lr_schedule(epoch, current_lr):
            if epoch < 5:  # Warmup phase
                return learning_rate * (epoch + 1) / 5
            elif epoch < 20:
                return learning_rate
            elif epoch < 40:
                return learning_rate * 0.5
            elif epoch < 60:
                return learning_rate * 0.1
            else:
                return learning_rate * 0.01
        
        self.lr_scheduler = LearningRateScheduler(lr_schedule)
        
        # Compile model with gradient clipping
        optimizer = Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,  # Gradient clipping
            clipvalue=0.5
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )

    def train(self, x_train, y_train, validation_data=None, epochs=10, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the LSTM model with advanced training techniques
        
        Args:
            x_train: Training data (numpy array or tuple of numpy arrays)
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation if validation_data is None
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            
        Returns:
            Dictionary containing training history and metrics
        """
        if self.model is None:
            self.build_model()

        # Create directory structure
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                min_delta=0.0001,  # Minimum change to qualify as improvement
                restore_best_weights=True,
                verbose=verbose
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=verbose
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            ),
            self.lr_scheduler
        ]

        # Print model summary
        if verbose > 0:
            print("\nModel Summary:")
            self.model.summary()
            
            # Print data shapes
            if isinstance(x_train, (list, tuple)):
                print("\nInput shapes:")
                for i, x in enumerate(x_train):
                    print(f"  Input {i+1}: {x.shape}")
            else:
                print(f"\nInput shape: {x_train.shape}")
            print(f"Output shape: {y_train.shape}")
            
            if validation_data is not None:
                val_x, val_y = validation_data
                if isinstance(val_x, (list, tuple)):
                    print("\nValidation input shapes:")
                    for i, x in enumerate(val_x):
                        print(f"  Val Input {i+1}: {x.shape}")
                else:
                    print(f"\nValidation input shape: {val_x.shape}")
                print(f"Validation output shape: {val_y.shape}")

        # Train the model
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series data
        )

        # Save final model using the recommended Keras format
        self.model.save(self.model_path, save_format='keras')
        
        # Store training history
        self.history = history.history
        
        # Calculate and store metrics
        if 'val_loss' in history.history:
            best_epoch = np.argmin(history.history['val_loss']) + 1
            self.metrics = {
                'history': history.history,
                'best_epoch': best_epoch,
                'best_val_loss': np.min(history.history['val_loss']),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            }
        else:
            best_epoch = np.argmin(history.history['loss']) + 1
            self.metrics = {
                'history': history.history,
                'best_epoch': best_epoch,
                'best_loss': np.min(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_mae': history.history['mae'][-1]
            }

        if verbose > 0:
            print("\nTraining completed!")
            if 'val_loss' in history.history:
                print(f"Best validation loss: {self.metrics['best_val_loss']:.4f} at epoch {best_epoch}")
            print(f"Final training loss: {self.metrics['final_loss']:.4f}")

        return self.metrics

    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            self.load_model()
            
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'loss': self.model.evaluate(X_test, y_test)[0],
            'mae': self.model.evaluate(X_test, y_test)[1],
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
            'r2': r2_score(y_test, predictions),
            'directional_accuracy': np.mean(np.sign(y_test[1:] - y_test[:-1]) == 
                                          np.sign(predictions[1:] - predictions[:-1])) * 100
        }
        
        self.metrics.update(metrics)
        
        return metrics

    def load_model(self):
        """
        Load a pre-trained model
        """
        if os.path.exists(self.model_path):
            from tensorflow.keras.models import load_model
            try:
                self.model = load_model(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def plot_training_history(self):
        """
        Advanced training history visualization
        """
        if not self.metrics:
            return
            
        history = self.metrics['history']
        
        plt.figure(figsize=(15, 12))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # MAE plot
        plt.subplot(2, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('MAE Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Learning rate plot
        plt.subplot(2, 2, 3)
        plt.plot([self.lr_scheduler(epoch) for epoch in range(len(history['loss']))],
                label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        # Combined metrics
        plt.subplot(2, 2, 4)
        plt.plot(history['loss'], label='Loss')
        plt.plot(history['mae'], label='MAE')
        plt.title('Combined Metrics')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def predict(self, data, days):
        """
        Advanced prediction with confidence intervals
        """
        if self.model is None:
            self.load_model()

        predictions = []
        confidence_intervals = []
        current_data = data[-self.sequence_length:]

        for _ in range(days):
            # Reshape data for prediction
            current_data = current_data.reshape((1, self.sequence_length, self.features))
            
            # Make prediction
            prediction = self.model.predict(current_data)
            predictions.append(prediction[0][0])
            
            # Estimate confidence interval
            std_dev = np.std(current_data)
            margin_error = 1.96 * std_dev / np.sqrt(self.sequence_length)
            confidence_intervals.append((prediction[0][0] - margin_error, 
                                      prediction[0][0] + margin_error))
            
            # Update current data with new prediction
            current_data = np.roll(current_data, -1, axis=1)
            current_data[0, -1, 0] = prediction[0][0]

        return np.array(predictions), np.array(confidence_intervals)
