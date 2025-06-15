import numpy as np
import pandas as pd
import os
import joblib
import logging
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Deep learning imports
from tensorflow.keras.models import load_model, Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Additional model imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

@dataclass
class ModelConfig:
    name: str
    model: object
    weight: float = 1.0
    requires_scaling: bool = False
    is_neural_net: bool = False

class EnsembleModel:
    def __init__(self, model_dir: str = "models/ensemble", use_meta_learner: bool = True):
        self.model_dir = model_dir
        self.use_meta_learner = use_meta_learner
        self.models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.model_weights = {}
        self._initialize_base_models()
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _initialize_base_models(self):
        """Initialize base models with optimized configurations"""
        self.models = {
            'lstm': self._create_lstm_model(),
            'xgb': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            'lgbm': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'gbr': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        self.model_weights = {name: 1.0/len(self.models) for name in self.models}

    def _create_lstm_model(self):
        """Create and compile LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def _calculate_weights(self, val_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate model weights based on validation performance"""
        inverse_scores = {k: 1.0 / (v + 1e-10) for k, v in val_scores.items()}
        total = sum(inverse_scores.values())
        return {k: v/total for k, v in inverse_scores.items()}

    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data

        val_scores = {}
        
        for name, model in self.models.items():
            try:
                if 'lstm' in name:
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                        ],
                        verbose=0
                    )
                    y_pred = model.predict(X_val).flatten()
                else:
                    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
                    y_pred = model.predict(X_val.reshape(X_val.shape[0], -1))
                
                val_scores[name] = np.sqrt(mean_squared_error(y_val, y_pred))
                
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                val_scores[name] = float('inf')
        
        self.model_weights = self._calculate_weights(val_scores)
        return val_scores
                
    def predict(self, X: np.ndarray, return_individual: bool = False) -> np.ndarray:
        """
        Make predictions using ensemble of models with dynamic weighting
        
        Args:
            X: Input features
            return_individual: If True, returns both ensemble and individual predictions
            
        Returns:
            Ensemble predictions (and individual predictions if return_individual=True)
        """
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                if 'lstm' in name:
                    pred = model.predict(X).flatten()
                else:
                    pred = model.predict(X.reshape(X.shape[0], -1))
                
                predictions.append(pred)
                weights.append(self.model_weights[name])
                
            except Exception as e:
                logging.error(f"Error predicting with {name}: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("No valid predictions from any model")
        
        # Weighted average of predictions
        final_pred = np.average(predictions, axis=0, weights=weights)
        
        if return_individual:
            return final_pred, predictions
        return final_pred
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble model performance with comprehensive metrics
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred, individual_preds = self.predict(X, return_individual=True)
        
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'directional_accuracy': np.mean(
                np.sign(y_pred[1:] - y_pred[:-1]) == 
                np.sign(y[1:] - y[:-1])
            ) * 100 if len(y) > 1 else float('nan')
        }
        
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current weights for each model in the ensemble
        
        Returns:
            Dictionary mapping model names to their current weights
        """
        return self.model_weights.copy()
        
    def save_models(self, base_path: str = None):
        """
        Save all models to disk
        
        Args:
            base_path: Base directory to save models (defaults to self.model_dir)
        """
        if base_path is None:
            base_path = self.model_dir
            
        os.makedirs(base_path, exist_ok=True)
        
        for name, model in self.models.items():
            try:
                if 'lstm' in name:
                    model.save(os.path.join(base_path, f"{name}.keras"))
                else:
                    joblib.dump(model, os.path.join(base_path, f"{name}.joblib"))
            except Exception as e:
                logging.error(f"Error saving model {name}: {e}")
    
    @classmethod
    def load_models(cls, model_dir: str = "models/ensemble") -> 'EnsembleModel':
        """
        Load all models from disk
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Instance of EnsembleModel with loaded models
        """
        ensemble = cls(model_dir=model_dir)
        
        for name in ensemble.models.keys():
            try:
                if 'lstm' in name:
                    model_path = os.path.join(model_dir, f"{name}.keras")
                    if os.path.exists(model_path):
                        ensemble.models[name] = load_model(model_path)
                else:
                    model_path = os.path.join(model_dir, f"{name}.joblib")
                    if os.path.exists(model_path):
                        ensemble.models[name] = joblib.load(model_path)
            except Exception as e:
                logging.error(f"Error loading model {name}: {e}")
        
        return ensemble

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        # Create and train ensemble
        ensemble = EnsembleModel()
        
        # Example data (replace with your actual data)
        X_train = np.random.rand(100, 60, 6)  # 100 samples, 60 time steps, 6 features
        y_train = np.random.rand(100, 1)      # Target values
        
        # Train the ensemble
        print("Training ensemble model...")
        val_scores = ensemble.train(X_train, y_train, epochs=50, batch_size=32)
        print("\nValidation scores:", val_scores)
        print("\nModel weights:", ensemble.get_model_weights())
        
        # Make predictions
        X_test = np.random.rand(10, 60, 6)
        y_test = np.random.rand(10, 1)
        
        predictions = ensemble.predict(X_test)
        metrics = ensemble.evaluate(X_test, y_test)
        
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save models
        ensemble.save_models()
        print("\nModels saved successfully.")
        
    except Exception as e:
        logging.exception("An error occurred during execution:")
