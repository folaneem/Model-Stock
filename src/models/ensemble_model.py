import numpy as np
import pandas as pd
import os
import joblib
import logging
import tensorflow as tf
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import time
from functools import partial

# Scikit-learn imports
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Deep learning imports
try:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_KERAS_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow Keras not available. LSTM models will not be supported.")
    TENSORFLOW_KERAS_AVAILABLE = False

# Additional model imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    logging.warning("CatBoost not available. CatBoost models will not be supported.")
    CATBOOST_AVAILABLE = False

# For advanced ensemble methods
try:
    from mlxtend.regressor import StackingCVRegressor
    MLXTEND_AVAILABLE = True
except ImportError:
    logging.warning("mlxtend not available. StackingCV will not be supported.")
    MLXTEND_AVAILABLE = False

@dataclass
class ModelConfig:
    """Configuration for a model in the ensemble"""
    name: str
    model: Any
    weight: float = 1.0
    requires_scaling: bool = False
    is_neural_net: bool = False
    hyperparams: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = {}

class EnsembleMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"  # Simple weighted average
    STACKING = "stacking"  # Stacking ensemble with meta-learner
    VOTING = "voting"  # Voting ensemble
    STACKING_CV = "stacking_cv"  # Stacking with cross-validation
    BLENDING = "blending"  # Blending ensemble
    BOOSTING = "boosting"  # Boosting-like sequential ensemble (boosting-like)

class EnsembleModel:
    """Ensemble model that combines multiple base models"""
    
    def __init__(self, 
                input_shape: Tuple[int, int] = None,
                ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
                meta_learner: Optional[BaseEstimator] = None,
                use_hyperopt: bool = False,
                cv_folds: int = 5):
        """Initialize ensemble model
        
        Args:
            input_shape: Shape of input data (samples, features)
            ensemble_method: Method to use for ensemble predictions
            meta_learner: Meta-learner for stacking ensemble (default: Ridge)
            use_hyperopt: Whether to use hyperparameter optimization
            cv_folds: Number of cross-validation folds for time series split
        """
        self.input_shape = input_shape
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner if meta_learner is not None else Ridge(alpha=1.0)
        self.use_hyperopt = use_hyperopt
        self.cv_folds = cv_folds
        
        self.models = {}
        self.model_weights = {}
        self.model_configs = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        self._initialize_base_models()
        os.makedirs("models/ensemble", exist_ok=True)
        
    def _initialize_base_models(self):
        """Initialize base models with optimized configurations"""
        # Define model configurations with default hyperparameters
        self.model_configs = {
            'xgb': ModelConfig(
                name='xgb',
                model=XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
                weight=1.0,
                requires_scaling=False,
                is_neural_net=False,
                hyperparams={
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            ),
            'lgbm': ModelConfig(
                name='lgbm',
                model=LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
                weight=1.0,
                requires_scaling=False,
                is_neural_net=False,
                hyperparams={
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'num_leaves': 31,
                    'subsample': 0.8
                }
            ),
            'rf': ModelConfig(
                name='rf',
                model=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                weight=1.0,
                requires_scaling=False,
                is_neural_net=False,
                hyperparams={
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            ),
            'gbr': ModelConfig(
                name='gbr',
                model=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
                weight=1.0,
                requires_scaling=False,
                is_neural_net=False,
                hyperparams={
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'subsample': 0.8
                }
            ),
            'ridge': ModelConfig(
                name='ridge',
                model=Ridge(alpha=1.0),
                weight=1.0,
                requires_scaling=True,
                is_neural_net=False,
                hyperparams={
                    'alpha': 1.0
                }
            ),
            'svr': ModelConfig(
                name='svr',
                model=SVR(kernel='rbf', C=1.0, epsilon=0.1),
                weight=1.0,
                requires_scaling=True,
                is_neural_net=False,
                hyperparams={
                    'C': 1.0,
                    'epsilon': 0.1,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                }
            )
        }
        
        # Add LSTM model if TensorFlow Keras is available
        if TENSORFLOW_KERAS_AVAILABLE:
            self.model_configs['lstm'] = ModelConfig(
                name='lstm',
                model=self._create_lstm_model(),
                weight=1.0,
                requires_scaling=True,
                is_neural_net=True,
                hyperparams={
                    'lstm_units': 100,
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            )
        
        # Initialize models and weights
        self.models = {name: config.model for name, config in self.model_configs.items()}
        self.model_weights = {name: config.weight for name, config in self.model_configs.items()}
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight

    def _create_lstm_model(self):
        """Create and compile LSTM model"""
        if not TENSORFLOW_KERAS_AVAILABLE:
            logging.warning("TensorFlow Keras not available. Cannot create LSTM model.")
            return None
            
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
        # Filter out invalid scores
        valid_scores = {k: v for k, v in val_scores.items() if v != float('inf') and not np.isnan(v)}
        
        if not valid_scores:
            logging.warning("No valid scores for weight calculation. Using equal weights.")
            return {k: 1.0/len(val_scores) for k in val_scores}
        
        # Calculate inverse scores (lower error = higher weight)
        inverse_scores = {k: 1.0 / (v + 1e-10) for k, v in valid_scores.items()}
        total = sum(inverse_scores.values())
        
        # Normalize weights
        weights = {k: v/total for k, v in inverse_scores.items()}
        
        # Add zero weights for invalid scores
        for k in val_scores:
            if k not in weights:
                weights[k] = 0.0
                
        return weights

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                            model_names: List[str] = None,
                            n_trials: int = 50,
                            timeout: int = 3600) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for selected models"""
        if not self.use_hyperopt:
            logging.warning("Hyperparameter optimization is disabled. Set use_hyperopt=True to enable.")
            return {}
            
        try:
            from src.utils.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
        except ImportError:
            logging.error("Could not import HyperparameterOptimizer. Make sure it's installed.")
            return {}
            
        if model_names is None:
            model_names = list(self.models.keys())
            
        # Configure optimizer
        config = OptimizationConfig(
            n_trials=n_trials,
            timeout=timeout,
            cv_folds=self.cv_folds,
            metric="rmse"
        )
        
        optimizer = HyperparameterOptimizer(config)
        best_params = {}
        
        for name in model_names:
            if name not in self.models:
                logging.warning(f"Model {name} not found in ensemble")
                continue
                
            logging.info(f"Optimizing hyperparameters for {name}...")
            
            # Reshape data for traditional ML models
            X_opt = X
            if name != 'lstm' and len(X.shape) > 2:
                X_opt = X.reshape(X.shape[0], -1)
                
            # Run optimization
            try:
                results = optimizer.optimize(name, X_opt, y)
                best_params[name] = results['best_params']
                
                # Update model with best parameters
                self.models[name] = optimizer.get_best_model(name)
                
                # Update model config
                if name in self.model_configs:
                    self.model_configs[name].hyperparams.update(best_params[name])
                    
                logging.info(f"Best parameters for {name}: {best_params[name]}")
                
            except Exception as e:
                logging.error(f"Error optimizing {name}: {str(e)}")
                
        return best_params
    
    def _train_ensemble_model(self, X: np.ndarray, y: np.ndarray, individual_preds: Dict[str, np.ndarray] = None):
        """Train the ensemble model based on the selected method"""
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Nothing to train for weighted average
            return
            
        elif self.ensemble_method == EnsembleMethod.STACKING:
            # Prepare data for meta-learner
            if individual_preds is None:
                individual_preds = self._get_individual_predictions(X)
                
            # Stack predictions as features for meta-learner
            meta_features = np.column_stack([individual_preds[name] for name in individual_preds])
            
            # Train meta-learner
            self.meta_learner.fit(meta_features, y)
            
        elif self.ensemble_method == EnsembleMethod.VOTING:
            # Create VotingRegressor
            estimators = [(name, model) for name, model in self.models.items() 
                         if name in self.model_weights and self.model_weights[name] > 0
                         and not (name == 'lstm' and TENSORFLOW_KERAS_AVAILABLE)]
            
            if not estimators:
                logging.warning("No valid models for VotingRegressor")
                return
                
            self.ensemble_model = VotingRegressor(
                estimators=estimators,
                weights=[self.model_weights[name] for name, _ in estimators]
            )
            
            # Reshape X for traditional ML models
            X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            self.ensemble_model.fit(X_reshaped, y)
            
        elif self.ensemble_method == EnsembleMethod.STACKING_CV and MLXTEND_AVAILABLE:
            # Create StackingCVRegressor
            estimators = [(name, model) for name, model in self.models.items() 
                         if name in self.model_weights and self.model_weights[name] > 0 
                         and not (name == 'lstm' and TENSORFLOW_KERAS_AVAILABLE)]
            
            if not estimators:
                logging.warning("No valid models for StackingCVRegressor")
                return
                
            self.ensemble_model = StackingCVRegressor(
                regressors=[model for _, model in estimators],
                meta_regressor=self.meta_learner,
                cv=TimeSeriesSplit(n_splits=self.cv_folds),
                use_features_in_secondary=True
            )
            
            # Reshape X for traditional ML models
            X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            self.ensemble_model.fit(X_reshaped, y)
            
        elif self.ensemble_method == EnsembleMethod.BLENDING:
            # Simple blending approach
            if individual_preds is None:
                individual_preds = self._get_individual_predictions(X)
                
            # Stack predictions as features
            meta_features = np.column_stack([individual_preds[name] for name in individual_preds])
            
            # Train meta-learner with elastic net for robustness
            if not isinstance(self.meta_learner, ElasticNet):
                self.meta_learner = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
                
            self.meta_learner.fit(meta_features, y)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = 100, batch_size: int = 32,
             optimize_hyperparams: bool = None) -> Dict[str, float]:
        """Train all models in the ensemble"""
        start_time = time.time()
        logging.info("Training ensemble model...")
        
        # Determine whether to optimize hyperparameters
        if optimize_hyperparams is None:
            optimize_hyperparams = self.use_hyperopt
            
        # Split data if validation set not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data

        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            self.optimize_hyperparameters(X_train, y_train)

        # Train individual models
        val_scores = {}
        individual_preds_val = {}
        
        for name, model in self.models.items():
            try:
                logging.info(f"Training {name} model...")
                config = self.model_configs.get(name, None)
                is_neural_net = config.is_neural_net if config else 'lstm' in name
                requires_scaling = config.requires_scaling if config else False
                
                # Prepare data
                X_train_model = X_train
                X_val_model = X_val
                
                # Reshape for traditional ML models
                if not is_neural_net and len(X_train.shape) > 2:
                    X_train_model = X_train.reshape(X_train.shape[0], -1)
                    X_val_model = X_val.reshape(X_val.shape[0], -1)
                
                # Apply scaling if needed
                if requires_scaling:
                    self.scaler.fit(X_train_model)
                    X_train_model = self.scaler.transform(X_train_model)
                    X_val_model = self.scaler.transform(X_val_model)
                
                # Train model
                if is_neural_net and TENSORFLOW_KERAS_AVAILABLE:
                    model.fit(
                        X_train_model, y_train,
                        validation_data=(X_val_model, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                        ],
                        verbose=0
                    )
                    y_pred = model.predict(X_val_model, verbose=0).flatten()
                else:
                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_val_model)
                
                # Store validation predictions and scores
                individual_preds_val[name] = y_pred
                val_scores[name] = np.sqrt(mean_squared_error(y_val, y_pred))
                logging.info(f"{name} validation RMSE: {val_scores[name]:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                val_scores[name] = float('inf')
        
        # Calculate model weights based on validation performance
        self.model_weights = self._calculate_weights(val_scores)
        logging.info(f"Model weights: {self.model_weights}")
        
        # Train ensemble model if using a method other than weighted average
        if self.ensemble_method != EnsembleMethod.WEIGHTED_AVERAGE:
            logging.info(f"Training {self.ensemble_method.value} ensemble...")
            self._train_ensemble_model(X_train, y_train, individual_preds_val)
        
        logging.info(f"Ensemble training completed in {time.time() - start_time:.2f} seconds")
        return val_scores
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the selected ensemble method"""
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Get individual model predictions
            predictions = self._get_individual_predictions(X)
            
            # Weighted average of predictions
            weighted_pred = np.zeros(X.shape[0])
            total_weight = 0
            
            for name, pred in predictions.items():
                if name in self.model_weights:
                    weighted_pred += pred * self.model_weights[name]
                    total_weight += self.model_weights[name]
            
            if total_weight > 0:
                weighted_pred /= total_weight
                
            return weighted_pred
            
        elif self.ensemble_method == EnsembleMethod.STACKING or self.ensemble_method == EnsembleMethod.BLENDING:
            # Get individual model predictions
            predictions = self._get_individual_predictions(X)
            
            # Stack predictions as features for meta-learner
            meta_features = np.column_stack([predictions[name] for name in predictions])
            
            # Use meta-learner to make final prediction
            return self.meta_learner.predict(meta_features)
            
        elif (self.ensemble_method == EnsembleMethod.VOTING or 
              self.ensemble_method == EnsembleMethod.STACKING_CV) and self.ensemble_model is not None:
            # Use the trained ensemble model directly
            X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            return self.ensemble_model.predict(X_reshaped)
            
        else:
            # Fallback to weighted average if ensemble model not available
            logging.warning(f"Ensemble method {self.ensemble_method.value} not fully implemented. Using weighted average.")
            return self.predict(X)  # Will use weighted average as default
        
    def _get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                config = self.model_configs.get(name, None)
                is_neural_net = config.is_neural_net if config else 'lstm' in name
                
                if is_neural_net and TENSORFLOW_KERAS_AVAILABLE:
                    pred = model.predict(X, verbose=0).flatten()
                else:
                    # Reshape for traditional ML models
                    X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
                    pred = model.predict(X_reshaped)
                    
                predictions[name] = pred
                
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
                
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble model performance with comprehensive metrics
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        individual_preds = self._get_individual_predictions(X)
        
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
        
    def save(self, model_dir: str = "models/ensemble"):
        """Save ensemble model to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            'ensemble_method': self.ensemble_method.value,
            'model_weights': self.model_weights,
            'cv_folds': self.cv_folds,
            'use_hyperopt': self.use_hyperopt
        }
        
        with open(os.path.join(model_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save model configurations
        model_configs = {}
        for name, config in self.model_configs.items():
            model_configs[name] = {
                'name': config.name,
                'weight': config.weight,
                'requires_scaling': config.requires_scaling,
                'is_neural_net': config.is_neural_net,
                'hyperparams': config.hyperparams
            }
            
        with open(os.path.join(model_dir, 'model_configs.json'), 'w') as f:
            json.dump(model_configs, f)
        
        # Save individual models
        for name, model in self.models.items():
            try:
                if name == 'lstm' and TENSORFLOW_KERAS_AVAILABLE:
                    # Save using the recommended Keras format
                    model.save(os.path.join(model_dir, f"{name}.keras"), save_format='keras')
                else:
                    joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))
            except Exception as e:
                logging.error(f"Error saving {name}: {str(e)}")
        
        # Save ensemble model if it exists
        if self.ensemble_model is not None:
            try:
                joblib.dump(self.ensemble_model, os.path.join(model_dir, "ensemble_model.joblib"))
            except Exception as e:
                logging.error(f"Error saving ensemble model: {str(e)}")
        
        # Save scaler
        try:
            joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        except Exception as e:
            logging.error(f"Error saving scaler: {str(e)}")
        
        # Save meta-learner if it exists
        if self.meta_learner is not None:
            try:
                joblib.dump(self.meta_learner, os.path.join(model_dir, "meta_learner.joblib"))
            except Exception as e:
                logging.error(f"Error saving meta-learner: {str(e)}")
                
        logging.info(f"Ensemble model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str = "models/ensemble"):
        """Load ensemble model from disk"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found")
        
        # Load ensemble configuration
        try:
            with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
                config = json.load(f)
                
            ensemble_method = EnsembleMethod(config.get('ensemble_method', 'weighted_average'))
            cv_folds = config.get('cv_folds', 5)
            use_hyperopt = config.get('use_hyperopt', False)
            
            # Create new instance
            instance = cls(
                ensemble_method=ensemble_method,
                use_hyperopt=use_hyperopt,
                cv_folds=cv_folds
            )
            
            # Load model weights
            instance.model_weights = config.get('model_weights', {})
            
        except Exception as e:
            logging.error(f"Error loading ensemble configuration: {str(e)}")
            instance = cls()  # Fallback to default
        
        # Load model configurations
        try:
            with open(os.path.join(model_dir, 'model_configs.json'), 'r') as f:
                model_configs = json.load(f)
                
            instance.model_configs = {}
            for name, config in model_configs.items():
                instance.model_configs[name] = ModelConfig(
                    name=config['name'],
                    model=None,  # Will be loaded below
                    weight=config['weight'],
                    requires_scaling=config['requires_scaling'],
                    is_neural_net=config['is_neural_net'],
                    hyperparams=config['hyperparams']
                )
                
        except Exception as e:
            logging.error(f"Error loading model configurations: {str(e)}")
        
        # Load individual models
        instance.models = {}
        for name in instance.model_configs.keys():
            try:
                if name == 'lstm' and TENSORFLOW_KERAS_AVAILABLE and os.path.exists(os.path.join(model_dir, f"{name}.h5")):
                    instance.models[name] = tf.keras.models.load_model(os.path.join(model_dir, f"{name}.h5"))
                elif os.path.exists(os.path.join(model_dir, f"{name}.joblib")):
                    instance.models[name] = joblib.load(os.path.join(model_dir, f"{name}.joblib"))
                else:
                    logging.warning(f"Model {name} not found in {model_dir}")
            except Exception as e:
                logging.error(f"Error loading {name}: {str(e)}")
        
        # Update model references in model_configs
        for name, model in instance.models.items():
            if name in instance.model_configs:
                instance.model_configs[name].model = model
        
        # Load ensemble model if it exists
        if os.path.exists(os.path.join(model_dir, "ensemble_model.joblib")):
            try:
                instance.ensemble_model = joblib.load(os.path.join(model_dir, "ensemble_model.joblib"))
            except Exception as e:
                logging.error(f"Error loading ensemble model: {str(e)}")
        
        # Load scaler
        if os.path.exists(os.path.join(model_dir, "scaler.joblib")):
            try:
                instance.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            except Exception as e:
                logging.error(f"Error loading scaler: {str(e)}")
                
        # Load meta-learner
        if os.path.exists(os.path.join(model_dir, "meta_learner.joblib")):
            try:
                instance.meta_learner = joblib.load(os.path.join(model_dir, "meta_learner.joblib"))
            except Exception as e:
                logging.error(f"Error loading meta-learner: {str(e)}")
                
        logging.info(f"Ensemble model loaded from {model_dir}")
        return instance
    
    @classmethod
    def load_models(cls, model_dir: str = "models/ensemble") -> 'EnsembleModel':
        """
        Load all models from disk, supporting both new and legacy formats
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Instance of EnsembleModel with loaded models
        """
        import json
        from tensorflow.keras.models import model_from_json
        
        ensemble = cls()
        
        if not os.path.exists(model_dir):
            logging.error(f"Model directory not found: {model_dir}")
            return ensemble
            
        # Load each model
        for model_name in os.listdir(model_dir):
            try:
                model_path = os.path.join(model_dir, model_name)
                
                # Check if it's a directory (new format)
                if os.path.isdir(model_path):
                    # Try to load Keras model
                    keras_model_path = os.path.join(model_path, 'model.keras')
                    if os.path.exists(keras_model_path):
                        from tensorflow.keras.models import load_model
                        model = load_model(keras_model_path, compile=False)
                        ensemble.models[model_name] = model
                        logging.info(f"Loaded Keras model: {model_name}")
                        continue
                        
                    # Try to load from JSON + weights
                    config_path = os.path.join(model_path, 'model_config.json')
                    weights_path = os.path.join(model_path, 'model_weights.weights.h5')
                    
                    if os.path.exists(config_path) and os.path.exists(weights_path):
                        with open(config_path, 'r') as json_file:
                            model = model_from_json(json_file.read())
                        model.load_weights(weights_path)
                        ensemble.models[model_name] = model
                        logging.info(f"Loaded model from config+weights: {model_name}")
                        continue
                
                # Legacy format support
                if model_name.endswith('.h5') or model_name.endswith('.keras'):
                    from tensorflow.keras.models import load_model
                    model = load_model(os.path.join(model_dir, model_name), compile=False)
                    ensemble.models[os.path.splitext(model_name)[0]] = model
                    logging.info(f"Loaded legacy model: {model_name}")
                    
                elif model_name.endswith('.joblib'):
                    import joblib
                    model = joblib.load(os.path.join(model_dir, model_name))
                    ensemble.models[os.path.splitext(model_name)[0]] = model
                    logging.info(f"Loaded model: {model_name}")
                    
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {e}", exc_info=True)
                
        return ensemble

    def get_feature_importances(self) -> Dict[str, np.ndarray]:
        """Get feature importances from the ensemble model
        
        Returns:
            Dictionary mapping model names to feature importance arrays
        """
        feature_importances = {}
        
        # Collect feature importances from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
        
        # For ensemble model
        if self.ensemble_model is not None and hasattr(self.ensemble_model, 'feature_importances_'):
            feature_importances['ensemble'] = self.ensemble_model.feature_importances_
            
        return feature_importances
        
    def plot_feature_importances(self, feature_names=None):
        """Plot feature importances for models that support it
        
        Args:
            feature_names: List of feature names for plotting
            
        Returns:
            Matplotlib figure object if plotting is successful, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            
            importances = self.get_feature_importances()
            if not importances:
                logging.warning("No feature importances available to plot")
                return None
                
            fig, axes = plt.subplots(nrows=len(importances), figsize=(10, 3*len(importances)))
            if len(importances) == 1:
                axes = [axes]
                
            for i, (name, importance) in enumerate(importances.items()):
                if feature_names is None:
                    feature_names = [f"Feature {j}" for j in range(len(importance))]
                    
                # Sort by importance
                indices = np.argsort(importance)
                sorted_names = [feature_names[j] for j in indices]
                sorted_importance = importance[indices]
                
                # Plot
                axes[i].barh(sorted_names, sorted_importance)
                axes[i].set_title(f"{name} Feature Importance")
                axes[i].set_xlabel('Importance')
                
            plt.tight_layout()
            return fig
            
        except ImportError:
            logging.warning("Matplotlib not available for plotting feature importances")
            return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        # Create and train ensemble
        ensemble = EnsembleModel(ensemble_method=EnsembleMethod.STACKING, use_hyperopt=True)
        
        # Example data (replace with your actual data)
        X_train = np.random.rand(100, 60, 6)  # 100 samples, 60 time steps, 6 features
        y_train = np.random.rand(100, 1).flatten()  # Target values
        
        # Train the ensemble
        print("Training ensemble model...")
        val_scores = ensemble.train(X_train, y_train, epochs=10, batch_size=32)
        print("\nValidation scores:", val_scores)
        print("\nModel weights:", ensemble.get_model_weights())
        
        # Make predictions
        X_test = np.random.rand(10, 60, 6)
        y_test = np.random.rand(10, 1).flatten()
        
        predictions = ensemble.predict(X_test)
        metrics = ensemble.evaluate(X_test, y_test)
        
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save models
        ensemble.save()
        print("\nModels saved successfully.")
        
        # Load models
        loaded_ensemble = EnsembleModel.load()
        loaded_predictions = loaded_ensemble.predict(X_test)
        print(f"\nLoaded model predictions match: {np.allclose(predictions, loaded_predictions)}")       
    except Exception as e:
        logging.exception("An error occurred during execution:")
