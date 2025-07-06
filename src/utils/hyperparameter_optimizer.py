import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from dataclasses import dataclass

# Hyperparameter optimization libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optimization libraries
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    n_trials: int = 50
    timeout: Optional[int] = 3600  # 1 hour
    n_jobs: int = -1  # Use all available cores
    cv_folds: int = 5
    early_stopping_rounds: int = 10
    random_state: int = 42
    study_name: str = "stock_prediction_optimization"
    direction: str = "minimize"  # "minimize" for error metrics, "maximize" for accuracy
    metric: str = "rmse"  # Options: "rmse", "mae", "r2"

class HyperparameterOptimizer:
    """Class for automated hyperparameter optimization of prediction models"""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize the optimizer with configuration"""
        self.config = config or OptimizationConfig()
        self.best_params = {}
        self.best_score = None
        self.study = None
        
    def _get_objective_function(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                              custom_objective: Optional[Callable] = None) -> Callable:
        """Create an objective function for Optuna based on model type"""
        
        if custom_objective is not None:
            return lambda trial: custom_objective(trial, X, y)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            if model_type == "lstm":
                params = {
                    'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=32),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200, step=25),
                }
                
                # Import here to avoid circular imports
                from src.models.lstm_model import LSTMModel
                
                # Initialize model with trial parameters
                model = LSTMModel(
                    lstm_units=params['lstm_units'],
                    dropout_rate=params['dropout_rate'],
                    recurrent_dropout=params['recurrent_dropout'],
                    learning_rate=params['learning_rate']
                )
                
            elif model_type == "two_stage":
                params = {
                    'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=32),
                    'attention_units': trial.suggest_int('attention_units', 16, 128, step=16),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200, step=25),
                }
                
                # Import here to avoid circular imports
                from src.models.two_stage_model import TwoStagePredictor
                
                # Initialize model with trial parameters
                model = TwoStagePredictor(
                    lstm_units=params['lstm_units'],
                    attention_units=params['attention_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
                
            elif model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    **params,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
                
            elif model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    **params,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
                
            elif model_type == "ensemble":
                params = {
                    'lstm_weight': trial.suggest_float('lstm_weight', 0.0, 1.0),
                    'xgb_weight': trial.suggest_float('xgb_weight', 0.0, 1.0),
                    'lgbm_weight': trial.suggest_float('lgbm_weight', 0.0, 1.0),
                    'rf_weight': trial.suggest_float('rf_weight', 0.0, 1.0),
                    'gbr_weight': trial.suggest_float('gbr_weight', 0.0, 1.0),
                    'ridge_weight': trial.suggest_float('ridge_weight', 0.0, 1.0),
                }
                
                # Import here to avoid circular imports
                from src.models.ensemble_model import EnsembleModel
                model = EnsembleModel()
                
                # Set model weights
                for name, weight in params.items():
                    model_name = name.replace('_weight', '')
                    if model_name in model.model_weights:
                        model.model_weights[model_name] = weight
                
                # Normalize weights to sum to 1
                total_weight = sum(model.model_weights.values())
                if total_weight > 0:
                    for name in model.model_weights:
                        model.model_weights[name] /= total_weight
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    if model_type in ["lstm", "two_stage"]:
                        # For deep learning models
                        history = model.train(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=params.get('epochs', 10),
                            batch_size=params.get('batch_size', 32),
                            verbose=0
                        )
                        y_pred = model.predict(X_val)
                    else:
                        # For traditional ML models
                        model.fit(X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train, y_train)
                        y_pred = model.predict(X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val)
                    
                    # Calculate metric
                    if self.config.metric == "rmse":
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    elif self.config.metric == "mae":
                        score = mean_absolute_error(y_val, y_pred)
                    elif self.config.metric == "r2":
                        score = -r2_score(y_val, y_pred)  # Negative because we're minimizing
                    else:
                        score = np.sqrt(mean_squared_error(y_val, y_pred))  # Default to RMSE
                    
                    cv_scores.append(score)
                    
                    # Report intermediate value for pruning
                    trial.report(np.mean(cv_scores), len(cv_scores) - 1)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                        
                except Exception as e:
                    logger.warning(f"Error in trial: {str(e)}")
                    return float('inf')  # Return a large error for failed trials
            
            return np.mean(cv_scores)
        
        return objective
    
    def optimize(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                custom_objective: Optional[Callable] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization for the specified model type"""
        
        start_time = time.time()
        logger.info(f"Starting hyperparameter optimization for {model_type} model")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=f"{self.config.study_name}_{model_type}",
            direction=self.config.direction,
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        # Get objective function
        objective = self._get_objective_function(model_type, X, y, custom_objective)
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        # Store results
        self.study = study
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        duration = time.time() - start_time
        logger.info(f"Optimization completed in {duration:.2f} seconds")
        logger.info(f"Best {self.config.metric}: {self.best_score:.6f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study,
            'duration': duration
        }
    
    def get_best_model(self, model_type: str) -> Any:
        """Create a model instance with the best parameters"""
        
        if not self.best_params:
            raise ValueError("No optimization has been performed yet")
        
        if model_type == "lstm":
            from src.models.lstm_model import LSTMModel
            return LSTMModel(
                lstm_units=self.best_params.get('lstm_units', 64),
                dropout_rate=self.best_params.get('dropout_rate', 0.2),
                recurrent_dropout=self.best_params.get('recurrent_dropout', 0.2),
                learning_rate=self.best_params.get('learning_rate', 0.001)
            )
        
        elif model_type == "two_stage":
            from src.models.two_stage_model import TwoStagePredictor
            return TwoStagePredictor(
                lstm_units=self.best_params.get('lstm_units', 64),
                attention_units=self.best_params.get('attention_units', 32),
                dropout_rate=self.best_params.get('dropout_rate', 0.2),
                learning_rate=self.best_params.get('learning_rate', 0.001)
            )
        
        elif model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=self.best_params.get('n_estimators', 100),
                max_depth=self.best_params.get('max_depth', 6),
                learning_rate=self.best_params.get('learning_rate', 0.1),
                subsample=self.best_params.get('subsample', 0.8),
                colsample_bytree=self.best_params.get('colsample_bytree', 0.8),
                min_child_weight=self.best_params.get('min_child_weight', 1),
                gamma=self.best_params.get('gamma', 0),
                reg_alpha=self.best_params.get('reg_alpha', 0),
                reg_lambda=self.best_params.get('reg_lambda', 1),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        elif model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=self.best_params.get('n_estimators', 100),
                max_depth=self.best_params.get('max_depth', 6),
                learning_rate=self.best_params.get('learning_rate', 0.1),
                num_leaves=self.best_params.get('num_leaves', 31),
                subsample=self.best_params.get('subsample', 0.8),
                colsample_bytree=self.best_params.get('colsample_bytree', 0.8),
                min_child_samples=self.best_params.get('min_child_samples', 20),
                reg_alpha=self.best_params.get('reg_alpha', 0),
                reg_lambda=self.best_params.get('reg_lambda', 1),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        elif model_type == "ensemble":
            from src.models.ensemble_model import EnsembleModel
            model = EnsembleModel()
            
            # Set model weights from best parameters
            for name, weight in self.best_params.items():
                model_name = name.replace('_weight', '')
                if model_name in model.model_weights:
                    model.model_weights[model_name] = weight
            
            # Normalize weights to sum to 1
            total_weight = sum(model.model_weights.values())
            if total_weight > 0:
                for name in model.model_weights:
                    model.model_weights[name] /= total_weight
                    
            return model
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No optimization has been performed yet")
            
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot optimization history
            plot_optimization_history(self.study).update_layout(
                title=f"Optimization History ({self.config.metric})",
                width=700, height=500
            )
            
            # Plot parameter importances
            plot_param_importances(self.study).update_layout(
                title="Parameter Importances",
                width=700, height=500
            )
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Optimization plots saved to {save_path}")
                
            plt.close()
            
        except ImportError:
            logger.warning("Plotting requires matplotlib and plotly. Install them to enable plotting.")
            
    def save_results(self, file_path: str):
        """Save optimization results to a file"""
        if self.study is None:
            raise ValueError("No optimization has been performed yet")
            
        import json
        
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'study_name': self.study.study_name,
            'direction': self.config.direction,
            'metric': self.config.metric,
            'n_trials': len(self.study.trials),
            'datetime': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Optimization results saved to {file_path}")
        
    @classmethod
    def load_results(cls, file_path: str) -> Dict[str, Any]:
        """Load optimization results from a file"""
        import json
        
        with open(file_path, 'r') as f:
            results = json.load(f)
            
        optimizer = cls()
        optimizer.best_params = results['best_params']
        optimizer.best_score = results['best_score']
        
        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(1000, 60, 5)  # 1000 samples, 60 time steps, 5 features
    y = np.random.rand(1000, 1)      # Target values
    
    # Configure optimizer
    config = OptimizationConfig(
        n_trials=10,  # Reduced for demonstration
        timeout=600,  # 10 minutes
        cv_folds=3,   # Reduced for demonstration
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization for XGBoost (faster than deep learning models for demonstration)
    results = optimizer.optimize("xgboost", X.reshape(X.shape[0], -1), y)
    
    # Get best model
    best_model = optimizer.get_best_model("xgboost")
    
    # Save results
    optimizer.save_results("xgboost_optimization_results.json")
