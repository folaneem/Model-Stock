import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from src.utils.data_processor import DataProcessor
from src.models.lstm_model import LSTMModel
from src.models.two_stage_model import TwoStagePredictor
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.yfinance_utils import fetch_daily_yfinance_data

def split_data(data, test_size=0.2, validation_split=0.0):
    """
    Split time series data into training, validation, and test sets while maintaining temporal order.
    
    Args:
        data: Input data to split
        test_size: Fraction of data to use for testing
        validation_split: Fraction of training data to use for validation
        
    Returns:
        If validation_split > 0: (train_data, val_data, test_data)
        Else: (train_data, test_data)
    """
    # First split into train+val and test
    split_idx = int(len(data) * (1 - test_size))
    train_val_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # If no validation split needed, return train and test
    if validation_split <= 0 or validation_split >= 1:
        return train_val_data, test_data
        
    # Split train into train and validation
    val_split_idx = int(len(train_val_data) * (1 - validation_split))
    train_data = train_val_data[:val_split_idx]
    val_data = train_val_data[val_split_idx:]
    
    return train_data, val_data, test_data

def plot_predictions(model, X_test_stock, X_test_add, y_test, scaler=None):
    # Make predictions
    predictions = model.predict(X_test_stock, X_test_add)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nDetailed Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Inverse transform if scaler is provided
    if scaler:
        try:
            # Reshape for inverse transform if needed
            y_test_reshaped = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
            predictions_reshaped = predictions.reshape(-1, 1) if len(predictions.shape) == 1 else predictions
            
            # Create dummy arrays with the same shape as the original data
            dummy_array_test = np.zeros((len(y_test_reshaped), scaler.n_features_in_))
            dummy_array_pred = np.zeros((len(predictions_reshaped), scaler.n_features_in_))
            
            # Put the values in the first column (assuming close price was the first column)
            dummy_array_test[:, 0] = y_test_reshaped.flatten()
            dummy_array_pred[:, 0] = predictions_reshaped.flatten()
            
            # Inverse transform
            y_test_inv = scaler.inverse_transform(dummy_array_test)[:, 0]
            predictions_inv = scaler.inverse_transform(dummy_array_pred)[:, 0]
            
            # Calculate metrics on original scale
            mse_orig = mean_squared_error(y_test_inv, predictions_inv)
            mae_orig = mean_absolute_error(y_test_inv, predictions_inv)
            rmse_orig = np.sqrt(mse_orig)
            
            print(f"\nOriginal Scale Metrics:")
            print(f"MSE (Original Scale): {mse_orig:.4f}")
            print(f"MAE (Original Scale): {mae_orig:.4f}")
            print(f"RMSE (Original Scale): {rmse_orig:.4f}")
            
            # Plot predictions vs actual in original scale
            plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.plot(y_test_inv, label='Actual', color='blue')
            plt.plot(predictions_inv, label='Predicted', color='red', linestyle='--')
            plt.title('Stock Price Prediction (Original Scale)')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot scaled values
            plt.subplot(1, 2, 2)
            plt.plot(y_test, label='Actual (Scaled)', color='blue')
            plt.plot(predictions, label='Predicted (Scaled)', color='red', linestyle='--')
            plt.title('Stock Price Prediction (Scaled)')
            plt.xlabel('Time Steps')
            plt.ylabel('Scaled Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error during inverse scaling: {e}")
            # Fall back to plotting scaled values only
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual (Scaled)', color='blue')
            plt.plot(predictions, label='Predicted (Scaled)', color='red', linestyle='--')
            plt.title('Stock Price Prediction (Scaled)')
            plt.xlabel('Time Steps')
            plt.ylabel('Scaled Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
    else:
        # Plot scaled values only
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual (Scaled)', color='blue')
        plt.plot(predictions, label='Predicted (Scaled)', color='red', linestyle='--')
        plt.title('Stock Price Prediction (Scaled)')
        plt.xlabel('Time Steps')
        plt.ylabel('Scaled Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.figure(figsize=(12, 6))
    errors = y_test.flatten() - predictions.flatten()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save all figures
    plt.figure(1).savefig('predictions_comparison.png')
    plt.figure(2).savefig('error_distribution.png')
    
    plt.show()

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Get stock ticker and model type from user
    ticker = input("Enter stock ticker (e.g., AAPL): ")
    model_choice = input("Select model (LSTM or TwoStage): ").strip().lower()

    # Load data
    print("\nLoading data...")
    # Calculate start date as 4 years ago from today
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)  # Approximately 4 years
    
    data = fetch_daily_yfinance_data(
        ticker=ticker,
        start=start_date,
        end=end_date
    )
    if data.empty:
        print("Error: No data found for the given ticker.")
        return

    # Model selection and data processing
    print("\nProcessing data...")
    try:
        if model_choice == 'twostage':
            def flatten_columns(df):
                def flatten(col):
                    if isinstance(col, tuple):
                        # Use first element if it's a standard OHLCV name
                        if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            return col[0]
                        else:
                            return '_'.join(str(c) for c in col if c)
                    return col
                df.columns = [flatten(col) for col in df.columns]
                return df
            model = TwoStagePredictor()
            # Flatten columns if needed
            data_flat = flatten_columns(data)
            X_stock, X_additional, y_full = model.preprocess_data(data_flat)
            # Split data
            print("\nSplitting data...")
            X_train_stock, X_test_stock = split_data(X_stock)
            X_train_add, X_test_add = split_data(X_additional)
            y_train_full, y_test_full = split_data(y_full)
            # Defensive checks before training
            print(f"X_train_stock shape: {X_train_stock.shape}")
            print(f"X_train_add shape: {X_train_add.shape}")
            print(f"y_train_full shape: {y_train_full.shape}")
            if (X_train_stock.shape[0] == 0 or X_train_add.shape[0] == 0 or y_train_full.shape[0] == 0):
                raise ValueError("No training samples available after preprocessing and splitting. Check your data length and sequence/window size.")
            import numpy as np
            print("NaNs in X_train_stock:", np.isnan(X_train_stock).sum())
            print("Infs in X_train_stock:", np.isinf(X_train_stock).sum())
            print("NaNs in X_train_add:", np.isnan(X_train_add).sum())
            print("Infs in X_train_add:", np.isinf(X_train_add).sum())
            print("NaNs in y_train_full:", np.isnan(y_train_full).sum())
            print("Infs in y_train_full:", np.isinf(y_train_full).sum())
            if (np.isnan(X_train_stock).any() or np.isinf(X_train_stock).any() or
                np.isnan(X_train_add).any() or np.isinf(X_train_add).any() or
                np.isnan(y_train_full).any() or np.isinf(y_train_full).any()):
                raise ValueError("Training data contains NaN or infinite values. Please check preprocessing.")
            print("Batch size for training:", 32)
            print("Training samples:", X_train_stock.shape[0])
            print("Validation split:", 0.2)
            # Reshape y_train_full to 1D if needed
            if y_train_full.ndim == 2 and y_train_full.shape[1] == 1:
                y_train_full = y_train_full.ravel()
                print("Reshaped y_train_full to:", y_train_full.shape)
            # Train, evaluate, predict
            print("\nTraining model...")
            # Split into train and validation sets
            val_split = 0.2  # 20% of training data for validation
            
            # Calculate split index
            split_idx = int(len(X_train_stock) * (1 - val_split))
            
            # Split stock data
            X_train_stock_final = X_train_stock[:split_idx]
            X_val_stock = X_train_stock[split_idx:]
            
            # Split additional features
            X_train_add_final = X_train_add[:split_idx]
            X_val_add = X_train_add[split_idx:]
            
            # Split targets
            y_train_final = y_train_full[:split_idx]
            y_val = y_train_full[split_idx:]
            
            print(f"\nData split:")
            print(f"- Training samples: {len(X_train_stock_final)}")
            print(f"- Validation samples: {len(X_val_stock)}")
            print(f"- Test samples: {len(X_test_stock)}")
            
            # Train the model with explicit validation data
            model.train(
                x_train=(X_train_stock_final, X_train_add_final),
                y_train=y_train_final,
                validation_dataset=((X_val_stock, X_val_add), y_val),
                epochs=100,  # Increased epochs for better training
                batch_size=32,
                verbose=1
            )
            model.evaluate((X_test_stock, X_test_add), y_test_full)
            print("X_test_stock shape:", X_test_stock.shape)
            print("X_test_add shape:", X_test_add.shape)
            assert X_test_stock.shape[1:] == (60, 1), f"X_test_stock shape mismatch: {X_test_stock.shape}"
            assert X_test_add.shape[1:] == (60, 6), f"X_test_add shape mismatch: {X_test_add.shape}"
            plot_predictions(model, X_test_stock, X_test_add, y_test_full, model.target_scaler)
            y_test_plot = y_test_full
        else:
            processor = DataProcessor()
            processed_data = processor.preprocess(data)
            if not processed_data or 'features' not in processed_data or 'returns' not in processed_data:
                print("Error: Data preprocessing failed or returned invalid format")
                return
            X = processed_data['features']
            y = processed_data['returns']
            # Split data
            print("\nSplitting data...")
            X_train, X_test = split_data(X)
            y_train, y_test = split_data(y)
            # Train, evaluate, predict
            print("\nTraining model...")
            model = LSTMModel(sequence_length=60, features=X_train.shape[2] if len(X_train.shape) > 2 else 1)
            if not hasattr(model, 'model') or model.model is None:
                model.build_model()
            if not hasattr(model, 'model') or model.model is None:
                raise RuntimeError("Failed to build the model")
            training_metrics = model.train(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            predictions = model.predict(X_test[-1:], 30)
            y_test_plot = y_test[-30:]
    except Exception as e:
        print(f"\nError during model training or preprocessing: {str(e)}")
        print("Please check the following:")
        print("1. Input data shapes - X_train: ", X_train.shape if 'X_train' in locals() else 'Not available')
        print("2. Output data shapes - y_train: ", y_train.shape if 'y_train' in locals() else 'Not available')
        print("3. Make sure all required packages are installed")
        print("4. Check for any NaN or infinite values in your data")
        return

    # Print metrics
    print("\nTraining Metrics:")
    if training_metrics and isinstance(training_metrics, dict):
        for k, v in training_metrics.items():
            print(f"{k}: {v}")
    print("\nTest Metrics:")
    if test_metrics and isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            print(f"{k}: {v}")

    # Plot training history if available
    if hasattr(model, 'plot_training_history'):
        print("\nPlotting training history...")
        model.plot_training_history()

    # Plot predictions vs actual
    print("\nPlotting predictions...")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_plot, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'plots/{ticker}_predictions.png')
    plt.close()

if __name__ == "__main__":
    main()
