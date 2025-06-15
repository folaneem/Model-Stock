import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from src.utils.data_processor import DataProcessor
from src.models.lstm_model import LSTMModel
import matplotlib.pyplot as plt
import os

def split_data(data, test_size=0.2):
    """
    Split time series data into training and test sets
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Get stock ticker from user
    ticker = input("Enter stock ticker (e.g., AAPL): ")

    # Load data
    print("\nLoading data...")
    data = yf.download(ticker, period="4y")
    if data.empty:
        print("Error: No data found for the given ticker.")
        return

    # Process data
    print("\nProcessing data...")
    processor = DataProcessor()
    X, y = processor.preprocess(data)

    # Split data
    print("\nSplitting data...")
    X_train, X_test = split_data(X)
    y_train, y_test = split_data(y)

    # Train model
    print("\nTraining model...")
    model = LSTMModel()
    training_metrics = model.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    test_metrics = model.evaluate(X_test, y_test)

    # Print metrics
    print("\nTraining Metrics:")
    print(f"Average Validation Loss: {training_metrics['avg_validation_loss']:.4f}")
    print(f"Average Validation MAE: {training_metrics['avg_validation_mae']:.4f}")
    
    print("\nTest Metrics:")
    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Test MAE: {test_metrics['test_mae']:.4f}")
    print(f"Test RMSE: {test_metrics['test_rmse']:.4f}")
    print(f"Test MAPE: {test_metrics['test_mape']:.2f}%")

    # Plot training history
    print("\nPlotting training history...")
    model.plot_training_history()

    # Make predictions for the next 30 days
    print("\nMaking predictions for the next 30 days...")
    predictions = model.predict(X_test[-1:], 30)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[-30:], label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'plots/{ticker}_predictions.png')
    plt.close()

if __name__ == "__main__":
    main()
