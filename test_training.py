import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from src.models.two_stage_model import TwoStagePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_test.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_sample_data(sequence_length=60, num_samples=1000, num_features=5):
    """Generate sample stock price and feature data for testing."""
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_samples + sequence_length)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random walk for stock prices
    stock_prices = np.cumsum(np.random.randn(len(dates)) * 0.01) + 100
    
    # Create DataFrame with technical indicators
    df = pd.DataFrame({
        'Date': dates,
        'Open': stock_prices,
        'High': stock_prices * (1 + np.random.rand(len(dates)) * 0.02),
        'Low': stock_prices * (1 - np.random.rand(len(dates)) * 0.02),
        'Close': stock_prices * (1 + (np.random.rand(len(dates)) * 0.04 - 0.02)),
        'Volume': np.random.lognormal(10, 1, len(dates)),
        'RSI': np.random.uniform(30, 70, len(dates)),
        'MACD': np.random.uniform(-2, 2, len(dates)),
        'BB_Upper': stock_prices * 1.02,
        'BB_Lower': stock_prices * 0.98
    })
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    return df

def test_training():
    """Test the model training pipeline with sample data."""
    try:
        logger.info("Starting model training test...")
        
        # Generate sample data
        logger.info("Generating sample data...")
        stock_data = generate_sample_data(sequence_length=60, num_samples=1000)
        
        # Initialize model
        logger.info("Initializing model...")
        model = TwoStagePredictor(
            lstm_units=32,  # Reduced for faster testing
            attention_units=16,
            dropout_rate=0.2,
            learning_rate=0.001,
            prediction_days=60
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_stock, X_additional, y = model.preprocess_data(stock_data)
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(
            x_train=(X_stock, X_additional),
            y_train=y,
            epochs=5,  # Reduced for testing
            batch_size=32,
            validation_split=0.2
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("=== Starting Model Training Test ===")
    success = test_training()
    if success:
        logger.info("=== Test Completed Successfully ===")
    else:
        logger.error("=== Test Failed ===")
    logger.info("====================================")
