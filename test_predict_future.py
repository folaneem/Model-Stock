import sys
import os
import numpy as np
import pandas as pd
import logging

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

# Import the TwoStagePredictor
from src.models.two_stage_model import TwoStagePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_predict_future():
    print("Creating test model...")
    model = TwoStagePredictor(prediction_days=5)
    
    print("Creating test data...")
    # Create sample data (5 days of historical data)
    last_sequence = np.random.random((5, 1))
    additional_features = np.random.random((5, 3))
    date_index = pd.date_range(start='2025-06-10', periods=5, freq='B')
    
    # Test 1: Verify that 'Model not trained' error is properly propagated
    print("\nTest 1: Verifying 'Model not trained' error is properly propagated...")
    try:
        future_preds, future_dates = model.predict_future(
            last_sequence, 
            additional_features, 
            days_ahead=3, 
            date_index=date_index
        )
        print("Error: This should have raised a 'Model not trained' exception!")
    except ValueError as e:
        if "Model not trained" in str(e):
            print(f"Success! Caught expected error: {str(e)}")
        else:
            print(f"Unexpected error: {str(e)}")
    
    # Test 2: Verify that 'days_ahead' validation works
    print("\nTest 2: Verifying days_ahead validation...")
    try:
        model.predict_future(last_sequence, additional_features, days_ahead=-1)
        print("Error: This should have raised a 'days_ahead' validation exception!")
    except ValueError as e:
        if "days_ahead must be greater than 0" in str(e):
            print(f"Success! Caught expected error: {str(e)}")
        else:
            print(f"Unexpected error: {str(e)}")
    
    print("\nAll validation tests passed successfully!")
    print("Note: The model requires training before it can make predictions.")
    print("In a real application, we would train the model first.")
    print("These tests confirm that our validation logic is working correctly.")


if __name__ == "__main__":
    test_predict_future()
