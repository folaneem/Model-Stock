import numpy as np
import tensorflow as tf
from unittest import TestCase, mock
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.two_stage_model import TwoStagePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPredictionAccuracy(TestCase):
    def setUp(self):
        """Set up test environment"""
        # Initialize predictor with small model size for testing
        self.predictor = TwoStagePredictor(
            lstm_units=16,
            attention_units=8,
            dropout_rate=0.1,
            learning_rate=0.001,
            prediction_days=5
        )
        
        # Create test data
        self.x_stock = np.random.rand(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
        self.x_features = np.random.rand(100, 10, 3)  # 100 samples, 10 timesteps, 3 features
        self.y = np.random.randint(0, 2, (100, 1))  # Binary classification
        
        # Train a simple model for testing predictions
        self.predictor.train(
            x_train=(self.x_stock, self.x_features),
            y_train=self.y,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

    def test_prediction_shape(self):
        """Test that predictions have the correct shape"""
        # Test batch prediction
        # Ensure stock_data has shape (samples, timesteps, 1)
        stock_data = self.x_stock[..., -1:]  # Use only the last feature (close price)
        predictions = self.predictor.predict(
            stock_data=stock_data[:5],  # First 5 samples
            additional_features=self.x_features[:5]  # (5, 10, 3)
        )
        
        # Should return one prediction per sample
        self.assertEqual(predictions.shape, (5, 1))
        
        # Test single sample prediction (2D input)
        single_pred = self.predictor.predict(
            stock_data=stock_data[0],  # (10, 1)
            additional_features=self.x_features[0]  # (10, 3)
        )
        self.assertEqual(single_pred.shape, (1, 1))

    def test_sequence_length_mismatch(self):
        """Test handling of sequence length mismatches"""
        # Create features with different sequence length
        x_stock = self.x_stock[..., -1:]  # (100, 10, 1)
        x_features_short = np.random.rand(5, 8, 3)  # 8 timesteps instead of 10
        
        # Should handle mismatch by padding/truncating
        predictions = self.predictor.predict(
            stock_data=x_stock[:5],  # (5, 10, 1)
            additional_features=x_features_short  # (5, 8, 3) - will be padded to (5, 10, 3)
        )
        self.assertEqual(predictions.shape, (5, 1))

    def test_missing_features(self):
        """Test prediction with missing additional features"""
        # Should handle None for additional_features
        predictions = self.predictor.predict(
            stock_data=self.x_stock[:5],
            additional_features=None
        )
        self.assertEqual(predictions.shape, (5, 1))

    def test_feature_dimension_mismatch(self):
        """Test handling of feature dimension mismatches"""
        # Features with different number of features
        x_stock = self.x_stock[..., -1:]  # (100, 10, 1)
        x_features_wrong_dims = np.random.rand(5, 10, 5)  # 5 features instead of 3
        
        # Should handle different feature dimensions
        predictions = self.predictor.predict(
            stock_data=x_stock[:5],  # (5, 10, 1)
            additional_features=x_features_wrong_dims  # (5, 10, 5)
        )
        self.assertEqual(predictions.shape, (5, 1))

    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input"""
        # Get predictions for the same input twice
        pred1 = self.predictor.predict(
            stock_data=self.x_stock[:5],
            additional_features=self.x_features[:5]
        )
        pred2 = self.predictor.predict(
            stock_data=self.x_stock[:5],
            additional_features=self.x_features[:5]
        )
        
        # Predictions should be very similar (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=5,
            err_msg="Predictions for same input should be consistent"
        )

if __name__ == '__main__':
    import unittest
    unittest.main()
