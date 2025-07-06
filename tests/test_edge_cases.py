import numpy as np
import tensorflow as tf
from unittest import TestCase, mock
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.two_stage_model import TwoStagePredictor

class TestEdgeCases(TestCase):
    def setUp(self):
        """Set up test environment"""
        # Initialize predictor with small model size for testing
        self.predictor = TwoStagePredictor(
            lstm_units=16,
            attention_units=8,
            dropout_rate=0.1,
            learning_rate=0.001,
            prediction_days=10  # Changed to match model's expected sequence length
        )
        
        # Create minimal valid training data with 10 timesteps
        self.x_stock = np.random.rand(20, 10, 5)  # (samples, timesteps, features)
        self.x_features = np.random.rand(20, 10, 3)  # (samples, timesteps, features)
        self.y = np.random.randint(0, 2, (20, 1))
        
        # Train a minimal model
        self.predictor.train(
            x_train=(self.x_stock, self.x_features),
            y_train=self.y,
            epochs=1,  # Single epoch is enough for these tests
            batch_size=4,
            validation_split=0.2,
            verbose=0
        )

    def test_single_sample_prediction(self):
        """Test prediction with a single sample"""
        # Single sample with correct shape (1, timesteps, features) for stock data
        single_stock = np.random.rand(1, 10, 5)  # (1, 10, 5) - batch of 1 sample, 10 timesteps, 5 features
        single_feat = np.random.rand(1, 10, 3)   # (1, 10, 3) - batch of 1 sample, 10 timesteps, 3 features
        
        pred = self.predictor.predict(
            stock_data=single_stock,    # (1, 5, 1)
            additional_features=single_feat  # (1, 5, 2)
        )
        self.assertEqual(pred.shape, (1, 1))
        
        # Single sample without batch dimension (2D input)
        pred = self.predictor.predict(
            stock_data=single_stock[0],    # (5, 1)
            additional_features=single_feat[0]  # (5, 2)
        )
        self.assertEqual(pred.shape, (1, 1))

    def test_varying_sequence_lengths(self):
        """Test handling of varying sequence lengths"""
        # Create batch with different sequence lengths
        batch_stock = [
            np.random.rand(3, 1),  # 3 timesteps, 1 feature
            np.random.rand(5, 1),  # 5 timesteps, 1 feature
            np.random.rand(4, 1)   # 4 timesteps, 1 feature
        ]
        
        batch_features = [
            np.random.rand(3, 2),  # 2 features
            np.random.rand(5, 2),  # 2 features
            np.random.rand(4, 2)   # 2 features
        ]
        
        # Should handle sequences of different lengths
        # The implementation should pad/truncate to match the stock data length
        preds = self.predictor.predict(
            stock_data=batch_stock,  # List of (timesteps, 1)
            additional_features=batch_features  # List of (timesteps, 2)
        )
        # Should return one prediction per sample
        self.assertEqual(preds.shape, (3, 1))

    def test_malformed_inputs(self):
        """Test handling of malformed inputs"""
        # Test empty input
        with self.assertRaises(ValueError):
            self.predictor.predict(stock_data=np.array([]))
            
        # Test None input
        with self.assertRaises(ValueError):
            self.predictor.predict(stock_data=None)
            
        # Test wrong number of dimensions
        with self.assertRaises(ValueError):
            self.predictor.predict(stock_data=np.random.rand(5))  # 1D
            
        # Test NaN values
        x_nan = self.x_stock.copy()
        x_nan[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.predict(stock_data=x_nan)

    def test_extreme_values(self):
        """Test handling of extreme input values"""
        # Very large values
        x_large = np.ones((2, 10, 5)) * 1e6  # (batch, timesteps, features)
        f_large = np.ones((2, 10, 3)) * 1e6  # (batch, timesteps, features)
        
        # Very small values (including zeros)
        x_small = np.ones((2, 10, 5)) * 1e-6
        f_small = np.zeros((2, 10, 3))  # Test with zeros
        
        # Should handle without crashing
        try:
            pred_large = self.predictor.predict(
                stock_data=x_large,
                additional_features=f_large
            )
            pred_small = self.predictor.predict(
                stock_data=x_small,
                additional_features=f_small
            )
            self.assertEqual(pred_large.shape, (2, 1))
            self.assertEqual(pred_small.shape, (2, 1))
            
            # Check that predictions are within expected range (0-1 for classification)
            self.assertTrue(np.all((pred_large >= 0) & (pred_large <= 1)))
            self.assertTrue(np.all((pred_small >= 0) & (pred_small <= 1)))
            
        except Exception as e:
            self.fail(f"Prediction failed with extreme values: {str(e)}")

    def test_missing_data_handling(self):
        """Test handling of missing data scenarios"""
        # Ensure stock_data has shape (samples, timesteps, features)
        x_stock = self.x_stock  # (20, 10, 5)
        
        # Test with None for additional features
        pred_none = self.predictor.predict(
            stock_data=x_stock[:2],  # (2, 10, 5)
            additional_features=None  # Will be created as zeros with shape (2, 10, 3)
        )
        self.assertEqual(pred_none.shape, (2, 1))
        
        # Test with empty additional features (0 features)
        pred_empty = self.predictor.predict(
            stock_data=x_stock[:2],
            additional_features=np.zeros((2, 10, 0))  # (2, 10, 0) - no features
        )
        self.assertEqual(pred_empty.shape, (2, 1))
        
        # Test with NaN values in stock data
        x_nan = x_stock.copy()
        x_nan[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.predict(
                stock_data=x_nan[:2],
                additional_features=self.x_features[:2]
            )

if __name__ == '__main__':
    import unittest
    unittest.main()
