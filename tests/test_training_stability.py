import os
import sys
import numpy as np
import tensorflow as tf
from unittest import TestCase, mock
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.two_stage_model import TwoStagePredictor, TrainingMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTrainingStability(TestCase):
    def setUp(self):
        """Set up test environment"""
        # Initialize predictor with small model size for testing
        self.predictor = TwoStagePredictor(
            lstm_units=16,
            attention_units=8,
            dropout_rate=0.1,
            learning_rate=0.001,
            prediction_days=5,
            clipnorm=1.0,  # Add gradient clipping
            clipvalue=0.5
        )
        
        # Create test data
        np.random.seed(42)  # For reproducibility
        
        # Stock data: (samples, timesteps, features)
        # Use only the last feature (close price) as per implementation
        self.x_stock = np.random.rand(100, 10, 1)  # 100 samples, 10 timesteps, 1 feature (close price)
        
        # Additional features: (samples, timesteps, n_features)
        self.x_features = np.random.rand(100, 10, 3)  # 100 samples, 10 timesteps, 3 features
        
        # Target: binary classification (0 or 1)
        self.y = np.random.randint(0, 2, (100, 1))  # Binary classification

    def test_training_monitor_initialization(self):
        """Test that TrainingMonitor initializes correctly"""
        # Create a mock logger
        mock_logger = mock.MagicMock()
        
        # Create an instance of the callback
        monitor = TrainingMonitor(logger=mock_logger, patience=10)
        
        # Test initialization
        self.assertIsNone(monitor.best_weights)
        self.assertEqual(monitor.best_val_loss, float('inf'))
        self.assertEqual(monitor.wait, 0)
        self.assertIsNone(monitor.initial_weights)
        
        # Test model property before setting
        with self.assertRaises(AttributeError):
            _ = monitor.model
            
        # Set up mock model with more complex weights
        mock_model = mock.MagicMock()
        mock_weights = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2x2 array
            np.array([0.5, 1.5])  # 1D array
        ]
        mock_model.get_weights.return_value = mock_weights
        
        # Set the model
        monitor.model = mock_model
        
        # Now test on_train_begin
        monitor.on_train_begin()
        
        # Verify weights are set correctly
        self.assertIsNotNone(monitor.initial_weights)
        self.assertIsNotNone(monitor.best_weights)
        self.assertEqual(monitor.best_val_loss, float('inf'))
        self.assertEqual(monitor.wait, 0)
        
        # Verify the weights were copied, not referenced
        self.assertIsNot(monitor.best_weights[0], mock_weights[0])
        np.testing.assert_array_equal(monitor.best_weights[0], mock_weights[0])
        
        # Verify the weights have the expected values
        np.testing.assert_array_equal(monitor.best_weights[0], np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(monitor.best_weights[1], np.array([0.5, 1.5]))
        
        # Verify logger was called
        mock_logger.info.assert_called()

    def test_training_process(self):
        """Test that training runs without errors"""
        logger.info("Testing training process...")
        
        try:
            # Create a fresh predictor for this test
            predictor = TwoStagePredictor(
                lstm_units=16,
                attention_units=8,
                dropout_rate=0.1,
                learning_rate=0.001,
                prediction_days=5
            )
            
            # Train the model with a small number of epochs for testing 
            history = predictor.train(
                (self.x_stock, self.x_features),
                self.y,
                epochs=3,
                batch_size=16,
                validation_split=0.2,  # Use 20% for validation
                verbose=1
            )
            
            # Check that history contains expected keys
            self.assertIn('loss', history)
            
            # Check that loss values are valid
            self.assertTrue(all(not np.isnan(loss) for loss in history['loss']))
            
            # Check that we have model weights
            self.assertIsNotNone(predictor.model)
            weights = predictor.model.get_weights()
            self.assertGreater(len(weights), 0)
            
            # Check that we have some non-zero weights
            has_nonzero = any(np.any(w != 0) for w in weights if w.size > 0)
            self.assertTrue(has_nonzero, "Model weights were not updated during training")
            
            # Log training completion
            logger.info(f"Training completed with final loss: {history['loss'][-1]:.4f}")
            if 'val_loss' in history:
                logger.info(f"Validation loss: {history['val_loss'][-1]:.4f}")
            else:
                logger.warning("No validation loss recorded in history")
            
            logger.info("Training process test completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}", exc_info=True)
            self.fail(f"Training failed with error: {str(e)}")

    def test_early_stopping(self):
        """Test that early stopping works as expected"""
        logger.info("Testing early stopping...")
        
        # Create a mock model
        mock_model = mock.MagicMock()
        mock_model.stop_training = False
        mock_weights = [np.random.rand(10, 10)]
        mock_model.get_weights.return_value = mock_weights
        
        # Setup optimizer with learning rate
        mock_optimizer = mock.MagicMock()
        mock_optimizer.learning_rate = 0.001
        mock_model.optimizer = mock_optimizer
        
        # Create a monitor with low patience for testing
        mock_logger = mock.MagicMock()
        monitor = TrainingMonitor(logger=mock_logger, patience=2)
        monitor.model = mock_model
        
        # Simulate training with no improvement
        monitor.on_train_begin()
        
        # First epoch - improvement
        logs = {'loss': 0.4, 'val_loss': 0.4}
        monitor.on_epoch_end(0, logs)
        self.assertFalse(mock_model.stop_training)
        
        # Second epoch - no improvement
        logs = {'loss': 0.5, 'val_loss': 0.5}
        monitor.on_epoch_end(1, logs)
        self.assertFalse(mock_model.stop_training)
        
        # Third epoch - still no improvement (patience=2)
        logs = {'loss': 0.6, 'val_loss': 0.6}
        monitor.on_epoch_end(2, logs)
        self.assertTrue(mock_model.stop_training)
        
        # Verify model weights were restored
        mock_model.set_weights.assert_called_once()
        
        # Verify logs were called
        self.assertTrue(mock_logger.info.called)
        logger.info("Early stopping test completed successfully")

    def test_nan_handling(self):
        """Test that NaNs in input are handled gracefully"""
        logger.info("Testing NaN handling...")
        
        # Create a new predictor
        predictor = TwoStagePredictor(
            lstm_units=16,
            attention_units=8,
            dropout_rate=0.1,
            learning_rate=0.001,
            prediction_days=5
        )
        
        # Create test data with NaN
        x_nan_stock = self.x_stock.copy()
        x_nan_stock[0, 0, 0] = np.nan
        
        # Call the method - it should replace NaNs with zeros
        x_stock_processed, x_features_processed, y_processed = predictor._validate_and_reshape_inputs(
            x_nan_stock,
            self.x_features,
            self.y
        )
        
        # Verify NaN was replaced with zero
        self.assertFalse(np.isnan(x_stock_processed).any(), "NaN values should be replaced with zeros")
        self.assertEqual(x_stock_processed[0, 0, 0], 0, "NaN should be replaced with zero")
        
        logger.info("NaN handling tests completed")

    def test_model_save_load(self):
        """Test that model can be saved and loaded correctly"""
        import tempfile
        import os
        
        # Skip this test if model hasn't been trained yet
        if not hasattr(self.predictor, 'model') or self.predictor.model is None:
            self.skipTest("Model not trained yet")
            
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.h5')
            
            # Save the model
            self.predictor.model.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model
            from tensorflow.keras.models import load_model
            loaded_model = load_model(model_path, compile=False)
            
            # Verify the loaded model has the same architecture
            self.assertEqual(len(self.predictor.model.layers), len(loaded_model.layers))
            
            # Test prediction with loaded model
            test_input = [self.x_stock[:1], self.x_features[:1]]
            
            # Mock the predict method to avoid actual prediction
            with mock.patch.object(self.predictor.model, 'predict') as mock_predict:
                mock_predict.return_value = np.array([[0.5]])  # Mock prediction
                orig_pred = self.predictor.model.predict(test_input)
                loaded_pred = loaded_model.predict(test_input)
                
                # Verify the prediction shape
                self.assertEqual(orig_pred.shape, (1, 1))
                self.assertEqual(loaded_pred.shape, (1, 1))

if __name__ == '__main__':
    import unittest
    unittest.main()
