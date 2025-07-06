import tensorflow as tf
import numpy as np
from src.models.two_stage_model import TwoStagePredictor

# Initialize the model
predictor = TwoStagePredictor()
print('TwoStagePredictor initialized successfully')

# Create some random test data
x_stock = np.random.random((100, 30, 5))
x_additional = np.random.random((100, 10))
y = np.random.random((100, 1))

# Try to train the model
try:
    print('Starting model training...')
    metrics = predictor.train_model(
        (x_stock, x_additional),
        y,
        epochs=1,
        batch_size=16,
        verbose=1
    )
    print('Training completed successfully!')
    print(f'Training metrics: {metrics}')
except Exception as e:
    print(f'Error during training: {str(e)}')
