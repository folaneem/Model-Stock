import sys
import pandas as pd
import numpy as np
import tensorflow as tf

def check_environment():
    print("\n=== Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check CUDA availability if using GPU
    print("\n=== GPU Check ===")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"GPU Device Name: {tf.test.gpu_device_name()}")
    
    # Check basic tensor operations
    print("\n=== TensorFlow Test ===")
    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication test successful!")
        print("Result:")
        print(c.numpy())
    except Exception as e:
        print(f"Error in TensorFlow operations: {e}")

if __name__ == "__main__":
    check_environment()
