#!/usr/bin/env python3
"""
Quick test script for PLS regression implementations.

This script provides a simple way to quickly test the PLS implementations
with small datasets.
"""

import sys
import os
import numpy as np
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pls
import jax.numpy as jnp


def quick_test():
    """Run a quick test with synthetic data."""
    print("Quick PLS Test")
    print("=" * 30)
    
    # Create simple test data
    X = jnp.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0], 
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ])
    
    Y = jnp.array([
        [2.5, 4.0],
        [3.5, 5.5],
        [4.5, 7.0], 
        [5.5, 8.5],
        [6.5, 10.0]
    ])
    
    print(f"Test data: X{X.shape}, Y{Y.shape}")
    
    # Test optimized implementation
    print("\nTesting optimized implementation...")
    start = time.time()
    model = pls.pls_regression(X, Y, n_components=1)
    duration = time.time() - start
    
    Y_pred = pls.predict(X, model)
    mse = float(np.mean((Y - Y_pred)**2))
    
    print(f"✓ Completed in {duration:.3f}s")
    print(f"  MSE: {mse:.6f}")
    
    # Show predictions
    print("\nPredictions vs Actual:")
    for i in range(X.shape[0]):
        print(f"  {i+1}: {Y[i, :]} → {Y_pred[i, :]} (error: {np.abs(Y[i, :] - Y_pred[i, :]).max():.3f})")
    
    print("\n✓ Quick test completed!")


def test_real_data():
    """Test with real dataset."""
    print("\nReal Data Test")
    print("=" * 30)
    
    data_dir = os.path.join(project_root, 'pls', 'examples')
    
    try:
        X = pls.parse_matrix(os.path.join(data_dir, 'X_data.csv'))
        Y = pls.parse_matrix(os.path.join(data_dir, 'Y_data.csv'))
        
        print(f"Full data: X{X.shape}, Y{Y.shape}")
        
        # Test with subset
        X_sub = X[:8, :5]
        Y_sub = Y[:8, :3]
        
        print(f"Subset: X{X_sub.shape}, Y{Y_sub.shape}")
        
        start = time.time()
        model = pls.pls_regression(X_sub, Y_sub, n_components=1)
        duration = time.time() - start
        
        Y_pred = pls.predict(X_sub, model)
        mse = float(np.mean((Y_sub - Y_pred)**2))
        
        print(f"✓ Completed in {duration:.3f}s")
        print(f"  MSE: {mse:.6f}")
        
        print("\n✓ Real data test completed!")
        
    except Exception as e:
        print(f"✗ Failed to test real data: {e}")


if __name__ == '__main__':
    quick_test()
    test_real_data()
