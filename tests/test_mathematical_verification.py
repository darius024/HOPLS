#!/usr/bin/env python3
"""
Mathematical Verification Test for Standard PLS Implementation

This test verifies that our PLS implementation correctly follows the 
standard mathematical formulation described in the literature.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pls
import jax.numpy as jnp


def test_mathematical_formulation():
    """
    Test that our implementation follows the standard PLS mathematical formulation:
    
    X = T P^T + E = Σ(r=1 to R) t_r * p_r^T + E     (Equation 2.28)
    Y = T D C^T + F = Σ(r=1 to R) d_r * t_r * c_r^T + F     (Equation 2.29)
    """
    print("Mathematical Formulation Verification Test")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    X = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0],
        [6.0, 7.0, 8.0, 9.0]
    ])
    
    Y = jnp.array([
        [10.0, 15.0],
        [12.0, 18.0],
        [14.0, 21.0],
        [16.0, 24.0],
        [18.0, 27.0],
        [20.0, 30.0]
    ])
    
    print(f"Test data: X{X.shape}, Y{Y.shape}")
    
    # Fit PLS model with 1 component (safer for small data)
    n_components = 1
    model = pls.pls_regression(X, Y, n_components=n_components)
    
    print(f"\n✓ PLS model fitted with {n_components} components")
    print(f"Model matrices:")
    print(f"  W (weights): {model['W'].shape}")
    print(f"  P (X loadings): {model['P'].shape}")
    print(f"  C (Y loadings): {model['C'].shape}")
    print(f"  T (scores): {model['T'].shape}")
    print(f"  D (scaling): {model['D'].shape}")
    
    # Center the original data as done in PLS
    X_centered = X - model['X_mean']
    Y_centered = Y - model['Y_mean']
    
    print(f"\nVerifying mathematical relationships:")
    
    # Test 1: Verify X decomposition: X = T P^T + E
    print(f"\n1. Testing X decomposition: X = T P^T + E")
    
    # Reconstruct X using X_reconstructed = T * P^T
    X_reconstructed = jnp.dot(model['T'], model['P'].T)
    X_residual = X_centered - X_reconstructed
    
    reconstruction_error_X = float(jnp.mean(X_residual**2))
    print(f"   X reconstruction MSE: {reconstruction_error_X:.6f}")
    
    if reconstruction_error_X < 1e-10:
        print(f"   ✓ X decomposition verified (perfect reconstruction)")
    elif reconstruction_error_X < 1e-3:
        print(f"   ✓ X decomposition verified (good reconstruction)")
    else:
        print(f"   ⚠ X reconstruction has significant error")
    
    # Test 2: Verify Y decomposition: Y = T D C^T + F
    print(f"\n2. Testing Y decomposition: Y = T D C^T + F")
    
    # Reconstruct Y using Y_reconstructed = T * diag(D) * C^T
    T_scaled = model['T'] * model['D']  # T * D (broadcasting)
    Y_reconstructed = jnp.dot(T_scaled, model['C'].T)
    Y_residual = Y_centered - Y_reconstructed
    
    reconstruction_error_Y = float(jnp.mean(Y_residual**2))
    print(f"   Y reconstruction MSE: {reconstruction_error_Y:.6f}")
    
    if reconstruction_error_Y < 1e-10:
        print(f"   ✓ Y decomposition verified (perfect reconstruction)")
    elif reconstruction_error_Y < 1e-3:
        print(f"   ✓ Y decomposition verified (good reconstruction)")
    else:
        print(f"   ⚠ Y reconstruction has significant error")
    
    # Test 3: Verify prediction formula: Y' = X' W D C^T + Y_mean
    print(f"\n3. Testing prediction formula: Y' = X' W D C^T + Y_mean")
    
    Y_pred = pls.predict(X, model)
    prediction_error = float(jnp.mean((Y - Y_pred)**2))
    print(f"   Prediction MSE: {prediction_error:.6f}")
    
    # Manual prediction to verify formula
    X_cent = X - model['X_mean']
    T_new = jnp.dot(X_cent, model['W'])
    T_scaled_new = T_new * model['D']
    Y_manual = jnp.dot(T_scaled_new, model['C'].T) + model['Y_mean']
    
    manual_vs_api = float(jnp.mean((Y_pred - Y_manual)**2))
    print(f"   API vs Manual prediction difference: {manual_vs_api:.10f}")
    
    if manual_vs_api < 1e-10:
        print(f"   ✓ Prediction formula verified")
    else:
        print(f"   ⚠ Prediction API differs from manual calculation")
    
    # Test 4: Check orthogonality properties
    print(f"\n4. Testing orthogonality properties")
    
    # Scores should be orthogonal: T^T * T should be diagonal-ish for standardized scores
    T_inner = jnp.dot(model['T'].T, model['T'])
    off_diagonal_sum = float(jnp.sum(jnp.abs(T_inner)) - jnp.sum(jnp.abs(jnp.diag(T_inner))))
    
    print(f"   Off-diagonal sum in T^T * T: {off_diagonal_sum:.6f}")
    
    if off_diagonal_sum < 1e-6:
        print(f"   ✓ Scores are approximately orthogonal")
    else:
        print(f"   ⚠ Scores may not be perfectly orthogonal")
    
    # Test 5: Check normalization constraints
    print(f"\n5. Testing normalization constraints")
    
    # Weight vectors should be normalized (||w|| = 1)
    w_norms = [float(jnp.linalg.norm(model['W'][:, i])) for i in range(n_components)]
    print(f"   Weight vector norms: {w_norms}")
    
    all_normalized = all(abs(norm - 1.0) < 1e-6 for norm in w_norms)
    if all_normalized:
        print(f"   ✓ Weight vectors are normalized")
    else:
        print(f"   ⚠ Some weight vectors are not properly normalized")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("MATHEMATICAL VERIFICATION SUMMARY")
    print("=" * 50)
    
    overall_success = (
        reconstruction_error_X < 1e-3 and
        reconstruction_error_Y < 1e-3 and
        prediction_error < 1e-3 and
        manual_vs_api < 1e-10 and
        all_normalized
    )
    
    if overall_success:
        print("✓ ALL MATHEMATICAL RELATIONSHIPS VERIFIED")
        print("✓ Implementation follows standard PLS formulation correctly")
    else:
        print("⚠ SOME MATHEMATICAL RELATIONSHIPS NEED ATTENTION")
    
    print(f"\nDetailed Results:")
    print(f"  X reconstruction error: {reconstruction_error_X:.8f}")
    print(f"  Y reconstruction error: {reconstruction_error_Y:.8f}")
    print(f"  Prediction error: {prediction_error:.8f}")
    print(f"  Formula consistency: {manual_vs_api:.10f}")
    print(f"  Weight normalization: {'✓' if all_normalized else '⚠'}")
    
    return overall_success


def test_component_interpretation():
    """Test the interpretation of individual PLS components."""
    print(f"\n" + "=" * 50)
    print("COMPONENT INTERPRETATION TEST")
    print("=" * 50)
    
    # Use the example data
    data_dir = os.path.join(project_root, 'pls', 'examples')
    X = pls.parse_matrix(os.path.join(data_dir, 'X_data.csv'))
    Y = pls.parse_matrix(os.path.join(data_dir, 'Y_data.csv'))
    
    # Fit model with 1 component for clear interpretation
    model = pls.pls_regression(X[:10, :5], Y[:10, :3], n_components=1)
    
    print(f"Analyzing first PLS component:")
    
    w1 = model['W'][:, 0]
    p1 = model['P'][:, 0]
    c1 = model['C'][:, 0]
    t1 = model['T'][:, 0]
    d1 = model['D'][0]
    
    print(f"  Weight vector w1: {w1}")
    print(f"  X loading p1: {p1}")
    print(f"  Y loading c1: {c1}")
    print(f"  First few score values t1: {t1[:5]}")
    print(f"  Scaling factor d1: {d1}")
    
    # Verify the relationships for this component
    X_centered = X[:10, :5] - model['X_mean']
    Y_centered = Y[:10, :3] - model['Y_mean']
    
    # t1 should equal X * w1
    t1_computed = jnp.dot(X_centered, w1)
    t1_error = float(jnp.mean((t1 - t1_computed)**2))
    print(f"  Score computation error |t1 - X*w1|²: {t1_error:.10f}")
    
    # p1 should equal X^T * t1 / (t1^T * t1)
    p1_computed = jnp.dot(X_centered.T, t1) / jnp.dot(t1, t1)
    p1_error = float(jnp.mean((p1 - p1_computed)**2))
    print(f"  X loading computation error |p1 - X^T*t1/(t1^T*t1)|²: {p1_error:.10f}")
    
    if t1_error < 1e-10 and p1_error < 1e-10:
        print(f"  ✓ Component relationships verified")
    else:
        print(f"  ⚠ Component relationships have errors")


if __name__ == '__main__':
    success1 = test_mathematical_formulation()
    test_component_interpretation()
    
    if success1:
        print(f"\n🎉 MATHEMATICAL VERIFICATION PASSED")
        print(f"The implementation correctly follows the standard PLS formulation!")
    else:
        print(f"\n⚠ MATHEMATICAL VERIFICATION NEEDS ATTENTION")
        print(f"Some relationships may need refinement.")
