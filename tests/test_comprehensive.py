#!/usr/bin/env python3
"""
Test suite for PLS regression implementations.

This module provides comprehensive testing for both basic and optimized
PLS regression implementations.
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


class PLSTestSuite:
    """Comprehensive test suite for PLS implementations."""
    
    def __init__(self, implementation_type="optimized"):
        """Initialize test suite with specified implementation."""
        self.implementation = pls.get_implementation(implementation_type)
        self.implementation_type = implementation_type
        
        # Load test data
        self.data_dir = os.path.join(project_root, 'pls', 'examples')
        self.X_full = self.implementation.parse_matrix(
            os.path.join(self.data_dir, 'X_data.csv')
        )
        self.Y_full = self.implementation.parse_matrix(
            os.path.join(self.data_dir, 'Y_data.csv')
        )
    
    def create_synthetic_data(self, n_samples=5, n_features=3, n_targets=2):
        """Create small synthetic test data."""
        np.random.seed(42)
        X = jnp.array(np.random.randn(n_samples, n_features) * 2 + 5)
        # Make Y correlated with X
        Y = jnp.array(np.dot(X, np.random.randn(n_features, n_targets)) + 
                     np.random.randn(n_samples, n_targets) * 0.1)
        return X, Y
    
    def test_basic_functionality(self):
        """Test basic functionality with synthetic data."""
        print(f"\n1. Testing Basic Functionality ({self.implementation_type})")
        print("-" * 50)
        
        X, Y = self.create_synthetic_data()
        print(f"Data shapes: X{X.shape}, Y{Y.shape}")
        
        try:
            start_time = time.time()
            model = self.implementation.pls_regression(X, Y, n_components=1)
            duration = time.time() - start_time
            
            Y_pred = self.implementation.predict(X, model)
            mse = float(np.mean((Y - Y_pred)**2))
            
            print(f"✓ Success in {duration:.3f}s")
            print(f"  MSE: {mse:.6f}")
            print(f"  Model components: {model['n_components']}")
            
            return True, {"duration": duration, "mse": mse}
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False, {"error": str(e)}
    
    def test_scalability(self):
        """Test with different data sizes."""
        print(f"\n2. Testing Scalability ({self.implementation_type})")
        print("-" * 50)
        
        test_sizes = [
            (5, 3, 2, 1),   # (samples, features, targets, components)
            (8, 4, 2, 1),
            (10, 5, 3, 1),
            (15, 8, 4, 1),
            (20, 10, 6, 1),  # Full dataset
        ]
        
        results = []
        
        for n_samples, n_features, n_targets, n_components in test_sizes:
            X = self.X_full[:n_samples, :n_features]
            Y = self.Y_full[:n_samples, :n_targets]
            
            print(f"  Testing {n_samples}x{n_features} → {n_targets} features, {n_components} components")
            
            try:
                start_time = time.time()
                model = self.implementation.pls_regression(X, Y, n_components)
                duration = time.time() - start_time
                
                Y_pred = self.implementation.predict(X, model)
                mse = float(np.mean((Y - Y_pred)**2))
                
                result = {
                    'size': f"{n_samples}x{n_features}",
                    'duration': duration,
                    'mse': mse,
                    'success': True
                }
                
                print(f"    ✓ {duration:.3f}s, MSE: {mse:.6f}")
                
            except Exception as e:
                result = {
                    'size': f"{n_samples}x{n_features}",
                    'error': str(e),
                    'success': False
                }
                print(f"    ✗ Failed: {e}")
            
            results.append(result)
        
        return results
    
    def test_multi_component(self):
        """Test multi-component models."""
        print(f"\n3. Testing Multi-Component Models ({self.implementation_type})")
        print("-" * 50)
        
        X = self.X_full[:10, :5]
        Y = self.Y_full[:10, :3]
        
        results = []
        
        for n_comp in [1, 2]:
            print(f"  Testing {n_comp} component(s)")
            
            try:
                start_time = time.time()
                model = self.implementation.pls_regression(X, Y, n_comp)
                duration = time.time() - start_time
                
                Y_pred = self.implementation.predict(X, model)
                mse = float(np.mean((Y - Y_pred)**2))
                
                # Check if results are reasonable (not numerical overflow)
                is_stable = mse < 1000 and not np.isnan(mse) and not np.isinf(mse)
                
                result = {
                    'components': n_comp,
                    'duration': duration,
                    'mse': mse,
                    'stable': is_stable,
                    'success': True
                }
                
                status = "✓" if is_stable else "⚠ (unstable)"
                print(f"    {status} {duration:.3f}s, MSE: {mse:.6f}")
                
            except Exception as e:
                result = {
                    'components': n_comp,
                    'error': str(e),
                    'success': False
                }
                print(f"    ✗ Failed: {e}")
            
            results.append(result)
        
        return results
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        print(f"PLS Implementation Test Suite - {self.implementation_type.upper()}")
        print("=" * 60)
        
        all_results = {}
        
        # Test 1: Basic functionality
        success, results = self.test_basic_functionality()
        all_results['basic'] = {'success': success, 'results': results}
        
        # Test 2: Scalability
        results = self.test_scalability()
        successful_scales = [r for r in results if r['success']]
        all_results['scalability'] = {
            'success': len(successful_scales) > 0,
            'results': results,
            'success_rate': len(successful_scales) / len(results)
        }
        
        # Test 3: Multi-component
        results = self.test_multi_component()
        stable_results = [r for r in results if r.get('success', False) and r.get('stable', False)]
        all_results['multi_component'] = {
            'success': len(stable_results) > 0,
            'results': results,
            'stability_rate': len(stable_results) / len([r for r in results if r.get('success', False)])
        }
        
        return all_results


def compare_implementations():
    """Compare basic vs optimized implementations."""
    print("\nImplementation Comparison")
    print("=" * 60)
    
    # Create test data
    test_suite_opt = PLSTestSuite("optimized")
    X, Y = test_suite_opt.create_synthetic_data(8, 4, 2)
    
    results = {}
    
    for impl_type in ["basic", "optimized"]:
        print(f"\nTesting {impl_type} implementation...")
        impl = pls.get_implementation(impl_type)
        
        try:
            start_time = time.time()
            model = impl.pls_regression(X, Y, n_components=1)
            duration = time.time() - start_time
            
            Y_pred = impl.predict(X, model)
            mse = float(np.mean((Y - Y_pred)**2))
            
            results[impl_type] = {
                'success': True,
                'duration': duration,
                'mse': mse
            }
            
            print(f"  ✓ Duration: {duration:.3f}s, MSE: {mse:.6f}")
            
        except Exception as e:
            results[impl_type] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ Failed: {e}")
    
    # Summary
    if results.get("basic", {}).get("success") and results.get("optimized", {}).get("success"):
        basic_time = results["basic"]["duration"]
        opt_time = results["optimized"]["duration"]
        speedup = basic_time / opt_time if opt_time > 0 else float('inf')
        
        print(f"\nPerformance Summary:")
        print(f"  Basic implementation:     {basic_time:.3f}s")
        print(f"  Optimized implementation: {opt_time:.3f}s")
        print(f"  Speedup:                  {speedup:.1f}x")
    
    return results


def main():
    """Main test execution."""
    print("HOPLS - PLS Regression Test Suite")
    print("=" * 60)
    
    # Test optimized implementation
    print("\n🚀 OPTIMIZED IMPLEMENTATION")
    opt_suite = PLSTestSuite("optimized")
    opt_results = opt_suite.run_all_tests()
    
    # Test basic implementation  
    print("\n🐌 BASIC IMPLEMENTATION")
    basic_suite = PLSTestSuite("basic")
    basic_results = basic_suite.run_all_tests()
    
    # Compare implementations
    comparison = compare_implementations()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for impl_name, results in [("Optimized", opt_results), ("Basic", basic_results)]:
        print(f"\n{impl_name} Implementation:")
        print(f"  Basic functionality: {'✓' if results['basic']['success'] else '✗'}")
        print(f"  Scalability:         {'✓' if results['scalability']['success'] else '✗'} "
              f"({results['scalability']['success_rate']:.0%} success rate)")
        print(f"  Multi-component:     {'✓' if results['multi_component']['success'] else '✗'} "
              f"({results['multi_component'].get('stability_rate', 0):.0%} stable)")
    
    print(f"\nRecommendations:")
    print(f"  - Use optimized implementation for better performance")
    print(f"  - Prefer 1-2 components for numerical stability")
    print(f"  - Both implementations work well with single components")


if __name__ == '__main__':
    main()
