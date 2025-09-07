#!/usr/bin/env python3
"""
HOPLS - Main Entry Point

This script provides a command-line interface for using the PLS regression algorithm.
It can be used to:
1. Run PLS regression on your own data files
2. See a demonstration with example data
3. Get help on usage

Usage:
    python main.py demo                    # Run demonstration
    python main.py run <X_file> <Y_file>   # Run on your data files
    python main.py --help                  # Show help
"""

import sys
import os
import argparse
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pls
import jax.numpy as jnp


def run_demo():
    """Run a demonstration of PLS regression."""
    print("🚀 HOPLS - PLS Regression Demonstration")
    print("=" * 50)
    
    print("\nCreating example dataset...")
    
    # Create simple example data
    np.random.seed(42)
    X = jnp.array([
        [1.2, 2.1, 0.8],
        [2.3, 1.9, 1.5],
        [3.1, 3.2, 2.1],
        [1.8, 2.5, 1.0],
        [4.0, 1.3, 2.8],
        [2.7, 3.8, 1.6],
        [3.5, 2.0, 2.3],
        [1.5, 3.1, 0.9],
        [4.2, 1.7, 2.9],
        [2.4, 2.8, 1.4]
    ])
    
    Y = jnp.array([
        [4.1, 1.8],
        [5.7, 2.3],
        [8.4, 4.2],
        [5.3, 2.1],
        [7.1, 4.5],
        [7.9, 3.4],
        [7.8, 3.8],
        [4.5, 2.0],
        [8.2, 4.7],
        [6.1, 2.9]
    ])
    
    print(f"Dataset: X{X.shape} -> Y{Y.shape}")
    
    # Fit PLS model
    print(f"\nFitting PLS model with 2 components...")
    model = pls.pls_regression(X, Y, n_components=2)
    
    # Make predictions
    print("Making predictions...")
    Y_pred = pls.predict(X, model)
    
    # Calculate metrics
    mse = float(np.mean((Y - Y_pred)**2))
    r2 = float(1 - np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y, axis=0))**2))
    
    print(f"\nResults:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  R-squared Score: {r2:.4f}")
    print(f"  Model Components: {model['n_components']}")
    
    print(f"\nFirst 5 predictions vs actual:")
    print(f"{'Actual':<15} {'Predicted':<15} {'Error':<15}")
    print("-" * 45)
    for i in range(min(5, len(Y))):
        actual = f"[{Y[i,0]:.2f}, {Y[i,1]:.2f}]"
        predicted = f"[{Y_pred[i,0]:.2f}, {Y_pred[i,1]:.2f}]"
        error = f"{np.linalg.norm(Y[i] - Y_pred[i]):.4f}"
        print(f"{actual:<15} {predicted:<15} {error:<15}")
    
    print(f"\n✅ Demo completed successfully!")
    return True


def run_on_files(x_file, y_file, n_components=2):
    """Run PLS regression on user-provided data files."""
    print(f"🚀 HOPLS - Running PLS Regression")
    print("=" * 50)
    
    try:
        print(f"Loading data files...")
        print(f"  X data: {x_file}")
        print(f"  Y data: {y_file}")
        
        # Load data files
        if x_file.endswith('.csv'):
            X = jnp.array(np.loadtxt(x_file, delimiter=','))
        else:
            X = jnp.array(np.loadtxt(x_file))
            
        if y_file.endswith('.csv'):
            Y = jnp.array(np.loadtxt(y_file, delimiter=','))
        else:
            Y = jnp.array(np.loadtxt(y_file))
        
        print(f"Data loaded: X{X.shape} -> Y{Y.shape}")
        
        # Fit PLS model
        print(f"\nFitting PLS model with {n_components} components...")
        model = pls.pls_regression(X, Y, n_components=n_components)
        
        # Make predictions
        print("Making predictions...")
        Y_pred = pls.predict(X, model)
        
        # Calculate metrics
        mse = float(np.mean((Y - Y_pred)**2))
        mae = float(np.mean(np.abs(Y - Y_pred)))
        
        # R-squared calculation
        ss_res = np.sum((Y - Y_pred)**2)
        ss_tot = np.sum((Y - np.mean(Y, axis=0))**2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  Mean Squared Error: {mse:.6f}")
        print(f"  Mean Absolute Error: {mae:.6f}")
        print(f"  R-squared Score: {r2:.4f}")
        print(f"  Model Components: {model['n_components']}")
        
        # Show model structure
        print(f"\n🔧 Model Structure:")
        print(f"  X loadings (P): {model['P'].shape}")
        print(f"  Y loadings (C): {model['C'].shape}")
        print(f"  Weights (W): {model['W'].shape}")
        print(f"  Scores (T): {model['T'].shape}")
        
        print(f"\n✅ PLS regression completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Please check your file paths and data format.")
        return False


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description='HOPLS - High-Performance Partial Least Squares Regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                                    # Run demonstration
  python main.py run data/X.csv data/Y.csv              # Run on CSV files  
  python main.py run data/X.txt data/Y.txt -c 3         # Run with 3 components
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run PLS regression demonstration')
    
    # Run command  
    run_parser = subparsers.add_parser('run', help='Run PLS regression on data files')
    run_parser.add_argument('x_file', help='Path to X data file (CSV or space-delimited)')
    run_parser.add_argument('y_file', help='Path to Y data file (CSV or space-delimited)')
    run_parser.add_argument('-c', '--components', type=int, default=2,
                           help='Number of PLS components (default: 2)')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        return run_demo()
    elif args.command == 'run':
        return run_on_files(args.x_file, args.y_file, args.components)
    else:
        parser.print_help()
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
