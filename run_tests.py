#!/usr/bin/env python3
"""
Test runner for HOPLS PLS regression implementations.

Usage:
    python run_tests.py [test_type]

Test types:
    quick       - Quick functionality test (default)
    comprehensive - Full comprehensive test suite  
    examples    - Multiple matrix examples test
    mathematical - Mathematical verification test
    benchmark   - Performance benchmarking
    all         - Run all tests
"""

import sys
import os
import subprocess
import argparse

def run_test(test_script, description):
    """Run a test script and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    script_path = os.path.join('tests', test_script)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run HOPLS tests')
    parser.add_argument('test_type', nargs='?', default='quick',
                       choices=['quick', 'comprehensive', 'examples', 'mathematical', 'benchmark', 'all'],
                       help='Type of test to run (default: quick)')
    
    args = parser.parse_args()
    
    print("HOPLS - PLS Regression Test Runner")
    print("=" * 60)
    
    results = []
    
    if args.test_type == 'quick' or args.test_type == 'all':
        success = run_test('test_quick.py', 'Quick Functionality Test')
        results.append(('Quick Test', success))
    
    if args.test_type == 'comprehensive' or args.test_type == 'all':
        success = run_test('test_comprehensive.py', 'Comprehensive Test Suite')
        results.append(('Comprehensive Test', success))
    
    if args.test_type == 'examples' or args.test_type == 'all':
        success = run_test('test_comprehensive_examples.py', 'Multiple Matrix Examples Test')
        results.append(('Examples Test', success))
    
    if args.test_type == 'mathematical' or args.test_type == 'all':
        success = run_test('test_mathematical_verification.py', 'Mathematical Verification Test')
        results.append(('Mathematical Test', success))
    
    if args.test_type == 'benchmark' or args.test_type == 'all':
        success = run_test('test_benchmark.py', 'Performance Benchmark')
        results.append(('Benchmark', success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<25} {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
