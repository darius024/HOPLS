#!/usr/bin/env python3
"""Residual analysis functions for PLS models."""

import numpy as np
from scipy import stats
from typing import Dict, Union


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> Dict[str, Union[float, bool]]:
    """Analyze residuals for model diagnostics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    residuals = y_true - y_pred
    
    if len(residuals) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 2.0
    
    abs_residuals = np.abs(residuals)
    if len(np.unique(y_pred)) > 1:
        hetero_corr, hetero_p = stats.pearsonr(y_pred, abs_residuals)
    else:
        hetero_corr, hetero_p = 0.0, 1.0
    
    results = {
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'shapiro_stat': float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
        'shapiro_p': float(shapiro_p) if not np.isnan(shapiro_p) else None,
        'normal_residuals': bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else None,
        'durbin_watson': float(dw_stat),
        'heteroscedasticity_corr': float(hetero_corr),
        'heteroscedasticity_p': float(hetero_p),
        'homoscedastic': bool(hetero_p > 0.05)
    }
    
    if verbose:
        print("🔍 Residual Analysis:")
        print(f"  Mean: {results['mean_residual']:.6f}, Std: {results['std_residual']:.6f}")
        print(f"  Range: [{results['min_residual']:.4f}, {results['max_residual']:.4f}]")
        if results['shapiro_p'] is not None:
            print(f"  Normality: p={results['shapiro_p']:.4f} ({'✓' if results['normal_residuals'] else '✗'})")
        print(f"  Durbin-Watson: {results['durbin_watson']:.4f}")
        print(f"  Homoscedasticity: p={results['heteroscedasticity_p']:.4f} ({'✓' if results['homoscedastic'] else '✗'})")
    
    return results
