#!/usr/bin/env python3
"""Basic PLS evaluation metrics."""

import numpy as np
import warnings
from typing import Dict


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """Calculate basic regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    n = len(y_true)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    max_error = np.max(np.abs(y_true - y_pred))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = mape if np.isfinite(mape) else np.inf
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'adj_r2': float(adj_r2),
        'max_error': float(max_error),
        'mape': float(mape)
    }
    
    if verbose:
        print("📊 Basic Metrics:")
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.4f}, Adj R²: {adj_r2:.4f}, Max Error: {max_error:.6f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return metrics


def quick_evaluate(y_true, y_pred, model=None, show_summary=True):
    """Quick evaluation function for basic metrics."""
    return basic_metrics(y_true, y_pred, verbose=show_summary)
