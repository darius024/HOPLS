#!/usr/bin/env python3
"""Cross-validation functions for PLS models."""

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from typing import Dict, List, Union


def cross_validation(X: np.ndarray, Y: np.ndarray, pls_func, n_components: int, 
                    cv_type: str = 'kfold', k: int = 5, verbose: bool = True) -> Dict[str, Union[float, List]]:
    """Perform cross-validation on PLS model."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if cv_type == 'kfold':
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_type == 'loo':
        cv = LeaveOneOut()
    else:
        raise ValueError("cv_type must be 'kfold' or 'loo'")
    
    cv_scores = []
    fold_predictions = []
    
    if verbose:
        print(f"🔄 Performing {cv_type.upper()} Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        try:
            model = pls_func(X_train, Y_train, n_components=n_components)
            X_test_centered = X_test - model['X_mean']
            Y_pred = X_test_centered @ model['W'] @ model['D'] @ model['C'].T + model['Y_mean']
            
            ss_res = np.sum((Y_test - Y_pred) ** 2)
            ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
            fold_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            cv_scores.append(fold_r2)
            fold_predictions.append({'true': Y_test, 'pred': Y_pred, 'r2': fold_r2})
            
            if verbose and cv_type == 'kfold':
                print(f"  Fold {fold+1}/{k}: R² = {fold_r2:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Fold {fold+1} failed: {e}")
            cv_scores.append(0.0)
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    results = {
        'cv_type': cv_type,
        'mean_r2': float(mean_score),
        'std_r2': float(std_score),
        'fold_scores': cv_scores,
        'fold_predictions': fold_predictions
    }
    
    if verbose:
        print(f"  Mean CV R²: {mean_score:.4f} (±{std_score:.4f})")
    
    return results


def cross_validate_pls(X, Y, pls_func, n_components_range=range(1, 6), cv_folds=5):
    """Cross-validate PLS with different numbers of components."""
    results = {}
    
    print(f"🔄 Cross-validating PLS with {cv_folds}-fold CV")
    print("-" * 50)
    
    for n_comp in n_components_range:
        print(f"\nTesting {n_comp} components:")
        cv_result = cross_validation(X, Y, pls_func, n_comp, cv_type='kfold', k=cv_folds)
        results[n_comp] = cv_result
        
    best_n_comp = max(results.keys(), key=lambda k: results[k]['mean_r2'])
    print(f"\n🏆 Best number of components: {best_n_comp} (R² = {results[best_n_comp]['mean_r2']:.4f})")
    
    return results, best_n_comp
