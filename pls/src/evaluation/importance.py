#!/usr/bin/env python3
"""Component importance analysis for PLS models."""

import numpy as np
from typing import Dict, List, Optional


def component_importance(model: Dict, feature_names: Optional[List[str]] = None, verbose: bool = True) -> Dict:
    """Analyze PLS component loadings for feature importance."""
    W = np.asarray(model['W'])
    P = np.asarray(model['P'])
    C = np.asarray(model['C'])
    
    n_features, n_components = W.shape
    
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    T = np.asarray(model['T'])
    explained_variance = np.var(T, axis=0)
    total_explained = np.sum(explained_variance)
    
    vip_scores = np.zeros(n_features)
    for i in range(n_features):
        weight_sum = 0
        for j in range(n_components):
            weight_sum += (W[i, j] ** 2) * explained_variance[j]
        vip_scores[i] = np.sqrt(n_features * weight_sum / total_explained)
    
    component_contrib = explained_variance / total_explained * 100
    
    results = {
        'vip_scores': vip_scores.tolist(),
        'feature_names': feature_names,
        'component_contributions': component_contrib.tolist(),
        'loadings_W': W.tolist(),
        'loadings_P': P.tolist(),
        'loadings_C': C.tolist(),
        'important_features': [feature_names[i] for i in np.where(vip_scores > 1.0)[0]]
    }
    
    if verbose:
        print("📊 Component Importance Analysis:")
        print(f"  Components: {n_components}")
        
        print("  Component Contributions:")
        for i, contrib in enumerate(component_contrib):
            print(f"    Component {i+1}: {contrib:.2f}%")
        
        print("  Variable Importance (VIP > 1.0):")
        sorted_indices = np.argsort(vip_scores)[::-1]
        for i in sorted_indices[:min(10, n_features)]:
            status = "⭐" if vip_scores[i] > 1.0 else "  "
            print(f"    {status} {feature_names[i]:<15}: {vip_scores[i]:.3f}")
    
    return results
