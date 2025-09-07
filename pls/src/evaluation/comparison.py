#!/usr/bin/env python3
"""Model comparison utilities for PLS models."""

import numpy as np
from typing import Dict, List, Union


def model_comparison(models_results: List[Dict[str, Union[Dict, str]]], verbose: bool = True) -> Dict:
    """Compare multiple PLS models with different parameters."""
    comparison = {
        'models': [],
        'best_model': None,
        'ranking': []
    }
    
    if verbose:
        print("🏆 Model Comparison:")
        print(f"{'Model':<20} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'Components':<10}")
        print("-" * 65)
    
    for model_result in models_results:
        model_info = {
            'name': model_result['name'],
            'metrics': model_result['metrics'],
            'n_components': model_result.get('n_components', 'Unknown')
        }
        comparison['models'].append(model_info)
        
        if verbose:
            metrics = model_result['metrics']
            print(f"{model_info['name']:<20} {metrics['r2']:<8.4f} {metrics['rmse']:<10.4f} "
                  f"{metrics['mae']:<10.4f} {model_info['n_components']:<10}")
    
    comparison['ranking'] = sorted(comparison['models'], key=lambda x: x['metrics']['r2'], reverse=True)
    comparison['best_model'] = comparison['ranking'][0]['name'] if comparison['ranking'] else None
    
    if verbose and comparison['best_model']:
        print(f"🥇 Best model: {comparison['best_model']}")
    
    return comparison
