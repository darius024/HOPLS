#!/usr/bin/env python3
"""Comprehensive PLS model evaluator."""

import numpy as np
from typing import Dict, Optional, List
from .metrics import basic_metrics
from .residuals import residual_analysis
from .importance import component_importance


class PLSEvaluator:
    """Comprehensive PLS model evaluation class."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the PLS evaluator."""
        self.verbose = verbose
        self.results = {}
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model: Dict, feature_names: Optional[List[str]] = None) -> Dict:
        """Perform comprehensive model evaluation."""
        if self.verbose:
            print("🔬 COMPREHENSIVE PLS MODEL EVALUATION")
            print("=" * 60)
        
        basic = basic_metrics(y_true, y_pred, verbose=self.verbose)
        residuals = residual_analysis(y_true, y_pred, verbose=self.verbose)
        importance = component_importance(model, feature_names, verbose=self.verbose)
        
        results = {
            'basic_metrics': basic,
            'residual_analysis': residuals,
            'component_importance': importance,
            'model_summary': {
                'n_components': model.get('n_components', 0),
                'n_features': model['W'].shape[0] if 'W' in model else 0,
                'n_targets': model['C'].shape[0] if 'C' in model else 0
            }
        }
        
        if self.verbose:
            print("=" * 60)
            print("✅ EVALUATION COMPLETED")
        
        self.results['comprehensive'] = results
        return results
