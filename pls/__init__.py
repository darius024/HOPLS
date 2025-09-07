# PLS Regression Implementations Package

__version__ = "1.0.0"
__author__ = "HOPLS Project"

# Import both implementations
from . import implementations

def get_implementation(implementation_type="optimized"):
    """
    Get a PLS implementation.
    
    Args:
        implementation_type (str): Either "basic" or "optimized"
        
    Returns:
        module: The requested PLS implementation module
    """
    if implementation_type == "basic":
        return implementations.basic
    elif implementation_type == "optimized":
        return implementations.optimized
    else:
        raise ValueError("implementation_type must be 'basic' or 'optimized'")

# Default to optimized implementation
pls_regression = implementations.optimized.pls_regression
predict = implementations.optimized.predict
parse_matrix = implementations.optimized.parse_matrix
