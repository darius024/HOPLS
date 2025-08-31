# PLS: Partial Least Squares

A comprehensive guide to understanding and implementing Partial Least Squares (PLS) regression from scratch.

## Table of Contents
- [PLS Theory and Principles](#pls-theory-and-principles)
- [Implementation From Scratch](#implementation-from-scratch)

## PLS Theory and Principles

### What is Partial Least Squares (PLS)?

Partial Least Squares (PLS) is a statistical method that finds the fundamental relations between two matrices (X and Y), i.e., a latent variable approach to modeling the covariance structures in these two spaces. PLS is particularly useful when:

- The number of predictors exceeds the number of observations
- Predictors are highly collinear
- You need interpretable latent components
- Traditional regression methods fail due to multicollinearity

### Mathematical Foundation

#### Core Concept
PLS finds linear combinations of the original variables that maximize the covariance between X and Y:

```
max cov(Xa, Yb) subject to ||a|| = ||b|| = 1
```

Where:
- `X` is the predictor matrix (n × p)
- `Y` is the response matrix (n × q)
- `a` and `b` are weight vectors
- `Xa` and `Yb` are the latent variables (scores)

#### Algorithm Steps (NIPALS Algorithm)

1. **Initialize**: Start with Y (or any column of Y)
2. **X-weights**: `w = X^T u / ||X^T u||` (normalize)
3. **X-scores**: `t = X w`
4. **Y-weights**: `c = Y^T t / ||Y^T t||` (normalize)
5. **Y-scores**: `u = Y c`
6. **Convergence check**: If `||u_old - u_new|| < tolerance`, continue; else goto step 2
7. **X-loadings**: `p = X^T t / (t^T t)`
8. **Y-loadings**: `q = Y^T t / (t^T t)`
9. **Deflation**: 
   - `X = X - t p^T`
   - `Y = Y - t q^T`
10. **Repeat** for next component

#### Key Properties

1. **Orthogonal Components**: Each successive component is orthogonal to previous ones
2. **Maximum Covariance**: Each component maximizes the covariance between X and Y spaces
3. **Dimensionality Reduction**: Reduces high-dimensional data to a few meaningful components
4. **Handles Multicollinearity**: Works well when predictors are highly correlated

### PLS Variants

#### PLS1 (Single Response)
- Y is a single variable (vector)
- Simpler case, often used for regression

#### PLS2 (Multiple Responses)
- Y is a matrix with multiple response variables
- More complex but handles multivariate outputs

#### Canonical PLS
- Focuses on maximizing correlation instead of covariance
- Uses normalized variables

## Implementation From Scratch

### Understanding the Data Structures Needed

Before implementing PLS, you need to understand what data structures and operations are required:

#### Core Data Structures:
1. **Input matrices**: X (predictors) and Y (responses)
2. **Weight vectors**: w (X-weights) and c (Y-weights) 
3. **Score vectors**: t (X-scores) and u (Y-scores)
4. **Loading vectors**: p (X-loadings) and q (Y-loadings)
5. **Mean vectors**: X_mean and Y_mean for centering

#### Essential Operations:
1. **Matrix-vector multiplication**: X^T * u, Y^T * t
2. **Vector normalization**: w / ||w||
3. **Outer product**: t * p^T for deflation
4. **Convergence checking**: ||u_old - u_new||

### Step-by-Step Implementation Approach

#### Step 1: Data Preparation
- **Input**: Raw matrices X and Y
- **Process**: Center the data by subtracting column means
- **Output**: X_centered, Y_centered, stored means
- **Key insight**: PLS requires mean-centered data to work properly

#### Step 2: Initialize First Component
- **Input**: Centered X and Y matrices
- **Process**: Start with first column of Y as initial u vector
- **Output**: Starting point u for NIPALS iteration
- **Key insight**: Choice of initialization affects convergence speed

#### Step 3: NIPALS Inner Loop (Single Component)
- **Input**: Current X_centered, Y_centered, initial u
- **Process**: Iterate until convergence:
  - Compute X-weights from X^T * u
  - Normalize weights
  - Compute X-scores from X * w
  - Compute Y-weights from Y^T * t  
  - Normalize Y-weights
  - Compute new Y-scores from Y * c
  - Check if u converged
- **Output**: Converged weight and score vectors
- **Key insight**: This finds the direction of maximum covariance

#### Step 4: Compute Loadings
- **Input**: Converged scores (t) and centered data
- **Process**: 
  - X-loadings: p = X^T * t / (t^T * t)
  - Y-loadings: q = Y^T * t / (t^T * t)
- **Output**: Loading vectors that describe original variables
- **Key insight**: Loadings show how original variables relate to components

#### Step 5: Deflation
- **Input**: Current X, Y matrices and component vectors t, p, q
- **Process**:
  - Remove component from X: X = X - t * p^T
  - Remove component from Y: Y = Y - t * q^T
- **Output**: Deflated matrices ready for next component
- **Key insight**: Deflation ensures orthogonal components

#### Step 6: Store Component and Repeat
- **Input**: All computed vectors for current component
- **Process**: Save w, c, p, q, t, u for this component
- **Output**: Component stored, ready for next iteration
- **Key insight**: Each component captures remaining variance

### Mathematical Requirements

#### Linear Algebra Operations Needed:
1. **Matrix transpose**: X^T
2. **Matrix-vector product**: X * w
3. **Vector dot product**: t^T * t
4. **Vector norm**: ||w||
5. **Outer product**: t * p^T

#### Convergence Criteria:
- **Tolerance**: Typically 1e-6
- **Max iterations**: Usually 500
- **Convergence check**: ||u_new - u_old|| < tolerance

#### Storage Requirements:
- **Per component**: 4 vectors (w, c, p, q) 
- **For prediction**: Weight matrix W, loading matrix Q
- **For centering**: Mean vectors

### Implementation Phases

#### Phase 1: Basic Structure
- Set up data containers
- Implement centering operation
- Create storage for components

#### Phase 2: Single Component
- Implement NIPALS inner loop
- Add convergence checking
- Test with simple 2D data

#### Phase 3: Multiple Components  
- Add deflation step
- Implement component storage
- Test orthogonality of components

#### Phase 4: Prediction
- Implement transform method
- Implement predict method
- Validate against known results

#### Phase 5: Validation
- Add cross-validation
- Implement performance metrics
- Compare with reference implementations

### Key Insights for Implementation

1. **Data centering is critical** - PLS will not work correctly without it
2. **Normalization prevents overflow** - Always normalize weight vectors
3. **Deflation ensures orthogonality** - Each component removes its contribution
4. **Convergence is usually fast** - Typically 2-10 iterations per component
5. **Storage order matters** - Keep components in order of extraction

### Common Implementation Pitfalls

1. **Forgetting to center data** - Will give incorrect results
2. **Not normalizing weights** - Can cause numerical instability  
3. **Wrong deflation** - Components won't be orthogonal
4. **Poor convergence checking** - May not find optimal solution
5. **Incorrect prediction** - Must use same centering as training

This step-by-step approach ensures you understand each piece before moving to the next, building a solid foundation for PLS implementation.

## Suggested Repository Structure

```
PLS/
├── README.md                    # Project overview and quick start
├── requirements.txt             # Python dependencies (numpy, matplotlib, etc.)
├── setup.py                     # Package installation script
├── src/
│   ├── __init__.py
│   ├── pls/
│   │   ├── __init__.py
│   │   ├── core.py              # Main PLSRegression class
│   │   ├── utils.py             # Data preprocessing utilities
│   │   └── metrics.py           # R², Q², VIP calculations
│   └── datasets/
│       ├── __init__.py
│       └── synthetic.py         # Generate test datasets
├── tests/
│   ├── __init__.py
│   ├── test_core.py             # Test main PLS implementation
│   ├── test_utils.py            # Test utility functions
│   ├── test_metrics.py          # Test evaluation metrics
│   └── test_integration.py      # Compare with scikit-learn
├── examples/
│   ├── basic_usage.py           # Simple PLS example
│   ├── cross_validation.py     # Model selection example
│   ├── comparison_sklearn.py    # Benchmark against scikit-learn
│   └── real_data_example.py     # Application to real dataset
├── docs/
│   ├── theory.md               # Mathematical background
│   ├── api.md                  # API documentation
│   └── tutorials/
│       ├── getting_started.md
│       └── advanced_usage.md
└── data/
    ├── sample_datasets/        # Small test datasets
    └── results/               # Output from experiments
```

### File-by-File Breakdown

#### Core Implementation (`src/pls/`)

**`core.py`** - Main PLS implementation:
- `PLSRegression` class with fit/predict/transform methods
- NIPALS algorithm implementation
- Component storage and management

**`utils.py`** - Helper functions:
- Data centering and scaling
- Matrix operations
- Input validation
- Data splitting utilities

**`metrics.py`** - Evaluation functions:
- R-squared calculation
- Q-squared (cross-validated R²)
- Variable Importance in Projection (VIP)
- Model selection criteria

#### Testing (`tests/`)

**`test_core.py`** - Core algorithm tests:
- NIPALS convergence
- Component orthogonality
- Prediction accuracy
- Edge cases (rank deficient, small samples)

**`test_integration.py`** - Comparison tests:
- Results match scikit-learn PLS
- Performance benchmarks
- Numerical precision tests

#### Examples (`examples/`)

**`basic_usage.py`** - Simple demonstration:
- Load data
- Fit PLS model
- Make predictions
- Visualize results

**`cross_validation.py`** - Model selection:
- K-fold cross-validation
- Optimal component selection
- Performance curves

#### Documentation (`docs/`)

**`theory.md`** - Mathematical details:
- NIPALS algorithm explanation
- Deflation mathematics
- Relationship to other methods

### Implementation Order

1. **Week 1**: Set up structure, implement `core.py` basic skeleton
2. **Week 2**: Complete NIPALS algorithm, add basic tests
3. **Week 3**: Add utilities, metrics, and comprehensive testing
4. **Week 4**: Create examples, documentation, and validation

### Key Benefits of This Structure

1. **Modular**: Each component has a clear responsibility
2. **Testable**: Comprehensive test coverage for all components
3. **Documented**: Clear examples and theory documentation
4. **Extensible**: Easy to add new features or variants
5. **Professional**: Follows Python package conventions

### Getting Started Commands

```bash
# Create the structure
mkdir -p PLS/src/pls PLS/src/datasets PLS/tests PLS/examples PLS/docs/tutorials PLS/data/sample_datasets PLS/data/results

# Initialize Python package files
touch PLS/src/__init__.py PLS/src/pls/__init__.py PLS/src/datasets/__init__.py PLS/tests/__init__.py

# Create main files
touch PLS/src/pls/core.py PLS/src/pls/utils.py PLS/src/pls/metrics.py
touch PLS/tests/test_core.py PLS/examples/basic_usage.py
touch PLS/README.md PLS/requirements.txt PLS/setup.py
```

This structure provides a solid foundation for developing, testing, and documenting your PLS implementation while keeping it organized and maintainable.
