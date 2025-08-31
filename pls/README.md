# PLS Implementation From Scratch: Step-by-Step Guide

This guide provides detailed instructions for implementing Partial Least Squares (PLS) regression from the ground up, telling you exactly what to do at each step without providing code.

## Table of Contents
- [Understanding What You Need to Build](#understanding-what-you-need-to-build)
- [Data Structures You Must Create](#data-structures-you-must-create)
- [Operations You Must Implement](#operations-you-must-implement)
- [Step-by-Step Implementation Plan](#step-by-step-implementation-plan)
- [Testing Approach](#testing-approach)
- [File Organization](#file-organization)

## Understanding What You Need to Build

### Core Concept
You are building a system that takes two matrices (X and Y) and finds the best linear combinations of X variables that predict Y variables. The algorithm finds these combinations iteratively, one "component" at a time.

### What PLS Does Mathematically
1. **Finds direction in X space** that has maximum covariance with Y
2. **Projects both X and Y** onto this direction to get "scores"
3. **Removes this pattern** from both X and Y (deflation)
4. **Repeats** to find additional orthogonal components
5. **Uses all components** together to make predictions

## Data Structures You Must Create

### Input Data Storage
You need variables to hold:
- **Original X matrix**: n_samples rows, n_features columns
- **Original Y matrix**: n_samples rows, n_targets columns  
- **Centered X matrix**: X with column means subtracted
- **Centered Y matrix**: Y with column means subtracted
- **X column means**: vector of length n_features (save for prediction)
- **Y column means**: vector of length n_targets (save for prediction)

### Per-Component Storage (for each component you extract)
For each component, you need to store 6 vectors:
- **X-weights (w)**: length n_features - how to combine X variables
- **Y-weights (c)**: length n_targets - how to combine Y variables
- **X-scores (t)**: length n_samples - X data projected onto w direction
- **Y-scores (u)**: length n_samples - Y data projected onto c direction
- **X-loadings (p)**: length n_features - how X variables relate to t scores
- **Y-loadings (q)**: length n_targets - how Y variables relate to t scores

### Final Model Storage (accumulated from all components)
You need matrices where each column is one component:
- **W matrix**: n_features rows, n_components columns (all X-weights)
- **P matrix**: n_features rows, n_components columns (all X-loadings)
- **Q matrix**: n_targets rows, n_components columns (all Y-loadings)
- **T matrix**: n_samples rows, n_components columns (all X-scores)
- **U matrix**: n_samples rows, n_components columns (all Y-scores)

### Algorithm Control Variables
You need:
- **Number of components to extract** (e.g., 3)
- **Convergence tolerance** (e.g., 1e-6)
- **Maximum iterations** (e.g., 500)
- **Current component number** (0, 1, 2, ...)
- **Convergence flag** (True/False for each iteration)

## Operations You Must Implement

### Data Preparation Operations

#### DataCentering Operation
**Operation name**: `center_data`
**What to do**: For any matrix, subtract the mean of each column from that column
**Why**: PLS requires mean-centered data to work correctly
**How**: Calculate column means, then subtract each mean from its column
**Save**: The column means for later use in prediction
**Returns**: The centered matrix and the column means vector

#### InputValidation Operation
**Operation name**: `validate_input`
**What to do**: Check that input data is valid before processing
**Check**: X and Y have same number of rows
**Check**: No infinite or NaN values exist
**Check**: Data is numeric and 2-dimensional
**Action**: Convert to proper numeric type and raise errors for invalid data
**Returns**: Validated and properly formatted X and Y matrices

### NIPALS Algorithm Operations

#### XWeightComputation Operation
**Operation name**: `compute_x_weights`
**Input**: Takes current Y-scores vector (u) and centered X matrix
**Process**: Multiply X-transpose by u vector, then normalize result to unit length
**Output**: The X-weights vector (w) with unit length
**Purpose**: Finds the direction in X space that best correlates with Y

#### YWeightComputation Operation
**Operation name**: `compute_y_weights`  
**Input**: Takes current X-scores vector (t) and centered Y matrix
**Process**: Multiply Y-transpose by t vector, then normalize result to unit length
**Output**: The Y-weights vector (c) with unit length
**Purpose**: Finds the direction in Y space that best correlates with X

#### XScoreComputation Operation
**Operation name**: `compute_x_scores`
**Input**: Takes X-weights vector (w) and centered X matrix
**Process**: Multiply X matrix by w vector
**Output**: The X-scores vector (t)
**Purpose**: Projects X data onto the weight direction

#### YScoreComputation Operation
**Operation name**: `compute_y_scores`
**Input**: Takes Y-weights vector (c) and centered Y matrix
**Process**: Multiply Y matrix by c vector
**Output**: The Y-scores vector (u)
**Purpose**: Projects Y data onto the weight direction

#### LoadingComputation Operation
**Operation names**: `compute_x_loadings` and `compute_y_loadings`
**Input**: Takes scores vector (t) and centered data matrix
**Process**: Multiply data-transpose by t, then divide by t-transpose times t
**Output**: The loadings vector (p for X, q for Y)
**Purpose**: Shows how original variables relate to the component

#### ConvergenceChecker Operation
**Operation name**: `check_convergence`
**Input**: Takes previous Y-scores (u_old) and new Y-scores (u_new)
**Process**: Calculate the norm (length) of the difference vector
**Output**: Boolean flag indicating if algorithm has converged
**Decision rule**: If difference is smaller than tolerance, you've converged

#### NIPALSSingleComponent Operation
**Operation name**: `extract_single_component`
**Input**: Centered X and Y matrices, tolerance, max iterations
**Process**: Runs the complete NIPALS loop for one component
**Steps**: Initialize u, then iterate (compute weights, scores, check convergence)
**Output**: All 6 component vectors (w, c, t, u, p, q)
**Purpose**: Extracts one complete PLS component

### Deflation Operations

#### MatrixDeflation Operation
**Operation name**: `deflate_matrix`
**Input**: Takes matrix, scores vector, and loadings vector
**Process**: Subtract (scores × loadings-transpose) from matrix
**Output**: The deflated matrix with component removed
**Purpose**: Removes the found pattern from data

#### DataDeflation Operation
**Operation name**: `deflate_data`
**Input**: X matrix, Y matrix, t-scores, p-loadings, q-loadings
**Process**: Apply matrix deflation to both X and Y
**Output**: Both matrices with current component removed
**Purpose**: Prepares data for finding next orthogonal component

### Prediction Operations

#### DataTransform Operation
**Operation name**: `transform_data`
**Input**: New X data, stored X means, W matrix (all X-weights)
**Process**: Center the new X data, then multiply by W matrix
**Output**: Matrix with n_components columns (PLS component scores)
**Purpose**: Converts new data to PLS component space

#### YPrediction Operation
**Operation name**: `predict_y`
**Input**: New X data, stored means, W matrix, Q matrix
**Process**: Transform X to PLS space, multiply by Q-transpose, add Y means
**Output**: Predicted Y matrix
**Purpose**: Makes actual predictions for new data

### Component Organization Operations

#### ComponentStorage Operation
**Operation name**: `store_component`
**Input**: Individual component vectors (w, c, t, u, p, q)
**Process**: Add vectors to a growing list or dictionary
**Output**: Updated component storage structure
**Purpose**: Keeps track of all extracted components

#### MatrixOrganization Operation
**Operation name**: `organize_components`
**Input**: List of all stored components
**Process**: Convert component list into organized matrices
**Output**: Final W, P, Q, T, U matrices where each column is one component
**Purpose**: Creates final model representation for prediction

## Step-by-Step Implementation Plan

### Phase 1: Basic Infrastructure
**Week 1 Tasks**:

1. **Create the main PLSRegression class**
   - Class name: `PLSRegression`
   - Initialization method: `__init__` that stores n_components, tolerance, max_iterations
   - Main methods to create: `fit`, `transform`, `predict`
   - Instance variables: `is_fitted` (boolean), `x_means_`, `y_means_`, `W_`, `P_`, `Q_`, `T_`, `U_`
   - Track if model is fitted using the `is_fitted` flag

2. **Create the InputValidator class**
   - Class name: `InputValidator`
   - Method: `validate_matrices` to check input matrix shapes match
   - Method: `check_for_invalid_values` to find NaN/infinite values
   - Method: `ensure_2d_numeric` to verify data is 2-dimensional and numeric
   - Method: `convert_to_float` to convert data to proper floating-point type

3. **Create the DataCenterer class**
   - Class name: `DataCenterer`
   - Method: `calculate_column_means` to get mean of each column
   - Method: `center_matrix` to subtract means from columns
   - Method: `restore_means` to add means back (for prediction)
   - Test this works correctly (centered data should have zero column means)

### Phase 2: NIPALS Algorithm Core
**Week 2 Tasks**:

1. **Create the NIPALSAlgorithm class**
   - Class name: `NIPALSAlgorithm`
   - Method: `extract_component` to extract one PLS component
   - Method: `initialize_u_vector` to start with first column of Y
   - Method: `run_nipals_iteration` to perform one iteration of the NIPALS loop
   - The iteration method should call: `compute_x_weights`, `compute_x_scores`, `compute_y_weights`, `compute_y_scores`, `check_convergence`

2. **Create the ComponentCalculator class**
   - Class name: `ComponentCalculator`
   - Method: `compute_x_weights` (calls XWeightComputation operation)
   - Method: `compute_y_weights` (calls YWeightComputation operation)
   - Method: `compute_x_scores` (calls XScoreComputation operation)
   - Method: `compute_y_scores` (calls YScoreComputation operation)
   - Method: `compute_x_loadings` (calls LoadingComputation operation)
   - Method: `compute_y_loadings` (calls LoadingComputation operation)

3. **Create the ConvergenceMonitor class**
   - Class name: `ConvergenceMonitor`
   - Method: `check_convergence` to compare old and new Y-scores
   - Method: `calculate_difference_norm` to compute vector difference magnitude
   - Instance variables: `tolerance`, `max_iterations`, `current_iteration`

### Phase 3: Multiple Components and Deflation
**Week 3 Tasks**:

1. **Create the MatrixDeflator class**
   - Class name: `MatrixDeflator`
   - Method: `deflate_x_matrix` to remove component from X matrix
   - Method: `deflate_y_matrix` to remove component from Y matrix
   - Method: `compute_deflation_term` to calculate (scores × loadings-transpose)
   - Test that deflated matrices have reduced rank

2. **Create the MultiComponentExtractor class**
   - Class name: `MultiComponentExtractor`
   - Method: `extract_all_components` to loop over desired number of components
   - Method: `extract_and_deflate` to extract one component then deflate
   - Uses: `NIPALSAlgorithm`, `MatrixDeflator`, `ComponentOrganizer`

3. **Create the ComponentOrganizer class**
   - Class name: `ComponentOrganizer`
   - Method: `initialize_storage` to create empty component storage
   - Method: `store_component` to add new component vectors
   - Method: `organize_into_matrices` to create final W, P, Q, T, U matrices
   - Method: `verify_orthogonality` to check components are orthogonal

### Phase 4: Prediction and Validation
**Week 4 Tasks**:

1. **Create the PLSPredictor class**
   - Class name: `PLSPredictor`
   - Method: `transform` to convert X to PLS space (calls DataTransform operation)
   - Method: `predict` to generate Y predictions (calls YPrediction operation)
   - Method: `validate_new_data` to check new data is compatible with trained model

2. **Create the ModelValidator class**
   - Class name: `ModelValidator`
   - Method: `test_on_synthetic_data` with known relationships
   - Method: `compare_with_sklearn` to match scikit-learn PLS results
   - Method: `test_edge_cases` for small datasets and rank-deficient data

3. **Create the EvaluationMetrics class**
   - Class name: `EvaluationMetrics`
   - Method: `calculate_r_squared` for model performance
   - Method: `cross_validate` for model validation
   - Method: `select_components` for choosing optimal number of components

## Testing Approach

### Unit Testing Strategy
**What to test**: Each individual operation in isolation
**Data centering test**: Verify centered data has zero column means
**Weight computation test**: Check weights have unit length
**Score computation test**: Verify score shapes and values
**Loading computation test**: Check loading calculations
**Deflation test**: Verify deflated matrices have correct properties

### Integration Testing Strategy  
**What to test**: Full algorithm working together
**Single component test**: Extract one component, verify all outputs
**Multi-component test**: Extract multiple components, check orthogonality
**Prediction test**: Fit on training data, predict on test data
**Comparison test**: Match results with scikit-learn implementation

### Validation Testing Strategy
**What to test**: Algorithm correctness on real problems
**Synthetic data test**: Create data with known structure, verify recovery
**Benchmark dataset test**: Use standard datasets, compare performance
**Cross-validation test**: Verify model selection works correctly

## File Organization

### Directory Structure to Create
Create these directories and files:
- **src/pls/** (main package directory)
- **src/pls/core.py** (main PLSRegression class)
- **src/pls/utils.py** (helper functions)
- **src/pls/metrics.py** (evaluation functions)
- **tests/** (test directory)
- **tests/test_core.py** (test main functionality)
- **tests/test_utils.py** (test helper functions)
- **examples/** (usage examples)
- **examples/basic_usage.py** (simple demonstration)

### Core Module (core.py) Structure
**What to include**:
- **PLSRegression class**: Main user interface class
  - Methods: `__init__`, `fit`, `transform`, `predict`, `score`
  - Instance variables: `W_`, `P_`, `Q_`, `T_`, `U_`, `x_means_`, `y_means_`, `is_fitted`
- **NIPALSAlgorithm class**: Core algorithm implementation
- **MultiComponentExtractor class**: Handles multiple component extraction
- **ComponentOrganizer class**: Manages component storage and organization

### Utils Module (utils.py) Structure  
**What to include**:
- **DataCenterer class**: Data centering functionality
  - Methods: `calculate_column_means`, `center_matrix`, `restore_means`
- **InputValidator class**: Input validation functionality
  - Methods: `validate_matrices`, `check_for_invalid_values`, `ensure_2d_numeric`
- **MatrixDeflator class**: Matrix deflation operations
  - Methods: `deflate_x_matrix`, `deflate_y_matrix`, `compute_deflation_term`
- **ComponentCalculator class**: Individual component calculations
  - Methods: `compute_x_weights`, `compute_y_weights`, `compute_x_scores`, `compute_y_scores`

### Metrics Module (metrics.py) Structure
**What to include**:
- **EvaluationMetrics class**: Model evaluation functionality
  - Methods: `calculate_r_squared`, `cross_validate`, `select_components`
- **ModelValidator class**: Testing and validation
  - Methods: `test_on_synthetic_data`, `compare_with_sklearn`, `test_edge_cases`
- **ConvergenceMonitor class**: Algorithm convergence tracking
  - Methods: `check_convergence`, `calculate_difference_norm`

## Implementation Tips

### Start Simple
- Begin with 2D toy data (few samples, few features)
- Get one component working before multiple components
- Test each operation individually before combining

### Debug Systematically
- Print intermediate values to verify calculations
- Check matrix shapes at each step
- Verify mathematical properties (orthogonality, unit norms)

### Build Incrementally
- Add one feature at a time
- Test after each addition
- Don't move to next phase until current phase works

### Validate Continuously
- Compare with known results when possible
- Use simple cases where you can calculate answers by hand
- Build confidence before tackling complex cases

This plan gives you everything you need to build PLS from scratch. Follow it step by step, and you'll have a complete, working implementation.
