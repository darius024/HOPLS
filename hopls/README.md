# HOPLS Implementation From Scratch: Step-by-Step Guide

This guide provides detailed instructions for implementing Higher Order Partial Least Squares (HOPLS) regression from the ground up, telling you exactly what to do at each step without providing code.

## Table of Contents
- [Understanding What You Need to Build](#understanding-what-you-need-to-build)
- [Data Structures You Must Create](#data-structures-you-must-create)
- [Operations You Must Implement](#operations-you-must-implement)
- [Step-by-Step Implementation Plan](#step-by-step-implementation-plan)
- [Testing Approach](#testing-approach)
- [File Organization](#file-organization)

## Understanding What You Need to Build

### Core Concept
You are building a system that takes tensor (multi-dimensional array) data and matrix data, then finds the best relationships between them. HOPLS extends regular PLS to handle higher-order data structures like 3D tensors while maintaining the interpretability of PLS.

### What HOPLS Does Mathematically
1. **Unfolds tensors** into matrices using different tensor modes
2. **Applies PLS** between unfolded tensor and response matrix
3. **Finds tensor components** that maximize covariance with response
4. **Deflates both tensor and response** to remove found patterns
5. **Repeats** to find additional orthogonal components
6. **Uses all components** together to make predictions on new tensor data

### Key Differences from PLS
- **Input**: 3D tensor X (samples × features × modes) + 2D matrix Y (samples × targets)
- **Tensor unfolding**: Different ways to reshape tensor into matrix
- **Mode-specific weights**: Separate weight vectors for each tensor mode
- **Tensor deflation**: More complex deflation procedure for 3D structures

## Data Structures You Must Create

### Input Data Storage
You need variables to hold:
- **Original X tensor**: n_samples × n_features × n_modes (3D array)
- **Original Y matrix**: n_samples × n_targets (2D array)
- **Unfolded X matrix**: Various reshapings of the tensor for processing
- **Centered X tensor**: X with appropriate centering applied
- **Centered Y matrix**: Y with column means subtracted
- **X centering parameters**: Mode-specific centering information
- **Y column means**: vector of length n_targets (save for prediction)

### Tensor Unfolding Storage
For tensor manipulation, you need:
- **Mode-1 unfolding**: n_samples × (n_features × n_modes) matrix
- **Mode-2 unfolding**: n_features × (n_samples × n_modes) matrix  
- **Mode-3 unfolding**: n_modes × (n_samples × n_features) matrix
- **Vectorized tensor**: Flattened version for certain calculations
- **Unfolding indices**: Track how to reshape back to tensor form

### Per-Component Storage (for each component you extract)
For each component, you need to store:
- **Tensor weights (W)**: n_features × n_modes matrix - how to combine tensor elements
- **Y-weights (c)**: length n_targets - how to combine Y variables
- **Tensor scores (t)**: length n_samples - tensor data projected onto W
- **Y-scores (u)**: length n_samples - Y data projected onto c direction
- **Tensor loadings (P)**: n_features × n_modes - how tensor relates to scores
- **Y-loadings (q)**: length n_targets - how Y variables relate to tensor scores

### Final Model Storage (accumulated from all components)
You need storage structures where each "slice" is one component:
- **W tensor**: n_features × n_modes × n_components (all tensor weights)
- **P tensor**: n_features × n_modes × n_components (all tensor loadings)
- **Q matrix**: n_targets × n_components (all Y-loadings)
- **T matrix**: n_samples × n_components (all tensor scores)
- **U matrix**: n_samples × n_components (all Y-scores)

### Algorithm Control Variables
You need:
- **Number of components to extract** (e.g., 3)
- **Convergence tolerance** (e.g., 1e-6)
- **Maximum iterations** (e.g., 500)
- **Tensor unfolding mode** (1, 2, or 3)
- **Current component number** (0, 1, 2, ...)
- **Convergence flag** (True/False for each iteration)

## Operations You Must Implement

### Tensor Manipulation Operations

#### TensorUnfolding Operation
**Operation name**: `unfold_tensor`
**Input**: 3D tensor and unfolding mode (1, 2, or 3)
**Process**: Reshape tensor into 2D matrix according to specified mode
**Output**: 2D matrix and information to reverse the operation
**Purpose**: Convert tensor to matrix for PLS-like operations

#### TensorRefolding Operation
**Operation name**: `refold_matrix`
**Input**: 2D matrix and original tensor dimensions
**Process**: Reshape matrix back into original tensor shape
**Output**: 3D tensor with original structure restored
**Purpose**: Convert processed matrix back to tensor form

#### TensorCentering Operation
**Operation name**: `center_tensor`
**Input**: 3D tensor
**Process**: Center tensor appropriately (mode-specific or global)
**Output**: Centered tensor and centering parameters
**Purpose**: Prepare tensor data for HOPLS algorithm

#### TensorVectorization Operation
**Operation name**: `vectorize_tensor`
**Input**: 3D tensor
**Process**: Flatten tensor into 1D vector in consistent order
**Output**: 1D vector and reshaping information
**Purpose**: Enable certain matrix operations on tensor data

### Data Preparation Operations

#### InputValidation Operation
**Operation name**: `validate_tensor_input`
**Input**: X tensor and Y matrix
**Check**: X and Y have same number of samples (first dimension)
**Check**: X is 3-dimensional, Y is 2-dimensional
**Check**: No infinite or NaN values exist
**Action**: Convert to proper numeric type and raise errors for invalid data
**Returns**: Validated tensor and matrix

#### TensorYAlignment Operation
**Operation name**: `align_tensor_matrix`
**Input**: X tensor and Y matrix
**Process**: Ensure sample dimensions match exactly
**Check**: First dimension of X equals first dimension of Y
**Output**: Aligned tensor and matrix ready for processing
**Purpose**: Verify data compatibility before algorithm starts

### HOPLS Algorithm Operations

#### TensorWeightComputation Operation
**Operation name**: `compute_tensor_weights`
**Input**: Current Y-scores vector (u) and unfolded tensor matrix
**Process**: Apply PLS weight computation adapted for tensor structure
**Steps**: Multiply unfolded-tensor-transpose by u, reshape to tensor form, normalize
**Output**: Tensor weights matrix (W) with appropriate normalization
**Purpose**: Finds the tensor pattern that best correlates with Y

#### TensorScoreComputation Operation
**Operation name**: `compute_tensor_scores`
**Input**: Tensor weights (W) and centered tensor
**Process**: Contract tensor with weights to get scalar projection for each sample
**Mathematical operation**: Sum over features and modes of (tensor × weights)
**Output**: Tensor scores vector (t) of length n_samples
**Purpose**: Projects tensor data onto the weight direction

#### YWeightComputation Operation
**Operation name**: `compute_y_weights`
**Input**: Tensor scores vector (t) and centered Y matrix
**Process**: Multiply Y-transpose by t vector, then normalize to unit length
**Output**: Y-weights vector (c) with unit length
**Purpose**: Finds direction in Y space that correlates with tensor

#### YScoreComputation Operation
**Operation name**: `compute_y_scores`
**Input**: Y-weights vector (c) and centered Y matrix
**Process**: Multiply Y matrix by c vector
**Output**: Y-scores vector (u) of length n_samples
**Purpose**: Projects Y data onto the weight direction

#### TensorLoadingComputation Operation
**Operation name**: `compute_tensor_loadings`
**Input**: Tensor scores vector (t) and centered tensor
**Process**: For each feature-mode combination, compute correlation with scores
**Mathematical operation**: Contract tensor with scores, normalize by score variance
**Output**: Tensor loadings matrix (P) showing variable-component relationships
**Purpose**: Shows how original tensor variables relate to the component

#### YLoadingComputation Operation
**Operation name**: `compute_y_loadings`
**Input**: Tensor scores vector (t) and centered Y matrix
**Process**: Multiply Y-transpose by t, divide by t-transpose times t
**Output**: Y-loadings vector (q) of length n_targets
**Purpose**: Shows how Y variables relate to tensor component

#### ConvergenceChecker Operation
**Operation name**: `check_hopls_convergence`
**Input**: Previous Y-scores (u_old) and new Y-scores (u_new)
**Process**: Calculate norm of difference vector
**Output**: Boolean convergence flag
**Decision rule**: Converged if difference smaller than tolerance
**Purpose**: Determines when HOPLS iteration should stop

### Tensor Deflation Operations

#### TensorDeflation Operation
**Operation name**: `deflate_tensor`
**Input**: Current tensor, tensor scores (t), tensor loadings (P)
**Process**: Remove component pattern from tensor structure
**Mathematical operation**: Subtract outer product of scores and loadings from tensor
**Output**: Deflated tensor with component removed
**Purpose**: Ensures next component is orthogonal to current one

#### YMatrixDeflation Operation
**Operation name**: `deflate_y_matrix`
**Input**: Y matrix, tensor scores (t), Y-loadings (q)
**Process**: Remove component pattern from Y using tensor-derived scores
**Mathematical operation**: Subtract (t × q-transpose) from Y
**Output**: Deflated Y matrix
**Purpose**: Removes Y variance explained by current tensor component

#### CoupledDeflation Operation
**Operation name**: `deflate_coupled_data`
**Input**: X tensor, Y matrix, all component vectors
**Process**: Apply deflation to both tensor and matrix simultaneously
**Output**: Both deflated tensor and matrix
**Purpose**: Maintains coupling between tensor and matrix deflation

### Prediction Operations

#### TensorTransform Operation
**Operation name**: `transform_tensor`
**Input**: New tensor data, stored centering parameters, W tensor (all weights)
**Process**: Center new tensor, then project onto all weight directions
**Output**: Matrix with n_components columns (HOPLS component scores)
**Purpose**: Converts new tensor data to HOPLS component space

#### TensorPrediction Operation
**Operation name**: `predict_from_tensor`
**Input**: New tensor data, stored parameters, W tensor, Q matrix
**Process**: Transform tensor to component space, multiply by Q-transpose, add Y means
**Output**: Predicted Y matrix for the new tensor data
**Purpose**: Makes predictions from new tensor observations

### Component Organization Operations

#### TensorComponentStorage Operation
**Operation name**: `store_tensor_component`
**Input**: Individual component tensors and vectors (W, P, t, c, u, q)
**Process**: Add to growing component storage structure
**Output**: Updated storage with new component added
**Purpose**: Organizes all extracted tensor components

#### TensorModelOrganization Operation
**Operation name**: `organize_tensor_model`
**Input**: List of all stored tensor components
**Process**: Arrange components into final prediction-ready structure
**Output**: Organized W tensor, P tensor, Q matrix, T matrix, U matrix
**Purpose**: Creates final model for efficient prediction

## Step-by-Step Implementation Plan

### Phase 1: Tensor Infrastructure
**Week 1-2 Tasks**:

1. **Create the main HOPLSRegression class**
   - Class name: `HOPLSRegression`
   - Initialization method: `__init__` that stores n_components, tolerance, max_iterations, unfolding_mode
   - Main methods to create: `fit`, `transform`, `predict`
   - Instance variables: `is_fitted`, `x_centering_params_`, `y_means_`, `W_tensor_`, `P_tensor_`, `Q_`, `T_`, `U_`

2. **Create the TensorHandler class**
   - Class name: `TensorHandler`
   - Method: `unfold_tensor` for different unfolding modes
   - Method: `refold_matrix` to restore tensor structure
   - Method: `validate_tensor_shape` to check tensor dimensions
   - Method: `vectorize_tensor` and `tensorize_vector` for conversions

3. **Create the TensorCenterer class**
   - Class name: `TensorCenterer`
   - Method: `calculate_tensor_means` for mode-specific centering
   - Method: `center_tensor` to apply centering
   - Method: `restore_tensor_means` for prediction phase

4. **Create the TensorValidator class**
   - Class name: `TensorValidator`
   - Method: `validate_tensor_matrix_input` to check X tensor and Y matrix compatibility
   - Method: `check_tensor_properties` to verify tensor has valid values
   - Method: `ensure_tensor_format` to convert to proper data types

### Phase 2: HOPLS Algorithm Core
**Week 3-4 Tasks**:

1. **Create the HOPLSAlgorithm class**
   - Class name: `HOPLSAlgorithm`
   - Method: `extract_tensor_component` to extract one HOPLS component
   - Method: `initialize_y_scores` to start iteration with Y column
   - Method: `run_hopls_iteration` to perform one HOPLS iteration loop
   - The iteration calls: tensor weight computation, tensor scores, Y weights, Y scores, convergence check

2. **Create the TensorComponentCalculator class**
   - Class name: `TensorComponentCalculator`
   - Method: `compute_tensor_weights` (implements TensorWeightComputation)
   - Method: `compute_tensor_scores` (implements TensorScoreComputation)
   - Method: `compute_y_weights` (implements YWeightComputation)
   - Method: `compute_y_scores` (implements YScoreComputation)
   - Method: `compute_tensor_loadings` (implements TensorLoadingComputation)
   - Method: `compute_y_loadings` (implements YLoadingComputation)

3. **Create the HOPLSConvergenceMonitor class**
   - Class name: `HOPLSConvergenceMonitor`
   - Method: `check_hopls_convergence` to monitor algorithm convergence
   - Method: `calculate_score_difference` to compute convergence metric
   - Instance variables: `tolerance`, `max_iterations`, `current_iteration`, `convergence_history`

### Phase 3: Tensor Deflation and Multiple Components
**Week 5-6 Tasks**:

1. **Create the TensorDeflator class**
   - Class name: `TensorDeflator`
   - Method: `deflate_tensor` to remove component from tensor
   - Method: `deflate_y_matrix` to remove component from Y matrix
   - Method: `compute_tensor_deflation_term` to calculate deflation amount
   - Method: `verify_deflation_orthogonality` to check deflation worked correctly

2. **Create the MultiTensorComponentExtractor class**
   - Class name: `MultiTensorComponentExtractor`
   - Method: `extract_all_tensor_components` to extract multiple components
   - Method: `extract_and_deflate_tensor` to extract one component then deflate
   - Uses: `HOPLSAlgorithm`, `TensorDeflator`, `TensorComponentOrganizer`

3. **Create the TensorComponentOrganizer class**
   - Class name: `TensorComponentOrganizer`
   - Method: `initialize_tensor_storage` to create empty component storage
   - Method: `store_tensor_component` to add new tensor component
   - Method: `organize_into_tensor_matrices` to create final W, P, Q, T, U structures
   - Method: `verify_tensor_orthogonality` to check components are orthogonal

### Phase 4: Tensor Prediction and Validation
**Week 7-8 Tasks**:

1. **Create the HOPLSPredictor class**
   - Class name: `HOPLSPredictor`
   - Method: `transform_tensor` to convert tensor to HOPLS space
   - Method: `predict_from_tensor` to generate Y predictions from tensor
   - Method: `validate_prediction_tensor` to check new tensor compatibility

2. **Create the TensorModelValidator class**
   - Class name: `TensorModelValidator`
   - Method: `test_on_synthetic_tensors` with known tensor-matrix relationships
   - Method: `compare_with_tensor_pls_methods` to validate against existing methods
   - Method: `test_tensor_edge_cases` for various tensor shapes and properties

3. **Create the HOPLSEvaluationMetrics class**
   - Class name: `HOPLSEvaluationMetrics`
   - Method: `calculate_tensor_r_squared` for model performance
   - Method: `tensor_cross_validate` for model validation
   - Method: `select_tensor_components` for choosing optimal components
   - Method: `analyze_tensor_variable_importance` for interpretation

## Testing Approach

### Unit Testing Strategy
**Tensor operation tests**: Verify unfolding/refolding operations preserve information
**Tensor centering tests**: Check centering works correctly for 3D structures
**Weight computation tests**: Verify tensor weights have correct properties
**Score computation tests**: Check tensor score calculations and shapes
**Loading computation tests**: Verify tensor loading calculations
**Deflation tests**: Ensure tensor deflation preserves orthogonality

### Integration Testing Strategy
**Single tensor component test**: Extract one component from synthetic tensor data
**Multi-component test**: Extract multiple components, verify orthogonality
**Tensor prediction test**: Fit on training tensors, predict on test tensors
**Memory efficiency test**: Verify algorithm handles large tensors efficiently

### Validation Testing Strategy
**Synthetic tensor data test**: Create tensors with known structure, verify recovery
**Comparison with matrix PLS**: When tensor reduces to matrix, results should match
**Real tensor dataset test**: Use actual 3D datasets, compare with existing methods
**Cross-validation test**: Verify tensor model selection works correctly

## File Organization

### Directory Structure to Create
Create these directories and files:
- **src/hopls/** (main package directory)
- **src/hopls/core.py** (main HOPLSRegression class)
- **src/hopls/tensor_utils.py** (tensor manipulation functions)
- **src/hopls/algorithms.py** (HOPLS algorithm implementations)
- **src/hopls/metrics.py** (evaluation functions)
- **tests/** (test directory)
- **tests/test_hopls_core.py** (test main functionality)
- **tests/test_tensor_utils.py** (test tensor operations)
- **tests/test_algorithms.py** (test algorithm components)
- **examples/** (usage examples)
- **examples/synthetic_tensor_data.py** (synthetic data generation)
- **examples/hopls_basic_usage.py** (simple demonstration)

### Core Module (core.py) Structure
**What to include**:
- **HOPLSRegression class**: Main user interface class
  - Methods: `__init__`, `fit`, `transform`, `predict`, `score`
  - Instance variables: `W_tensor_`, `P_tensor_`, `Q_`, `T_`, `U_`, `x_centering_params_`, `y_means_`, `is_fitted`
- **HOPLSAlgorithm class**: Core algorithm implementation
- **MultiTensorComponentExtractor class**: Multiple component extraction
- **TensorComponentOrganizer class**: Component storage and organization

### Tensor Utils Module (tensor_utils.py) Structure
**What to include**:
- **TensorHandler class**: Tensor manipulation functionality
  - Methods: `unfold_tensor`, `refold_matrix`, `vectorize_tensor`, `tensorize_vector`
- **TensorCenterer class**: Tensor centering functionality
  - Methods: `calculate_tensor_means`, `center_tensor`, `restore_tensor_means`
- **TensorValidator class**: Tensor validation functionality
  - Methods: `validate_tensor_matrix_input`, `check_tensor_properties`, `ensure_tensor_format`
- **TensorDeflator class**: Tensor deflation operations
  - Methods: `deflate_tensor`, `deflate_y_matrix`, `compute_tensor_deflation_term`

### Algorithms Module (algorithms.py) Structure
**What to include**:
- **TensorComponentCalculator class**: Individual component calculations
  - Methods: `compute_tensor_weights`, `compute_tensor_scores`, `compute_y_weights`, `compute_y_scores`
- **HOPLSConvergenceMonitor class**: Algorithm convergence tracking
  - Methods: `check_hopls_convergence`, `calculate_score_difference`
- **HOPLSPredictor class**: Prediction operations
  - Methods: `transform_tensor`, `predict_from_tensor`, `validate_prediction_tensor`

### Metrics Module (metrics.py) Structure
**What to include**:
- **HOPLSEvaluationMetrics class**: Model evaluation functionality
  - Methods: `calculate_tensor_r_squared`, `tensor_cross_validate`, `select_tensor_components`
- **TensorModelValidator class**: Testing and validation
  - Methods: `test_on_synthetic_tensors`, `compare_with_tensor_pls_methods`, `test_tensor_edge_cases`

## Implementation Tips

### Start with Simple Tensors
- Begin with small 3D tensors (e.g., 10×5×3)
- Get tensor unfolding/refolding working perfectly first
- Test each tensor operation individually before combining

### Handle Memory Efficiently
- Tensors can be large - be mindful of memory usage
- Consider in-place operations where possible
- Use appropriate data types (float32 vs float64)

### Debug Tensor Operations Systematically
- Print tensor shapes at each step
- Verify tensor unfolding preserves all information
- Check that deflation actually removes components

### Build Incrementally
- Get 1-component HOPLS working before multiple components
- Verify each phase works before moving to next
- Compare with matrix PLS on simple cases

### Validate Tensor Mathematics
- Ensure tensor operations are mathematically correct
- Test on cases where you know the answer
- Verify orthogonality properties are maintained

This comprehensive plan provides everything needed to build HOPLS from scratch. The tensor operations add complexity, but following this step-by-step approach will lead to a complete, working implementation.
