# HOPLS Theory and Mathematical Foundation

This document provides comprehensive theoretical background for Higher Order Partial Least Squares (HOPLS) regression, explaining the mathematical principles and algorithmic approach.

## Table of Contents
- [Mathematical Foundation](#mathematical-foundation)
- [Tensor Operations Theory](#tensor-operations-theory)
- [HOPLS Algorithm Theory](#hopls-algorithm-theory)
- [Comparison with PLS](#comparison-with-pls)
- [Advanced Topics](#advanced-topics)

## Mathematical Foundation

### Tensor Basics

#### What is a Tensor?
A tensor is a multi-dimensional array that generalizes vectors (1D) and matrices (2D) to higher dimensions. In HOPLS, we primarily work with 3-way tensors.

**Notation:**
- **Scalar**: a (0-way tensor)
- **Vector**: **a** (1-way tensor, bold lowercase)
- **Matrix**: **A** (2-way tensor, bold uppercase)
- **3-way Tensor**: 𝒳 (script letter)

#### Tensor Dimensions and Modes
A 3-way tensor 𝒳 ∈ ℝ^(I×J×K) has three modes:
- **Mode 1**: I elements (often samples/observations)
- **Mode 2**: J elements (often features/variables)
- **Mode 3**: K elements (often time points/conditions)

### Tensor Unfolding (Matricization)

#### Mode-n Unfolding
Tensor unfolding reshapes a tensor into a matrix by fixing one mode and arranging the others.

**Mode-1 unfolding**: 𝒳₍₁₎ ∈ ℝ^(I×JK)
- Rows: Mode 1 (samples)
- Columns: Modes 2 and 3 combined (features×conditions)

**Mode-2 unfolding**: 𝒳₍₂₎ ∈ ℝ^(J×IK)
- Rows: Mode 2 (features)
- Columns: Modes 1 and 3 combined (samples×conditions)

**Mode-3 unfolding**: 𝒳₍₃₎ ∈ ℝ^(K×IJ)
- Rows: Mode 3 (conditions)
- Columns: Modes 1 and 2 combined (samples×features)

#### Vectorization
The vectorization operation vec(𝒳) creates a column vector by stacking all tensor elements in a consistent order.

### Tensor Products and Contractions

#### Mode-n Product
The mode-n product between tensor 𝒳 ∈ ℝ^(I₁×I₂×...×Iₙ×...×Iₙ) and matrix **A** ∈ ℝ^(J×Iₙ) is:

𝒴 = 𝒳 ×ₙ **A**

This operation multiplies the tensor with the matrix along mode n.

#### Tensor Contraction
For HOPLS score computation, we use tensor contraction:
t_i = 𝒳ᵢ : **W**

Where : denotes the Frobenius inner product (element-wise multiplication then sum).

## Tensor Operations Theory

### Centering 3-Way Tensors

#### Global Centering
Subtract the overall mean from all elements:
𝒳ᶜᵉⁿᵗ = 𝒳 - mean(𝒳)

#### Mode-Specific Centering
Center across specific modes while preserving structure:

**Mode-1 centering** (across samples):
𝒳ᶜᵉⁿᵗᵢⱼₖ = 𝒳ᵢⱼₖ - (1/I)∑ᵢ 𝒳ᵢⱼₖ

**Mode-2 centering** (across features):
𝒳ᶜᵉⁿᵗᵢⱼₖ = 𝒳ᵢⱼₖ - (1/J)∑ⱼ 𝒳ᵢⱼₖ

**Mode-3 centering** (across conditions):
𝒳ᶜᵉⁿᵗᵢⱼₖ = 𝒳ᵢⱼₖ - (1/K)∑ₖ 𝒳ᵢⱼₖ

### Tensor Rank and Decomposition

#### CP Decomposition
The CANDECOMP/PARAFAC (CP) decomposition represents a tensor as sum of rank-1 tensors:

𝒳 ≈ ∑ᵣ₌₁ᴿ λᵣ **aᵣ** ∘ **bᵣ** ∘ **cᵣ**

Where ∘ denotes the outer product and λᵣ are scaling factors.

#### Tucker Decomposition
Tucker decomposition represents a tensor as:

𝒳 ≈ 𝒢 ×₁ **A** ×₂ **B** ×₃ **C**

Where 𝒢 is the core tensor and **A**, **B**, **C** are factor matrices.

## HOPLS Algorithm Theory

### Problem Formulation

#### Input Structure
- **Tensor predictor**: 𝒳 ∈ ℝ^(n×p×q) (samples × features × modes)
- **Matrix response**: **Y** ∈ ℝ^(n×m) (samples × targets)

#### Objective
Find tensor weight matrix **W** ∈ ℝ^(p×q) and response weight vector **c** ∈ ℝᵐ that maximize:

cov(𝒳 : **W**, **Y****c**)

Subject to: ‖**W**‖_F = 1 and ‖**c**‖ = 1

### HOPLS Components

#### Tensor Weights
The tensor weight matrix **W** defines how to combine tensor elements to create scores:
tᵢ = 𝒳ᵢ : **W** = ∑ⱼ∑ₖ 𝒳ᵢⱼₖ Wⱼₖ

#### Tensor Scores
Tensor scores project the tensor data onto the weight direction:
**t** = [t₁, t₂, ..., tₙ]ᵀ

#### Response Weights
Response weights **c** combine response variables:
uᵢ = ∑ⱼ Yᵢⱼ cⱼ

#### Response Scores
Response scores project response data onto weight direction:
**u** = [u₁, u₂, ..., uₙ]ᵀ

### NIPALS for Tensors

#### Initialization
1. Initialize **u** with first column of **Y**
2. Set convergence tolerance ε and maximum iterations

#### Iteration Loop
Repeat until convergence:

1. **Compute tensor weights**:
   **W** = 𝒳₍₁₎ᵀ **u** (reshaped to p×q)
   **W** = **W**/‖**W**‖_F

2. **Compute tensor scores**:
   tᵢ = 𝒳ᵢ : **W**

3. **Compute response weights**:
   **c** = **Y**ᵀ**t**/(**t**ᵀ**t**)
   **c** = **c**/‖**c**‖

4. **Compute response scores**:
   **u**ⁿᵉʷ = **Y****c**

5. **Check convergence**:
   If ‖**u**ⁿᵉʷ - **u**‖ < ε, stop
   Else **u** = **u**ⁿᵉʷ, continue

### Tensor Loadings

#### Tensor Loadings Computation
After convergence, compute tensor loadings:
**P** = (**t**ᵀ**t**)⁻¹ **t**ᵀ 𝒳₍₁₎ (reshaped to p×q)

#### Response Loadings Computation
Response loadings:
**q** = **Y**ᵀ**t**/(**t**ᵀ**t**)

### Tensor Deflation

#### Tensor Deflation Formula
Remove component from tensor:
𝒳ⁿᵉʷ = 𝒳 - **t** ∘ **P**

Where ∘ denotes tensor outer product: (**t** ∘ **P**)ᵢⱼₖ = tᵢ Pⱼₖ

#### Response Deflation Formula
Remove component from response:
**Y**ⁿᵉʷ = **Y** - **t****q**ᵀ

### Multi-Component HOPLS

#### Component Extraction Loop
For h = 1, 2, ..., H components:

1. Extract component h using NIPALS
2. Store: **Wₕ**, **tₕ**, **cₕ**, **uₕ**, **Pₕ**, **qₕ**
3. Deflate: 𝒳 and **Y**
4. Repeat with deflated data

#### Final Model Organization
Organize components into:
- **Weight tensor**: 𝒲 ∈ ℝ^(p×q×H)
- **Loading tensor**: 𝒫 ∈ ℝ^(p×q×H)
- **Score matrix**: **T** ∈ ℝ^(n×H)
- **Response loading matrix**: **Q** ∈ ℝ^(m×H)

## Comparison with PLS

### Similarities
- **Iterative algorithm**: Both use NIPALS-type iterations
- **Deflation procedure**: Both remove found components
- **Covariance maximization**: Both maximize covariance between predictor and response
- **Dimensionality reduction**: Both create lower-dimensional representations

### Key Differences

#### Data Structure
- **PLS**: Matrix predictor **X** ∈ ℝ^(n×p)
- **HOPLS**: Tensor predictor 𝒳 ∈ ℝ^(n×p×q)

#### Weight Structure
- **PLS**: Weight vector **w** ∈ ℝᵖ
- **HOPLS**: Weight matrix **W** ∈ ℝ^(p×q)

#### Score Computation
- **PLS**: **t** = **X****w** (matrix-vector product)
- **HOPLS**: tᵢ = 𝒳ᵢ : **W** (tensor contraction)

#### Deflation
- **PLS**: **X** - **t****p**ᵀ (rank-1 matrix update)
- **HOPLS**: 𝒳 - **t** ∘ **P** (rank-1 tensor update)

### When to Use HOPLS vs PLS

#### Use HOPLS when:
- Data has natural 3-way structure (samples × features × conditions)
- Want to preserve multi-way relationships
- Need to analyze mode-specific patterns
- Data is naturally tensorial (e.g., time-varying features)

#### Use PLS when:
- Data is naturally 2-way (samples × features)
- Computational efficiency is critical
- Simpler interpretation is desired
- Limited tensor analysis expertise

## Advanced Topics

### Computational Complexity

#### Memory Requirements
- **PLS**: O(np + nm) for data storage
- **HOPLS**: O(npq + nm) for data storage
- **HOPLS advantage**: Often npq << n(pq) for unfolded approach

#### Time Complexity
- **PLS component**: O(np + nm) per iteration
- **HOPLS component**: O(npq + nm) per iteration
- **Scaling**: HOPLS scales linearly with tensor size

### Theoretical Properties

#### Convergence Guarantees
HOPLS inherits convergence properties from PLS:
- Algorithm converges to stationary point
- Convergence rate is linear
- Unique solution for each component (up to sign)

#### Optimality
Each HOPLS component solves:
max **W**,**c** cov(𝒳:**W**, **Y****c**)
subject to ‖**W**‖_F = ‖**c**‖ = 1

#### Orthogonality
HOPLS components are orthogonal:
- **tₕ**ᵀ**tₕ'** = 0 for h ≠ h'
- Tensor scores are mutually orthogonal
- Enables additive model interpretation

### Extensions and Variations

#### Multi-way PLS (N-PLS)
Generalization to N-way tensors:
- Handle 4-way, 5-way, etc. tensors
- Multiple tensor inputs
- Tensor-tensor regression

#### Sparse HOPLS
Add sparsity constraints:
- L1 regularization on weights
- Automatic feature selection
- Improved interpretation

#### Robust HOPLS
Handle outliers:
- Robust loss functions
- Iteratively reweighted algorithms
- Outlier detection

### Implementation Considerations

#### Numerical Stability
- Use SVD for weight normalization
- Handle near-singular cases
- Monitor conditioning

#### Initialization Strategies
- Random initialization
- SVD-based initialization
- Warm start from previous solutions

#### Stopping Criteria
- Relative change in scores
- Absolute tolerance
- Maximum iteration limits
- Multiple criteria combination

This theoretical foundation provides the mathematical background needed to understand and implement HOPLS effectively.
