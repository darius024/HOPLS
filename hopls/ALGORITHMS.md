# HOPLS Algorithms Pseudocode

This document provides pseudocode for all HOPLS algorithms and helper functions.

## Main Algorithm: HOPLS Regression

```
ALGORITHM: HOPLS_Regression(X_tensor, Y, n_components)
INPUT: 
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    Y: matrix (n_samples × n_targets)
    n_components: integer
OUTPUT:
    model: HOPLS model with all components
BEGIN
    // Data preparation
    X_tensor_mean ← compute_tensor_mean(X_tensor)
    Y_mean ← compute_column_means(Y)
    X_tensor_centered ← center_tensor(X_tensor, X_tensor_mean)
    Y_centered ← center_matrix(Y, Y_mean)
    
    // Initialize storage
    W_tensor ← empty_tensor(n_features, n_modes, n_components)
    P_tensor ← empty_tensor(n_features, n_modes, n_components)
    Q ← empty_matrix(n_targets, n_components)
    T ← empty_matrix(n_samples, n_components)
    
    X_work ← X_tensor_centered
    Y_work ← Y_centered
    
    // Extract components
    FOR h = 1 TO n_components DO
        W, c, t, u, P, q ← extract_hopls_component(X_work, Y_work)
        
        // Store component
        W_tensor[:, :, h] ← W
        P_tensor[:, :, h] ← P
        Q[:, h] ← q
        T[:, h] ← t
        
        // Tensor deflation
        X_work ← deflate_tensor(X_work, t, P)
        Y_work ← deflate_Y(Y_work, t, q)
    END FOR
    
    RETURN create_hopls_model(W_tensor, P_tensor, Q, T, X_tensor_mean, Y_mean)
END
```

## Core Algorithm: Extract Single HOPLS Component (Tensor NIPALS)

```
ALGORITHM: extract_hopls_component(X_tensor, Y)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    Y: matrix (n_samples × n_targets)
OUTPUT:
    W: tensor weight matrix (n_features × n_modes)
    c: Y-weights vector
    t: tensor scores vector
    u: Y-scores vector
    P: tensor loadings matrix (n_features × n_modes)
    q: Y-loadings vector
BEGIN
    // Initialize
    u ← first_column(Y)
    tolerance ← 1e-6
    max_iter ← 500
    
    FOR iter = 1 TO max_iter DO
        u_old ← u
        
        // Compute tensor weights
        W ← compute_tensor_weights(X_tensor, u)
        W ← normalize_tensor_matrix(W)
        
        // Compute tensor scores
        t ← compute_tensor_scores(X_tensor, W)
        
        // Y-weights and scores
        c ← matrix_vector_multiply(transpose(Y), t)
        c ← normalize_vector(c)
        u ← matrix_vector_multiply(Y, c)
        
        // Check convergence
        IF vector_norm(u - u_old) < tolerance THEN
            BREAK
        END IF
    END FOR
    
    // Compute tensor loadings
    t_norm_squared ← dot_product(t, t)
    P ← compute_tensor_loadings(X_tensor, t, t_norm_squared)
    q ← matrix_vector_multiply(transpose(Y), t) / t_norm_squared
    
    RETURN W, c, t, u, P, q
END
```

## Tensor-Specific Helper Functions

### Tensor Operations

```
ALGORITHM: compute_tensor_weights(X_tensor, u)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    u: vector (n_samples)
OUTPUT:
    W: matrix (n_features × n_modes)
BEGIN
    W ← empty_matrix(n_features, n_modes)
    FOR j = 1 TO n_features DO
        FOR k = 1 TO n_modes DO
            W[j, k] ← sum(X_tensor[i, j, k] * u[i] for i = 1 to n_samples)
        END FOR
    END FOR
    RETURN W
END

ALGORITHM: compute_tensor_scores(X_tensor, W)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    W: matrix (n_features × n_modes)
OUTPUT:
    t: vector (n_samples)
BEGIN
    t ← empty_vector(n_samples)
    FOR i = 1 TO n_samples DO
        t[i] ← tensor_contraction(X_tensor[i, :, :], W)
    END FOR
    RETURN t
END

ALGORITHM: tensor_contraction(X_slice, W)
INPUT:
    X_slice: matrix (n_features × n_modes)
    W: matrix (n_features × n_modes)
OUTPUT:
    result: scalar
BEGIN
    result ← 0
    FOR j = 1 TO n_features DO
        FOR k = 1 TO n_modes DO
            result ← result + X_slice[j, k] * W[j, k]
        END FOR
    END FOR
    RETURN result
END

ALGORITHM: compute_tensor_loadings(X_tensor, t, t_norm_squared)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    t: vector (n_samples)
    t_norm_squared: scalar
OUTPUT:
    P: matrix (n_features × n_modes)
BEGIN
    P ← empty_matrix(n_features, n_modes)
    FOR j = 1 TO n_features DO
        FOR k = 1 TO n_modes DO
            P[j, k] ← sum(X_tensor[i, j, k] * t[i] for i = 1 to n_samples) / t_norm_squared
        END FOR
    END FOR
    RETURN P
END
```

### Tensor Centering and Normalization

```
ALGORITHM: center_tensor(X_tensor, mean_value)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    mean_value: scalar (global mean)
OUTPUT:
    X_centered: 3D tensor (n_samples × n_features × n_modes)
BEGIN
    X_centered ← empty_tensor(n_samples, n_features, n_modes)
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_centered[i, j, k] ← X_tensor[i, j, k] - mean_value
            END FOR
        END FOR
    END FOR
    RETURN X_centered
END

ALGORITHM: compute_tensor_mean(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    mean_value: scalar
BEGIN
    total_sum ← 0
    total_elements ← n_samples * n_features * n_modes
    
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                total_sum ← total_sum + X_tensor[i, j, k]
            END FOR
        END FOR
    END FOR
    
    RETURN total_sum / total_elements
END

ALGORITHM: normalize_tensor_matrix(W)
INPUT:
    W: matrix (n_features × n_modes)
OUTPUT:
    W_normalized: matrix (n_features × n_modes)
BEGIN
    frobenius_norm ← 0
    FOR j = 1 TO n_features DO
        FOR k = 1 TO n_modes DO
            frobenius_norm ← frobenius_norm + W[j, k]^2
        END FOR
    END FOR
    frobenius_norm ← sqrt(frobenius_norm)
    
    IF frobenius_norm > 0 THEN
        RETURN W / frobenius_norm
    ELSE
        RETURN W
    END IF
END
```

### Tensor Unfolding Operations

```
ALGORITHM: unfold_mode_1(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_unfolded: matrix (n_samples × n_features*n_modes)
BEGIN
    X_unfolded ← empty_matrix(n_samples, n_features * n_modes)
    FOR i = 1 TO n_samples DO
        column_index ← 1
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_unfolded[i, column_index] ← X_tensor[i, j, k]
                column_index ← column_index + 1
            END FOR
        END FOR
    END FOR
    RETURN X_unfolded
END

ALGORITHM: unfold_mode_2(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_unfolded: matrix (n_features × n_samples*n_modes)
BEGIN
    X_unfolded ← empty_matrix(n_features, n_samples * n_modes)
    FOR j = 1 TO n_features DO
        column_index ← 1
        FOR i = 1 TO n_samples DO
            FOR k = 1 TO n_modes DO
                X_unfolded[j, column_index] ← X_tensor[i, j, k]
                column_index ← column_index + 1
            END FOR
        END FOR
    END FOR
    RETURN X_unfolded
END

ALGORITHM: unfold_mode_3(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_unfolded: matrix (n_modes × n_samples*n_features)
BEGIN
    X_unfolded ← empty_matrix(n_modes, n_samples * n_features)
    FOR k = 1 TO n_modes DO
        column_index ← 1
        FOR i = 1 TO n_samples DO
            FOR j = 1 TO n_features DO
                X_unfolded[k, column_index] ← X_tensor[i, j, k]
                column_index ← column_index + 1
            END FOR
        END FOR
    END FOR
    RETURN X_unfolded
END
```

### Tensor Deflation

```
ALGORITHM: deflate_tensor(X_tensor, t, P)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
    t: vector (n_samples)
    P: matrix (n_features × n_modes)
OUTPUT:
    X_deflated: 3D tensor (n_samples × n_features × n_modes)
BEGIN
    X_deflated ← empty_tensor(n_samples, n_features, n_modes)
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_deflated[i, j, k] ← X_tensor[i, j, k] - t[i] * P[j, k]
            END FOR
        END FOR
    END FOR
    RETURN X_deflated
END
```

## Advanced Tensor Centering (Mode-specific)

```
ALGORITHM: center_tensor_mode_1(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_centered: 3D tensor (n_samples × n_features × n_modes)
    means: matrix (n_features × n_modes)
BEGIN
    means ← empty_matrix(n_features, n_modes)
    X_centered ← empty_tensor(n_samples, n_features, n_modes)
    
    // Compute means across samples (mode 1)
    FOR j = 1 TO n_features DO
        FOR k = 1 TO n_modes DO
            means[j, k] ← sum(X_tensor[i, j, k] for i = 1 to n_samples) / n_samples
        END FOR
    END FOR
    
    // Center tensor
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_centered[i, j, k] ← X_tensor[i, j, k] - means[j, k]
            END FOR
        END FOR
    END FOR
    
    RETURN X_centered, means
END

ALGORITHM: center_tensor_mode_2(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_centered: 3D tensor (n_samples × n_features × n_modes)
    means: matrix (n_samples × n_modes)
BEGIN
    means ← empty_matrix(n_samples, n_modes)
    X_centered ← empty_tensor(n_samples, n_features, n_modes)
    
    // Compute means across features (mode 2)
    FOR i = 1 TO n_samples DO
        FOR k = 1 TO n_modes DO
            means[i, k] ← sum(X_tensor[i, j, k] for j = 1 to n_features) / n_features
        END FOR
    END FOR
    
    // Center tensor
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_centered[i, j, k] ← X_tensor[i, j, k] - means[i, k]
            END FOR
        END FOR
    END FOR
    
    RETURN X_centered, means
END

ALGORITHM: center_tensor_mode_3(X_tensor)
INPUT:
    X_tensor: 3D tensor (n_samples × n_features × n_modes)
OUTPUT:
    X_centered: 3D tensor (n_samples × n_features × n_modes)
    means: matrix (n_samples × n_features)
BEGIN
    means ← empty_matrix(n_samples, n_features)
    X_centered ← empty_tensor(n_samples, n_features, n_modes)
    
    // Compute means across modes (mode 3)
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            means[i, j] ← sum(X_tensor[i, j, k] for k = 1 to n_modes) / n_modes
        END FOR
    END FOR
    
    // Center tensor
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            FOR k = 1 TO n_modes DO
                X_centered[i, j, k] ← X_tensor[i, j, k] - means[i, j]
            END FOR
        END FOR
    END FOR
    
    RETURN X_centered, means
END
```

## HOPLS Prediction Algorithm

```
ALGORITHM: predict_hopls(X_tensor_new, model)
INPUT:
    X_tensor_new: 3D tensor (n_new_samples × n_features × n_modes)
    model: trained HOPLS model
OUTPUT:
    Y_pred: matrix (n_new_samples × n_targets)
BEGIN
    // Center new tensor data using training parameters
    X_centered ← center_tensor(X_tensor_new, model.X_tensor_mean)
    
    // Transform to HOPLS space
    T_new ← empty_matrix(n_new_samples, model.n_components)
    FOR h = 1 TO model.n_components DO
        FOR i = 1 TO n_new_samples DO
            T_new[i, h] ← tensor_contraction(X_centered[i, :, :], model.W_tensor[:, :, h])
        END FOR
    END FOR
    
    // Predict Y in centered space
    Y_centered_pred ← matrix_multiply(T_new, transpose(model.Q))
    
    // Add back Y means
    Y_pred ← empty_matrix(n_new_samples, n_targets)
    FOR i = 1 TO n_new_samples DO
        FOR j = 1 TO n_targets DO
            Y_pred[i, j] ← Y_centered_pred[i, j] + model.Y_mean[j]
        END FOR
    END FOR
    
    RETURN Y_pred
END
```

## Model Creation

```
ALGORITHM: create_hopls_model(W_tensor, P_tensor, Q, T, X_tensor_mean, Y_mean)
INPUT:
    W_tensor, P_tensor: component tensors
    Q, T: component matrices
    X_tensor_mean, Y_mean: centering parameters
OUTPUT:
    model: HOPLS model structure
BEGIN
    model.W_tensor ← W_tensor
    model.P_tensor ← P_tensor
    model.Q ← Q
    model.T ← T
    model.X_tensor_mean ← X_tensor_mean
    model.Y_mean ← Y_mean
    model.n_components ← size(W_tensor, 3)
    RETURN model
END
```
