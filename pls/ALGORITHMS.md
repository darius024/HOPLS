# PLS Algorithms Pseudocode

This document provides pseudocode for all PLS algorithms and helper functions.

## Main Algorithm: PLS Regression

```
ALGORITHM: PLS_Regression(X, Y, n_components)
INPUT: 
    X: matrix (n_samples × n_features)
    Y: matrix (n_samples × n_targets)
    n_components: integer
OUTPUT:
    model: PLS model with all components
BEGIN
    // Data preparation
    X_mean ← compute_column_means(X)
    Y_mean ← compute_column_means(Y)
    X_centered ← center_matrix(X, X_mean)
    Y_centered ← center_matrix(Y, Y_mean)
    
    // Initialize storage
    W ← empty_matrix(n_features, n_components)
    P ← empty_matrix(n_features, n_components)
    Q ← empty_matrix(n_targets, n_components)
    T ← empty_matrix(n_samples, n_components)
    
    X_work ← X_centered
    Y_work ← Y_centered
    
    // Extract components
    FOR h = 1 TO n_components DO
        w, c, t, u, p, q ← extract_pls_component(X_work, Y_work)
        
        // Store component
        W[:, h] ← w
        P[:, h] ← p
        Q[:, h] ← q
        T[:, h] ← t
        
        // Deflation
        X_work ← deflate_X(X_work, t, p)
        Y_work ← deflate_Y(Y_work, t, q)
    END FOR
    
    RETURN create_model(W, P, Q, T, X_mean, Y_mean)
END
```

## Core Algorithm: Extract Single PLS Component (NIPALS)

```
ALGORITHM: extract_pls_component(X, Y)
INPUT:
    X: matrix (n_samples × n_features)
    Y: matrix (n_samples × n_targets)
OUTPUT:
    w: X-weights vector
    c: Y-weights vector
    t: X-scores vector
    u: Y-scores vector
    p: X-loadings vector
    q: Y-loadings vector
BEGIN
    // Initialize
    u ← first_column(Y)
    tolerance ← 1e-6
    max_iter ← 500
    
    FOR iter = 1 TO max_iter DO
        u_old ← u
        
        // X-weights and scores
        w ← matrix_vector_multiply(transpose(X), u)
        w ← normalize_vector(w)
        t ← matrix_vector_multiply(X, w)
        
        // Y-weights and scores
        c ← matrix_vector_multiply(transpose(Y), t)
        c ← normalize_vector(c)
        u ← matrix_vector_multiply(Y, c)
        
        // Check convergence
        IF vector_norm(u - u_old) < tolerance THEN
            BREAK
        END IF
    END FOR
    
    // Compute loadings
    t_norm_squared ← dot_product(t, t)
    p ← matrix_vector_multiply(transpose(X), t) / t_norm_squared
    q ← matrix_vector_multiply(transpose(Y), t) / t_norm_squared
    
    RETURN w, c, t, u, p, q
END
```

## Helper Functions

### Data Centering

```
ALGORITHM: center_matrix(X, means)
INPUT:
    X: matrix (n_samples × n_features)
    means: vector (n_features)
OUTPUT:
    X_centered: matrix (n_samples × n_features)
BEGIN
    X_centered ← X
    FOR j = 1 TO n_features DO
        X_centered[:, j] ← X[:, j] - means[j]
    END FOR
    RETURN X_centered
END

ALGORITHM: compute_column_means(X)
INPUT:
    X: matrix (n_samples × n_features)
OUTPUT:
    means: vector (n_features)
BEGIN
    means ← empty_vector(n_features)
    FOR j = 1 TO n_features DO
        means[j] ← sum(X[:, j]) / n_samples
    END FOR
    RETURN means
END
```

### Vector Operations

```
ALGORITHM: normalize_vector(v)
INPUT:
    v: vector
OUTPUT:
    v_normalized: vector
BEGIN
    norm ← sqrt(sum(v[i]^2 for all i))
    IF norm > 0 THEN
        RETURN v / norm
    ELSE
        RETURN v
    END IF
END

ALGORITHM: vector_norm(v)
INPUT:
    v: vector
OUTPUT:
    norm: scalar
BEGIN
    RETURN sqrt(sum(v[i]^2 for all i))
END

ALGORITHM: dot_product(v1, v2)
INPUT:
    v1, v2: vectors of same length
OUTPUT:
    result: scalar
BEGIN
    RETURN sum(v1[i] * v2[i] for all i)
END
```

### Matrix Operations

```
ALGORITHM: matrix_vector_multiply(A, v)
INPUT:
    A: matrix (m × n)
    v: vector (n)
OUTPUT:
    result: vector (m)
BEGIN
    result ← empty_vector(m)
    FOR i = 1 TO m DO
        result[i] ← sum(A[i, j] * v[j] for j = 1 to n)
    END FOR
    RETURN result
END

ALGORITHM: transpose(A)
INPUT:
    A: matrix (m × n)
OUTPUT:
    A_T: matrix (n × m)
BEGIN
    A_T ← empty_matrix(n, m)
    FOR i = 1 TO m DO
        FOR j = 1 TO n DO
            A_T[j, i] ← A[i, j]
        END FOR
    END FOR
    RETURN A_T
END
```

### Deflation Operations

```
ALGORITHM: deflate_X(X, t, p)
INPUT:
    X: matrix (n_samples × n_features)
    t: vector (n_samples)
    p: vector (n_features)
OUTPUT:
    X_deflated: matrix (n_samples × n_features)
BEGIN
    // Compute outer product t * p^T and subtract from X
    X_deflated ← X
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_features DO
            X_deflated[i, j] ← X[i, j] - t[i] * p[j]
        END FOR
    END FOR
    RETURN X_deflated
END

ALGORITHM: deflate_Y(Y, t, q)
INPUT:
    Y: matrix (n_samples × n_targets)
    t: vector (n_samples)
    q: vector (n_targets)
OUTPUT:
    Y_deflated: matrix (n_samples × n_targets)
BEGIN
    // Compute outer product t * q^T and subtract from Y
    Y_deflated ← Y
    FOR i = 1 TO n_samples DO
        FOR j = 1 TO n_targets DO
            Y_deflated[i, j] ← Y[i, j] - t[i] * q[j]
        END FOR
    END FOR
    RETURN Y_deflated
END
```

## Prediction Algorithm

```
ALGORITHM: predict(X_new, model)
INPUT:
    X_new: matrix (n_new_samples × n_features)
    model: trained PLS model
OUTPUT:
    Y_pred: matrix (n_new_samples × n_targets)
BEGIN
    // Center new data using training means
    X_centered ← center_matrix(X_new, model.X_mean)
    
    // Transform to PLS space
    T_new ← matrix_multiply(X_centered, model.W)
    
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
ALGORITHM: create_model(W, P, Q, T, X_mean, Y_mean)
INPUT:
    W, P, Q, T: component matrices
    X_mean, Y_mean: centering parameters
OUTPUT:
    model: PLS model structure
BEGIN
    model.W ← W
    model.P ← P
    model.Q ← Q
    model.T ← T
    model.X_mean ← X_mean
    model.Y_mean ← Y_mean
    model.n_components ← number_of_columns(W)
    RETURN model
END
```
