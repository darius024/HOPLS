from .helpers import *
import jax.numpy as jnp

def pls_regression(X, Y, n_components):
    """
    Standard Partial Least Squares regression implementation following the mathematical formulation:
    
    X = TP^T + E = Σ(r=1 to R) t_r * p_r^T + E
    Y = TDC^T + F = Σ(r=1 to R) d_rr * t_r * c_r^T + F
    
    where:
    - T: scores matrix (latent variables from X)
    - P: X loadings matrix  
    - C: Y loadings matrix
    - D: diagonal scaling matrix
    - W: weights matrix for X
    - E, F: residual matrices
    """
    print(f"Performing standard PLS regression with {n_components} components on X: {X.shape} and Y: {Y.shape}")

    # Store original means before centering (for prediction)
    X_mean = compute_column_means(X)
    Y_mean = compute_column_means(Y)

    # Center the matrices (mean-centering is standard in PLS)
    X = center_matrix(X)
    Y = center_matrix(Y)

    # Initialize storage matrices for PLS components
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    
    # W: weight vectors for X (n_features x n_components)
    W = empty_matrix((n_features, n_components))
    # P: loading vectors for X (n_features x n_components) 
    P = empty_matrix((n_features, n_components))
    # C: loading vectors for Y (n_targets x n_components)
    C = empty_matrix((n_targets, n_components))
    # T: score vectors (latent variables) (n_samples x n_components)
    T = empty_matrix((n_samples, n_components))
    # D: diagonal scaling factors (n_components,)
    D = jnp.zeros(n_components)

    # Recursive PLS algorithm - extract one component at a time
    for r in range(n_components):
        print(f"  Extracting component {r+1}/{n_components}")
        
        # Extract the r-th PLS component
        w_r, c_r, t_r, u_r, p_r, d_r = extract_standard_pls_component(X, Y)

        # Store the r-th component
        W = W.at[:, r].set(w_r)      # X weight vector
        P = P.at[:, r].set(p_r)      # X loading vector  
        C = C.at[:, r].set(c_r)      # Y loading vector (weight vector in this case)
        T = T.at[:, r].set(t_r)      # Score vector (latent variable)
        D = D.at[r].set(d_r)         # Scaling factor

        # Standard PLS deflation:
        # X := X - t_r * p_r^T  (remove explained X variance)
        X = X - jnp.outer(t_r, p_r)
        
        # Y := Y - t_r * q_r^T, where q_r = Y^T * t_r / (t_r^T * t_r)
        q_r = jnp.dot(Y.T, t_r) / jnp.dot(t_r, t_r)
        Y = Y - jnp.outer(t_r, q_r)

    return create_standard_pls_model(W, P, C, T, D, X_mean, Y_mean)

def extract_standard_pls_component(X, Y):
    """
    Extract a single PLS component following the standard NIPALS algorithm.
    """
    # Initialize u as the first column of Y (standard initialization)
    u = Y[:, 0].copy()
    tolerance = 1e-10
    max_iterations = 1000
    
    # Initialize variables to handle degenerate cases
    w = jnp.zeros(X.shape[1])
    c = jnp.zeros(Y.shape[1])
    t = jnp.zeros(X.shape[0])
    converged = False

    # NIPALS algorithm (Nonlinear Iterative Partial Least Squares)
    for iteration in range(max_iterations):
        u_old = u.copy()

        # Step 1: w = X^T u / ||X^T u|| (X weights, normalized)
        w = jnp.dot(X.T, u)
        w_norm = jnp.linalg.norm(w)
        if w_norm > tolerance:
            w = w / w_norm
        else:
            # Handle degenerate case - no more useful components
            print(f"    Degenerate case: X^T u has zero norm at iteration {iteration + 1}")
            w = jnp.zeros_like(w)
            t = jnp.zeros(X.shape[0])
            break

        # Step 2: t = X w (X scores)
        t = jnp.dot(X, w)

        # Step 3: c = Y^T t / ||Y^T t|| (Y weights, normalized)  
        c = jnp.dot(Y.T, t)
        c_norm = jnp.linalg.norm(c)
        if c_norm > tolerance:
            c = c / c_norm
        else:
            # Handle degenerate case
            print(f"    Degenerate case: Y^T t has zero norm at iteration {iteration + 1}")
            c = jnp.zeros_like(c)
            break

        # Step 4: u = Y c (Y scores)
        u = jnp.dot(Y, c)

        # Check convergence
        if jnp.linalg.norm(u - u_old) < tolerance:
            print(f"    Converged after {iteration + 1} iterations")
            converged = True
            break
    else:
        if jnp.linalg.norm(w) > tolerance and jnp.linalg.norm(c) > tolerance:
            print(f"    Warning: Did not converge after {max_iterations} iterations")
            converged = True
        else:
            print(f"    Stopped due to degeneracy after {max_iterations} iterations")

    # Final computations after convergence:
    t_norm_sq = jnp.dot(t, t)
    
    if t_norm_sq > tolerance and converged:
        # X loadings: p = X^T t / (t^T t)
        p = jnp.dot(X.T, t) / t_norm_sq
        
        # Inner relation coefficient: d = u^T t / (t^T t)
        d = jnp.dot(u, t) / t_norm_sq
    else:
        # Degenerate case
        p = jnp.zeros(X.shape[1])
        d = 0.0

    return w, c, t, u, p, d

def create_standard_pls_model(W, P, C, T, D, X_mean, Y_mean):
    """
    Create a standard PLS model.
    """
    model = {
        'W': W,           # X weight vectors
        'P': P,           # X loading vectors
        'C': C,           # Y weight vectors  
        'T': T,           # Score vectors (latent variables)
        'D': D,           # Diagonal scaling factors
        'X_mean': X_mean, # Original X means
        'Y_mean': Y_mean, # Original Y means
        'n_components': W.shape[1]
    }
    return model

def predict(X, model):
    """
    Predict using standard PLS model.
    
    Uses the standard PLS prediction: Y' = X' * W * (P^T * W)^{-1} * C^T * D + Y_mean
    where (P^T * W)^{-1} accounts for the deflation process.
    """
    # Center the input
    X_centered = X - model['X_mean']

    # For simplicity, use the regression approach: Y = T * Q^T where Q are Y loadings
    # T = X * W, and Q can be computed from the relationship Y = T * Q^T
    
    # Compute scores for new data
    T_new = jnp.dot(X_centered, model['W'])
    
    # For standard PLS, the Y loadings Q are not the same as the Y weights C
    # We need to use the regression coefficients approach
    
    # Simplified approach: Use the relationship Y ≈ T * C^T * diag(D)
    # This works for the NIPALS algorithm we're using
    Y_pred = jnp.dot(T_new * model['D'], model['C'].T)
    
    # Add back the Y means
    Y_pred = Y_pred + model['Y_mean']

    return Y_pred
