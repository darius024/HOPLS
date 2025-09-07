from .helpers import *

def pls_regression(X, Y, n_components):
    """
    Standard PLS regression using manual matrix operations (basic implementation).
    """
    print(f"Performing standard PLS regression (basic) with {n_components} components on X: {X.shape} and Y: {Y.shape}")

    # Store original means before centering
    X_mean = compute_column_means(X)
    Y_mean = compute_column_means(Y)

    # Center the matrices
    X = center_matrix(X)
    Y = center_matrix(Y)

    # Initialize storage matrices
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    W = empty_matrix((n_features, n_components))
    P = empty_matrix((n_features, n_components))
    C = empty_matrix((n_targets, n_components))
    T = empty_matrix((n_samples, n_components))

    # Main recursive loop
    for i in range(n_components):
        print(f"  Extracting component {i+1}/{n_components}")
        
        # Extract the i-th PLS component
        w, c, t, u, p, d = extract_standard_pls_component(X, Y)

        # Store the component
        W = W.at[:, i].set(w)
        P = P.at[:, i].set(p)
        C = C.at[:, i].set(c)
        T = T.at[:, i].set(t)

        # Deflate matrices
        X = deflate_matrix(X, t, p)
        Y = deflate_matrix(Y, t, c)

    return create_model(W, P, C, T, X_mean, Y_mean)

def extract_standard_pls_component(X, Y):
    """
    Extract a single PLS component using manual matrix operations (basic implementation).
    """
    # Initialize Y score vector u with first column of Y
    u = Y[:, 0]
    tolerance = 1e-10
    max_iterations = 1000

    # Power iteration to solve optimization problem
    for iteration in range(max_iterations):
        u_old = u

        # Step 1: Compute X weight vector w = X^T * u, normalize
        w = matrix_vector_multiply(transpose(X), u)
        w = normalise_vector(w)

        # Step 2: Compute X score vector t = X * w
        t = matrix_vector_multiply(X, w)

        # Step 3: Compute Y weight vector c = Y^T * t, normalize
        c = matrix_vector_multiply(transpose(Y), t)
        c = normalise_vector(c)

        # Step 4: Compute Y score vector u = Y * c
        u = matrix_vector_multiply(Y, c)

        # Check convergence
        if (vector_norm(u - u_old) < tolerance):
            print(f"    Converged after {iteration + 1} iterations")
            break
    else:
        print(f"    Warning: Did not converge after {max_iterations} iterations")

    # Final computations:
    # X loading vector: p = X^T * t / (t^T * t)
    t_norm_sq = vector_dot(t, t)
    t_t_inv = 1.0 / t_norm_sq
    p = matrix_vector_multiply(transpose(X), t)

    # Manual scalar multiplication for basic implementation
    for i in range(p.shape[0]):
        p = p.at[i].set(p[i] * t_t_inv)
    
    # Scaling factor: d = u^T * t / (t^T * t)  
    d = vector_dot(u, t) * t_t_inv

    return w, c, t, u, p, d

def predict(X, model):
    """
    Predict using standard PLS model with manual operations.
    """
    # Center input
    X_centered = X - model['X_mean']

    # Compute scores T = X_centered * W
    T = matrix_multiply(X_centered, model['W'])

    # For backward compatibility, use 'Q' if 'C' not available
    if 'C' in model:
        loadings = model['C']
    else:
        loadings = model['Q']
    
    # Predict: Y = T * C^T
    Y_pred = matrix_multiply(T, transpose(loadings))

    # Add back Y means
    Y_pred = Y_pred + model['Y_mean']

    return Y_pred
