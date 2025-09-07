import numpy as np
import jax.numpy as jnp

def parse_matrix(file_path):
    # The matrix is in the form of a numpy array stored in a text file
    # Support both comma and space delimited files
    if file_path.endswith('.csv'):
        return jnp.array(np.loadtxt(file_path, delimiter=','))
    else:
        return jnp.array(np.loadtxt(file_path))

def center_matrix(X):
    # Center the matrix by subtracting the mean of each column
    # X = X - X.mean(axis=0)
    means = compute_column_means(X)
    
    result = jnp.zeros_like(X)
    for i in range(X.shape[1]):
        result = result.at[:, i].set(X[:, i] - means[i])

    return result

def compute_column_means(X):
    # Compute the mean of each column in the matrix
    means = jnp.zeros(X.shape[1])
    for i in range(X.shape[1]):
        means = means.at[i].set(jnp.mean(X[:, i]))

    return means

def empty_matrix(shape):
    # Create an empty matrix of the given shape
    return jnp.zeros(shape)

def deflate_matrix(X, t, p):
    # Deflate the matrix X by removing the information explained by the component (t, p)
    # Compute the outer product of t and p and subtract it from X
    # X = X - jnp.outer(t, p)

    result = jnp.zeros_like(X)
    for i in range(X.shape[1]):
        result = result.at[:, i].set(X[:, i] - t * p[i])

    return result

def create_model(W, P, Q, T, X_mean, Y_mean):
    # Create a model object to store the PLS regression results
    model = {
        'W': W,
        'P': P,
        'Q': Q,
        'T': T,
        'X_mean': X_mean,
        'Y_mean': Y_mean,
        'n_components': W.shape[1]
    }
    return model

def normalise_vector(v):
    # Normalize a vector to have unit length
    norm = vector_norm(v)
    if norm == 0:
        return v
    return v / norm

def transpose(A):
    # Transpose a matrix A
    # jnp.transpose(A)
    result = jnp.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result = result.at[j, i].set(A[i, j])
    return result

def vector_dot(u, v):
    # Compute the dot product of two vectors
    # jnp.dot(u, v)
    sum = 0.0
    for i in range(u.shape[0]):
        sum += u[i] * v[i]
    return sum

def vector_norm(v):
    # Compute the Euclidean norm of a vector
    # jnp.sqrt(vector_dot(v, v))
    norm = 0.0
    for i in range(v.shape[0]):
        norm += v[i] * v[i]
    return jnp.sqrt(norm)

def matrix_vector_multiply(A, v):
    # Multiply a matrix A by a vector v
    # jnp.dot(A, v)
    result = jnp.zeros(A.shape[0])
    for i in range(A.shape[0]):
        sum = 0.0
        for j in range(A.shape[1]):
            sum += A[i, j] * v[j]
        result = result.at[i].set(sum)
    return result

def matrix_multiply(A, B):
    # Multiply two matrices A and B
    # jnp.dot(A, B)
    result = jnp.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[i, k] * B[k, j]
            result = result.at[i, j].set(sum)
    return result
