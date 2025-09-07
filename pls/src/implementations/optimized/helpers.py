import jax.numpy as jnp
import numpy as np

def parse_matrix(file_path):
    # The matrix is in the form of a numpy array stored in a text file
    # Support both comma and space delimited files
    if file_path.endswith('.csv'):
        return jnp.array(np.loadtxt(file_path, delimiter=','))
    else:
        return jnp.array(np.loadtxt(file_path))

def center_matrix(X):
    # Center the matrix by subtracting the mean of each column
    return X - jnp.mean(X, axis=0)

def compute_column_means(X):
    # Compute the mean of each column in the matrix
    return jnp.mean(X, axis=0)

def empty_matrix(shape):
    # Create an empty matrix of the given shape
    return jnp.zeros(shape)

def deflate_matrix(X, t, p):
    # Deflate the matrix X by removing the information explained by the component (t, p)
    # X = X - jnp.outer(t, p)
    return X - jnp.outer(t, p)

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
    norm = jnp.linalg.norm(v)
    return jnp.where(norm == 0, v, v / norm)

def transpose(A):
    # Transpose a matrix A
    return jnp.transpose(A)

def vector_dot(u, v):
    # Compute the dot product of two vectors
    return jnp.dot(u, v)

def vector_norm(v):
    # Compute the Euclidean norm of a vector
    return jnp.linalg.norm(v)

def matrix_vector_multiply(A, v):
    # Multiply a matrix A by a vector v
    return jnp.dot(A, v)

def matrix_multiply(A, B):
    # Multiply two matrices A and B
    return jnp.dot(A, B)
