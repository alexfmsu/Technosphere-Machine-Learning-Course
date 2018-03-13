from numpy.linalg import norm
import numpy as np


def matrix_multiply(X, Y):
    """ Matrix multiplication
    Inputs:
      - X: A numpy array of shape (N, M)
      - Y: A numpy array of shape (M, K)
    Output:
      - out: A numpy array of shape (N, K)
    """
    out = np.dot(X, Y)

    return out


def matrix_rowmean(X, weights=np.empty(0)):
    """ Calculate mean of each row.
    In case of weights do weighted mean.
    For example, for matrix [[1, 2, 3]] and weights [0, 1, 2]
    weighted mean equals 2.6666 (while ordinary mean equals 2)
    Inputs:
      - X: A numpy array of shape (N, M)
      - weights: A numpy array of shape (M,) or an emty array of shape (0,)
    Output:
      - out: A numpy array of shape (N,)
    """
    if len(np.shape(weights)) == 1 and weights.shape[0]:
        if weights.shape[0] == X.shape[1]:
            out = np.average(X, weights=weights, axis=1)
        else:
            print("Error: size mismatch")
            exit(1)
    else:
        out = np.mean(X, axis=1)

    return out


def cosine_row_similarity(row, X):
    return np.dot(X, row) / (norm(row) * np.apply_along_axis(norm, 1, X))


def cosine_similarity(X, top_n=10, with_mean=True, with_std=True):
    """ Calculate cosine similarity between each pair of row.
    1. In case of with_mean: subtract mean of each row from row
    2. In case of with_std: divide each row on it's std
    3. Select top_n best elements in each row or set other to zero.
    4. Compute cosine similarity between each pair of rows.
    Inputs:
      - X: A numpy array of shape (N, M)
      - top_n: int, number of best elements in each row
      - with_mean: bool, in case of subtracting each row's mean
      - with_std: bool, in case of subtracting each row's std
    Output:
      - out: A numpy array of shape (N, N)

    Example (with top_n=1, with_mean=True, with_std=True):
        X = array([[1, 2], [4, 3]])
        after mean and std transform:
        X = array([[-1.,  1.], [ 1., -1.]])
        after top n choice
        X = array([[0.,  1.], [ 1., 0]])
        cosine similarity:
        X = array([[ 1.,  0.], [ 0.,  1.]])

    """
    if with_mean:
        X = X - np.mean(X, axis=1).reshape(-1, 1)

    if with_std:
        X = X / np.std(X, axis=1).reshape(-1, 1)

    i = np.arange(X.shape[0]).reshape(-1, 1)
    j = np.argsort(X, axis=1)[:, :-top_n]

    X[i, j] = 0

    out = np.apply_along_axis(cosine_row_similarity, 1, X, X)

    return out

a = np.array([[1, 2, 11], [4, 3, 12]])
b = np.array([[5, 6], [7, 8], [9, 10]])
print(matrix_multiply(a, b))
print()

a = np.array([[1, 2, 3]])
weights = np.array([0, 1, 2])
print(matrix_rowmean(a, weights))
print(matrix_rowmean(a))
print()

a = np.array([[1, 2], [4, 3]])
print(cosine_similarity(a, top_n=1))
print()
