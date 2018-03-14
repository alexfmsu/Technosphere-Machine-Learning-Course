from numba import jit
import numpy as np
from math import sqrt


@jit(nopython=True)
def matrix_multiply(X, Y):
    """ Matrix multiplication
    Inputs:
      - X: A numpy array of shape (N, M)
      - Y: A numpy array of shape (M, K)
    Output:
      - out: A numpy array of shape (N, K)
    """
    if X.shape[1] != Y.shape[0]:
        print("Error: size mismatch")
        return None

    N = X.shape[0]
    M = Y.shape[0]
    K = Y.shape[1]

    out = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        for j in range(K):
            val = .0

            for k in range(M):
                val += X[i, k] * Y[k, j]

            out[i, j] = val

    return out


@jit(nopython=True)
def matrix_rowmean(X, weights=np.empty(0)):
    """ Calculate mean of each row.
    In case of weights do weighted mean.
    For example, for matrix [[1, 2, 3]] and weights [0, 1, 2]
    weighted mean equals 2.6666 (while ordinary mean equals 2)
    Inputs:
      - X: A numpy array of shape (N, M)
      - weights: A numpy array of shape (M,)
    Output:
      - out: A numpy array of shape (N,)
    """
    M = X.shape[0]
    N = X.shape[1]

    out = np.zeros(M, dtype=np.float32)

    weights_tmp = np.zeros(N, dtype=np.float32)

    if weights.shape == (0,):
        weights_tmp = np.ones(N, dtype=np.float32)
    else:
        for i in range(len(weights)):
            weights_tmp[i] = weights[i]
    w = .0

    for i in range(N):
        w += weights_tmp[i]

    for i in range(M):
        val = .0

        for j in range(N):
            if weights is None:
                val += X[i, j]
            else:
                val += X[i, j] * weights_tmp[j]

        out[i] = val / w

    return out


@jit(nopython=True)
def array_mean(X):
    out = X.sum() / len(X)

    return out


@jit(nopython=True)
def array_std(X):
    mean = array_mean(X)

    out = np.zeros(len(X), dtype=np.float32)

    for i in range(len(X)):
        out[i] = (X[i] - mean) ** 2

    out = sqrt(array_mean(out))

    return out


@jit(nopython=True)
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
    M = X.shape[0]
    N = X.shape[1]

    x_tmp = np.zeros(X.shape, dtype=np.float32)

    x_row = np.zeros(N)
    x_col = np.zeros(M)

    out = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        for j in range(N):
            x_tmp[i, j] = X[i, j]

    if with_mean:
        for i in range(M):
            for j in range(N):
                x_row[j] = x_tmp[i, j]

            for j in range(N):
                x_tmp[i, j] -= array_mean(x_row)

    if with_std:
        for i in range(M):
            for j in range(N):
                x_row[j] = x_tmp[i, j]

            for j in range(N):
                x_tmp[i, j] /= array_std(x_row)

    for j in range(M):
        for p in range(N):
            x_row[p] = x_tmp[j, p]

        args = np.argsort(x_row)

        for i in range(N - top_n):
            x_tmp[j, args[i]] = .0

    for i in range(M):
        for j in range(N):
            x_col[i] += x_tmp[i, j] ** 2

        x_col[i] = sqrt(x_col[i])

    for i in range(M):
        for j in range(M):
            for k in range(N):
                out[i, k] += x_tmp[i, k] * x_tmp[j, k]

            out[i, j] /= x_col[i] * x_col[j]

    return out
