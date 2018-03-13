from numba import jit
import numpy as np


@jit(nopython=True)
def matrix_multiply(X, Y):
    """ Matrix multiplication
    Inputs:
      - X: A numpy array of shape (N, M)
      - Y: A numpy array of shape (M, K)
    Output:
      - out: A numpy array of shape (N, K)
    """
    out = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float64)

    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            val = 0

            for k in range(Y.shape[0]):
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
    out = np.zeros((X.shape[0]), dtype=np.float64)

    for i in range(X.shape[0]):
        val = 0

        for j in range(X.shape[1]):
            if weights is None:
                val += X[i, j]
            else:
                val += X[i, j] * weights[j]

        out[i] = val / X.shape[1]

    return out


@jit(nopython=True)
def arr_mean(X):
    out = 0.0

    for i in range(X.size):
        out += X[i]

    out /= X.size

    return out


@jit(nopython=True)
def arr_std(X):
    mean = arr_mean(X)

    out = np.zeros(X.shape)

    for i in range(X.size):
        out[i] = X[i]

    for i in range(X.size):
        out[i] -= mean
        out[i] = abs(out[i]) ** 2

    return arr_mean(out) ** 0.5


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
    out = np.zeros(X.shape, dtype=np.float64)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j] = X[i, j]

    x = np.zeros(X.shape[1])

    if with_mean:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x[j] = out[i, j]
            for j in range(X.shape[1]):
                out[i, j] -= arr_mean(x)

    if with_std:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x[j] = out[i, j]
            for j in range(X.shape[1]):
                out[i, j] /= arr_std(x)

    if top_n < 0:
        top_n = 0
    elif top_n > X.shape[1]:
        top_n = X.shape[1]

    for j in range(X.shape[0]):
        for q in range(X.shape[1]):
            x[q] = out[j, q]

        args = np.argsort(x)

        for i in range(X.shape[1] - top_n):
            out[j, args[i]] = 0

    v = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v[i] += out[i, j] ** 2

        v[i] = v[i] ** 0.5

    OUT = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for l in range(X.shape[1]):
                OUT[i, j] += out[i, l] * out[j, l]

            OUT[i, j] /= v[i] * v[j]

    return OUT
