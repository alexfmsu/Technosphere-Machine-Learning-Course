import numpy as np


cpdef primes(int kmax):
    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result

cpdef matrix_multiply2(X, Y):
    """ Matrix multiplication
    Inputs:
      - X: A numpy array of shape (N, M)
      - Y: A numpy array of shape (M, K)
    Output:
      - out: A numpy array of shape (N, K)
    """
    cdef np.int N = X.shape[0]
    cdef np.int M = Y.shape[0]
    cdef np.int K = Y.shape[1]

    cdef np.ndarray[np.float32_t, ndim = 2] out

    out = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        for j in range(K):
            val = .0

            for k in range(M):
                val += X[i, k] * Y[k, j]

            out[i, j] = val

    return out


cpdef matrix_rowmean(X, weights=np.empty(0)):
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
    if weights.shape == (0,):
        weights = np.ones(X.shape[1], dtype=np.float32)

    cdef np.float32_t s = 0

    for i in range(X.shape[1]):
        s += weights[i]

    cdef np.ndarray[np.float32_t, ndim = 1] out = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i] += X[i, j] * weights[j]
        out[i] /= s

    return out

cpdef arr_mean(np.ndarray[np.float32_t, ndim=1] X):
    cdef np.float32_t res = float(0)
    for i in range(X.size):
        res += X[i]

    res /= X.size
    return res

cpdef arr_std(np.ndarray[np.float32_t, ndim=1] X):
    cdef np.float32_t mean = arr_mean(X)

    cdef np.ndarray[np.float32_t, ndim = 1] tmp
    tmp = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(X.size):
        tmp[i] = X[i]

    for i in range(X.size):
        tmp[i] -= mean
        tmp[i] = abs(tmp[i]) ** 2

    return arr_mean(tmp) ** 0.5


cpdef cosine_similarity(X, top_n=10, with_mean=True, with_std=True):
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
    cdef np.ndarray[np.float32_t, ndim = 2] tmp
    tmp = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tmp[i, j] = X[i, j]

    cdef np.ndarray[np.float32_t, ndim = 1] x

    x = np.zeros(X.shape[1], dtype=np.float32)

    if with_mean:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x[j] = tmp[i, j]
            for j in range(X.shape[1]):
                tmp[i, j] -= arr_mean(x)

    if with_std:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x[j] = tmp[i, j]
            for j in range(X.shape[1]):
                tmp[i, j] /= arr_std(x)

    if top_n < 0:
        top_n = 0
    elif top_n > X.shape[1]:
        top_n = X.shape[1]

    for j in range(X.shape[0]):
        for q in range(X.shape[1]):
            x[q] = tmp[j, q]

        args = np.argsort(x)

        for i in range(X.shape[1] - top_n):
            tmp[j, args[i]] = 0

    cdef np.ndarray[np.float32_t, ndim = 1] v

    v = np.zeros(X.shape[0], dtype=np.float32)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v[i] += tmp[i, j] ** 2
        v[i] = v[i] ** 0.5

    cdef np.ndarray[np.float32_t, ndim = 2] res
    res = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for l in range(X.shape[1]):
                res[i, j] += tmp[i, l] * tmp[j, l]
            res[i, j] /= v[i] * v[j]

    return res
