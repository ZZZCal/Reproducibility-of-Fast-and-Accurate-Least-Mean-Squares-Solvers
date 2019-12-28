"""
"""

import numpy as np
import scipy.linalg as linalg


def caratheodory(P, u=None, size=None, tol=1e-8, dtype=np.float64):
    """
    Construct a Caratheordoy-set.
    Input: a set of points P as a numpy array
    Output: a new vector of size |P| of the computed weights
    """
    (n, d) = P.shape

    if u is None:
        u = np.ones(n, dtype)
    else:
        u = u.astype(dtype)

    u.reshape(-1,1)

    mask = u != 0
    n = np.sum(mask)

    while n > (size or d+1):
        A = P[mask]
        v = np.linalg.svd((A[:-1] - A[-1]).T, full_matrices=True)[2][-1]
        v = np.append(v, -np.sum(v))

        positive = v > 0
        w = u[mask]
        w = w - np.min(w[positive]/v[positive]) * v
        w[np.isclose(w, 0, atol=tol)] = 0
        u[mask] = w
        # unrolled recursive step
        mask = u != 0
        n = np.sum(mask)

    return u


def fast_caratheodory(P, u=None, k = None, size=None, tol=1e-8, dtype=np.float64):
    """
    Construct a Caratheodory-set using a novel approach described in
    'Fast and Accurate Least-Mean-Square Solvers (2019)'.
    Input: a set of points P as a numpy array.
    Output: a vector of size |P| of the computed weights.
    """
    n, d = P.shape
    n_ = n # save original size

    if u is None:
        u = np.ones(n, dtype)
    else:
        u = u.astype(dtype)

    mask = u != 0
    P = P[mask]
    u = u[mask]
    idx = np.arange(n_, dtype=np.int)[mask] # idx of the computed weights

    n = np.sum(mask)

    # default fastest value for k
    if not k:
        k = 2 *d + 2
    elif k <= d + 1:
        return u
    k_ = k

    # scaling, see original implementation
    u_sum = np.sum(u)
    u /= u_sum

    while n > (size or d+1):
        # discretize cluster count and size
        cluster_size = int(np.ceil(n/k_))
        k = int(np.ceil(n/cluster_size))
        # fill data to match cluster size
        fill = cluster_size - n % cluster_size
        if fill != cluster_size:
            P = np.append(P, np.zeros((fill, d), dtype))
            u = np.append(u, np.zeros(fill, dtype))
            idx = np.append(idx, np.zeros(fill, dtype=np.int32))
        # partition into clusters
        clusters = P.reshape(k, cluster_size, d)
        cluster_weights = u.reshape(k, cluster_size)
        cluster_idx = idx.reshape(k, cluster_size)

        # weighted means of the clusters
        means = np.einsum('ijk,ij->ik', clusters, cluster_weights)
        # call to caratheodory using weighted cluster means
        w = caratheodory(means, np.ones(k, dtype), None, tol, dtype)

        cluster_mask = w != 0
        P = clusters[cluster_mask].reshape(-1,d)
        u = (cluster_weights[cluster_mask] * w[cluster_mask][:, np.newaxis]).reshape(-1)
        idx = cluster_idx[cluster_mask].reshape(-1)

        n = u.shape[0]

    u_new = np.zeros(n_, dtype)
    u_new[idx] = u
    u_new *= u_sum
    return u_new


def coreset(X, y=None, weights=None, k=None, size=None, tol=1e-8, dtype=np.float64):
    """
    Computes a corset for the given data and weights.
    The result preserves the covariance matrix of the input set.
    """
    n, d = X.shape
    P = X[:,:,np.newaxis].astype(dtype)
    P = np.einsum('ikj,ijk->ijk', P, P)
    P = P.reshape(n,(d)**2)
    #P = np.append(X, y[:,np.newaxis], axis=1)[:,:,np.newaxis].astype(dtype)
    #P = np.einsum('ikj,ijk->ijk', P, P)
    #P = P.reshape(n,(d+1)**2)

    w = np.sqrt(fast_caratheodory(P, weights, k, size, tol, dtype))
    idx_mask = w != 0

    if y is not None:
        return X[idx_mask], y[idx_mask], w[idx_mask][:,np.newaxis]
    else:
        return X[idx_mask], w[idx_mask][:,np.newaxis]



def main():
    from ablation import load_datasets
    from Booster import linregcoreset

    import time
    def timeit(f, *args, **kwargs):
        """ function used to time runs"""
        t0 = time.perf_counter()
        ret = f(*args, **kwargs)
        return ret, time.perf_counter() - t0

    import pandas as pd

    data_range = 100

    df0 = pd.DataFrame(columns=('n','d','k','ratio','time'))

    # test re-implementation
    for n in range(250000, 2250000, 250000):
        u = np.ones(n)
        for d in (3, 5, 7):
            A = np.floor(np.random.rand(n, d) * data_range)
            for k in (2*(d**2)+2, 4*(d**2)+4, 8*(d**2)+8):
                for _ in range(5):
                    (C, w), t = timeit(lambda: coreset(A, weights=u, k=k))
                    S = w * C
                    assert(np.allclose(A.T @ A, S.T @ S)) # test theorem

                    row = (n, d, k, C.shape[0] / A.shape[0], t)
                    df0.loc[len(df0)] = row

                    print(row)

    df1 = pd.DataFrame(columns=('n','d','k','ratio','time'))

    # test original implementation
    for n in range(250000, 2250000, 250000):
        u = np.ones(n)
        for d in (3, 5, 7):
            A = np.floor(np.random.rand(n, d) * data_range)
            #for k in (2*(d**2)+2, 4*(d**2)+4, 8*(d**2)+8): can't change k
            for _ in range(5):
                (C, w), t = timeit(lambda: linregcoreset(A, u, None))
                t += timeit(lambda: np.sqrt(w))[1] # this is not done in original implementation

                row = (n, d, 2*(d**2)+2, C.shape[0] / A.shape[0], t)
                df1.loc[len(df1)] = row

                print(row)

    df0.to_csv('nani_coreset_performance.csv')
    df1.to_csv('ibrahim_coreset_performance.csv')


if __name__ == '__main__':
    import cProfile

    main()
