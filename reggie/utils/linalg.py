"""
Exact inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from scipy.linalg import lapack
from scipy.linalg import LinAlgError
from numpy.lib.stride_tricks import as_strided


__all__ = []


def add_diagonal(A, d, copy=False):
    if copy:
        A = A.copy()
    shape = (A.shape[0],)
    strides = ((A.shape[0]+1)*A.itemsize,)
    D = as_strided(A, shape, strides)
    D += d
    return A


def cholesky(A, maxtries=5):
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        d = A.diagonal()
        if np.any(d <= 0):
            raise LinAlgError('Matrix has non-positive diagonal elements')
        j = d.mean() * 1e-6
        for _ in xrange(maxtries):
            L, info = lapack.dpotrf(add_diagonal(A, j, True), lower=1)
            if info == 0:
                return L
            else:
                j *= 10
        raise LinAlgError('Matrix is not positive definite, even with jitter')


def cholesky_inverse(A):
    X, info = lapack.dpotri(A, lower=1)
    if info != 0:
        raise LinAlgError('Matrix is not invertible')
    triu = np.triu_indices_from(X, k=1)
    X[triu] = X.T[triu]
    return X


def solve_triangular(A, B, trans=0):
    X, info = lapack.dtrtrs(A, B, lower=1, trans=trans)
    if info != 0:
        raise LinAlgError('Matrix is singular')
    return X


def solve_cholesky(A, B):
    X, _ = lapack.dpotrs(A, B, lower=1)
    return X
