"""
Linear algebra helpers.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings

from scipy.linalg import lapack
from scipy.linalg import LinAlgError
from numpy.lib.stride_tricks import as_strided


__all__ = ['add_diagonal', 'cholesky', 'cholesky_inverse', 'cholesky_update',
           'solve_triangular', 'solve_cholesky']


def add_diagonal(A, d, copy=True):
    """
    Return the matrix `A` where its diagonal has had the vector `d` added to
    it. If `copy` is False perform the addition in place.
    """
    if copy:
        A = A.copy()
    shape = (A.shape[0],)
    strides = ((A.shape[0]+1)*A.itemsize,)
    D = as_strided(A, shape, strides)
    D += d
    return A


def cholesky(A, maxtries=5):
    """
    Compute the cholesky of A. If the matrix is singular make `maxtries`
    additional attempts to invert it by adding small amounts to the diagonal
    before giving up.
    """
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
                message = 'jitter of {:s} required to compute the cholesky'
                warnings.warn(message.format(str(j)), stacklevel=2)
                return L
            else:
                j *= 10
        raise LinAlgError('Matrix is not positive definite, even with jitter')


def cholesky_update(L, B, C, x=None, c=None):
    """
    Update the cholesky decomposition of a growing matrix.

    This computes the lower-triangular cholesky decomposition of [A B'; B C]
    where L, the cholesky decomposition of A, has been previously computed.
    """
    # grow the new cholesky
    B = solve_triangular(L, B.T).T
    C = cholesky(C - np.dot(B, B.T))
    L = np.r_[np.c_[L, np.zeros_like(B.T)], np.c_[B, C]]

    if x is None:
        return L

    else:
        x = np.r_[x, solve_triangular(C, c - np.dot(B, x))]
        return L, x


def cholesky_inverse(A):
    """
    Compute the inverse of the matrix AA' given its cholesky decomposition A.
    """
    X, info = lapack.dpotri(A, lower=1)
    if info != 0:
        raise LinAlgError('Matrix is not invertible')
    triu = np.triu_indices_from(X, k=1)
    X[triu] = X.T[triu]
    return X


def solve_triangular(A, B, trans=False):
    """
    Solve the system Ax=B where A is lower triangular. If `trans` is True then
    solve the system A'x=B.
    """
    X, info = lapack.dtrtrs(A, B, lower=1, trans=int(trans))
    if info != 0:
        raise LinAlgError('Matrix is singular')
    return X


def solve_cholesky(A, B):
    """
    Solve the system Mx=B where A is the lower-triangular cholesky
    decomposition of the matrix M.
    """
    X, _ = lapack.dpotrs(A, B, lower=1)
    return X
