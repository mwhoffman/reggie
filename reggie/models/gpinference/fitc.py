"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as la

from ._core import Inference

__all__ = ['FITC']


def dott(A):
    return np.dot(A, A.T)


class FITC(Inference):
    def __init__(self, U):
        self._U = np.array(U, ndmin=2, dtype=float, copy=True)
        self.init()

    def init(self):
        self._L1 = None
        self._L2 = None
        self._a = None

    def update(self, like, kern, mean, X, Y):
        sn2 = like.get_variance()
        su2 = sn2 / 1e6

        # get the kernel matrices.
        Kux = kern.get_kernel(self._U, X)
        kxx = kern.get_dkernel(X) + sn2
        Kuu = la.add_diagonal(kern.get_kernel(self._U), su2)

        # kernel wrt the inducing points.
        self._L1 = la.cholesky(Kuu)

        # get the residuals and the cholesky of Q.
        r = Y - mean.get_function(X)
        V = la.solve_triangular(self._L1, Kux)

        # rescaling factor
        ell = np.sqrt(kxx - np.sum(V**2, axis=0))

        A = la.add_diagonal(dott(V/ell), 1)
        a = np.dot(Kux/ell, r/ell)

        # update the posterior.
        self._L2 = np.dot(self._L1, la.cholesky(A))
        self._a = la.solve_triangular(self._L2, a)
