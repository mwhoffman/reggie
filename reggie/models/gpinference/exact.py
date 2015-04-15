"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as la

from ._core import Inference

__all__ = ['Exact']


class Exact(Inference):
    def init(self):
        self.L = None
        self.a = None
        self.lZ = None
        self.dlZ = None

    def update(self, like, kern, mean, X, Y):
        K = la.add_diagonal(kern.get_kernel(X), like.get_variance())
        r = Y - mean.get_function(X)

        # the posterior parameterization
        L = la.cholesky(K)
        a = la.solve_triangular(L, r)

        # the log-likelihood
        lZ = -0.5 * np.inner(a, a)
        lZ -= 0.5 * np.log(2 * np.pi) * len(X)
        lZ -= np.sum(np.log(L.diagonal()))

        alpha = la.solve_triangular(L, a, trans=1)
        Q = la.cholesky_inverse(L)
        Q -= np.outer(alpha, alpha)

        dlZ = np.r_[
            # derivative wrt the likelihood's noise term.
            -0.5*np.trace(Q),

            # derivative wrt each kernel hyperparameter.
            [-0.5*np.sum(Q*dK)
             for dK in kern.get_grad(X)],

            # derivative wrt the mean.
            [np.dot(dmu, alpha)
             for dmu in mean.get_grad(X)]]

        self.L = L
        self.a = a
        self.lZ = lZ
        self.dlZ = dlZ
