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
        super(Exact, self).init()
        self.L = None
        self.a = None

    def update(self, X, Y):
        K = la.add_diagonal(self.kern.get_kernel(X), self.like.get_variance())
        r = Y - self.mean.get_function(X)

        # the posterior parameterization
        L = la.cholesky(K)
        a = la.solve_cholesky(L, r)
        Q = la.cholesky_inverse(L) - np.outer(a, a)

        # the log-likelihood
        lZ = -0.5 * np.inner(a, r)
        lZ -= 0.5 * np.log(2 * np.pi) * len(X)
        lZ -= np.sum(np.log(L.diagonal()))

        dlZ = np.r_[
            # derivative wrt the likelihood's noise term.
            -0.5*np.trace(Q),

            # derivative wrt each kernel hyperparameter.
            [-0.5*np.sum(Q*dK) for dK in self.kern.get_grad(X)],

            # derivative wrt the mean.
            [np.dot(dmu, a) for dmu in self.mean.get_grad(X)]]

        self.L = L
        self.a = a
        self.lZ = lZ
        self.dlZ = dlZ
