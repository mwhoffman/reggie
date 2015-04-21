"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as spop
import mwhutils.linalg as la

from ._core import Inference

__all__ = ['Laplace']


MAXIT = 60


class Laplace(Inference):
    def init(self):
        super(Laplace, self).init()
        self.L = None
        self.a = None

    def update(self, X, y):
        # create a likelihood function so we don't pass around y
        likefun = lambda f: self.like.get_logprob(y, f)

        # grab the kernel, mean, and initialize the weights
        K = self.kern.get_kernel(X)
        m = self.mean.get_function(X)
        a = np.zeros_like(K.shape[1])

        # get the current mode, its likelihood, and the relevant derivatives
        f = np.dot(K, a) + m
        p, g, W = likefun(f)
        W = np.sqrt(-W)

        def psi(a):
            r = np.dot(K, a)
            p, _, _ = likefun(r+m)
            return 0.5 * np.inner(r, a) - np.sum(p)

        for i in xrange(MAXIT):
            # take a single step
            L = la.cholesky(la.add_diagonal(np.outer(W, W)*K, 1))
            b = W**2 * (f-m) + g
            d = b - a - W*la.solve_cholesky(L, W*np.dot(K, b))
            s = spop.brent(lambda s: psi(a+s*d), tol=1e-4, maxiter=12)

            # update the parameters
            a += s*d
            f = np.dot(K, a) + m
            p, g, W = likefun(f)
            W = np.sqrt(-W)
