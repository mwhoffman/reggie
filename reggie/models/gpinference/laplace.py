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
MINTOL = 1e-6


class Laplace(Inference):
    def update(self, X, y):
        # grab the kernel, mean, and initialize the weights
        K = self.kern.get_kernel(X)
        L = None
        m = self.mean.get_function(X)
        a = np.zeros(K.shape[1])

        def psi(a):
            # define the linesearch objective
            r = np.dot(K, a)
            p, g, w = self.like.get_logprob(y, r+m)
            w = np.sqrt(-w)
            psi = 0.5 * np.inner(r, a) - np.sum(p)
            return psi, r, p, g, w

        psi1, r, p, g, w = psi(a)
        psi0 = np.inf

        for _ in xrange(MAXIT):
            # attempt to breakout early
            if np.abs(psi1 - psi0) < MINTOL:
                break
            psi0 = psi1

            # take a single step
            L = la.cholesky(la.add_diagonal(np.outer(w, w)*K, 1))
            b = w**2 * r + g
            d = b - a - w*la.solve_cholesky(L, w*np.dot(K, b))
            s = spop.brent(lambda s: psi(a+s*d)[0], tol=1e-4, maxiter=12)

            # update the parameters
            a += s*d
            psi1, r, p, g, w = psi(a)

        lZ = -psi1 - np.sum(np.log(np.diag(L)))

        self.L = L
        self.a = a
        self.w = w
