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
            lp, d1, d2, d3 = self.like.get_logprob(y, r+m)
            psi = 0.5 * np.inner(r, a) - np.sum(lp)
            return psi, r, d1, d2, d3

        psi1, r, dy1, dy2, dy3 = psi(a)
        psi0 = np.inf

        for _ in xrange(MAXIT):
            # attempt to breakout early
            if np.abs(psi1 - psi0) < MINTOL:
                break
            psi0 = psi1

            # find the step direction
            w = np.sqrt(-dy2)
            L = la.cholesky(la.add_diagonal(np.outer(w, w)*K, 1))
            b = w**2 * r + dy1

            # find the step size
            delta = b - a - w*la.solve_cholesky(L, w*np.dot(K, b))
            s = spop.brent(lambda s: psi(a+s*delta)[0], tol=1e-4, maxiter=12)

            # update the parameters
            a += s*delta
            psi1, r, dy1, dy2, dy3 = psi(a)

        # update the posterior parameters
        w = np.sqrt(-dy2)
        L = la.cholesky(la.add_diagonal(np.outer(w, w)*K, 1))

        # compute the marginal log-likelihood
        lZ = -psi1 - np.sum(np.log(np.diag(L)))

        # compute parameters needed for the hyperparameter gradients
        R = w * la.solve_cholesky(L, np.diag(w))
        C = la.solve_triangular(L, w*K)
        g = 0.5 * (np.diag(K) - np.sum(C**2, axis=0))
        f = r+m
        df = g * dy3

        # define the implicit part of the gradients
        implicit = lambda b: np.dot(df, b - np.dot(K, np.dot(R, b)))

        # allocate space for the gradients
        dlZ = np.zeros(self.nparams)

        # the likelihood derivatives
        i = 0
        for dl0, dl1, dl2 in self.like.get_grad(y, f):
            dlZ[i] = np.dot(g, dl2) + np.sum(dl0)
            dlZ[i] += implicit(np.dot(K, dl1))
            i += 1

        # covariance derivatives
        for dK in self.kern.get_grad(X):
            dlZ[i] = 0.5 * (np.dot(a, np.dot(dK, a)) - np.sum(R*dK))
            dlZ[i] += implicit(np.dot(dK, dy1))
            i += 1

        # mean derivatives
        for dm in self.mean.get_grad(X):
            dlZ[i] = np.dot(dm, a) + implicit(dm)
            i += 1

        self.L = L
        self.a = a
        self.w = w
        self.lZ = lZ
        self.dlZ = dlZ
