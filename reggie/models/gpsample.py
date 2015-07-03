"""
Approximate finite-dimensional samples from a GP.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as la
import mwhutils.random as random


class FourierSample(object):
    """
    Encapsulation of a continuous function sampled from a Gaussian process
    where this infinitely-parameterized object is approximated using a weighted
    sum of finitely many Fourier samples.
    """
    def __init__(self, gp, n, rng=None):
        rng = random.rstate(rng)

        # randomize the feature
        W, a = gp.kern.sample_spectrum(n, rng)

        self._W = W
        self._b = rng.rand(n) * 2 * np.pi
        self._a = np.sqrt(2*a/n)
        self._mean = gp.mean.copy()
        self._theta = None

        if gp.ndata > 0:
            X, Y = gp.data
            Z = np.dot(X, self._W.T) + self._b
            Phi = np.cos(Z) * self._a

            # get the components for regression
            A = np.dot(Phi.T, Phi)
            A = la.add_diagonal(A, gp.like.get_variance())

            L = la.cholesky(A)
            r = Y - self._mean.get_function(X)
            p = np.sqrt(gp.like.get_variance()) * rng.randn(n)

            self._theta = la.solve_cholesky(L, np.dot(Phi.T, r))
            self._theta += la.solve_triangular(L, p, True)

        else:
            self._theta = rng.randn(n)

    def __call__(self, x, grad=False):
        if grad:
            F, G = self.get(x, True)
            return F[0], G[0]
        else:
            return self.get(x)[0]

    def get(self, X, grad=False):
        X = np.array(X, ndmin=2, copy=False)
        Z = np.dot(X, self._W.T) + self._b

        F = self._mean.get_function(X)
        F += np.dot(self._a * np.cos(Z), self._theta)

        if not grad:
            return F

        d = (-self._a * np.sin(Z))[:, :, None] * self._W[None]
        G = np.einsum('ijk,j', d, self._theta)

        return F, G
