"""
Exact inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla

from ..core.models import PosteriorModel
from ..kernels.kernel import Kernel

__all__ = ['ExactGP']


### BASE KERNEL INTERFACE #####################################################

class ExactGP(PosteriorModel):
    """
    Implementation of exact GP inference.
    """
    def __init__(self, logsn, kernel, mean=0):
        self._register('logsn', logsn, ndim=0)
        self._register('kernel', kernel, Kernel)
        self._register('mean', mean, ndim=0)

        # cached sufficient statistics
        self._R = None
        self._a = None

    def _update(self):
        sn2 = np.exp(self._logsn*2)
        K = self.kernel.get_kernel(self._X, self._X)
        K = K + sn2 * np.eye(len(self._X))
        r = self._Y - self._mean
        self._R = sla.cholesky(K)
        self._a = sla.solve_triangular(self._R, r, trans=True)

    def _get_loglike(self):
        lZ = -0.5 * np.inner(self._a, self._a)
        lZ -= 0.5 * np.log(2 * np.pi) * self.ndata
        lZ -= np.sum(np.log(self._R.diagonal()))

        alpha = sla.solve_triangular(self._R, self._a, trans=False)
        Q = sla.cho_solve((self._R, False), np.eye(self.ndata))
        Q -= np.outer(alpha, alpha)

        sn2 = np.exp(self._logsn*2)
        dlZ = np.r_[
            # derivative wrt the likelihood's noise term.
            -sn2 * np.trace(Q),

            # derivative wrt each kernel hyperparameter.
            [-0.5*np.sum(Q*dK)
             for dK in self.kernel.get_grad(self._X, self._X)],

            # derivative wrt the mean.
            np.sum(alpha)]

        return lZ, dlZ

    def get_posterior(self, X):
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self._mean)
        s2 = self.kernel.get_dkernel(X)

        if self._X is not None:
            K = self.kernel.get_kernel(self._X, X)
            RK = sla.solve_triangular(self._R, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(RK.T, self._a)
            s2 -= np.sum(RK**2, axis=0)

        return mu, s2
