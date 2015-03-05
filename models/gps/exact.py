"""
Exact inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla

from ..kernels.kernel import Kernel
from ..functions.function import Function
from ..core.models import PosteriorModel
from ..core.transforms import Log

__all__ = ['ExactGP']


### BASE KERNEL INTERFACE #####################################################

class ExactGP(PosteriorModel):
    """
    Implementation of exact GP inference.
    """
    def __init__(self, sn2, kernel, mean):
        self._sn2 = self._register('sn2', sn2, transform=Log())
        self._kernel = self._register('kernel', kernel, Kernel)
        self._mean = self._register('mean', mean, Function)

        # cached sufficient statistics
        self._R = None
        self._a = None

    def _update(self):
        if self.ndata > 0:
            K = self._kernel.get_kernel(self._X)
            K = K + self._sn2 * np.eye(len(self._X))
            r = self._Y - self._mean.get_function(self._X)
            self._R = sla.cholesky(K)
            self._a = sla.solve_triangular(self._R, r, trans=True)

        else:
            self._R = None
            self._a = None

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0

        lZ = -0.5 * np.inner(self._a, self._a)
        lZ -= 0.5 * np.log(2 * np.pi) * self.ndata
        lZ -= np.sum(np.log(self._R.diagonal()))

        if not grad:
            return lZ

        alpha = sla.solve_triangular(self._R, self._a, trans=False)
        Q = sla.cho_solve((self._R, False), np.eye(self.ndata))
        Q -= np.outer(alpha, alpha)

        dlZ = np.r_[
            # derivative wrt the likelihood's noise term.
            -0.5*np.trace(Q),

            # derivative wrt each kernel hyperparameter.
            [-0.5*np.sum(Q*dK)
             for dK in self._kernel.get_grad(self._X)],

            # derivative wrt the mean.
            [np.dot(dmu, alpha)
             for dmu in self._mean.get_grad(self._X)]]

        return lZ, dlZ

    def get_posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = self._mean.get_function(X)
        s2 = self._kernel.get_dkernel(X)

        if self.ndata > 0:
            K = self._kernel.get_kernel(self._X, X)
            RK = sla.solve_triangular(self._R, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(RK.T, self._a)
            s2 -= np.sum(RK**2, axis=0)

        if not grad:
            return mu, s2

        # Get the prior gradients.
        dmu = self._mean.get_gradx(X)
        ds2 = np.zeros_like(X)

        # NOTE: the above assumes a constant mean and stationary kernel (which
        # we satisfy, but should we change either assumption...).

        if self.ndata > 0:
            dK = np.rollaxis(self._kernel.get_gradx(X, self._X), 1)
            dK = dK.reshape(self.ndata, -1)

            RdK = sla.solve_triangular(self._R, dK, trans=True)
            dmu += np.dot(RdK.T, self._a).reshape(X.shape)

            RdK = np.rollaxis(np.reshape(RdK, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(RdK * RK, axis=1).T

        return mu, s2, dmu, ds2
