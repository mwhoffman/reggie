"""
Exact inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.domains import POSITIVE
from ..core.models import Model
from ..utils import linalg
from ..kernels.kernel import Kernel
from ..functions.function import Function

from ..kernels import SE
from ..functions import Constant

__all__ = ['ExactGP', 'BasicGP']


class ExactGP(Model):
    """
    Implementation of exact GP inference.
    """
    def __init__(self, sn2, kernel, mean):
        self._sn2 = self._register('sn2', sn2, domain=POSITIVE)
        self._kernel = self._register('kernel', kernel, Kernel)
        self._mean = self._register('mean', mean, Function)

        # cached sufficient statistics
        self._L = None
        self._a = None

    def _update(self):
        if self.ndata > 0:
            K = self._kernel.get_kernel(self._X)
            K = linalg.add_diagonal(K, self._sn2)
            r = self._Y - self._mean.get_function(self._X)
            self._L = linalg.cholesky(K)
            self._a = linalg.solve_triangular(self._L, r)

        else:
            self._L = None
            self._a = None

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0

        lZ = -0.5 * np.inner(self._a, self._a)
        lZ -= 0.5 * np.log(2 * np.pi) * self.ndata
        lZ -= np.sum(np.log(self._L.diagonal()))

        if not grad:
            return lZ

        alpha = linalg.solve_triangular(self._L, self._a, trans=1)
        Q = linalg.cholesky_inverse(self._L) - np.outer(alpha, alpha)

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

    def get_posterior(self, X, grad=False, predictive=False):
        # grab the prior mean and variance.
        mu = self._mean.get_function(X)
        s2 = self._kernel.get_dkernel(X)

        if self.ndata > 0:
            K = self._kernel.get_kernel(self._X, X)
            LK = linalg.solve_triangular(self._L, K)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(LK.T, self._a)
            s2 -= np.sum(LK**2, axis=0)

        if predictive:
            s2 += self._sn2

        if not grad:
            return mu, s2

        # Get the prior gradients. NOTE: this assumes a stationary kernel.
        dmu = None
        ds2 = np.zeros_like(X)

        if hasattr(self._mean, 'get_gradx'):
            # if the mean has real-valued inputs then it should define this
            # method to get the gradient of the mean function wrt its inputs.
            dmu = self._mean.get_gradx(X)
        else:
            # however, constant functions can take any inputs, not just
            # real-valued. but their gradients are zeros anyway.
            dmu = np.zeros_like(X)

        if self.ndata > 0:
            dK = np.rollaxis(self._kernel.get_gradx(X, self._X), 1)
            dK = dK.reshape(self.ndata, -1)

            LdK = linalg.solve_triangular(self._L, dK)
            dmu += np.dot(LdK.T, self._a).reshape(X.shape)

            LdK = np.rollaxis(np.reshape(LdK, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(LdK * LK, axis=1).T

        return mu, s2, dmu, ds2


class BasicGP(ExactGP):
    """
    Thin wrapper around exact GP inference which only provides for Iso or ARD
    kernels with constant mean.
    """
    def __init__(self, sn2, rho, ell, mean=0.0, ndim=None):
        super(BasicGP, self).__init__(sn2, SE(rho, ell, ndim), Constant(mean))

        # flatten the parameters and rename them
        self._flatten({'kernel.rho': 'rho',
                       'kernel.ell': 'ell',
                       'mean.bias': 'mean'})

    def __repr__(self):
        if self._kernel._iso:
            return super(BasicGP, self).__repr__(ndim=self._kernel.ndim)
        else:
            return super(BasicGP, self).__repr__()
