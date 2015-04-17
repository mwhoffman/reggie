"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as la
import mwhutils.random as random

from ..likelihoods._core import Likelihood
from ..kernels._core import Kernel
from ..functions._core import Function
from .gpinference._core import Inference

from .. import likelihoods
from .. import kernels
from .. import functions

from ._core import Model
from .gpinference import Exact

__all__ = ['GP', 'BasicGP']


class GP(Model):
    """
    Implementation of GP inference.
    """
    def __init__(self, like, kern, mean, post):
        self._like = self._register('like', like, Likelihood)
        self._kern = self._register('kern', kern, Kernel)
        self._mean = self._register('mean', mean, Function)
        self._post = self._register('post', post, Inference)

    def _update(self):
        if self.ndata == 0:
            self._post.init()
        else:
            self._post.update(self._like, self._kern, self._mean,
                              self._X, self._Y)

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0
        else:
            return (self._post.lZ, self._post.dlZ) if grad else self._post.lZ

    def _predict(self, X, joint=False, grad=False):
        # get the prior mean and variance
        mu = self._mean.get_function(X)
        s2 = self._kern.get_kernel(X) if joint else self._kern.get_dkernel(X)

        # if we have data compute the posterior.
        if self.ndata > 0:
            if hasattr(self._post, 'U'):
                K = self._kern.get_kernel(self._post.U, X)
                V1 = la.solve_triangular(self._post.L1, K)
                V2 = la.solve_triangular(self._post.L2, K)
                mu += np.dot(V2.T, self._post.a)
                s2 += (np.dot(V2.T, V2) - np.dot(V1.T, V1)) if joint else \
                      (np.sum(V2**2, axis=0) - np.sum(V1**2, axis=0))

            else:
                K = self._kern.get_kernel(self._X, X)
                V = la.solve_triangular(self._post.L, K)
                mu += np.dot(V.T, self._post.a)
                s2 -= np.dot(V.T, V) if joint else np.sum(V**2, axis=0)

        if not grad:
            return mu, s2

        if joint:
            raise ValueError('cannot compute gradients of joint predictions')

        # Get the prior gradients. NOTE: this assumes a stationary kernel.
        dmu = self._mean.get_gradx(X)
        ds2 = np.zeros_like(X)

        if self.ndata > 0:
            if hasattr(self._post, 'U'):
                m = self._post.U.shape[0]
                dK = np.rollaxis(self._kern.get_gradx(X, self._post.U), 1)
                dK = dK.reshape(m, -1)

                # compute the mean
                dV1 = la.solve_triangular(self._post.L1, dK)
                dV2 = la.solve_triangular(self._post.L2, dK)
                dmu += np.dot(dV2.T, self._post.a).reshape(X.shape)

                # compute the variance
                dV1 = np.rollaxis(np.reshape(dV1, (m,) + X.shape), 2)
                dV2 = np.rollaxis(np.reshape(dV2, (m,) + X.shape), 2)
                ds2 += 2 * np.sum(dV2 * V2, axis=1).T
                ds2 -= 2 * np.sum(dV1 * V1, axis=1).T

            else:
                dK = np.rollaxis(self._kern.get_gradx(X, self._X), 1)
                dK = dK.reshape(self.ndata, -1)
                dV = la.solve_triangular(self._post.L, dK)
                dmu += np.dot(dV.T, self._post.a).reshape(X.shape)
                dV = np.rollaxis(np.reshape(dV, (-1,) + X.shape), 2)
                ds2 -= 2 * np.sum(dV * V, axis=1).T

        return mu, s2, dmu, ds2

    def sample(self, X, size=None, latent=True, rng=None):
        mu, Sigma = self._predict(X, joint=True)
        rng = random.rstate(rng)

        L = la.cholesky(la.add_diagonal(Sigma, 1e-10))
        m = 1 if (size is None) else size
        n = len(X)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), L.T)

        if latent is False:
            f = self._like.sample(f.ravel(), rng).reshape(f.shape)
        if size is None:
            f = f.ravel()
        return f

    def predict(self, X, grad=False):
        if grad:
            return self._predict(X, grad=True)
        else:
            return self._predict(X)


class BasicGP(GP):
    """
    Thin wrapper around exact GP inference which only provides for Iso or ARD
    kernels with constant mean.
    """
    def __init__(self, sn2, rho, ell, mean=0.0, ndim=None, kernel='se'):
        # create the mean/likelihood objects
        like = likelihoods.Gaussian(sn2)
        mean = functions.Constant(mean)

        # create a kernel object which depends on the string identifier
        kern = (
            kernels.SE(rho, ell, ndim) if (kernel == 'se') else
            kernels.Matern(rho, ell, 1, ndim) if (kernel == 'matern1') else
            kernels.Matern(rho, ell, 3, ndim) if (kernel == 'matern3') else
            kernels.Matern(rho, ell, 5, ndim) if (kernel == 'matern5') else
            None)

        if kernel is None:
            raise ValueError('Unknown kernel type')

        super(BasicGP, self).__init__(like, kern, mean, Exact())

        # flatten the parameters and rename them
        self._rename({'like.sn2': 'sn2',
                      'kern.rho': 'rho',
                      'kern.ell': 'ell',
                      'mean.bias': 'mean'})

    def __repr__(self):
        kwargs = {}
        if self._kern._iso:
            kwargs['ndim'] = self._kern.ndim
        if isinstance(self._kern, kernels.Matern):
            kwargs['kernel'] = 'matern{:d}'.format(self._kern._d)
        return super(BasicGP, self).__repr__(**kwargs)
