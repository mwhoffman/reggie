"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as linalg
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

    def _updateinc(self, X, Y):
        try:
            self._post.updateinc(self._like, self._kern, self._mean,
                                 self._X, X, Y)
        except NotImplementedError:
            raise

    def switch_inference(self, post):
        gp = GP(self._like, self._kern, self._mean, post)
        if self.ndata > 0:
            gp.add_data(self._X, self._Y)
        return gp

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0

        lZ = -0.5 * np.inner(self._post.a, self._post.a)
        lZ -= 0.5 * np.log(2 * np.pi) * self.ndata
        lZ -= np.sum(np.log(self._post.L.diagonal()))

        if not grad:
            return lZ

        alpha = linalg.solve_triangular(self._post.L, self._post.a, trans=1)
        Q = linalg.cholesky_inverse(self._post.L) - np.outer(alpha, alpha)

        dlZ = np.r_[
            # derivative wrt the likelihood's noise term.
            -0.5*np.trace(Q),

            # derivative wrt each kernel hyperparameter.
            [-0.5*np.sum(Q*dK)
             for dK in self._kern.get_grad(self._X)],

            # derivative wrt the mean.
            [np.dot(dmu, alpha)
             for dmu in self._mean.get_grad(self._X)]]

        return lZ, dlZ

    def sample(self, X, size=None, latent=True, rng=None):
        rng = random.rstate(rng)
        m = 1 if (size is None) else size
        n = len(X)

        # get the prior.
        mu = self._mean.get_function(X)
        Sigma = self._kern.get_kernel(X)

        # get the posterior.
        if self.ndata > 0:
            K = self._kern.get_kernel(self._X, X)
            V = linalg.solve_triangular(self._post.L, K)
            mu += np.dot(V.T, self._post.a)
            Sigma -= np.dot(V.T, V)

        # compute the cholesky and sample from this multivariate Normal. Note
        # that here we add a small amount to the diagonal since this
        # corresponds to sampling from the noise-free process.
        L = linalg.cholesky(linalg.add_diagonal(Sigma, 1e-10))
        f = mu[None] + np.dot(rng.normal(size=(m, n)), L.T)

        if latent is False:
            f = self._like.sample(f.ravel(), rng).reshape(f.shape)

        if size is None:
            f = f.ravel()

        return f

    def get_posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = self._mean.get_function(X)
        s2 = self._kern.get_dkernel(X)

        if self.ndata > 0:
            K = self._kern.get_kernel(self._X, X)
            V = linalg.solve_triangular(self._post.L, K)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(V.T, self._post.a)
            s2 -= np.sum(V**2, axis=0)

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
            dK = np.rollaxis(self._kern.get_gradx(X, self._X), 1)
            dK = dK.reshape(self.ndata, -1)

            dV = linalg.solve_triangular(self._post.L, dK)
            dmu += np.dot(dV.T, self._post.a).reshape(X.shape)

            dV = np.rollaxis(np.reshape(dV, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(dV * V, axis=1).T

        return mu, s2, dmu, ds2


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
