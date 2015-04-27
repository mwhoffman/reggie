"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.linalg as la
import mwhutils.random as random

from .. import likelihoods
from .. import kernels
from .. import functions

from ._core import Model
from . import gpinference

__all__ = ['GP', 'make_gp']


class GP(Model):
    """
    Implementation of GP inference.
    """
    def __init__(self, like, kern, mean, inference, *args, **kwargs):
        # look up the inference method and initialize the posterior
        inference = gpinference.INFERENCE[inference]
        post = inference(like, kern, mean, *args, **kwargs)

        # register hyperparameters
        self._post = self._register(None, post)

    def __info__(self):
        info = super(GP, self).__info__()
        info.append(('inference', type(self._post).__name__.lower()))
        info.extend(self._post.__info__())
        return info

    def _update(self):
        if self.ndata == 0:
            self._post.init()
        else:
            self._post.update(self._X, self._Y)

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0
        else:
            return (self._post.lZ, self._post.dlZ) if grad else self._post.lZ

    def _predict(self, X, joint=False, grad=False):
        # get the prior mean and variance
        mu = self._post.mean.get_function(X)
        s2 = (self._post.kern.get_kernel(X) if joint else
              self._post.kern.get_dkernel(X))

        # if we have data compute the posterior
        if self.ndata > 0:
            if hasattr(self._post, 'U'):
                K = self._post.kern.get_kernel(self._post.U, X)
            else:
                K = self._post.kern.get_kernel(self._X, X)

            # compute the mean and variance
            w = self._post.w.reshape(-1, 1)
            V = la.solve_triangular(self._post.L, w*K)
            mu += np.dot(K.T, self._post.a)
            s2 -= np.dot(V.T, V) if joint else np.sum(V**2, axis=0)

            # add on a correction factor if necessary
            if self._post.C is not None:
                VC = la.solve_triangular(self._post.C, K)
                s2 += np.dot(VC.T, VC) if joint else np.sum(VC**2, axis=0)

        if not grad:
            return mu, s2

        if joint:
            raise ValueError('cannot compute gradients of joint predictions')

        # Get the prior gradients. NOTE: this assumes a stationary kernel.
        dmu = self._post.mean.get_gradx(X)
        ds2 = np.zeros_like(X)

        if self.ndata > 0:
            # get the kernel gradients
            if hasattr(self._post, 'U'):
                dK = self._post.kern.get_gradx(X, self._post.U)
            else:
                dK = self._post.kern.get_gradx(X, self._X)

            # reshape them to make it a 2d-array
            dK = np.rollaxis(dK, 1)
            dK = np.reshape(dK, (dK.shape[0], -1))

            # compute the mean gradients
            dmu += np.dot(dK.T, self._post.a).reshape(X.shape)

            # compute the variance gradients
            dV = la.solve_triangular(self._post.L, w*dK)
            dV = np.rollaxis(np.reshape(dV, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(dV * V, axis=1).T

            # add in a correction factor
            if self._post.C is not None:
                dVC = la.solve_triangular(self._post.C, dK)
                dVC = np.rollaxis(np.reshape(dVC, (-1,) + X.shape), 2)
                ds2 += 2 * np.sum(dVC * VC, axis=1).T

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


def make_gp(sn2, rho, ell, mean=0.0, ndim=None, kernel='se',
            inference='exact', **kwargs):
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

    return GP(like, kern, mean, inference, **kwargs)
