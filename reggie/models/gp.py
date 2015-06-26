"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

import mwhutils.linalg as la
import mwhutils.random as random

from .. import likelihoods
from .. import kernels
from .. import functions

from ._core import ParameterizedModel
from . import gpinference

__all__ = ['GP', 'make_gp']


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


class GP(ParameterizedModel):
    """
    Implementation of GP inference.
    """
    def __init__(self, like, kern, mean, infer, U=None):
        # initialize
        super(GP, self).__init__()

        # store the component objects
        self.like = self._register_obj('like', like)
        self.kern = self._register_obj('kern', kern)
        self.mean = self._register_obj('kern', mean)

        self._U = U
        self._post = None
        self._infer = infer

    def __info__(self):
        info = [
            ('like', self.like),
            ('kern', self.kern),
            ('mean', self.mean),
            ('infer', self._infer)]
        if self._U is not None:
            info.append(('U', self._U))
        return info

    def _update(self):
        args = (self.like, self.kern, self.mean, self._X, self._Y)
        if self._U is not None:
            args += (self._U, )
        self._post = self._infer(*args)

    def _predict(self, X, joint=False, grad=False):
        # get the prior mean and variance
        mu = self.mean.get_function(X)
        s2 = (self.kern.get_kernel(X) if joint else
              self.kern.get_dkernel(X))

        # if we have data compute the posterior
        if self.ndata > 0:
            if self._U is not None:
                K = self.kern.get_kernel(self._U, X)
            else:
                K = self.kern.get_kernel(self._X, X)

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
        dmu = self.mean.get_gradx(X)
        ds2 = np.zeros_like(X)

        if self.ndata > 0:
            # get the kernel gradients
            if hasattr(self._post, 'U'):
                dK = self.kern.get_gradx(X, self._post.U)
            else:
                dK = self.kern.get_gradx(X, self._X)

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

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.params.size)) if grad else 0.0
        else:
            return (self._post.lZ, self._post.dlZ) if grad else self._post.lZ

    def sample(self, X, size=None, latent=True, rng=None):
        mu, Sigma = self._predict(X, joint=True)
        rng = random.rstate(rng)

        L = la.cholesky(la.add_diagonal(Sigma, 1e-10))
        m = 1 if (size is None) else size
        n = len(X)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), L.T)

        if latent is False:
            f = self.like.sample(f.ravel(), rng).reshape(f.shape)
        if size is None:
            f = f.ravel()
        return f

    def sample_f(self, n, rng=None):
        return FourierSample(self, n, rng)

    def predict(self, X, grad=False):
        return self._predict(X, grad=grad)

    def get_tail(self, X, f, grad=False):
        # get the posterior (possibly with gradients) and standardize
        post = self.predict(X, grad=grad)
        mu, s2 = post[:2]
        a = mu - f
        s = np.sqrt(s2)
        z = a / s

        # get the cdf
        cdf = ss.norm.cdf(z)

        if not grad:
            return cdf
        else:
            dmu, ds2 = post[2:]
            dcdf = dmu / s[:, None] - 0.5 * ds2 * z[:, None] / s2[:, None]
            return cdf, dcdf

    def get_improvement(self, X, f, grad=False):
        # get the posterior (possibly with gradients) and standardize
        post = self.predict(X, grad=grad)
        mu, s2 = post[:2]
        a = mu - f
        s = np.sqrt(s2)
        z = a / s

        # get the cdf, pdf, and ei
        cdf = ss.norm.cdf(z)
        pdf = ss.norm.pdf(z)
        ei = a * cdf + s * pdf

        if not grad:
            return ei
        else:
            dmu, ds2 = post[2:]
            dei = 0.5 * ds2 / s2[:, None]
            dei *= (ei - s * z * cdf)[:, None] + cdf[:, None] * dmu
            return ei, dei


def make_gp(sn2, rho, ell,
            mean=0.0, ndim=None, kernel='se', inference='exact', U=None):
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

    if inference in ['exact', 'laplace', 'fitc']:
        infer = getattr(gpinference, inference)
    else:
        raise ValueError('Unknown inference method')

    return GP(like, kern, mean, infer, U)
