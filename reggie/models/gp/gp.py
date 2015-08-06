"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

from ...utils.misc import rstate
from ...utils import linalg as la
from .._core import ParameterizedModel

from .fourier import FourierSample
from .fstar import GP_fstar
from .xstar import GP_xstar

from ... import likelihoods
from ... import kernels
from ... import means

from . import inference

__all__ = ['GP', 'make_gp']


class GP(ParameterizedModel):
    """
    Implementation of GP inference.
    """
    def __init__(self, like, kern, mean, inf='exact', U=None):
        # initialize
        super(GP, self).__init__()

        # store the component objects
        self._like = self._register_obj('like', like)
        self._kern = self._register_obj('kern', kern)
        self._mean = self._register_obj('kern', mean)

        if isinstance(inf, basestring):
            if inf in inference.__all__:
                inf = getattr(inference, inf)
            else:
                raise ValueError('Unknown inference method')

        # FIXME: there should probably be a check here to see if the inference
        # method supports inducing points. Also we should check whether the
        # inference method requires Gaussian likelihoods.

        # store the inference method, the posterior sufficient statistics (None
        # so far) and any inducing points. inducing points.
        self._infer = inf
        self._post = None
        self._U = U

    def __info__(self):
        info = [
            ('like', self._like),
            ('kern', self._kern),
            ('mean', self._mean)]

        inf = self._infer.__name__

        # append the inference method if it is non-default
        if inf in inference.__all__:
            if inf is not 'exact':
                info.append(('inf', inference))
        else:
            info.append(('inf', self._infer))

        # append if we have any inducing points.
        if self._U is not None:
            info.append(('U', self._U))

        return info

    def _update(self):
        if self._X is None:
            self._post = None
        else:
            args = (self._like, self._kern, self._mean, self._X, self._Y)
            if self._U is not None:
                args += (self._U, )
            self._post = self._infer(*args)

    def _predict(self, X, joint=False, grad=False):
        # get the prior mean and variance
        mu = self._mean.get_mean(X)
        s2 = (self._kern.get_kernel(X) if joint else
              self._kern.get_dkernel(X))

        # if we have data compute the posterior
        if self._post is not None:
            if self._U is not None:
                K = self._kern.get_kernel(self._U, X)
            else:
                K = self._kern.get_kernel(self._X, X)

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

        dmu = self._mean.get_gradx(X)
        ds2 = self._kern.get_dgradx(X)

        if self._post is not None:
            # get the kernel gradients
            if hasattr(self._post, 'U'):
                dK = self._kern.get_gradx(X, self._post.U)
            else:
                dK = self._kern.get_gradx(X, self._X)

            # reshape them to make it a 2d-array
            dK = self._kern.get_gradx(X, self._X)
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
        if self._post is None:
            return (0.0, np.zeros(self.params.size)) if grad else 0.0
        else:
            return (self._post.lZ, self._post.dlZ) if grad else self._post.lZ

    def sample(self, X, size=None, latent=True, rng=None):
        mu, Sigma = self._predict(X, joint=True)
        rng = rstate(rng)

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

    def get_entropy(self, X):
        """
        Return the marginal predictive entropy.
        """
        s2 = self.predict(X)[1] + self._like.get_variance()
        return 0.5 * np.log(2 * np.pi * np.e * s2)

    def sample_f(self, n, rng=None):
        return FourierSample(self._like, self._kern, self._mean,
                             self._X, self._Y, n, rng)

    def condition_xstar(self, xstar):
        return GP_xstar(self._like, self._kern, self._mean,
                        self._X, self._Y, xstar)

    def condition_fstar(self, fstar):
        return GP_fstar(self._like, self._kern, self._mean,
                        self._X, self._Y, fstar)


def make_gp(sn2, rho, ell,
            mean=0.0, ndim=None, kernel='se', inf='exact', U=None):
    # create the mean/likelihood objects
    like = likelihoods.Gaussian(sn2)
    mean = means.Constant(mean)

    # create a kernel object which depends on the string identifier
    kern = (
        kernels.SE(rho, ell, ndim) if (kernel == 'se') else
        kernels.Matern(rho, ell, 1, ndim) if (kernel == 'matern1') else
        kernels.Matern(rho, ell, 3, ndim) if (kernel == 'matern3') else
        kernels.Matern(rho, ell, 5, ndim) if (kernel == 'matern5') else
        None)

    if kernel is None:
        raise ValueError('Unknown kernel type')

    return GP(like, kern, mean, inf, U)
