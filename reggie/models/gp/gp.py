"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss
import warnings

from ...utils.misc import rstate
from ...utils import linalg as la
from .._core import ParameterizedModel

from .fourier import FourierSample

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

        # this is a non-parametric model so we'll need to store the data
        self._X = None
        self._Y = None

        # store the component objects
        self._like = self._register_obj('like', like)
        self._kern = self._register_obj('kern', kern)
        self._mean = self._register_obj('mean', mean)

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
                info.append(('inf', inf))
        else:
            info.append(('inf', self._infer))
        # append if we have any inducing points.
        if self._U is not None:
            info.append(('U', self._U))
        return info

    def __deepcopy__(self, memo):
        # don't make a copy of the data.
        memo[id(self._X)] = self._X
        memo[id(self._Y)] = self._Y
        return super(GP, self).__deepcopy__(memo)

    def add_data(self, X, Y):
        X = np.array(X, copy=False, ndmin=2, dtype=float)
        Y = np.array(Y, copy=False, ndmin=1, dtype=float)
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]
        self._update()

    def _update(self):
        # NOTE: this method is called both after adding data as well as any
        # time that the parameters change.
        if self._X is None:
            self._post = None
        else:
            args = (self._like, self._kern, self._mean, self._X, self._Y)
            if self._U is not None:
                args += (self._U, )
            self._post = self._infer(*args)

    def _predict(self, X, joint=False, grad=False):
        """
        Internal method used to make both joint and marginal predictions.
        """
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

        # make sure s2 isn't zero. this is almost equivalent to using a nugget
        # parameter, but after the fact if the predictive variance is too
        # small.
        # s2 = np.clip(s2, 1e-100, np.inf)

        if not grad:
            return mu, s2

        if joint:
            raise ValueError('cannot compute gradients of joint predictions')

        dmu = self._mean.get_gradx(X)
        ds2 = self._kern.get_dgradx(X)

        if self._post is not None:
            # get the kernel gradients
            if self._U is not None:
                dK = self._kern.get_gradx(X, self._U)
            else:
                dK = self._kern.get_gradx(X, self._X)

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
        if self._post is None:
            return (0.0, np.zeros(self.params.size)) if grad else 0.0
        else:
            return (self._post.lZ, self._post.dlZ) if grad else self._post.lZ

    def sample(self, X, size=None, latent=True, rng=None):
        mu, Sigma = self._predict(X, joint=True)
        rng = rstate(rng)

        # the covariance here is without noise, so the cholesky code may add to
        # the diagonal and raise a warning. since we know this may happe, we
        # can just ignore this.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = la.cholesky(Sigma)

        m = 1 if (size is None) else size
        n = len(X)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), L.T)

        if latent is False:
            f = self._like.sample(f.ravel(), rng).reshape(f.shape)
        if size is None:
            f = f.ravel()
        return f

    def predict(self, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Return mean and variance predictions `(mu, s2)` at inputs `X`. If
        `grad` is true, also compute gradients of these quantities with respect
        to the inputs.
        """
        return self._predict(X, grad=grad)

    def get_tail(self, f, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the expected improvement in value at inputs `X` over the target
        value `f`. If `grad` is true, also compute gradients with respect to
        the inputs.
        """
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

    def get_improvement(self, f, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the expected improvement in value at inputs `X` over the target
        value `f`. If `grad` is true, also compute gradients with respect to
        the inputs.
        """
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

    def get_entropy(self, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the predictive entropy evaluated at inputs `X`. If `grad` is
        true, also compute gradients quantity with respect to the inputs.
        """
        vals = self.predict(X, grad)
        s2 = vals[1]
        sp2 = s2 + self._like.get_variance()
        H = 0.5 * np.log(2 * np.pi * np.e * sp2)

        if not grad:
            return H

        # get the derivative of the entropy
        ds2 = vals[3]
        dH = 0.5 * ds2 / sp2[:, None]

        return H, dH

    def sample_f(self, n, rng=None):
        """
        Return a function or object `f` implementing `__call__` which can be
        used as a sample of the latent function. The argument `n` specifies the
        number of approximate features to use.
        """
        return FourierSample(self._like, self._kern, self._mean,
                             self._X, self._Y, n, rng)


def make_gp(sn2, rho, ell,
            mean=0.0, ndim=None, kernel='se', inf='exact', U=None):
    """
    Simple interface for creating either an isotropic or ARD GP.
    """
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
