"""
Condition a GP on an observed maximizer location.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

from ...utils import linalg as la


def get_latent(m0, v0, ymax, sn2):
    """
    Given a Gaussian (m0, v0) for the value of the latent maximizer return an
    approximate Gaussian posterior (m, v) subject to the constraint that the
    value is greater than ymax, where the noise varaince sn2 is used to soften
    this constraint.
    """
    s = np.sqrt(v0 + sn2)
    t = m0 - ymax

    alpha = t / s
    ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
    beta = ratio * (alpha + ratio) / s / s
    kappa = (alpha + ratio) / s

    m = m0 + 1. / kappa
    v = (1 - beta*v0) / beta

    return m, v


class GP_xstar(object):
    def __init__(self, like, kern, mean, X, Y, xstar):
        # format the optimum location as a (1,d) array.
        Z = np.array(xstar, ndmin=2)

        # condition on our observations. NOTE: if this is an exact GP, then
        # we've already computed these quantities.
        sn2 = like.get_variance()
        Kxx = la.add_diagonal(kern.get_kernel(X), sn2)
        L = la.cholesky(Kxx)
        a = la.solve_triangular(L, Y - mean.get_mean(X))

        # condition on the gradient being zero.
        Kgx = kern.get_gradx(Z, X)[0]
        Kgg = kern.get_gradxy(Z, Z)[0, 0]
        L, a = la.cholesky_update(L, Kgx.T, Kgg, a, np.zeros_like(xstar))

        # evaluate the kernel so we can test at the latent optimizer.
        Kzz = kern.get_kernel(Z)
        Kzc = np.c_[
            kern.get_kernel(Z, X),
            kern.get_gradx(Z, Z)[0]]

        # make predictions at the optimizer.
        B = la.solve_triangular(L, Kzc.T)
        m0 = float(np.dot(B.T, a)) + mean.get_mean(Z)
        v0 = float(Kzz - np.dot(B.T, B))

        # get the approximate factors and use this to update the cholesky,
        # which should now be wrt the covariance between [y; g; f(z)].
        m, v = get_latent(m0, v0, max(Y), sn2)
        L, a = la.cholesky_update(L, Kzc, Kzz + v, a, m - mean.get_mean(Z))
        Bstar = la.solve_triangular(L, np.c_[Kzc, Kzz].T)

        # save the model
        self._like = like
        self._kern = kern
        self._mean = mean
        self._X = X
        self._Z = Z

        # save the cholesky
        self._L = L
        self._a = a

        # get predictions at the optimum.
        self._Bstar = Bstar
        self._mstar = float(np.dot(Bstar.T, a)) + mean.get_mean(Z)
        self._vstar = float(kern.get_dkernel(Z) - np.sum(Bstar**2, axis=0))

    def predict(self, X, grad=False):
        # evaluate the covariance between our test points and both the analytic
        # constraints and z.
        Ktc = np.c_[
            self._kern.get_kernel(X, self._X),
            self._kern.get_gradx(self._Z, X)[0],
            self._kern.get_kernel(X, self._Z)]

        # get the marginal posterior without the constraint that the function
        # at the optimum is better than the function at test points.
        B = la.solve_triangular(self._L, Ktc.T)
        m = self._mean.get_mean(X) + np.dot(B.T, self._a)
        v = self._kern.get_dkernel(X) - np.sum(B**2, axis=0)

        # the covariance between each test point and xstar.
        r = Ktc[:, -1] - np.dot(B.T, self._Bstar).flatten()
        s = v + self._vstar - 2*r

        while any(s < 1e-10):
            r[s < 1e-10] *= 1 - 1e-10
            s = v + self._vstar - 2*r

        a = (self._mstar - m) / np.sqrt(s)
        b = np.exp(ss.norm.logpdf(a) - ss.norm.logcdf(a))

        m += b * (r - v) / np.sqrt(s)
        v -= b * (b + a) * (r - v)**2 / s

        return m, v
