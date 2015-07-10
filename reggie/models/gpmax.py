"""
Predictive entropy search.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss
import mwhutils.linalg as la


def get_factors(m0, V0, fstar):
    """
    Given a Gaussian distribution with mean and covariance (m0, V0) use EP to
    find a Gaussian approximating the constraint that each latent variable is
    below fstar. Return the approximate factors (tau_, rho_) in canonical form.
    """
    # initialize the current state of the posterior as well as the EP factors
    # in canonical form.
    m, V = m0, V0
    rho_ = np.zeros_like(m0)
    tau_ = np.zeros_like(m0)

    # no damping on the first iteration
    damping = 1

    while True:
        # get the canonical form marginals
        tau = 1 / V.diagonal()
        rho = m / V.diagonal()

        # eliminate the contribution of the EP factors
        v = (tau - tau_) ** -1
        m = v * (rho - rho_)

        sigma = np.sqrt(v)
        alpha = (fstar - m) / sigma
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        kappa = (ratio + alpha) / sigma
        gamma = ratio * kappa / sigma

        # get the new factors
        tauNew = gamma / (1 - gamma*v)
        rhoNew = (m - 1 / kappa) * tauNew

        # update the EP factors with damping
        tau_ = tauNew * damping + tau_ * (1-damping)
        rho_ = rhoNew * damping + rho_ * (1-damping)

        # the new posterior.
        t = np.sqrt(tau_)
        L = la.cholesky(la.add_diagonal(t*V0*t[:, None], 1))
        V = la.solve_triangular(L, V0*t[:, None])
        V = V0 - np.dot(V.T, V)
        m = np.dot(V, rho_) + la.solve_cholesky(L, t*m0) / t

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            damping *= 0.99
        else:
            break

    return tau_, rho_


class GP_fstar(object):
    def __init__(self, like, kern, mean, X, Y, fstar):
        # get the data and the noise variance
        R = Y - mean.get_mean(X)
        sn2 = like.get_variance()

        # get the mean and kernel at our latents
        m = mean.get_mean(X)
        K = kern.get_kernel(X)

        # compute intermediate terms.
        L = la.cholesky(la.add_diagonal(K, sn2))
        A = la.solve_triangular(L, K)
        a = la.solve_triangular(L, R)

        # get the initial predictions
        m0 = m + np.dot(A.T, a)
        V0 = K - np.dot(A.T, A)

        # get the EP factors and construct convolving factor
        tau, rho = get_factors(m0, V0, fstar)
        omega = sn2 / (1 + sn2*tau)

        # save the model
        self.like = like
        self.kern = kern
        self.mean = mean

        # save the data
        self.X = X
        self.fstar = fstar

        # get the new posterior
        self.L = la.cholesky(la.add_diagonal(K, omega))
        self.a = la.solve_triangular(self.L, omega * (R/sn2 + rho))

    def predict(self, X, grad=False):
        # now evaluate the kernel at the new points and compute intermediate
        # terms
        K = self.kern.get_kernel(self.X, X)
        A = la.solve_triangular(self.L, K)

        # get the predictions before the final constraint
        m1 = self.mean.get_mean(X) + np.dot(A.T, self.a)
        v1 = self.kern.get_dkernel(X) - np.sum(A**2, axis=0)

        # get terms necessary for the final constraint
        sigma = np.sqrt(v1)
        alpha = (self.fstar - m1) / sigma
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        kappa = ratio + alpha
        delta = ratio - alpha
        gamma = ratio * (delta*kappa + ratio*delta + 1)

        # incorporatee the prior constraint
        m2 = m1 - ratio * sigma
        v2 = v1 - v1 * ratio * (ratio + alpha)

        if grad is False:
            return m2, v2

        # get the "prior" gradient at X
        dm1 = self.mean.get_gradx(X)
        dv1 = self.kern.get_dgradx(X)

        # get the kernel gradient and reshape it so we can do linear algebra
        dK = self.kern.get_gradx(X, self.X)
        dK = np.rollaxis(dK, 1)
        dK = np.reshape(dK, (dK.shape[0], -1))

        # compute the mean gradients
        dm1 += np.dot(dK.T, self.a).reshape(X.shape)

        # compute the variance gradients
        dA = la.solve_triangular(self.L, dK)
        dA = np.rollaxis(np.reshape(dA, (-1,) + X.shape), 2)
        dv1 -= 2 * np.sum(dA * A, axis=1).T

        dm2 = (1 + ratio**2 - ratio * alpha)[:, None] * dm1
        dm2 += 0.5 * ((ratio*alpha*kappa - ratio**2) / sigma)[:, None] * dv1

        dv2 = (1 - ratio*kappa - 0.5*alpha*gamma)[:, None] * dv1
        dv2 -= (gamma*sigma)[:, None] * dm1

        return m2, v2, dm2, dv2

    def get_entropy(self, X):
        s2 = self.predict(X)[1] + self.like.get_variance()
        return 0.5 * np.log(2 * np.pi * np.e * s2)
