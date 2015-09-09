"""
Definitions of various priors. Note that while priors can be defined over a
vector of values, all those implemented in this module are assumed to be
independent.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random
import mwhutils.pretty as pretty

from .domains import EPSILON

__all__ = ['Uniform', 'LogNormal', 'Normal', 'Horseshoe']


class Prior(object):
    """
    Interface for prior distributions.
    """
    def sample(self, size=None, rng=None):
        """
        Sample from the prior. If `size` is None, return a single sample,
        otherwise return `size`-by-d array of samples.
        """
        raise NotImplementedError

    def get_logprior(self, theta, grad=False):
        """
        Compute the log prior evaluated at the parameter values `theta`. If
        `grad` is True return the gradient of this quantity wrt `theta` as
        well.
        """
        raise NotImplementedError


class Uniform(Prior):
    """
    Uniform prior distribution with bounds a[i] and b[i].
    """
    def __init__(self, a, b):
        a = np.array(a, ndmin=1, dtype=float)
        b = np.array(b, ndmin=1, dtype=float)

        self.bounds = np.c_[a, b]
        self.ndim = len(self.bounds)

        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("malformed upper/lower bounds")

    def __repr__(self):
        return pretty.repr_args(self,
                                self.bounds[:, 0].squeeze(),
                                self.bounds[:, 1].squeeze())

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        a = self.bounds[:, 0]
        b = self.bounds[:, 1] - a
        if size is None:
            return a + b * rng.rand(self.ndim)
        else:
            return a + b * rng.rand(size, self.ndim)

    def get_logprior(self, theta, grad=False):
        return (0.0, np.zeros_like(theta)) if grad else 0.0


class LogNormal(Prior):
    """
    Log-normal prior where log of each parameter value has an independent
    normal distribution with mean mu[i] and variance s2[i].
    """

    bounds = (EPSILON, np.inf)

    def __init__(self, mu=0, s2=1):
        self._mu = np.array(mu, dtype=float, copy=True, ndmin=1)
        self._s2 = np.array(s2, dtype=float, copy=True, ndmin=1)
        self.ndim = len(self._mu)

    def __repr__(self):
        return pretty.repr_args(self,
                                self._mu.squeeze(),
                                self._s2.squeeze())

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        m, s = self._mu, np.sqrt(self._s2)
        if size is not None:
            m = np.tile(m, (size, 1))
            s = np.tile(s, (size, 1))
        return rng.lognormal(m, s)

    def get_logprior(self, theta, grad=False):
        logp = np.sum(
            - np.log(theta)
            - 0.5 * np.log(2 * np.pi * self._s2) * self.ndim
            - 0.5 * np.square(np.log(theta) - self._mu) / self._s2)

        if grad:
            dlogp = -((np.log(theta) - self._mu) / self._s2 + 1) / theta
            return logp, dlogp

        else:
            return logp


class Normal(Prior):
    """
    Normal prior where each parameter value has an independent normal
    distribution with mean mu[i] and variance s2[i].
    """

    bounds = (-np.inf, np.inf)

    def __init__(self, mu=0, s2=1):
        self._mu = np.array(mu, dtype=float, copy=True, ndmin=1)
        self._s2 = np.array(s2, dtype=float, copy=True, ndmin=1)
        self.ndim = len(self._mu)

    def __repr__(self):
        return pretty.repr_args(self,
                                self._mu.squeeze(),
                                self._s2.squeeze())

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        m, s = self._mu, np.sqrt(self._s2)
        if size is not None:
            m = np.tile(m, (size, 1))
            s = np.tile(s, (size, 1))
        return rng.normal(m, s)

    def get_logprior(self, theta, grad=False):
        logp = np.sum(
            - 0.5 * np.log(2 * np.pi * self._s2)
            - 0.5 * np.square(theta - self._mu) / self._s2)

        if grad:
            dlogp = -(theta - self._mu) / self._s2
            return logp, dlogp

        else:
            return logp

class Horseshoe(Prior):
    """Horseshoe prior where each parameter value is independently distributed
    with scale parameter scale[i]."""

    bounds = (EPSILON, np.inf)

    def __init__(self, scale=1.):
        self._scale = np.array(scale, copy=True, ndmin=1)
        self.ndim = len(self._scale)

    def __repr__(self):
        return pretty.repr_args(self, self._scale.squeeze())

    def get_logprior(self, theta, grad=False):
        theta2_inv = (self._scale / theta)**2
        inner = np.log1p(theta2_inv)

        logp = np.sum(np.log(inner))

        if grad:
            dlogp = theta2_inv / (1 + theta2_inv)
            dlogp /= inner
            dlogp *= -2 / theta

            return logp, dlogp
        else:
            return logp

# get a dictionary mapping a string to each prior
PRIORS = dict()
for _ in __all__:
    PRIORS[_.lower()] = globals()[_]
