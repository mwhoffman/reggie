"""
Prior distributions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from .domains import EPSILON
from ..utils.pretty import repr_args

__all__ = ['Uniform', 'LogNormal', 'Normal']


class Prior(object):
    """
    Interface for prior distributions.
    """
    def sample(self, size, rng=None):
        raise NotImplementedError

    def get_logprior(self, theta, grad=False):
        raise NotImplementedError


class Uniform(Prior):
    def __init__(self, a, b):
        a = np.array(a, ndmin=1, dtype=float)
        b = np.array(b, ndmin=1, dtype=float)

        self.bounds = np.c_[a, b]
        self.ndim = len(self.bounds)

        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("malformed upper/lower bounds")

    def __repr__(self):
        return repr_args(self,
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
    bounds = np.array((EPSILON, np.inf))

    def __init__(self, mu=0, s2=1):
        self._mu = np.array(mu, dtype=float, copy=True, ndmin=1)
        self._s2 = np.array(s2, dtype=float, copy=True, ndmin=1)
        self.ndim = len(self._mu)

    def __repr__(self):
        return repr_args(self, self._mu.squeeze(), self._s2.squeeze())

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
    bounds = np.array((-np.inf, np.inf))

    def __init__(self, mu=0, s2=1):
        self._mu = np.array(mu, dtype=float, copy=True, ndmin=1)
        self._s2 = np.array(s2, dtype=float, copy=True, ndmin=1)
        self.ndim = len(self._mu)

    def __repr__(self):
        return repr_args(self, self._mu.squeeze(), self._s2.squeeze())

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        m, s = self._mu, np.sqrt(self._s2)
        if size is not None:
            m = np.tile(m, (size, 1))
            s = np.tile(s, (size, 1))
        return rng.normal(m, s)

    def get_logprior(self, theta, grad=False):
        logp = (
            - 0.5 * np.log(2 * np.pi * self._s2) * self.ndim
            - 0.5 * np.sum(np.square(theta - self._mu) / self._s2))

        if grad:
            dlogp = -(theta - self._mu) / self._s2
            return logp, dlogp

        else:
            return logp


# get a dictionary mapping a string to each prior
PRIORS = dict()
for _ in __all__:
    PRIORS[_.lower()] = globals()[_]
