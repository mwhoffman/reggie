"""
Prior distributions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ..utils.pretty import repr_args

__all__ = ['Uniform']


EPSILON = np.finfo(np.float64).resolution


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

    def project(self, theta):
        return np.clip(theta, self.bounds[:, 0], self.bounds[:, 1])

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        a = self.bounds[:, 0]
        b = self.bounds[:, 1] - a
        if size is None:
            return a + b * rng.rand(self.ndim)
        else:
            return a + b * rng.rand(size, self.ndim)

    def get_logprior(self, theta, grad=False):
        for (a, b), t in zip(self.bounds, theta):
            if (t < a) or (t > b):
                logp = -np.inf
                break
        else:
            logp = 0.0
        return (logp, np.zeros_like(theta)) if grad else logp


# get a dictionary mapping a string to each prior
PRIORS = dict()
for _ in __all__:
    PRIORS[_.lower()] = globals()[_]
