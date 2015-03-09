"""
Prior distributions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ..utils.pretty import repr_args
from .domains import outside_bounds

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

    def sample(self, size=None, rng=None):
        rng = random.rstate(rng)
        a = self.bounds[:, 0]
        b = self.bounds[:, 1] - a
        if size is None:
            return a + b * rng.rand(self.ndim)
        else:
            return a + b * rng.rand(size, self.ndim)

    def get_logprior(self, theta, grad=False):
        logp = -np.inf if outside_bounds(self.bounds, theta) else 0.0
        return (logp, np.zeros_like(theta)) if grad else logp


# get a dictionary mapping a string to each prior
PRIORS = dict()
for _ in __all__:
    PRIORS[_.lower()] = globals()[_]
