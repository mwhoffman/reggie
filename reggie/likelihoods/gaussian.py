"""
Implementation of the Gaussian likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ._core import Likelihood
from ..core.domains import POSITIVE
from ..utils.misc import rstate

__all__ = ['Gaussian']


class Gaussian(Likelihood):
    """
    The Gaussian likelihood function for regression. This likelihood can be
    written as::

        p(y|f) = 1 / sqrt(2*pi*sn2) * exp(-0.5 * (y-f)^2 / sn2) 

    where `sn2` is the noise variance.
    """
    def __init__(self, sn2):
        super(Gaussian, self).__init__()
        self._sn2 = self._register('sn2', sn2, POSITIVE)

    def __info__(self):
        info = []
        info.append(('sn2', self._sn2))
        return info

    def get_variance(self):
        return float(self._sn2)

    def sample(self, f, rng=None):
        rng = rstate(rng)
        return f + rng.normal(size=len(f), scale=np.sqrt(self._sn2))

    def get_logprob(self, y, f):
        r = y-f
        lp = -0.5 * (r**2 / self._sn2 + np.log(2 * np.pi * self._sn2))
        d1 = r / self._sn2
        d2 = np.full_like(r, -1/self._sn2)
        d3 = np.zeros_like(r)
        return lp, d1, d2, d3

    def get_laplace_grad(self, y, f):
        r = y-f
        s = self._sn2**2
        d0 = 0.5 * (r**2/s - 1/self._sn2)
        d1 = -r/s
        d2 = np.full_like(r, 1/s)
        yield d0, d1, d2
