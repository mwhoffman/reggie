"""
Implementation of the Gaussian likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ._core import Likelihood
from ..core.domains import POSITIVE

__all__ = ['Gaussian']


class Gaussian(Likelihood):
    """
    The Gaussian likelihood function for regression. This likelihood can be
    written as::

        p(y|f) = 1 / sqrt(2*pi*sn2) * exp(-0.5 * (y-f)^2 / sn2) 

    where `sn2` is the noise variance.
    """
    def __init__(self, sn2):
        # register our parameters
        self._sn2 = self._register('sn2', sn2, domain=POSITIVE)

    def sample(self, f, rng=None):
        rng = random.rstate(rng)
        return f + rng.normal(size=len(f), scale=np.sqrt(self._sn2))

    def get_variance(self):
        """
        Return the variance of the observation model; this is used for
        performing exact inference.
        """
        return float(self._sn2)

    def get_logprob(self, y, f):
        yhat = y - f
        lp = -0.5 * (yhat**2 / self._sn2 + np.log(2 * np.pi * self._sn2))
        dlp = yhat / self._sn2
        d2lp = np.full_like(yhat, -1/self._sn2)
        return lp, dlp, d2lp
