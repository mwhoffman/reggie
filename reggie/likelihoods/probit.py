
"""
Implementation of the probit likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.special as ss
import mwhutils.random as random

from ._core import Likelihood

__all__ = ['Probit']


def logphi(z, grad=False):
    lp = np.log(ss.erfc(-z/np.sqrt(2))/2)
    if not grad:
        return lp
    d1 = np.exp(-z**2/2 - lp) / np.sqrt(2*np.pi)
    d2 = -d1 * np.abs(z+d1)
    d3 = -d2 * np.abs(z+2*d1) - d1
    return lp, d1, d2, d3


class Probit(Likelihood):
    def sample(self, f, rng=None):
        rng = random.rstate(rng)
        i = np.log(rng.rand(len(f))) < logphi(f)
        y = 1*i - 1*(-i)
        return y

    def get_logprob(self, y, f):
        # make sure we have only +/- signals
        i = (y == 1)
        y = 1*i - 1*(-i)
        lp, d1, d2, d3 = logphi(y*f, grad=True)
        d1 *= y
        d3 *= y
        return lp, d1, d2, d3

    def get_laplace_grad(self, y, f):
        return iter([])
