
"""
Implementation of the probit likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.special as ss

from ._core import Likelihood

__all__ = ['Probit']


def logphi(z):
    lp = np.log(ss.erfc(z/np.sqrt(2))/2)
    d1 = np.exp(-z**2/2 - lp) / np.sqrt(2*np.pi)
    d2 = -d1 * np.abs(z+d1)
    d3 = -d2 * np.abs(z+2*d1) - d1
    return lp, d1, d2, d3


class Probit(Likelihood):
    def get_logprob(self, y, f):
        # make sure we have only +/- signals
        i = (f == 1)
        y = y*i - y*(-i)
        lp, d1, d2, d3 = logphi(y*f)
        d1 *= y
        d3 *= y
        return lp, d1, d2, d3

    def get_grad(self, y, f):
        return iter([])
