"""
Objects representing model implementations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from .params import Parameter, Model, Likelihood

__all__ = []


def get_logp(obj, grad, *args):
    args += (grad,)
    return \
        (obj.get_logp(*args)) if grad else \
        (obj.get_logp(*args), None, None)


class Gaussian(Likelihood):
    def __init__(self, s2):
        self._register('s2', Parameter(s2))

    def get_loglike(self, F, Y, grad=False):
        loglike = np.log(2*np.pi) + np.log(self._s2)    # normalization
        loglike += np.sum((Y - F)**2) / self._s2        # distance metric
        loglike /= -2

        if grad:
            raise NotImplementedError
        else:
            return loglike


class GLM(Model):
    def __init__(self, theta, likelihood):
        self._register('theta', Parameter(theta))
        self._register('likelihood', likelihood)
