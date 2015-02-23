"""
Parameter transformations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = []


class Transform(object):
    def get_transform(self, x):
        raise NotImplementedError

    def get_inverse(self, fx):
        raise NotImplementedError

    def get_inverse_grad(self, fx):
        raise NotImplementedError


class Log(Transform):
    def get_transform(self, x):
        return np.log(x)

    def get_inverse(self, fx):
        return np.exp(fx)

    def get_inverse_grad(self, fx):
        return np.exp(fx)
