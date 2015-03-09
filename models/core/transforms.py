"""
Parameter transformations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils.pretty import repr_args

__all__ = ['Log']


EPSILON = np.finfo(np.float64).resolution


class Transform(object):
    """
    Interface for parameter transformations.
    """
    def get_transform(self, x):
        """
        Transform a parameter with value `x` from its original space into the
        transformed space `f(x)`.
        """
        raise NotImplementedError

    def get_gradfactor(self, x):
        """
        Get the gradient factor for the transformation. This computes and
        returns the gradient of `f^{-1}(t)` evaluated at `t=f(x)`.
        """
        raise NotImplementedError

    def get_inverse(self, t):
        """
        Apply the inverse transformation which takes a transformed parameter
        `t=f(x)` and returns the original value `x`.
        """
        raise NotImplementedError


class Log(Transform):
    bounds = np.array((EPSILON, np.inf))

    def __repr__(self):
        return repr_args(self)

    def get_transform(self, x):
        return np.log(x)

    def get_gradfactor(self, x):
        return x.copy()

    def get_inverse(self, t):
        return np.clip(np.exp(t), EPSILON, np.inf)


# get a dictionary mapping a string to each transform
TRANSFORMS = dict()
for _ in __all__:
    TRANSFORMS[_.lower()] = globals()[_]
