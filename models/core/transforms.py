"""
Parameter transformations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = []


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

    def get_inverse(self, fx):
        """
        Apply the inverse transformation which takes a transformed parameter
        `f(x)` and returns the original value `x`.
        """
        raise NotImplementedError

    def get_gradfactor(self, x):
        """
        Get the gradient factor for the transformation. This vector is
        evaluated at input x and can be used to multiply a gradient vector in
        order to obtain a gradient with respect to the transformed inputs.
        """
        raise NotImplementedError


class Log(Transform):
    def get_transform(self, x):
        return np.log(x)

    def get_inverse(self, fx):
        return np.exp(fx)

    def get_gradfactor(self, x):
        return x.copy()
