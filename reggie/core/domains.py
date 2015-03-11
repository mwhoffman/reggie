"""
Helper code for dealing with domains.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['BOUNDS', 'TRANSFORMS', 'REAL', 'POSITIVE']


# numerical constants
EPSILON = np.finfo(np.float64).resolution

# domain identifiers
REAL = 'real'
POSITIVE = 'positive'

# boundaries for each of the domains
BOUNDS = {
    REAL: (-np.inf, np.inf),
    POSITIVE: (EPSILON, np.inf)
}


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
    def get_transform(self, x):
        return np.log(x)

    def get_gradfactor(self, x):
        return x.copy()

    def get_inverse(self, t):
        return np.clip(np.exp(t), EPSILON, np.inf)


class Identity(Transform):
    def get_transform(self, x):
        return x.copy()

    def get_gradfactor(self, x):
        return np.ones_like(x)

    def get_inverse(self, t):
        return t.copy()


# default parameter transformations
TRANSFORMS = {
    REAL: Identity(),
    POSITIVE: Log()
}
