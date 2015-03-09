"""
Helper code for dealing with domains.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['outside_bounds']


EPSILON = np.finfo(np.float64).resolution


def outside_bounds(bounds, theta):
    """
    Check whether a vector is inside the given bounds.
    """
    if bounds is None:
        return False
    else:
        bounds = np.array(bounds, ndmin=2)
        return np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1])
