"""
Definition of the function interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized

__all__ = ['Function', 'RealFunction']


### BASE KERNEL INTERFACE #####################################################

class Function(Parameterized):
    """
    The base Function interface.
    """
    def __call__(self, x):
        X = np.array(x, ndmin=1)[None]
        return self.get_function(X)[0]

    def get_function(self, X):
        """
        Evaluate the function at input points X.
        """
        raise NotImplementedError

    def get_grad(self, X):
        """
        Get the gradient of the function with respect to any hyperparameters,
        evaluated at input points X. Return a generator yielding each gradient
        component.
        """
        raise NotImplementedError


class RealFunction(Function):
    """
    Interface for functions defined over real input spaces.
    """
    def get_gradx(self, X):
        """
        Get the gradient of the function with respect to the input space,
        evaluated at X.
        """
        raise NotImplementedError
