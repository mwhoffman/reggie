"""
Definition of the function interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized

__all__ = ['Function']


### BASE KERNEL INTERFACE #####################################################

class Function(Parameterized):
    """
    The base Function interface.
    """
    def __call__(self, x):
        X = np.array(x, ndmin=1)[None]
        return self.get_function(X)[0]

    def get_function(self, X):
        raise NotImplementedError

    def get_grad(self, X):
        raise NotImplementedError
