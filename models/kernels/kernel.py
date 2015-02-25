"""
Definition of the kernel interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized

__all__ = ['Kernel']


### BASE KERNEL INTERFACE #####################################################

class Kernel(Parameterized):
    """
    The base Kernel interface.
    """
    def __call__(self, x1, x2):
        X1 = np.array(x1, ndmin=1)[None]
        X2 = np.array(x2, ndmin=1)[None]
        return self.get_kernel(X1, X2)[0, 0]

    def get_kernel(self, X1, X2=None):
        raise NotImplementedError

    def get_grad(self, X1, X2=None):
        raise NotImplementedError

    def get_dkernel(self, X1):
        raise NotImplementedError

    def get_dgrad(self, X1):
        raise NotImplementedError
