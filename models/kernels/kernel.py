"""
Definition of the kernel interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ..core.params import Parameterized

__all__ = ['Kernel']


### BASE KERNEL INTERFACE #####################################################

class Kernel(Parameterized):
    """
    The base Kernel interface.
    """
    def get_kernel(self, X1, X2):
        raise NotImplementedError

    def get_grad(self, X1, X2):
        raise NotImplementedError

    def get_dkernel(self, X1):
        raise NotImplementedError

    def get_dgrad(self, X1):
        raise NotImplementedError
