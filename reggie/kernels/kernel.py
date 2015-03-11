"""
Definition of the kernel interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized

__all__ = ['Kernel', 'RealKernel']


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
        """
        Compute the kernel matrix for inputs X1 and X2. If X2 is None, return
        the covariance between X1 and itself.
        """
        raise NotImplementedError

    def get_grad(self, X1, X2=None):
        """
        Compute the gradient of the kernel matrix with respect to any
        hyperparameters for inputs X1 and X2. If X2 is None, compute this for
        the the covariance between X1 and itself. Return a generator which
        yields each gradient component.
        """
        raise NotImplementedError

    def get_dkernel(self, X1):
        """
        Compute the diagonal of the self-covariance matrix.
        """
        raise NotImplementedError

    def get_dgrad(self, X1):
        """
        Compute the gradient of the diagonal of the self-covariance matrix.
        Return a generator yielding each component.
        """
        raise NotImplementedError


class RealKernel(Kernel):
    """
    Kernel defined over a real-valued input space.
    """
    def get_gradx(self, X1, X2=None):
        """
        Compute the gradient of the kernel k(x,y) with respect to the first
        input x, and evaluate it on input points X1 and X2.
        """
        raise NotImplementedError
