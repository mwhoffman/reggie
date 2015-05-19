"""
Definition of a base class which provides an interface for inference in GP
models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ...core.params import Parameterized
from ...likelihoods._core import Likelihood
from ...kernels._core import Kernel
from ...functions._core import Function


__all__ = ['Inference']


class Inference(Parameterized):
    """
    Base interface for inference methods.
    """
    def __init__(self, like, kern, mean):
        super(Inference, self).__init__()
        self.like = self._register_obj('like', like, Likelihood)
        self.kern = self._register_obj('kern', kern, Kernel)
        self.mean = self._register_obj('mean', mean, Function)
        self.init()

    def __info__(self):
        info = []
        info.append(('like', self.like))
        info.append(('kern', self.kern))
        info.append(('mean', self.mean))
        return info

    def init(self):
        """
        Initialize the posterior parameterization.
        """
        self.L = None
        self.C = None
        self.a = None
        self.w = None
        self.lZ = None
        self.dlZ = None

    def update(self, X, Y):
        """
        Update the posterior given a complete set of data.
        """
        raise NotImplementedError
