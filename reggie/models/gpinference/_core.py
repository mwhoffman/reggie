"""
Definition of a base class which provides an interface for inference in GP
models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ...core.params import Parameterized


__all__ = ['Inference']


class Inference(Parameterized):
    """
    Base interface for inference methods.
    """
    def __init__(self):
        self.init()

    def init(self):
        """Initialize the posterior parameterization."""
        raise NotImplementedError

    def update(self, like, kern, mean, X, Y):
        """Update the posterior given a complete set of data."""
        raise NotImplementedError
