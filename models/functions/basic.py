"""
Implementation of basic functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from .function import RealFunction

__all__ = ['Zero', 'Constant']


class Zero(RealFunction):
    """
    Function which returns zero on any input.
    """
    def get_function(self, X):
        return np.zeros(len(X))

    def get_grad(self, X):
        return iter([])

    def get_gradx(self, X):
        return np.zeros_like(X)


class Constant(RealFunction):
    """
    Function which returns a constant value on any input.
    """
    def __init__(self, bias=0):
        self._register('bias', bias)

    def get_function(self, X):
        return np.full(len(X), self._bias)

    def get_grad(self, X):
        yield np.ones(len(X))

    def get_gradx(self, X):
        return np.zeros_like(X)
