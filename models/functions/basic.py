"""
Implementation of basic functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from .function import Function

__all__ = ['Zero', 'Constant']


class Zero(Function):
    def get_function(self, X):
        return np.zeros(len(X))

    def get_grad(self, X):
        return iter([])


class Constant(Function):
    def __init__(self, bias=0):
        self._register('bias', bias, ndim=0)

    def get_function(self, X):
        return np.full(len(X), self._bias)

    def get_grad(self, X):
        yield np.ones(len(X))
