"""
Implementation of basic functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ._core import Mean
from ..kernels._distances import rescale, dist

__all__ = ['Zero', 'Constant', 'Linear', 'Quadratic']


class Zero(Mean):
    """
    Function which returns zero on any input.
    """
    def get_function(self, X):
        return np.zeros(len(X))

    def get_grad(self, X):
        return iter([])

    def get_gradx(self, X):
        return np.zeros_like(X)


class Constant(Mean):
    """
    Function which returns a constant value on any input.
    """
    def __init__(self, bias=0):
        super(Constant, self).__init__()
        self._bias = self._register('bias', bias)

    def __info__(self):
        info = []
        info.append(('bias', self._bias))
        return info

    def get_function(self, X):
        return np.full(len(X), self._bias)

    def get_grad(self, X):
        yield np.ones(len(X))

    def get_gradx(self, X):
        return np.zeros_like(X)


class Linear(Mean):
    """
    Mean function for linear trends
    """
    def __init__(self, bias=0, slopes=0):
        super(Linear, self).__init__()
        self._bias = self._register('bias', bias)
        self._slopes = self._register('slopes', np.array(slopes, ndmin=1))

    def __info__(self):
        info = []
        info.append(('bias', self._bias))
        info.append(('slopes', self._slopes))
        return info

    def get_function(self, X):
        slopes = self._slopes if (len(self._slopes) == len(X)) else \
                 np.full(len(X), self._slopes)
        return X.dot(slopes) + self._bias

    def get_grad(self, X):
        for i in X.shape[1]:
            yield X[:,i]

    def get_gradx(self, X):
        slopes = self._slopes if (len(self._slopes) == len(X)) else \
            np.full(len(X), self._slopes)
        return slopes

class Quadratic(Mean):
    def __init__(self, bias, centre, widths, ndim=None):
        super(Quadratic, self).__init__()
        self._bias = self._register('bias', bias)
        self._centre = self._register('centre', np.array(centre, ndmin=1))
        self._widths = self._register('widths', np.array(widths, ndmin=1))

        self._iso = False
        self.ndim = np.size(self._widths)

    def __info__(self):
        info = []
        info.append(('bias', self._bias))
        info.append(('centre', self._centre))
        info.append(('widths', self._widths))
        return info

    def get_function(self, X):
        X0 = np.array(self._centre, ndmin=2)
        X, X0 = rescale(self._widths, X, X0)
        return self._bias - dist(X, X0).ravel() ** 2

    def get_grad(self, X):
        """Gradient wrt the value of the constant mean."""
        yield np.ones((1, X.shape[0]))                  # derivative wrt bias

        X0 = np.array(self._centre, ndmin=2)
        D = (X - X0) / (self._widths ** 2)
        yield 2 * D.T                                   # derivative wrt centre

        X, X0 = rescale(self._widths, X, X0)
        D2 = (X - X0) ** 2
        K = 2 / self._widths
        if self._iso:
            yield K * np.sum(D2, axis=1).T              # derivative wrt widths
        else:
            G = K * D2
            yield G.T

    def get_gradx(self, X):
        """Gradient wrt the inputs X."""
        X0 = np.array(self._centre, ndmin=2)
        D = X - X0
        return -2 * D / (self._widths ** 2)