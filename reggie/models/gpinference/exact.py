"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import mwhutils.linalg as linalg

from ._core import Inference

__all__ = ['Exact']


class Exact(Inference):
    def __init__(self):
        self.init()

    def init(self):
        self.L = None
        self.a = None

    def update(self, like, kern, mean, X, Y):
        K = linalg.add_diagonal(kern.get_kernel(X), like.get_variance())
        r = Y - mean.get_function(X)
        self.L = linalg.cholesky(K)
        self.a = linalg.solve_triangular(self.L, r)

    def updateinc(self, like, kern, mean, X_, X, Y):
        B = kern.get_kernel(X, X_)
        C = linalg.add_diagonal(kern.get_kernel(X), like.get_variance())
        r = Y - mean.get_function(X)
        self.L, self.a = linalg.cholesky_update(self.L, B, C, self.a, r)
