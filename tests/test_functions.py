"""
Tests for function objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import reggie.functions as functions


### BASE TEST CLASSES #########################################################

class FunctionTest(object):
    def __init__(self, function, X):
        self.function = function
        self.X = X

    def test_repr(self):
        _ = repr(self.function)

    def test_call(self):
        F1 = self.function.get_function(self.X)
        F2 = np.array([self.function(x) for x in self.X])
        nt.assert_equal(F1, F2)

    def test_get_grad(self):
        if self.function.nparams > 0:
            x = self.function.get_params()
            f = lambda x, x_: self.function.copy(x)(x_)
            G1 = np.array(list(self.function.get_grad(self.X)))
            G2 = np.array([spop.approx_fprime(x, f, 1e-8, x_)
                           for x_ in self.X]).T
            nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_get_gradx(self):
        f = self.function
        G1 = self.function.get_gradx(self.X)
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)


### PER-INSTANCE TEST CLASSES #################################################

class TestZero(FunctionTest):
    def __init__(self):
        f = functions.Zero()
        X = np.random.rand(5, 2)
        FunctionTest.__init__(self, f, X)


class TestConstant(FunctionTest):
    def __init__(self):
        f = functions.Constant(1.0)
        X = np.random.rand(5, 2)
        FunctionTest.__init__(self, f, X)
