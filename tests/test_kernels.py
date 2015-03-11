"""
Tests for kernel objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import reggie.kernels as rk


### BASE TEST CLASSES #########################################################

class KernelTest(object):
    def __init__(self, kernel, X1, X2):
        self.kernel = kernel
        self.X1 = X1
        self.X2 = X2

    def test_kernel(self):
        K = self.kernel.get_kernel(self.X1, self.X1)
        k = self.kernel.get_dkernel(self.X1)
        nt.assert_allclose(k, K.diagonal())

    def test_dgrad(self):
        g = np.array(list(self.kernel.get_dgrad(self.X1)))
        G = np.array([_.diagonal() for _ in self.kernel.get_grad(self.X1,
                                                                 self.X1)])
        nt.assert_allclose(g, G)

    def test_call(self):
        m = self.X1.shape[0]
        n = self.X2.shape[0]
        K = self.kernel.get_kernel(self.X1, self.X2)
        K_ = np.array([self.kernel(x1, x2)
                       for x1 in self.X1
                       for x2 in self.X2]).reshape(m, n)
        nt.assert_equal(K, K_)

    def test_grad(self):
        x = self.kernel.get_params()
        k = lambda x, x1, x2: self.kernel.copy(x)(x1, x2)

        G = np.array(list(self.kernel.get_grad(self.X1, self.X2)))
        m = self.X1.shape[0]
        n = self.X2.shape[0]

        G_ = np.array([spop.approx_fprime(x, k, 1e-8, x1, x2)
                       for x1 in self.X1
                       for x2 in self.X2]).swapaxes(0, 1).reshape(-1, m, n)

        nt.assert_allclose(G, G_, rtol=1e-6, atol=1e-6)


class RealKernelTest(KernelTest):
    def __init__(self, kernel):
        rng = np.random.RandomState(0)
        X1 = rng.rand(5, kernel.ndim)
        X2 = rng.rand(3, kernel.ndim)
        super(RealKernelTest, self).__init__(kernel, X1, X2)


### PER-INSTANCE TEST CLASSES #################################################

class TestSEARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, rk.SE(0.8, [0.3, 0.4]))


class TestSEIso(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, rk.SE(0.8, 0.3, ndim=2))
