"""
Tests for kernel objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

import models.kernels as mk


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


class RealKernelTest(KernelTest):
    def __init__(self, kernel):
        X1 = np.random.rand(5, kernel._ndim)
        X2 = np.random.rand(5, kernel._ndim)
        super(RealKernelTest, self).__init__(kernel, X1, X2)


### PER-INSTANCE TEST CLASSES #################################################

class TestSEARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, mk.SE(0.8, [0.3, 0.4]))


class TestSEIso(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, mk.SE(0.8, 0.3, ndim=2))
