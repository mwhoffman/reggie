"""
Tests for priors.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import reggie.core.priors as priors


### BASE TEST CLASS ###########################################################

class PriorTest(object):
    def __init__(self, prior):
        self.prior = prior

    def test_repr(self):
        _ = repr(self.prior)

    def test_bounds(self):
        bshape = np.shape(self.prior.bounds)
        assert bshape == (2,) or bshape == (self.prior.ndim, 2)

    def test_sample(self):
        assert np.shape(self.prior.sample()) == (self.prior.ndim,)
        assert np.shape(self.prior.sample(5)) == (5, self.prior.ndim)

    def test_logprior(self):
        for theta in self.prior.sample(5, 0):
            g1 = spop.approx_fprime(theta, self.prior.get_logprior, 1e-8)
            _, g2 = self.prior.get_logprior(theta, True)
            nt.assert_allclose(g1, g2, rtol=1e-6)


### PER-INSTANCE TESTS ########################################################

class TestUniform(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, priors.Uniform([0, 0], [1, 1]))


class TestNormal(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, priors.Normal([0, 0], [1, 1]))


class TestLogNormal(PriorTest):
    def __init__(self):
        PriorTest.__init__(self, priors.LogNormal([0, 0], [1, 1]))


def test_uniform():
    nt.assert_raises(ValueError, priors.Uniform, 0, -1)
