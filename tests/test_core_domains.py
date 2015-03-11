"""
Tests for domain helpers.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop
import copy

import reggie.core.domains as domains


### BASE TEST CLASS ###########################################################

class TransformTest(object):
    def __init__(self, domain, transform):
        self.bounds = domains.BOUNDS[domain]
        self.transform = transform

    def test_singleton(self):
        t1 = type(self.transform)()
        t2 = type(self.transform)()
        t3 = copy.copy(self.transform)
        t4 = copy.deepcopy(self.transform)
        assert t1 == t2 == t3 == t4

    def test_bounds(self):
        dbounds = np.array(self.bounds)
        tbounds = self.transform.get_transform(dbounds)
        nt.assert_allclose(self.transform.get_inverse(tbounds), dbounds)

    def test_gradfactor(self):
        for t in [0.1, 0.5, 1, 2]:
            t = np.array([t])
            x = self.transform.get_inverse(t)
            d1 = self.transform.get_gradfactor(x)
            d2 = spop.approx_fprime(t, self.transform.get_inverse, 1e-8)
            nt.assert_allclose(d1, d2, rtol=1e-6)


### PER-INSTANCE TESTS ########################################################

class TestLogTransform(TransformTest):
    def __init__(self):
        TransformTest.__init__(self, domains.POSITIVE, domains.Log())


class TestIdentityTransform(TransformTest):
    def __init__(self):
        TransformTest.__init__(self, domains.REAL, domains.Identity())
