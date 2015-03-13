"""
Tests for models.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import reggie.gps as gps


### BASE TEST CLASSES #########################################################

class ModelTest(object):
    def __init__(self, model):
        self.model = model

    def test_repr(self):
        _ = repr(self.model)

    def test_add_data(self):
        X, Y = self.model.data
        n = self.model.ndata // 2

        model = self.model.copy(reset=True)
        model.add_data(X[:n], Y[:n])
        model.add_data(X[n:], Y[n:])
        nt.assert_allclose(model.get_posterior(X), self.model.get_posterior(X))

    def test_get_loglike(self):
        # first make sure we can call it with zero data.
        _ = self.model.copy(reset=True).get_loglike()
        _, _ = self.model.copy(reset=True).get_loglike(True)

        # and with some data
        _ = self.model.get_loglike()
        _, _ = self.model.get_loglike(True)

        # and test the gradients
        x = self.model.get_params()
        f = lambda x: self.model.copy(x).get_loglike()
        _, g1 = self.model.get_loglike(grad=True)
        g2 = spop.approx_fprime(x, f, 1e-8)
        nt.assert_allclose(g1, g2, rtol=1e-6, atol=1e-6)

    def test_get_posterior(self):
        # first check that we can even evaluate the posterior.
        X, _ = self.model.data
        _ = self.model.get_posterior(X)
        _ = self.model.get_posterior(X, predictive=True)

        # check the mu gradients
        f = lambda x: self.model.get_posterior(x[None])[0]
        G1 = self.model.get_posterior(X, grad=True)[2]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

        # check the s2 gradients
        f = lambda x: self.model.get_posterior(x[None])[1]
        G1 = self.model.get_posterior(X, grad=True)[3]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)


### PER-INSTANCE TEST CLASSES #################################################

class TestGP(ModelTest):
    def __init__(self):
        gp = gps.BasicGP(1, 1, [1., 1.])
        gp.add_data(np.random.rand(10, 2), np.random.rand(10))
        ModelTest.__init__(self, gp)