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

import reggie.likelihoods as likelihoods


def check_grad_one(g, f, x, *args):
    if len(args) == 0:
        g1 = spop.approx_fprime(x, f, 1e-8)
    else:
        g1 = np.array([
            spop.approx_fprime(x, f, 1e-8, *[_[None, i] for _ in args])
            for i in xrange(len(args[0]))])
    nt.assert_allclose(g1, g.reshape(g1.shape), rtol=1e-6, atol=1e-6)


def check_grad(g, f, x, *args):
    g1 = np.array([
        spop.approx_fprime(x[None, i], f, 1e-8, *[_[None, i] for _ in args])
        for i in xrange(len(x))])
    nt.assert_allclose(g1, g.reshape(g1.shape), rtol=1e-6, atol=1e-6)


### BASE TEST CLASSES #########################################################

class LikelihoodTest(object):
    def __init__(self, like):
        self.like = like
        self.f = 2*np.random.rand(50) - 1
        self.y = self.like.sample(self.f)

    def test_repr(self):
        _ = repr(self.like)

    def test_get_logprob(self):
        lp = lambda f, y: self.like.get_logprob(y, f)[0]
        d1 = lambda f, y: self.like.get_logprob(y, f)[1]
        d2 = lambda f, y: self.like.get_logprob(y, f)[2]
        d3 = lambda f, y: self.like.get_logprob(y, f)[3]

        check_grad(d1(self.f, self.y), lp, self.f, self.y)
        check_grad(d2(self.f, self.y), d1, self.f, self.y)
        check_grad(d3(self.f, self.y), d2, self.f, self.y)

    def test_get_laplace_grad(self):
        if self.like.params.size > 0:
            lp = lambda x, y, f: self.like.copy(x[0]).get_logprob(y, f)[0]
            d1 = lambda x, y, f: self.like.copy(x[0]).get_logprob(y, f)[1]
            d2 = lambda x, y, f: self.like.copy(x[0]).get_logprob(y, f)[2]

            theta = self.like.params.get_value()
            grads = list(self.like.get_laplace_grad(self.y, self.f))
            g0, g1, g2 = np.rollaxis(np.array(grads), 0, 3)

            check_grad_one(g0, lp, theta, self.y, self.f)
            check_grad_one(g1, d1, theta, self.y, self.f)
            check_grad_one(g2, d2, theta, self.y, self.f)

        else:
            assert list(self.like.get_laplace_grad(self.y, self.f)) == []


### PER-INSTANCE TEST CLASSES #################################################

class TestGaussian(LikelihoodTest):
    def __init__(self):
        LikelihoodTest.__init__(self, likelihoods.Gaussian(0.8))


class TestProbit(LikelihoodTest):
    def __init__(self):
        LikelihoodTest.__init__(self, likelihoods.Probit())
