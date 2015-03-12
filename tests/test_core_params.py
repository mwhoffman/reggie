"""
Tests for the Parameter/Parameterized objects.
"""

# pylint: disable=wildcard-import
# pylint: disable=missing-docstring
# pylint: disable=protected-access

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

from reggie.core.domains import *
from reggie.core.params import Parameterized


class Child(Parameterized):
    def __init__(self, a, b):
        self.a = self._register('a', a, domain=REAL, shape='dd')
        self.b = self._register('b', b, domain=POSITIVE)


class Parent(Parameterized):
    def __init__(self, a, b):
        self.a = self._register('a', a, Child)
        self.b = self._register('b', b, Child)


class Empty(Parameterized):
    pass


def test_init():
    nt.assert_raises(ValueError, Child, np.random.rand(4, 3), 1)
    nt.assert_raises(ValueError, Parent, 1, 2)

    a = Child(1, 1)
    obj = Parent(a, a)

    assert id(obj.a) != id(obj.b)
    nt.assert_raises(ValueError, Child, 1, 'asdf')


class TestParams(object):
    def __init__(self):
        a = Child(1, 1)
        b = Child(np.ones((2, 2)), 1)

        # create a parent object to test
        self.obj = Parent(a, b)
        self.obj.set_prior('a.a', 'normal', 1, 2)
        self.obj.set_prior('a.b', 'uniform', 1, 2)
        self.obj.set_prior('b.a', 'uniform', np.full(4, 1), np.full(4, 2))
        self.obj.set_prior('b.b', None)

        # create an empty object
        self.empty = Empty()

    def test_pretty(self):
        _ = repr(self.obj)
        _ = self.obj.describe()

    def test_copy(self):
        obj = self.obj.copy()
        nt.assert_equal(obj.a.a, self.obj.a.a)
        assert id(obj.a.a) != id(self.obj.a.a)

        theta = 1.5 * np.ones(self.obj.nparams)
        obj = self.obj.copy(theta)
        nt.assert_equal(obj.get_params(), theta)

    def test_flatten(self):
        obj1 = self.obj.copy()
        obj1._flatten()

        obj2 = self.obj.copy()
        obj2._flatten({'a.a': 'a', 'a.b': 'b', 'b.a': 'c', 'b.b': 'd'})

        nt.assert_equal(obj1.get_names(), self.obj.get_names())
        nt.assert_equal(obj2.get_names(), ['a', 'b', 'c[0]', 'c[1]', 'c[2]',
                                           'c[3]', 'd'])

    def test_transform(self):
        obj = self.obj.copy(self.obj.get_params(True), True)
        nt.assert_equal(obj.get_params(), self.obj.get_params())

    def test_set_param(self):
        nt.assert_raises(ValueError, self.obj.set_param, 'a', 0)
        nt.assert_raises(ValueError, self.obj.set_param, 'a.b', 0)
        nt.assert_raises(ValueError, self.obj.set_param, 'asdf', 0)
        self.obj.set_param('a.a', 0)
        nt.assert_equal(self.obj.a.a, 0)

    def test_set_params(self):
        nt.assert_raises(ValueError, self.obj.set_params, 1)

        theta = 1.5 * np.ones(self.obj.nparams)
        self.obj.set_params(theta)

        nt.assert_equal(self.obj.get_params(), 1.5)
        nt.assert_equal(self.obj.a.a, 1.5)

    def test_get_params(self):
        nt.assert_equal(self.obj.get_params(), np.ones(self.obj.nparams))
        nt.assert_equal(self.empty.get_params(), np.array([]))

    def test_gradfactor(self):
        nt.assert_equal(self.obj.get_gradfactor(), np.ones(self.obj.nparams))
        nt.assert_equal(self.empty.get_gradfactor(), np.array([]))

    def test_priors(self):
        set_prior = self.obj.set_prior
        nt.assert_raises(ValueError, set_prior, 'a.b', 'uniform', -1, 1)
        _ = self.obj.get_logprior()
        _, _ = self.obj.get_logprior(True)

        nt.assert_equal(self.empty.get_logprior(), 0)
        nt.assert_equal(self.empty.get_logprior(True), (0, np.array([])))

    def test_bounds(self):
        assert not np.any(np.isnan(self.obj.get_bounds()))
        assert not np.any(np.isnan(self.obj.get_bounds(True)))

    def test_blocks(self):
        self.obj.set_block('a.a', 1)
        self.obj.set_block('a.b', 1)
        blocks1 = {(0, 1), tuple(range(2, 7))}
        blocks2 = set(map(tuple, self.obj.get_blocks()))
        nt.assert_equal(blocks1, blocks2)

    def test_names(self):
        names = ['a.a', 'a.b']
        names.extend(['b.a[{:d}]'.format(i) for i in xrange(4)])
        names.append('b.b')
        nt.assert_equal(self.obj.get_names(), names)
