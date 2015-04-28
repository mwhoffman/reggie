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
        super(Child, self).__init__()
        self.a = self._register('a', a, REAL, 'dd')
        self.b = self._register('b', b, POSITIVE)


class Parent(Parameterized):
    def __init__(self, a, b):
        super(Parent, self).__init__()
        self.a = self._pregister('a', a, Child)
        self.b = self._pregister('b', b, Child)


class Empty(Parameterized):
    pass


class Clashing1(Parameterized):
    def __init__(self):
        super(Clashing1, self).__init__()
        self._register('a', 1)
        self._register('a', 1)


class Clashing2(Parameterized):
    def __init__(self):
        super(Clashing2, self).__init__()
        self._register('a', 1)
        self._pregister(None, Child(1, 1))


class NonArray(Parameterized):
    def __init__(self):
        super(NonArray, self).__init__()
        self._register('a', 'asdf')


def test_empty():
    params = Empty().params
    nt.assert_equal(params.gradfactor, np.array([]))
    nt.assert_equal(params.get_value(), np.array([]))
    nt.assert_equal(params.get_logprior(), 0)
    nt.assert_equal(params.get_logprior(True), (0, np.array([])))


def test_register():
    # the following classes should raise errors when instantiated due to the
    # fact that they call register incorrectly
    nt.assert_raises(ValueError, Clashing1)
    nt.assert_raises(ValueError, Clashing2)
    nt.assert_raises(ValueError, NonArray)

    # trying to register something that is not an instance of Parameter or
    # Parameters should fail; however this shouldn't be called directly.
    nt.assert_raises(ValueError, Child(1, 1).params._register, 'foo', 1)

    # this should fail due to incorrect size of the first parameter
    nt.assert_raises(ValueError, Child, np.random.rand(5, 2), 1)

    # the following should fail since Parent expects a Child in first position
    a = Child(1, 1)
    b = Parent(a, a)
    nt.assert_raises(ValueError, Parent, b, a)


class TestParams(object):
    def __init__(self):
        a = Child(1, 1)
        b = Child(np.ones((2, 2)), 1)

        # create a parent object to test
        self.obj = Parent(a, b)
        self.params = self.obj.params

        # set some priors
        self.params['a.a'].set_prior('normal', 1, 2)
        self.params['a.b'].set_prior('uniform', 1, 2)
        self.params['b.a'].set_prior(None)
        self.params['b.a'].set_prior('uniform', np.full(4, 1), np.full(4, 2))

    def test_getitem(self):
        nt.assert_raises(ValueError, self.params.__getitem__, ('a.a', 'a.a'))
        nt.assert_raises(ValueError, self.params.__getitem__, 'foo')

    def test_pretty(self):
        _ = repr(self.obj)
        _ = self.params.describe()

    def test_copy(self):
        obj = self.obj.copy()
        nt.assert_equal(obj.a.a, self.obj.a.a)
        assert id(obj.a.a) != id(self.obj.a.a)

        theta = 1.5 * np.ones(self.obj.nparams)
        params = self.obj.copy(theta).params
        nt.assert_equal(params.get_value(), theta)

    def test_transform(self):
        params = self.obj.copy(self.params.get_value(True), True).params
        nt.assert_equal(params.get_value(),
                        self.params.get_value())

    def test_set_param(self):
        # try to set out of bounds
        set_value = self.params['a.b'].set_value
        nt.assert_raises(ValueError, set_value, 0)

        # make sure the value changes
        set_value(1)
        nt.assert_equal(self.obj.a.a, 1)

    def test_set_params(self):
        nt.assert_raises(ValueError, self.params.set_value, 1)

        theta = 1.5 * np.ones(self.obj.nparams)
        self.params.set_value(theta)

        nt.assert_equal(self.params.get_value(), 1.5)
        nt.assert_equal(self.obj.a.a, 1.5)

    def test_get_params(self):
        nt.assert_equal(self.params.get_value(), np.ones(self.obj.nparams))

    def test_gradfactor(self):
        nt.assert_equal(self.params.gradfactor, np.ones(self.obj.nparams))

    def test_priors(self):
        set_prior = self.params['a.a', 'a.b'].set_prior
        nt.assert_raises(RuntimeError, set_prior, 'uniform', 1.3, 2)

        set_prior = self.params['a.b'].set_prior
        nt.assert_raises(ValueError, set_prior, 'uniform', -1, 1)

        set_prior = self.params['a.a'].set_prior
        nt.assert_warns(UserWarning, set_prior, 'uniform', 1.3, 2)

        _ = self.params.get_logprior()
        _, _ = self.params.get_logprior(True)

    def test_bounds(self):
        assert not np.any(np.isnan(self.params.get_bounds()))
        assert not np.any(np.isnan(self.params.get_bounds(True)))

    def test_blocks(self):
        self.params['a.a', 'a.b'].block = 1
        blocks1 = {(0, 1), tuple(range(2, 7))}
        blocks2 = set(map(tuple, self.params.blocks))
        nt.assert_equal([1, 1, 0, 0], self.params.block)
        nt.assert_equal(blocks1, blocks2)

        # invalid block assignment
        def assign():
            self.params.block = [1, 1]
        nt.assert_raises(ValueError, assign)

    def test_names(self):
        names = ['a.a', 'a.b']
        names.extend(['b.a[{:d}]'.format(i) for i in xrange(4)])
        names.append('b.b')
        nt.assert_equal(self.params.names, names)
