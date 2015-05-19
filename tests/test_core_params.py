"""
Tests for the Parameter/Parameterized objects.
"""

# pylint: disable=wildcard-import
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import copy
import numpy as np
import numpy.testing as nt

from reggie.core.domains import REAL
from reggie.core.params import Parameter, Parameters, Parameterized


class ParameterTest(object):
    def __init__(self, value, domain):
        self.param = Parameter(value, domain)

    def test_deepcopy(self):
        param = copy.deepcopy(self.param)
        nt.assert_equal(param.value, self.param.value)
        assert isinstance(param.value, np.ndarray)
        assert id(param.value) != id(self.param.value)

    def test_gradfactor(self):
        assert self.param.gradfactor.shape == self.param.value.ravel().shape

    def test_get_value(self):
        # test untransformed
        value = self.param.get_value()
        nt.assert_equal(value, self.param.value.ravel())
        assert isinstance(value, np.ndarray)
        assert id(value) != id(self.param.value)

        # test transformed
        value = self.param.get_value(True)
        assert isinstance(value, np.ndarray)
        assert value.shape == self.param.value.ravel().shape

    def test_set_value(self):
        # test untransformed
        value1 = self.param.get_value()
        self.param.set_value(value1)
        value2 = self.param.get_value()
        nt.assert_equal(value2, value1)

        # test transformed
        value1 = self.param.get_value(True)
        self.param.set_value(self.param.get_value(True), True)
        value2 = self.param.get_value()
        nt.assert_equal(value2, value1)

    def test_set_prior(self):
        ones = np.ones(self.param.size)

        # set a uniform prior which should change the bounds to [2, 3] and
        # raise a warning. this should also make set_value raise an error if we
        # try to set the value to all ones.
        nt.assert_warns(UserWarning, self.param.set_prior, 'uniform', 2, 3)
        nt.assert_raises(ValueError, self.param.set_value, ones)

        # removing the prior we should then be able to set the parameters.
        self.param.set_prior(None)
        self.param.set_value(ones)

    def test_get_logprior(self):
        zeros = np.zeros(self.param.size)
        nt.assert_equal(self.param.get_logprior(), 0)
        nt.assert_equal(self.param.get_logprior(True), (0, zeros))

        self.param.set_prior('uniform', 0, 2)
        nt.assert_equal(self.param.get_logprior(), 0)
        nt.assert_equal(self.param.get_logprior(True), (0, zeros))


class TestParameter(ParameterTest):
    def __init__(self):
        ParameterTest.__init__(self, np.ones(2), REAL)


class TestParameterArray(ParameterTest):
    def __init__(self):
        ParameterTest.__init__(self, np.ones((2, 2)), REAL)


class TestParameterScalar(ParameterTest):
    def __init__(self):
        ParameterTest.__init__(self, np.array(1.), REAL)


class Dummy(object):
    # dummy object which will be used to test an instance of the Parameters
    # object without requiring a Parameterized object.
    def __init__(self):
        self.updates = 0

    def _update(self):
        self.updates += 1


class TestParameters(object):
    def __init__(self):
        # create an object
        self.obj = Dummy()
        self.params = Parameters(self.obj)
        self.params._register('foo', Parameter(np.zeros((2, 2)), REAL))
        self.params._register('bar', Parameter(np.array(0.), REAL))

    def test_deepcopy(self):
        params = copy.deepcopy(self.params)
        for param1, param2 in zip(self.params._Parameters__params.values(),
                                  params._Parameters__params.values()):
            nt.assert_equal(param2.value, param1.value)
            assert isinstance(param1.value, np.ndarray)
            assert isinstance(param2.value, np.ndarray)
            assert id(param2.value) != id(param1.value)

    def test_getitem(self):
        nt.assert_raises(ValueError, self.params.__getitem__, ('foo', 'foo'))
        nt.assert_raises(ValueError, self.params.__getitem__, ('baz',))

        params = self.params['foo']
        assert isinstance(params, Parameters)
        assert params.size == 4

    def test_register(self):
        param = Parameter(np.zeros(2), REAL)
        nt.assert_raises(ValueError, self.params._register, 'foo', param)

        params = Parameters(Dummy())
        params._register('foo', Parameter(np.zeros(2), REAL))
        params._register('bar', Parameter(np.zeros(1), REAL))
        self.params._register('foo', params)

        nt.assert_raises(ValueError, self.params._register, None, params)
        nt.assert_raises(ValueError, self.params._register, None, 1)

    def test_gradfactor(self):
        assert self.params.gradfactor.shape == (self.params.size,)
        assert Parameters(Dummy()).gradfactor.shape == (0,)

    def test_blocks(self):
        # everything should be in the same block
        nt.assert_equal(self.params.block, [0, 0])
        nt.assert_equal(self.params.blocks, [range(self.params.size)])

        # set one of the blocks
        self.params['bar'].block = 1
        nt.assert_equal([0, 1], self.params.block)
        blocks1 = set(map(tuple, self.params.blocks))
        blocks2 = set(map(tuple, [range(4), [4]]))
        nt.assert_equal(blocks1, blocks2)

        # make sure the setter raises an error if it gets an incorrect number
        # of identifiers for the blocks.
        def setter(value):
            self.params.block = value
        nt.assert_raises(ValueError, setter, [1, 1, 1])

    def test_names(self):
        names = ['foo[{:d}]'.format(_) for _ in range(4)] + ['bar']
        nt.assert_equal(self.params.names, names)

    def test_describe(self):
        self.params.describe()

    def test_set_prior(self):
        nt.assert_raises(RuntimeError, self.params.set_prior, 'uniform', 0, 1)
        self.params['foo'].set_prior('uniform', 0, 1)
        self.params['bar'].set_prior('uniform', 0, 1)
        assert self.obj.updates == 2

    def test_get_value(self):
        assert self.params.get_value().shape == (self.params.size,)
        assert Parameters(Dummy()).get_value().shape == (0,)

    def test_set_value(self):
        nt.assert_raises(ValueError, self.params.set_value, np.zeros(3))
        self.params.set_value(self.params.get_value())
        assert self.obj.updates == 1

    def test_get_bounds(self):
        assert self.params.get_bounds().shape == (self.params.size, 2)
        assert self.params.get_bounds(True).shape == (self.params.size, 2)

    def test_get_logprior(self):
        zeros = np.zeros(self.params.size)
        nt.assert_equal(self.params.get_logprior(), 0)
        nt.assert_equal(self.params.get_logprior(True), (0, zeros))

        params = Parameters(Dummy())
        nt.assert_equal(params.get_logprior(), 0)
        nt.assert_equal(params.get_logprior(True), (0, np.zeros(0)))


class Null(Parameterized):
    pass


class Inner(Parameterized):
    def __init__(self):
        super(Inner, self).__init__()
        self._a = self._register('a', 1.0)
        self._b = self._register('b', 2.0)

    def __info__(self):
        info = super(Inner, self).__info__()
        info.append(('a', self._a))
        info.append(('b', self._b))
        return info


class Outer(Parameterized):
    def __init__(self):
        super(Outer, self).__init__()
        self._a = self._register('a', np.ones(5), shape=(5,))
        self._b = self._register('b', np.ones(2), shape='d')
        self._c = self._register('c', np.ones((2, 2)), shape='dd')
        self._d = self._register('d', np.ones((3, 2)), shape='mn')
        self._e = self._register('e', 1)
        self._inner = self._register_obj('inner', Inner())

    def __info__(self):
        info = super(Outer, self).__info__()
        info.append(('a', self._a))
        info.append(('b', self._b))
        info.append(('c', self._c))
        info.append(('d', self._d))
        info.append(('e', self._e))
        info.append(('inner', self._inner))
        return info


class TestParameterized(object):
    def __init__(self):
        self.obj = Outer()

    def test_repr(self):
        assert isinstance(repr(self.obj), str)

    def test_deepcopy(self):
        obj1 = self.obj
        obj2 = copy.deepcopy(self.obj)
        for param1, param2 in zip(obj1.params._Parameters__params.values(),
                                  obj2.params._Parameters__params.values()):
            nt.assert_equal(param2.value, param1.value)
            assert isinstance(param1.value, np.ndarray)
            assert isinstance(param2.value, np.ndarray)
            assert id(param2.value) != id(param1.value)

    def test_copy(self):
        obj1 = self.obj.copy()
        obj2 = self.obj.copy(self.obj.params.get_value())
        obj3 = self.obj.copy(self.obj.params.get_value(True), True)

        nt.assert_equal(obj1.params.get_value(), self.obj.params.get_value())
        nt.assert_equal(obj2.params.get_value(), self.obj.params.get_value())
        nt.assert_equal(obj3.params.get_value(), self.obj.params.get_value())

    def test_register(self):
        nt.assert_raises(ValueError, self.obj._register, 'a', 1)
        nt.assert_raises(ValueError, self.obj._register, 'z', 'asdf')
        nt.assert_raises(ValueError, self.obj._register, 'z', 1, shape=2)

    def test_register_obj(self):
        null = Null()
        nt.assert_raises(ValueError, self.obj._register_obj, 'z', 'asdf')
        nt.assert_raises(ValueError, self.obj._register_obj, 'z', 'asdf', str)
        nt.assert_raises(ValueError, self.obj._register_obj, 'z', null, Inner)
