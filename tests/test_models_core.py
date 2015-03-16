"""
Tests for base model objects.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

import reggie.models._core as models


class TmpModel(models.Model):
    pass


class TmpModelInc(models.Model):
    def __init__(self):
        self.updated = False

    def _updateinc(self, X, Y):
        self.updated = True


def test_updateinc():
    model = TmpModelInc()
    X = np.random.rand(5, 2)
    Y = np.random.rand(5)
    model.add_data(X, Y)
    model.add_data(X, Y)
    assert model.updated


class TestModels(object):
    def __init__(self):
        self.model = TmpModel()
        self.model.add_data(np.random.rand(5, 2), np.random.rand(5))

    def test_ndata(self):
        nt.assert_equal(self.model.ndata, 5)

    def test_reset(self):
        self.model.reset()
        nt.assert_equal(self.model.ndata, 0)

    def test_copy(self):
        model = self.model.copy()
        assert id(self.model.data[0]) == id(model.data[0])
        assert id(self.model.data[1]) == id(model.data[1])
        nt.assert_equal(self.model.copy(reset=True).ndata, 0)

    def test_add_data(self):
        self.model.add_data(np.random.rand(5, 2), np.random.rand(5))
        nt.assert_equal(self.model.ndata, 10)

    def test_optimize(self):
        self.model.optimize()
