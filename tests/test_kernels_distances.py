"""
Tests for distance helpers.
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt

import reggie.kernels._distances as distances


def test_rescale():
    X1 = np.random.rand(5, 2)
    X2 = np.random.rand(10, 2)
    ell = np.array([5, 5.])

    nt.assert_equal(distances.rescale(5, X1, X2),
                    distances.rescale(ell, X1, X2))

    assert distances.rescale(5, X1, None)[1] is None


def test_diff():
    X1 = np.random.rand(5, 2)
    X2 = np.random.rand(10, 2)

    D1 = distances.diff(X1, X2)
    D2 = np.array([[x1-x2 for x2 in X2] for x1 in X1])
    nt.assert_equal(D1, D2)
    nt.assert_equal(distances.diff(X1), distances.diff(X1, X1))


def test_dist():
    X1 = np.random.rand(5, 2)
    nt.assert_equal(distances.dist(X1), distances.dist(X1, X1))


def test_dist_foreach():
    X1 = np.random.rand(5, 2)
    X2 = np.random.rand(10, 2)

    D1 = np.array(list(distances.dist_foreach(X1, X2)))
    D2 = np.array([distances.dist(x1[:, None], x2[:, None])
                   for x1, x2 in zip(X1.T, X2.T)])

    nt.assert_equal(D1, D2)
    nt.assert_equal(np.array(list(distances.dist_foreach(X1))),
                    np.array(list(distances.dist_foreach(X1, X1))))
