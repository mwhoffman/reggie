"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import scipy.spatial.distance as ssd

__all__ = ['rescale', 'diff', 'dist', 'dist_foreach']


def rescale(ell, X1, X2=None):
    """
    Rescale the input data contained in X1 and X2 with lengthscales ell. Return
    the tuple of rescaled data, or if X2 is None return (X1, None).
    """
    X1 = (X1 / ell)
    X2 = (X2 / ell) if (X2 is not None) else None
    return X1, X2


def diff(X1, X2=None):
    """
    Compute the pairwise differences between input data contained in X1 and X2.
    If X2 is None then compute the differences between X1 and itself.
    """
    X2 = X1 if (X2 is None) else X2
    return X1[:, None, :] - X2[None, :, :]


def dist(X1, X2=None, metric='sqeuclidean'):
    """
    Compute the pairwise distances between input data contained in X1 and X2
    with some given metric (passed directly to scipy's cdist). If X2 is None
    then compute the distances between X1 and itself.
    """
    X2 = X1 if (X2 is None) else X2
    return ssd.cdist(X1, X2, metric)


def dist_foreach(X1, X2=None, metric='sqeuclidean'):
    """
    Return a generator which yields the pairwise distances between each
    dimension of the data contained in X1 and X2.  If X2 is None then compute
    the distances between X1 and itself.
    """
    X2 = X1 if (X2 is None) else X2
    for i in xrange(X1.shape[1]):
        yield ssd.cdist(X1[:, i, None], X2[:, i, None], metric)
