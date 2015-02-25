"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.spatial.distance as ssd

from .kernel import Kernel
from ..core.transforms import Log

__all__ = ['SE']


def rescale(ell, X1, X2=None):
    X1 = (X1 / ell)
    X2 = (X2 / ell) if (X2 is not None) else None
    return X1, X2


def dist(X1, X2=None, metric='sqeuclidean'):
    X2 = X1 if (X2 is None) else X2
    return ssd.cdist(X1, X2, metric)


def dist_foreach(X1, X2=None, metric='sqeuclidean'):
    X2 = X1 if (X2 is None) else X2
    for i in xrange(X1.shape[1]):
        yield ssd.cdist(X1[:, i, None], X2[:, i, None], metric)


class SE(Kernel):
    def __init__(self, rho, ell, ndim=None):
        self._register('rho', rho, ndim=0)
        self._register('ell', ell, ndim=1 if (ndim is None) else 0)

        self._ndim = self.ell.nparams if (ndim is None) else ndim
        self._iso = ndim is not None

        self.rho.set_transform(Log())
        self.ell.set_transform(Log())

    def get_kernel(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        K = self._rho * np.exp(-D/2)
        return K

    def get_grad(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        E = np.exp(-D/2)
        K = self._rho * E

        yield E                                 # derivative wrt rho
        if self._iso:
            yield K * D / self._ell             # derivative wrt ell (iso)
        else:
            for i, D in enumerate(dist_foreach(X1, X2)):
                yield K * D / self._ell[i]      # derivative wrt ell (ard)

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

    def get_dgrad(self, X1):
        yield np.ones(len(X1))
        for _ in xrange(self.nparams-1):
            yield np.zeros(len(X1))
