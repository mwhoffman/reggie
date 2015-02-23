"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.spatial.distance as ssd

from ..core.transforms import Log
from .kernel import Kernel

__all__ = ['SE']


class SE_(Kernel):
    def __init__(self, logsf, logell, ndim=None):
        self._register('logsf', logsf, ndim=0)
        self._register('logell', logell, ndim=1 if (ndim is None) else 0)

        self._ndim = self.logell.nparams if (ndim is None) else ndim
        self._iso = ndim is not None

    def get_kernel(self, X1, X2):
        X1 = X1 / np.exp(self._logell)
        X2 = X2 / np.exp(self._logell)
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        K = np.exp(self._logsf*2 - D/2)
        return K

    def get_grad(self, X1, X2):
        X1 = X1 / np.exp(self._logell)
        X2 = X2 / np.exp(self._logell)
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        K = np.exp(self._logsf*2 - D/2)

        yield 2*K           # derivative wrt logsf
        if self._iso:
            yield K*D       # derivative wrt logell (iso)
        else:
            for i in xrange(X1.shape[1]):
                D = ssd.cdist(X1[:, i, None], X2[:, i, None], 'sqeuclidean')
                yield K*D   # derivatives wrt logell (ard)

    def get_dkernel(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def get_dgrad(self, X1):
        yield 2 * self.get_dkernel(X1)
        for _ in xrange(self.nparams-1):
            yield np.zeros(len(X1))


class SE(Kernel):
    def __init__(self, logsf, logell, ndim=None):
        self._register('sf2', np.exp(logsf*2), ndim=0)
        self._register('ell', np.exp(logell), ndim=1 if (ndim is None) else 0)

        self._ndim = self.ell.nparams if (ndim is None) else ndim
        self._iso = ndim is not None

        self.sf2.set_transform(Log())
        self.ell.set_transform(Log())

    def get_kernel(self, X1, X2):
        X1 = X1 / self._ell
        X2 = X2 / self._ell
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        K = self._sf2 * np.exp(-D/2)
        return K

    def get_grad(self, X1, X2):
        X1 = X1 / self._ell
        X2 = X2 / self._ell
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        E = np.exp(-D/2)
        K = self._sf2 * E

        yield E                                 # derivative wrt sf2
        if self._iso:
            yield K * D / self._ell             # derivative wrt ell (iso)
        else:
            for i, ell in enumerate(self._ell):
                D = ssd.cdist(X1[:, i, None],
                              X2[:, i, None],
                              'sqeuclidean')
                yield K * D / ell               # derivative wrt ell (ard)

    def get_dkernel(self, X1):
        return np.full(len(X1), self._sf2)

    def get_dgrad(self, X1):
        yield np.ones(len(X1))
        for _ in xrange(self.nparams-1):
            yield np.zeros(len(X1))
