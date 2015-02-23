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


class SE(Kernel):
    def __init__(self, rho, ell, ndim=None):
        self._register('rho', rho, ndim=0)
        self._register('ell', ell, ndim=1 if (ndim is None) else 0)

        self._ndim = self.ell.nparams if (ndim is None) else ndim
        self._iso = ndim is not None

        self.rho.set_transform(Log())
        self.ell.set_transform(Log())

    def get_kernel(self, X1, X2):
        X1 = X1 / self._ell
        X2 = X2 / self._ell
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        K = self._rho * np.exp(-D/2)
        return K

    def get_grad(self, X1, X2):
        X1 = X1 / self._ell
        X2 = X2 / self._ell
        D = ssd.cdist(X1, X2, 'sqeuclidean')
        E = np.exp(-D/2)
        K = self._rho * E

        yield E                                 # derivative wrt rho
        if self._iso:
            yield K * D / self._ell             # derivative wrt ell (iso)
        else:
            for i, ell in enumerate(self._ell):
                D = ssd.cdist(X1[:, i, None],
                              X2[:, i, None],
                              'sqeuclidean')
                yield K * D / ell               # derivative wrt ell (ard)

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

    def get_dgrad(self, X1):
        yield np.ones(len(X1))
        for _ in xrange(self.nparams-1):
            yield np.zeros(len(X1))
