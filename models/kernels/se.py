"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.spatial.distance as ssd

from .kernel import Kernel

__all__ = ['SE']


class SE(Kernel):
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
