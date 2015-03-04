"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from . import _distances as dist
from .kernel import RealKernel
from ..core.transforms import Log

__all__ = ['SE']


class SE(RealKernel):
    """
    The squared-exponential kernel with lengthscales ell and signal variance
    rho. If ndim is None then this will be an ARD kernel over an input space
    whose dimensions are given by the size of ell; otherwise the kernel will be
    isotropic and ell must be a scalar.
    """
    def __init__(self, rho, ell, ndim=None):
        self._register('rho', rho)
        self._register('ell', ell, shape=('d',) if (ndim is None) else ())

        self._iso = ndim is not None
        self._ndim = ndim if self._iso else self.ell.nparams

        if self._iso:
            self._kwarg('ndim', self._ndim)

        self.rho.set_transform(Log())
        self.ell.set_transform(Log())

    def get_kernel(self, X1, X2=None):
        X1, X2 = dist.rescale(self._ell, X1, X2)
        D = dist.dist(X1, X2)
        K = self._rho * np.exp(-D/2)
        return K

    def get_grad(self, X1, X2=None):
        X1, X2 = dist.rescale(self._ell, X1, X2)
        D = dist.dist(X1, X2)
        E = np.exp(-D/2)
        K = self._rho * E

        yield E                                 # derivative wrt rho
        if self._iso:
            yield K * D / self._ell             # derivative wrt ell (iso)
        else:
            for i, D in enumerate(dist.dist_foreach(X1, X2)):
                yield K * D / self._ell[i]      # derivative wrt ell (ard)

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

    def get_dgrad(self, X1):
        yield np.ones(len(X1))
        for _ in xrange(self.nparams-1):
            yield np.zeros(len(X1))

    def get_gradx(self, X1, X2=None):
        X1, X2 = dist.rescale(self._ell, X1, X2)
        D = dist.diff(X1, X2)
        K = self._rho * np.exp(-0.5 * np.sum(D**2, axis=-1))
        G = -K[:, :, None] * D / self._ell
        return G
