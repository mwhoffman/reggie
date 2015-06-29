"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ._core import RealKernel
from ._distances import rescale, dist, dist_foreach, diff
from ..core.domains import POSITIVE

__all__ = ['SE']


class SE(RealKernel):
    """
    The squared-exponential kernel with lengthscales ell and signal variance
    rho. If ndim is None then this will be an ARD kernel over an input space
    whose dimensions are given by the size of ell; otherwise the kernel will be
    isotropic and ell must be a scalar.
    """
    def __init__(self, rho, ell, ndim=None):
        super(SE, self).__init__()

        # get the shape that ell should be
        shape = ('d',) if (ndim is None) else ()

        # register our parameters
        self._rho = self._register('rho', rho, POSITIVE)
        self._ell = self._register('ell', ell, POSITIVE, shape)

        # save flags for iso/ndim
        self._iso = ndim is not None
        self.ndim = ndim if self._iso else self._ell.size

    def __info__(self):
        info = []
        info.append(('rho', self._rho))
        info.append(('ell', self._ell))
        if self._iso:
            info.append(('ndim', self.ndim))
        return info

    def get_kernel(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        K = self._rho * np.exp(-D/2)
        return K

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

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

    def get_dgrad(self, X1):
        yield np.ones(len(X1))
        for _ in xrange(self.params.size-1):
            yield np.zeros(len(X1))

    def get_gradx(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = diff(X1, X2)
        K = self._rho * np.exp(-0.5 * np.sum(D**2, axis=-1))
        G = -K[:, :, None] * D / self._ell
        return G

    def get_dgradx(self, X1):
        return np.zeros_like(X1)

    def get_gradxy(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = diff(X1, X2)
        K = self._rho * np.exp(-0.5 * np.sum(D**2, axis=-1))
        D /= self._ell
        M = np.eye(self.ndim) / self._ell**2 - D[:, :, None] * D[:, :, :, None]
        G = M * K[:, :, None, None]
        return G

    def sample_spectrum(self, N, rng=None):
        rng = random.rstate(rng)
        W = rng.randn(N, self.ndim) / self._ell
        alpha = float(self._rho)
        return W, alpha
