"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import itertools as it
import mwhutils.linalg as la

from ._core import Inference

__all__ = ['FITC']


class FITC(Inference):
    def __init__(self, like, kern, mean, U):
        super(FITC, self).__init__(like, kern, mean)
        self.U = np.array(U, ndmin=2, dtype=float, copy=True)

    def __info__(self):
        info = super(FITC, self).__info__()
        info.append(('U', self.U))
        return info

    def init(self):
        super(FITC, self).init()
        self.L = None
        self.C = None
        self.a = None

    def update(self, X, Y):
        sn2 = self.like.get_variance()
        su2 = sn2 / 1e6

        # get the kernel matrices
        Kux = self.kern.get_kernel(self.U, X)
        kxx = self.kern.get_dkernel(X) + sn2
        Kuu = la.add_diagonal(self.kern.get_kernel(self.U), su2)
        Luu = la.cholesky(Kuu)

        V = la.solve_triangular(Luu, Kux)
        r = (Y - self.mean.get_function(X))

        ell = np.sqrt(kxx - np.sum(V**2, axis=0))
        V /= ell
        r /= ell

        L = la.cholesky(la.add_diagonal(np.dot(V, V.T), 1))
        b = la.solve_triangular(L, np.dot(V, r))
        a = (r - np.dot(V.T, la.solve_triangular(L, b, True))) / ell

        # the log-likelihood
        lZ = -np.sum(np.log(L.diagonal())) - np.sum(np.log(ell))
        lZ -= 0.5 * (np.inner(r, r) - np.inner(b, b))
        lZ -= 0.5 * len(X)*np.log(2*np.pi)

        # components needed for the gradient
        B = la.solve_triangular(Luu, V*ell, True)
        W = la.solve_triangular(L, V/ell)
        w = np.dot(B, a)
        v = 2 * su2 * np.sum(B**2, axis=0)

        # allocate space for the derivatives
        dlZ = np.zeros(self.nparams)

        # derivative wrt sn2
        dlZ[0] = 0.5 * (
            - (np.sum(ell**-2) - np.sum(W**2) - np.inner(a, a))
            - (np.sum(w**2) + np.sum(np.dot(B, W.T)**2)) / 1e6
            + (np.inner(a, v*a) + np.inner(np.sum(W**2, axis=0), v)) / 2 / sn2)

        # iterator over gradients of the kernels
        dK = it.izip(
            self.kern.get_grad(self.U),
            self.kern.get_grad(self.U, X),
            self.kern.get_dgrad(X))

        # we need to keep track of how many gradients we've already computed.
        # note also that at the end of the next loop this variable will have
        # changed to track the current number of gradients.
        i = self.like.nparams

        for i, (dKuu, dKux, dkxx) in enumerate(dK, i):
            M = 2*dKux - np.dot(dKuu, B)
            v = dkxx - np.sum(M*B, axis=0)
            dlZ[i] = (
                - np.sum(dkxx/ell**2)
                - np.inner(w, dKuu.dot(w) - 2*dKux.dot(a))
                + np.inner(a, v*a) + np.inner(np.sum(W**2, axis=0), v)
                + np.sum(M.dot(W.T) * B.dot(W.T))) / 2.0

        for i, dmu in enumerate(self.mean.get_grad(X), i+1):
            dlZ[i] = np.dot(dmu, a)

        # save the posterior
        self.L = Luu
        self.C = np.dot(Luu, L)
        self.a = la.solve_cholesky(self.C, np.dot(Kux, r/ell))

        # save the log-likelihood and its gradient
        self.lZ = lZ
        self.dlZ = dlZ
