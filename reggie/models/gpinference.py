"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as spop
import itertools as it
import collections

import mwhutils.linalg as la

__all__ = ['exact']


# create a named tuple which functions as storage for the posterior sufficient
# statistics.
Statistics = collections.namedtuple('Statistics', 'L, a, w, lZ, dlZ, C')
Statistics.__new__.__defaults__ = (None, )


def exact(like, kern, mean, X, Y):
    K = kern.get_kernel(X)
    K = la.add_diagonal(K, like.get_variance())
    r = Y - mean.get_function(X)

    # the posterior parameterization
    L = la.cholesky(K)
    a = la.solve_cholesky(L, r)
    w = np.ones_like(a)

    # we'll need this to compute the log-likelihood derivatives
    Q = la.cholesky_inverse(L) - np.outer(a, a)

    # the log-likelihood
    lZ = -0.5 * np.inner(a, r)
    lZ -= 0.5 * np.log(2 * np.pi) * len(X)
    lZ -= np.sum(np.log(L.diagonal()))

    dlZ = np.r_[
        # derivative wrt the likelihood's noise term.
        -0.5*np.trace(Q),

        # derivative wrt each kernel hyperparameter.
        [-0.5*np.sum(Q*dK) for dK in kern.get_grad(X)],

        # derivative wrt the mean.
        [np.dot(dmu, a) for dmu in mean.get_grad(X)]]

    return Statistics(L, a, w, lZ, dlZ)


def laplace(like, kern, mean, X, Y):
    MAXIT = 60
    MINTOL = 1e-6

    # grab the kernel, mean, and initialize the weights
    K = kern.get_kernel(X)
    L = None
    m = mean.get_function(X)
    a = np.zeros(K.shape[1])

    def psi(a):
        # define the linesearch objective
        r = np.dot(K, a)
        lp, d1, d2, d3 = like.get_logprob(Y, r+m)
        psi = 0.5 * np.inner(r, a) - np.sum(lp)
        return psi, r, d1, d2, d3

    psi1, r, dy1, dy2, dy3 = psi(a)
    psi0 = np.inf

    for _ in xrange(MAXIT):
        # attempt to breakout early
        if np.abs(psi1 - psi0) < MINTOL:
            break
        psi0 = psi1

        # find the step direction
        w = np.sqrt(-dy2)
        L = la.cholesky(la.add_diagonal(np.outer(w, w)*K, 1))
        b = w**2 * r + dy1

        # find the step size
        d = b - a - w*la.solve_cholesky(L, w*np.dot(K, b))
        s = spop.brent(lambda s: psi(a+s*d)[0], tol=1e-4, maxiter=12)

        # update the parameters
        a += s*d
        psi1, r, dy1, dy2, dy3 = psi(a)

    # update the posterior parameters
    w = np.sqrt(-dy2)
    L = la.cholesky(la.add_diagonal(np.outer(w, w)*K, 1))

    # compute the marginal log-likelihood
    lZ = -psi1 - np.sum(np.log(np.diag(L)))

    # compute parameters needed for the hyperparameter gradients
    R = w * la.solve_cholesky(L, np.diag(w))
    C = la.solve_triangular(L, w*K)
    g = 0.5 * (np.diag(K) - np.sum(C**2, axis=0))
    f = r+m
    df = g * dy3

    # define the implicit part of the gradients
    implicit = lambda b: np.dot(df, b - np.dot(K, np.dot(R, b)))

    # allocate space for the gradients
    dlZ = np.zeros(like.params.size + kern.params.size + mean.params.size)

    # the likelihood derivatives
    i = 0
    for dl0, dl1, dl2 in like.get_laplace_grad(Y, f):
        dlZ[i] = np.dot(g, dl2) + np.sum(dl0)
        dlZ[i] += implicit(np.dot(K, dl1))
        i += 1

    # covariance derivatives
    for dK in kern.get_grad(X):
        dlZ[i] = 0.5 * (np.dot(a, np.dot(dK, a)) - np.sum(R*dK))
        dlZ[i] += implicit(np.dot(dK, dy1))
        i += 1

    # mean derivatives
    for dm in mean.get_grad(X):
        dlZ[i] = np.dot(dm, a) + implicit(dm)
        i += 1

    return Statistics(L, a, w, lZ, dlZ)


def fitc(like, kern, mean, X, Y, U):
    sn2 = like.get_variance()
    su2 = sn2 / 1e6

    # get the kernel matrices
    Kux = kern.get_kernel(U, X)
    kxx = kern.get_dkernel(X) + sn2
    Kuu = la.add_diagonal(kern.get_kernel(U), su2)
    Luu = la.cholesky(Kuu)

    V = la.solve_triangular(Luu, Kux)
    r = (Y - mean.get_function(X))

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
    dlZ = np.zeros(like.params.size + kern.params.size + mean.params.size)

    # derivative wrt sn2
    dlZ[0] = 0.5 * (
        - (np.sum(ell**-2) - np.sum(W**2) - np.inner(a, a))
        - (np.sum(w**2) + np.sum(np.dot(B, W.T)**2)) / 1e6
        + (np.inner(a, v*a) + np.inner(np.sum(W**2, axis=0), v)) / 2 / sn2)

    # iterator over gradients of the kernels
    dK = it.izip(kern.get_grad(U), kern.get_grad(U, X), kern.get_dgrad(X))

    # we need to keep track of how many gradients we've already computed.
    # note also that at the end of the next loop this variable will have
    # changed to track the current number of gradients.
    i = like.params.size

    for i, (dKuu, dKux, dkxx) in enumerate(dK, i):
        M = 2*dKux - np.dot(dKuu, B)
        v = dkxx - np.sum(M*B, axis=0)
        dlZ[i] = (
            - np.sum(dkxx/ell**2)
            - np.inner(w, dKuu.dot(w) - 2*dKux.dot(a))
            + np.inner(a, v*a) + np.inner(np.sum(W**2, axis=0), v)
            + np.sum(M.dot(W.T) * B.dot(W.T))) / 2.0

    for i, dmu in enumerate(mean.get_grad(X), i+1):
        dlZ[i] = np.dot(dmu, a)

    C = np.dot(Luu, L)
    a = la.solve_cholesky(C, np.dot(Kux, r/ell))
    w = np.ones_like(a)

    return Statistics(Luu, a, w, lZ, dlZ, C)
