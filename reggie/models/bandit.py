"""
Definition of simple, independent arm bandit models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.special as special
import scipy.stats as stats

from ..utils.misc import rstate
from ._core import Model

__all__ = ['BetaBernoulli']


class BetaBernoulli(Model):
    """
    Beta-Bernoulli bandit model where the payout probability for each arm is
    assumed to follow a Beta(alpha, beta) prior and observations are Bernoulli
    with this probability.
    """
    def __init__(self, alpha, beta=None):
        self._alpha = np.array(alpha, ndmin=1, dtype=int)
        self._beta = np.array(alpha if (beta is None) else beta,
                              ndmin=1, dtype=int)
        if len(self._alpha) != len(self._beta):
            raise ValueError('alpha and beta must have the same size')

    def _get_input(self, X):
        """
        Format and verify the given input locations.
        """
        X = np.array(X, dtype=int, ndmin=1)
        if np.any(np.logical_or(X < 0, X > len(self._alpha))):
            raise ValueError('invalid inputs')
        return X

    def _get_output(self, Y):
        """
        Format and verify the observed outputs.
        """
        Y = np.array(Y, dtype=int, ndmin=1)
        if np.any(np.logical_and(Y != 0, Y != 1)):
            raise ValueError('invalid outputs')
        return Y

    def _get_alphabeta(self, X=None):
        """
        Get a subset of the alpha/beta parameters, or all of them if the inputs
        X are None.
        """
        X = slice(len(self._alpha)) if (X is None) else self._get_input(X)
        return self._alpha[X], self._beta[X]

    def add_data(self, X, Y):
        X = self._get_input(X)
        Y = self._get_output(Y)
        if len(X) != len(Y):
            raise ValueError('X and Y must have the same size')
        for x, y in zip(X, Y):
            self._alpha[x] += y
            self._beta[x] += 1 - y

    def sample(self, X=None, size=None, latent=True, rng=None):
        alpha, beta = self._get_alphabeta(X)
        rng = rstate(rng)
        f = rng.beta(alpha, beta,
                     size=None if (size is None) else (size, len(alpha)))
        if latent is False:
            f = np.array(rng.uniform(size=f.shape) < f, dtype=int)
        return f

    def predict(self, X=None):
        alpha, beta = self._get_alphabeta(X)
        mu = alpha / (alpha + beta)
        s2 = alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)
        return mu, s2

    def get_tail(self, f, X=None):
        alpha, beta = self._get_alphabeta(X)
        return 1 - stats.beta.cdf(f, alpha, beta)

    def get_improvement(self, f, X=None):
        alpha, beta = self._get_alphabeta(X)
        ei = alpha / (alpha + beta) * (1 - stats.beta.cdf(f, alpha+1, beta))
        ei -= f * (1 - stats.beta.cdf(f, alpha, beta))
        return ei

    def get_entropy(self, X=None):
        a, b = self._get_alphabeta(X)
        return (special.betaln(a, b) -
                special.digamma(a) * (a-1) -
                special.digamma(b) * (b-1) +
                special.digamma(a+b) * (a+b-2))

    def get_quantile(self, q, X=None):
        a, b = self._get_alphabeta(X)
        return special.betaincinv(a, b, q)
