"""
Meta-models for learning.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mwhutils.random import rstate

from ._core import Model
from ..learning import sample

__all__ = ['MCMC']


class MCMC(Model):
    """
    Model which implements MCMC to produce a posterior over parameterized
    models.
    """
    def __init__(self, model, n=100, burn=100, rng=None):
        self._n = n
        self._burn = burn
        self._rng = rstate(rng)
        self._models = self._sample(model.copy(), burn=True)

    @property
    def samples(self):
        """
        An array of the parameter values produced by MCMC."""
        return np.array(list(m.params.get_value() for m in self._models))

    def _sample(self, model, burn=False):
        """
        Resample the hyperparameters with burnin if requested."""
        if burn:
            model = sample(model, self._burn, False, self._rng)[-1]
        return sample(model, self._n, False, self._rng)

    @property
    def ndata(self):
        return self._models[-1].ndata

    @property
    def data(self):
        return self._models[-1].data

    def reset(self):
        model = self._models.pop()
        model.reset()
        self._models = self._sample(model, burn=True)

    def add_data(self, X, Y):
        # add the data
        nprev = self.ndata
        model = self._models.pop()
        model.add_data(X, Y)
        self._models = self._sample(model, burn=(model.ndata > 2*nprev))

    def predict(self, X, grad=False):
        parts = map(np.array, zip(*[_.predict(X, grad) for _ in self._models]))
        mu_, s2_ = parts[:2]
        mu = np.mean(mu_, axis=0)
        s2 = np.mean(s2_ + (mu_ - mu)**2, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.mean(dmu_, axis=0)
        Dmu = dmu_ - dmu
        ds2 = np.mean(ds2_
                      + 2 * mu_[:, :, None] * Dmu
                      - 2 * mu[None, :, None] * Dmu, axis=0)

        return mu, s2, dmu, ds2

    def sample(self, X, size=None, latent=True, rng=None):
        rng = rstate(rng)
        model = self._models[rng.randint(self._n)]
        return model.sample(X, size, latent, rng)

    def get_loglike(self):
        return np.mean([m.get_loglike() for m in self._models])

    def get_improvement(self, X, x, xi=0, grad=False, pi=False):
        args = (X, x, xi, grad, pi)
        parts = [m.get_improvement(*args) for m in self._models]
        if grad:
            return tuple([np.mean(_, axis=0) for _ in zip(*parts)])
        else:
            return np.mean(parts, axis=0)
