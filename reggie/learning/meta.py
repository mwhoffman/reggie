"""
Meta-models for learning.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mwhutils.random import rstate
from .sampling import sample

__all__ = ['MetaMCMC']


class MetaMCMC(object):
    def __init__(self, model, n=100, burn=100, rng=None):
        self._n = n
        self._burn = burn
        self._rng = rstate(rng)
        self._models = [model.copy()]
        self._resample(True)

    @property
    def ndata(self):
        return self._models[-1].ndata

    @property
    def data(self):
        return self._models[-1].data

    def _resample(self, burn=False):
        model = self._models.pop()
        if burn:
            sample(model, self._burn, False, self._rng)
        self._models = sample(model, self._n, False, self._rng)

    def add_data(self, X, Y):
        # add the data
        nprev = self.ndata
        model = self._models.pop()
        model.add_data(X, Y)
        self._models = [model]
        self._resample(model.ndata > 2*nprev)

    def predict(self, X, grad=False, predictive=False):
        parts = map(np.array, zip(*[_.predict(X, grad, predictive)
                                    for _ in self._models]))

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

    def get_samples(self):
        return np.array([m.get_params() for m in self._models])

    @property
    def names(self):
        return self._models[0].names
