"""
Implementation of posterior GP mean (e.g., from a previous run)
as a prior mean.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ._core import Mean

__all__ = ['GPMean']


class GPMean(Mean):
    """
    Allows prescribing a posterior GP mean as the prior mean for
    another GP.
    """
    def __init__(self, gp):
        super(GPMean, self).__init__()
        self._gp = gp.copy()

    def __info__(self):
        info = []
        info.append(('gp', self._gp))
        return info

    def get_mean(self, X):
        mu, _ = self._gp.predict(X)
        return mu

    def get_grad(self, X):
        return iter([])

    def get_gradx(self, X):
        _, _, dmu, _ = self._gp.predict(X, grad=True)
        return dmu
