"""
Objects representing models which implement some form of supervised learning.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from .params import Parameterized

__all__ = ['Model']


class Model(Parameterized):
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls, *args, **kwargs)
        self._X = None
        self._Y = None
        return self

    @property
    def ndata(self):
        """
        The number of independent observations added to the model.
        """
        return 0 if self._X is None else self._X.shape[0]

    @property
    def data(self):
        return (self._X, self._Y)

    @property
    def bounds(self):
        return zip(self._X.min(axis=0), self._X.max(axis=0))

    def add_data(self, X, Y):
        """
        Add a new set of input/output data to the model.
        """
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
            self._update()

        elif hasattr(self, '_updateinc'):
            self._updateinc(X, Y)
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]

        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]
            self._update()

    def get_loglike(self, grad=False):
        if self.ndata == 0:
            return (0.0, np.zeros(self.nparams)) if grad else 0.0
        else:
            return self._get_loglike(grad)

    def set_params(self, theta, transform=False):
        super(Model, self).set_params(theta, transform)
        self._update()

    def _update(self):
        """
        Update any internal parameters (ie sufficient statistics) given the
        entire set of current data.
        """
        pass

    def _get_loglike(self, grad=False):
        raise NotImplementedError


class PosteriorModel(Model):
    def get_posterior(self, X):
        """
        Compute the first two moments of the marginal posterior, evaluated at
        input points X.
        """
        raise NotImplementedError
