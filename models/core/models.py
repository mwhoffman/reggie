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
        """
        Tuple containing the observed input- and output-data.
        """
        return (self._X, self._Y)

    def add_data(self, X, Y):
        """
        Add a new set of input/output data to the model.
        """
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]

    def get_loglike(self):
        if self.ndata == 0:
            return 0.0, np.zeros(self.nparams)
        else:
            return self._get_loglike()

    def _get_loglike(self):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError
