"""
Definition of a base model class which provides an interface for supervised
learning over some latent function class.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized
from ..learning.optimization import optimize

__all__ = ['Model']


class Model(Parameterized):
    """
    Base class for parameterized posterior models.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls, *args, **kwargs)
        # pylint: disable=W0212
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
        The data stored for the posterior model.
        """
        return (self._X, self._Y)

    def reset(self):
        """
        Reset the model by removing all data and recomputing or resetting any
        internal statistics.
        """
        self._X = None
        self._Y = None
        self._update()

    def __deepcopy__(self, memo):
        # don't make a copy of the data.
        memo[id(self._X)] = self._X
        memo[id(self._Y)] = self._Y
        return super(Model, self).__deepcopy__(memo)

    def copy(self, theta=None, transform=False, reset=False):
        """
        Copy the model structure. If `theta` is given then also modify the
        parameters of the copied model; if `transform` is True these parameters
        will be in the transformed space. If `reset` is True then copy the
        model, but remove any data.
        """
        # pylint: disable=arguments-differ
        obj = super(Model, self).copy(theta, transform)
        if reset:
            obj.reset()
        return obj

    def add_data(self, X, Y):
        """
        Add a new set of input/output data to the model.
        """
        X = np.array(X, copy=False, ndmin=2, dtype=float)
        Y = np.array(Y, copy=False, ndmin=1, dtype=float)

        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
            self._update()

        else:
            try:
                self._updateinc(X, Y)
                self._X = np.r_[self._X, X]
                self._Y = np.r_[self._Y, Y]

            except NotImplementedError:
                self._X = np.r_[self._X, X]
                self._Y = np.r_[self._Y, Y]
                self._update()

    def _updateinc(self, X, Y):
        """
        Update the sufficient-statistics of a model given new data instances.
        """
        raise NotImplementedError

    def optimize(self):
        """
        Set the parameters to their MAP estimates.
        """
        self.set_params(optimize(self, True), True)

    def sample(self, X, m=None, latent=True, rng=None):
        """
        Sample from the model at points X.
        """
        raise NotImplementedError

    def get_loglike(self, grad=False):
        """
        Get the log-likelihood of the model (and its gradient if requested).
        """
        raise NotImplementedError

    def get_posterior(self, X, grad=False):
        """
        Compute the first two moments of the marginal posterior, evaluated at
        input points X.
        """
        raise NotImplementedError
