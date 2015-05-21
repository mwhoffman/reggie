"""
Definition of a base model class which provides an interface for supervised
learning over some latent function class.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import copy

from ..core.params import Parameterized
from ..learning.optimization import optimize

__all__ = ['Model']


class Model(object):
    @property
    def ndata(self):
        """
        The number of observations added to the model."""
        raise NotImplementedError

    @property
    def data(self):
        """
        The observed data (X, Y) stored for the model."""
        raise NotImplementedError

    def copy(self, reset=False):
        """
        Copy the model and optionally reset the result."""
        obj = copy.deepcopy(self)
        if reset:
            obj.reset()
        return obj

    def add_data(self, X, Y):
        """
        Add input/output data X and Y to the model."""
        raise NotImplementedError

    def reset(self):
        """
        Reset the model to its state before data was added."""
        raise NotImplementedError

    def predict(self, X, grad=False):
        """
        Return predictions (and possibly their gradients) at inputs X.

        Return a tuple (mu, s2) containing vectors of the predicted mean and
        variance at input locations X. If grad is True return a 4-tuple whose
        first two components are the same as above and where the final two
        components are the derivatives of the mean and variance with respect to
        the inputs.
        """
        raise NotImplementedError

    def sample(self, X, size=None, latent=True, rng=None):
        """
        Return a sample of the model at input locations X.

        If size is None then return a vector of the function values, otherwise
        return an (size, n) array where n is the length of X. If latent is True
        then return samples of the latent function f(x) and otherwise return
        samples of the outputs Y. Finally, use the random state rng if given.
        """
        raise NotImplementedError

    def get_loglike(self):
        """
        Return the log-likelihood of the observed data."""
        raise NotImplementedError

    def get_improvement(self, X, xi=0, grad=False, pi=False):
        """
        Return the expected or probability of improvement at each point X.

        The improvement must be of a level of at least xi over the current
        incumbent. If pi is True this will return the probability of
        improvement, otherwise returning the expected improvement. Finally, if
        grad is True return the gradients of this function at each input
        location.
        """
        raise NotImplementedError


class BasicModel(Model):
    def __init__(self):
        super(BasicModel, self).__init__()
        self._X = None
        self._Y = None

    def _update(self):
        """
        Update any internal model parameters or statistics."""
        pass

    @property
    def ndata(self):
        return 0 if self._X is None else self._X.shape[0]

    @property
    def data(self):
        return (self._X, self._Y)

    def reset(self):
        self._X = None
        self._Y = None
        self._update()

    def add_data(self, X, Y):
        X = np.array(X, copy=False, ndmin=2, dtype=float)
        Y = np.array(Y, copy=False, ndmin=1, dtype=float)
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]
        self._update()


class ParameterizedModel(Parameterized, BasicModel):
    def __deepcopy__(self, memo):
        # don't make a copy of the data.
        memo[id(self._X)] = self._X
        memo[id(self._Y)] = self._Y
        return super(ParameterizedModel, self).__deepcopy__(memo)

    def copy(self, theta=None, transform=False, reset=False):
        # pylint: disable=arguments-differ
        """
        Return a copy of the model.

        If theta/transform are given this will modify the parameters of the
        copy; if reset is True this will reset the resulting model.
        """
        # NOTE: this will call the version of copy defined by Parameterized and
        # not Model's, hence we can pass theta/transform and we need to add in
        # the reset logic again.
        obj = super(ParameterizedModel, self).copy(theta, transform)
        if reset:
            obj.reset()
        return obj

    def optimize(self):
        """
        Set the model parameters to their MAP values."""
        self.params.set_value(optimize(self, raw=True), transform=True)

    def get_loglike(self, grad=False):
        # pylint: disable=arguments-differ
        """
        Return the log-likelihood of the data and if requested the derivatives
        of the log-likelihood with respect to the model hyperparameters.
        """
        raise NotImplementedError
