"""
Definition of a base model class which provides an interface for supervised
learning over some latent function class.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import copy

from ..core.params import Parameterized
from ..learning.optimization import optimize

__all__ = ['Model']


class Model(object):
    """
    Interface for probabilistic models of latent functions. This interface
    includes the ability to add input/output data, evaluate the log-likelihood
    of this data, make predictions and sample in both the latent and
    observed space, evaluate tail probabilities, expectations, and entropy.
    """
    def copy(self):
        """
        Copy the model; this is just a convenience wrapper around deepcopy.
        """
        return copy.deepcopy(self)

    def add_data(self, X, Y):
        """
        Add input/output data `X` and `Y` to the model.
        """
        raise NotImplementedError

    def get_loglike(self):
        """
        Return the log-likelihood of the observed data.
        """
        raise NotImplementedError

    def sample(self, X, size=None, latent=True, rng=None):
        """
        Return a sample of the model at input locations `X`.

        If `size` is not given this will return an n-vector where n is the
        length of `X`; otherwise it will return an array of shape `(size, n)`.
        If `latent` is true the samples will be in the latent space, otherwise
        they will be sampled in the output space.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Return mean and variance predictions `(mu, s2)` at inputs `X`.
        """
        raise NotImplementedError

    def get_tail(self, f, X):
        """
        Compute the probability that the latent function at inputs `X` exceeds
        the target value `f`.
        """
        raise NotImplementedError

    def get_improvement(self, f, X):
        """
        Compute the expected improvement in value at inputs `X` over the target
        value `f`.
        """
        raise NotImplementedError

    def get_entropy(self, X):
        """
        Compute the predictive entropy evaluated at inputs `X`.
        """
        raise NotImplementedError


class ParameterizedModel(Parameterized, Model):
    """
    Interface for a model that is also parameterized. This adds additional
    methods to optimize the log-likelhiood, get gradients of the log-
    likelihood, as well as copy the model while changing the paraemters.
    """
    def copy(self, theta=None, transform=False):
        # pylint: disable=arguments-differ
        """
        Return a copy of the model; optionally changing the parameters."""
        # NOTE: this will call the version of copy defined by Parameterized.
        return super(ParameterizedModel, self).copy(theta, transform)

    def get_loglike(self, grad=False):
        # pylint: disable=arguments-differ
        """
        Return the log-likelihood of the data and if requested the derivatives
        of the log-likelihood with respect to the model hyperparameters.
        """
        raise NotImplementedError

    def optimize(self):
        """
        Set the model parameters to their MAP values.
        """
        self.params.set_value(optimize(self, raw=True), transform=True)
