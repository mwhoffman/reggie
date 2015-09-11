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
        Add input/output data X and Y to the model.
        """
        raise NotImplementedError

    def get_loglike(self):
        """
        Return the log-likelihood of the observed data.
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

    def get_tail(self, X, f, grad=False):
        """
        Compute the probability that latent function exceeds some target `f`.

        Return a vector containing the probability that the latent function
        f(x) exceeds the given target `v` for each point `X[i]`. If grad is
        True return the gradients of this function at each input location.
        """
        raise NotImplementedError

    def get_improvement(self, X, f, grad=False):
        """
        Compute expected improvement over some target `f`.

        Return a vector containing the expected improvement over a given target
        `f` for each point `X[i]`. If grad is True return the gradients of this
        function at each input location.
        """
        raise NotImplementedError

    def get_entropy(self, X, grad=False):
        """
        Compute the predictive entropy evaluated at inputs X.
        """
        raise NotImplementedError

class ParameterizedModel(Parameterized, Model):
    """
    Interface for a model that is also parameterized. This adds additional
    methods to optimize the log-likelhiood, get gradients of the log-likelihood,
    as well as copy the model while changing the paraemters.
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
