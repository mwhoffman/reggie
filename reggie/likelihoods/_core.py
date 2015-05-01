"""
Definition of the likelihood interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ..core.params import Parameterized

__all__ = ['Likelihood']


### BASE LIKELIHOOD INTERFACE #################################################

class Likelihood(Parameterized):
    """
    The base Likelihood interface.
    """
    def get_variance(self):
        """
        Return the variance of the observation model; this is used for
        performing exact inference and should only be implemented by Gaussian
        models.
        """
        raise NotImplementedError

    def sample(self, f, rng=None):
        """
        Sample observations y given evaluations of the latent function f.
        """
        raise NotImplementedError

    def get_logprob(self, y, f):
        """
        Get the log marginal probability log p(y|f) along with the first three
        derivatives of this quantity wrt f; returns a 4-tuple.
        """
        raise NotImplementedError

    def get_laplace_grad(self, y, f):
        """
        Get the gradients necessary to compute a Laplace approximation. This
        should yield, for each likelihood parameter, a 3-tuple containing the
        derivatives of::

                log p(y|f)
            d   log p(y|f) / df
            d^2 log p(y|f) / df^2

        with respect to the ith such likelihood parameter.
        """
        raise NotImplementedError
