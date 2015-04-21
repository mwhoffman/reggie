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
    def sample(self, f, rng=None):
        """
        Sample observations y given evaluations of the latent function f.
        """
        raise NotImplementedError
