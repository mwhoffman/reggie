"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = []


class Parameter(object):
    """
    Representation of a parameter vector.
    """
    def __init__(self, value, prior=None, transform=None):
        self.value = np.array(value, dtype=float)
        self.size = self.value.size
        self.prior = prior
        self.transform = transform

    def __repr__(self):
        return np.array2string(self.value, separator=',')

    # NOTE: for now I will assume that any priors are not themselves
    # parameterized. Adding this would require the get/set functions to modify
    # the prior parameters as well and logp would have to take this into
    # account if gradients are asked for.

    def get_params(self):
        """
        Return the parameters.
        """
        return self.value.copy()

    def set_params(self, hyper):
        """
        Set the parameters.
        """
        self.value.flat[:] = hyper

    def logp(self, grad=False):
        """
        Return the log probability of parameter assignments for this parameter
        vector. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if self.prior is None:
            return (0.0, np.zeros_like(self.value.ravel())) if grad else 0.0
        else:
            return self.prior.logp(self.value.ravel(), grad)


class Parameterized(object):
    """
    Base class for parameterized objects.
    """
    def __init__(self):
        self.__params = []

    def __repr__(self):
        return (self.__class__.__name__ + '(' +
                ', '.join('{:s}={:s}'.format(name, obj)
                          for (name, obj) in self.__params) + ')')

    @property
    def size(self):
        """
        Return the number of parameters for this object.
        """
        return sum(obj.size for (_, obj) in self.__params)

    def _register(self, name, value):
        """
        Register a parameter.
        """
        param = Parameter(value)
        self.__params.append((name, param))
        self.__setattr__('_' + name, param.value)

    def get_params(self):
        """
        Return a flattened vector consisting of the parameters for the object.
        """
        if len(self.__params) == 0:
            return np.array([])
        else:
            return np.hstack(obj.get_params() for (_, obj) in self.__params)

    def set_params(self, theta):
        """
        Given a parameter vector of the appropriate size, assign the values of
        this vector to the internal parameters.
        """
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        if theta.shape != (self.size,):
            raise ValueError('incorrect number of parameters')
        offset = 0
        for _, obj in self.__params:
            obj.set_params(theta[offset:offset+obj.size])

    def logp(self, grad=False):
        """
        Return the log probability of parameter assignments to a parameterized
        object. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if grad is False:
            return sum((obj.logp(False) for (_, obj) in self.__params), 0.0)
        else:
            logp, dlogp = 0.0, []
            for (_, obj) in self.__params:
                elem = obj.logp(True)
                logp += elem[0]
                dlogp.append(elem[1])
            return logp, (np.hstack(dlogp) if len(dlogp) > 0 else np.array([]))
