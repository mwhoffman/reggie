"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import copy

__all__ = ['Parameter', 'Parameterized']


class Parameter(object):
    """
    Representation of a parameter vector.
    """
    def __init__(self, value, prior=None, transform=None):
        self.value = np.array(value, dtype=float)
        self.nparams = self.value.size
        self.prior = prior
        self.transform = transform

    def __repr__(self):
        return np.array2string(self.value, separator=',')

    def copy(self):
        return copy.deepcopy(self)

    def get_params(self):
        """Return the parameters."""
        return self.value.copy()

    def set_params(self, hyper):
        """Set the parameters."""
        self.value.flat[:] = hyper

    def get_logprior(self):
        """
        Return the log probability of parameter assignments for this parameter
        vector. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if self.prior is None:
            return (0.0, np.zeros_like(self.value.ravel()))
        else:
            return self.prior.get_logprior(self.value.ravel())


class Parameterized(object):
    """
    Representation of a parameterized object.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Parameterized, cls).__new__(cls, *args, **kwargs)
        self.__params = []
        return self

    def __repr__(self):
        return (self.__class__.__name__ + '(' +
                ', '.join('{:s}={:s}'.format(name, obj)
                          for (name, obj) in self.__params) + ')')

    def copy(self):
        return copy.deepcopy(self)

    @property
    def nparams(self):
        """
        Return the number of parameters for this object.
        """
        return sum(obj.nparams for (_, obj) in self.__params)

    def _register(self, name, param):
        """
        Register a parameter.
        """
        if not isinstance(param, (Parameter, Parameterized)):
            raise ValueError('only objects of type Parameter or Parameterized '
                             'can be registered')
        self.__params.append((name, param))
        self.__setattr__(name, param)
        if isinstance(param, Parameter):
            setattr(self, '_' + name, param.value)

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
        if theta.shape != (self.nparams,):
            raise ValueError('incorrect number of parameters')
        offset = 0
        for _, obj in self.__params:
            obj.set_params(theta[offset:offset+obj.nparams])

    def get_logprior(self):
        """
        Return the log probability of parameter assignments to a parameterized
        object as well as the gradient with respect to those parameters.
        """
        logp = 0.0
        dlogp = []
        for (_, obj) in self.__params:
            elem = obj.get_logprior()
            logp += elem[0]
            dlogp.append(elem[1])
        return logp, (np.hstack(dlogp) if len(dlogp) > 0 else np.array([]))
