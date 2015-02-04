"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['Parameter', 'Parameterized', 'Model', 'Prior', 'Likelihood']


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

    # NOTE: for now I will ignore any parameterization of priors.  All this
    # would really mean is that the parameter "vector" is itself composite. so
    # get/set methods would have to pull values from the prior objects and
    # logprior would have to take into account the additional parameters when
    # returning the gradient. self.nparams would change too.

    def logprior(self, grad=False):
        """
        Return the log probability of parameter assignments for this parameter
        vector. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if self.prior is None:
            if grad:
                return (0.0, np.zeros_like(self.value.ravel()))
            else:
                return 0.0
        else:
            logp = self.prior.logp(self.value.ravel(), grad)
            if grad:
                return logp[0], logp[1]
            else:
                return logp


class Parameterized(object):
    """
    Base class for parameterized objects.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Parameterized, cls).__new__(cls, *args, **kwargs)
        self.__params = []
        return self

    def __repr__(self):
        return (self.__class__.__name__ + '(' +
                ', '.join('{:s}={:s}'.format(name, obj)
                          for (name, obj) in self.__params) + ')')

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
        self.__setattr__('_' + name,
                         param.value if isinstance(param, Parameter) else
                         param)

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

    def logprior(self, grad=False):
        """
        Return the log probability of parameter assignments to a parameterized
        object. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if grad is False:
            return sum((obj.logprior() for (_, obj) in self.__params), 0.0)
        else:
            logp, dlogp = 0.0, []
            for (_, obj) in self.__params:
                elem = obj.logprior(True)
                logp += elem[0]
                dlogp.append(elem[1])
            return logp, (np.hstack(dlogp) if len(dlogp) > 0 else np.array([]))


class Model(Parameterized):
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls, *args, **kwargs)
        self._X = None
        self._Y = None
        return self

    @property
    def ndata(self):
        return 0 if self._X is None else self._X.shape[0]

    @property
    def data(self):
        return (self._X, self._Y)

    def add_data(self, X, Y):
        if self._X is None:
            self._X = X.copy()
            self._Y = X.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]

    def logposterior(self, grad=False):
        if grad:
            logp0, dlogp0 = self.logprior(True)
            logp1, dlogp1 = self.logp(True)
            return logp0+logp1, dlogp0+dlogp1
        else:
            return self.logprior() + self.logp()

    def logp(self, grad=False):
        raise NotImplementedError


class Prior(Parameterized):
    def logp(self, theta, grad=False):
        raise NotImplementedError


class Likelihood(Parameterized):
    def logp(self, F, Y, grad=False):
        raise NotImplementedError
