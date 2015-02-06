"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import prettytable

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
    # get_logprior would have to take into account the additional parameters when
    # returning the gradient. self.nparams would change too.

    def get_logprior(self, grad=False):
        """
        Return the log probability of parameter assignments for this parameter
        vector. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if self.prior is None:
            return (0.0, np.zeros_like(self.value.ravel())) if grad else 0.0
        else:
            return self.prior.get_logp(self.value.ravel(), grad)


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

    def _walk_params(self):
        for name, param in self.__params:
            if isinstance(param, Parameterized):
                for name_, param_ in param._walk_params():
                    yield name + '.' + name_, param_
            else:
                yield name, param

    def describe(self):
        t = prettytable.PrettyTable(['name', 'value', 'prior'])
        t.align['name'] = 'l'
        for name, param in self._walk_params():
            t.add_row([name, str(param.value), param.prior])
        print(t)

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

    def get_logprior(self, grad=False):
        """
        Return the log probability of parameter assignments to a parameterized
        object. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if grad is False:
            return sum((obj.get_logprior() for (_, obj) in self.__params), 0.0)
        else:
            get_logp, dget_logp = 0.0, []
            for (_, obj) in self.__params:
                elem = obj.get_logprior(True)
                get_logp += elem[0]
                dget_logp.append(elem[1])
            return get_logp, (np.hstack(dget_logp) if len(dget_logp) > 0 else
                              np.array([]))


class Model(Parameterized):
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls, *args, **kwargs)
        self._X = None
        self._Y = None
        return self

    @property
    def ndata(self):
        """The number of independent observations added to the model."""
        return 0 if self._X is None else self._X.shape[0]

    @property
    def data(self):
        """A tuple containing the observed input- and output-data."""
        return (self._X, self._Y)

    def add_data(self, X, Y):
        """
        Add a new set of input/output data to the model.
        """
        if self._X is None:
            self._X = X.copy()
            self._Y = X.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]

    def get_logposterior(self, grad=False):
        """
        Compute the log posterior of the model.
        """
        if grad:
            logp0, dlogp0 = self.get_logprior(True)
            logp1, dlogp1 = self.get_logp(True)
            return logp0+logp1, dlogp0+dlogp1
        else:
            return self.get_logprior() + self.get_logp()

    def get_logp(self, grad=False):
        """
        Compute the log likelihood of the model.
        """
        raise NotImplementedError


class Prior(object):
    """
    Base class for prior probability distributions.
    """
    def get_logp(self, theta, grad=False):
        raise NotImplementedError


class Likelihood(Parameterized):
    """
    Base class for likelihood models.
    """
    def get_logp(self, F, Y, grad=False):
        raise NotImplementedError
