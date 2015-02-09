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
            return self.prior.get_logprior(self.value.ravel(), grad)


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
            logp, dlogp = 0.0, []
            for (_, obj) in self.__params:
                elem = obj.get_logprior(True)
                logp += elem[0]
                dlogp.append(elem[1])
            return logp, (np.hstack(dlogp) if len(dlogp) > 0 else
                          np.array([]))


class Model(Parameterized):
    """
    Base class for model objects. These objects should implement some form of
    supervised learning wherein they can have data added to them and can
    compute the marginal likelihood of the model (which can in turn be used for
    optimizing or sampling hyperparameters).
    """
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls, *args, **kwargs)
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
        Tuple containing the observed input- and output-data.
        """
        return (self._X, self._Y)

    def add_data(self, X, Y):
        """
        Add a new set of input/output data to the model.
        """
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]

    def get_logpost(self, grad=False):
        """
        Compute the log posterior of the model.
        """
        if grad:
            logp0, dlogp0 = self.get_logprior(True)
            logp1, dlogp1 = self.get_loglike(True)
            return logp0+logp1, dlogp0+dlogp1
        else:
            return self.get_logprior() + self.get_loglike()

    def get_loglike(self, grad=False):
        """
        Compute the log likelihood of the model.
        """
        raise NotImplementedError

    def get_meanvar(self, X, grad=False):
        """
        Compute the marginal mean and variance of the model, evaluated at input
        locations X[i].

        If grad is True then compute the derivative of these with respect to
        the input location. Return either a 2-tuple or a 4-tuple of the form
        (mu, s2, dmu, ds2) where the last two are omitted if grad is False.
        """
        raise NotImplementedError

    def sample(self, X, m=None, latent=True, rng=None):
        """
        Sample values from the model, evaluated at input locations X[i].

        If m is not None return an (m,n) array where n is the number of input
        values in X; otherwise return an n-array.  If latent is True the sample
        is only of the latent function, otherwise it will be corrupted by
        observation noise. Finally, rng can be used to seed the randomness.
        """
        raise NotImplementedError


class Prior(object):
    """
    Base class for prior objects.
    """
    def get_logprior(self, theta, grad=False):
        raise NotImplementedError


class Likelihood(Parameterized):
    """
    Base class for likelihood objects.
    """
    def get_loglike(self, F, Y, grad=False):
        raise NotImplementedError
