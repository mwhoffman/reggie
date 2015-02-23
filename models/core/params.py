"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import copy
import tabulate

from .transforms import Transform

__all__ = ['Parameterized']


# CONSTANTS FOR ADJUSTING PARAMETER FORMATTING
PRECISION = 2


def _deepcopy(obj, memo):
    """
    Method equivalent to `copy.deepcopy` except that it ignored the
    implementation of `obj.__deepcopy__`. This allows for an object to
    implement deepcopy, populate its memo dictionary, and then call this
    version of deepcopy.
    """
    ret = type(obj).__new__(type(obj))
    memo[id(obj)] = ret
    for key, val in obj.__dict__.items():
        setattr(ret, key, copy.deepcopy(val, memo))
    return ret


def _get_offsets(params):
    """
    Given a list of (name, param) tuples, iterate through this list and
    generate a sequence of (param, a, b) tuples where a and b are the indices
    into an external parameter vector.
    """
    a = 0
    for _, param in params:
        b = a + param.nparams
        yield param, a, b
        a = b


class Parameter(object):
    """
    Representation of a parameter vector.
    """
    def __init__(self, value, prior=None, transform=None):
        self.nparams = value.size
        self._value = value
        self._prior = prior
        self._transform = transform

    def __repr__(self):
        if self._value.shape == ():
            return np.array2string(self._value.ravel(),
                                   precision=PRECISION,
                                   suppress_small=True)[1:-1].strip()
        else:
            return np.array2string(self._value,
                                   separator=',',
                                   precision=PRECISION,
                                   suppress_small=True)

    def __deepcopy__(self, memo):
        # this gets around a bug where copy.deepcopy(array) does not return an
        # array when called on a 0-dimensional object.
        memo[id(self._value)] = self._value.copy()
        return _deepcopy(self, memo)

    def copy(self, theta=None, transform=False):
        obj = copy.deepcopy(self)
        if theta is not None:
            obj.set_params(theta, transform)
        return obj

    def get_params(self, transform=False):
        """Return the parameters."""
        if transform and self._transform is not None:
            return self._transform.get_transform(self._value)
        else:
            return self._value.copy()

    def set_params(self, theta, transform=False):
        """Set the parameters."""
        if transform and self._transform is not None:
            theta = self._transform.get_inverse(theta)
        self._value.flat[:] = theta

    def set_transform(self, transform):
        if not isinstance(transform, Transform):
            raise ValueError('transform must be an instance of Transform')
        self._transform = transform

    def transform_grad(self, theta, grad):
        if self._transform is None:
            return grad.copy()
        else:
            return grad * self._transform.get_inverse_grad(theta)

    def get_logprior(self):
        """
        Return the log probability of parameter assignments for this parameter
        vector. Also if requested return the gradient of this probability with
        respect to the parameter values.
        """
        if self._prior is None:
            return (0.0, np.zeros_like(self._value.ravel()))
        else:
            return self._prior.get_logprior(self._value.ravel())


class Parameterized(object):
    """
    Representation of a parameterized object.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Parameterized, cls).__new__(cls, *args, **kwargs)
        self.__params = []
        return self

    def __repr__(self):
        typename = self.__class__.__name__
        parts = ['{:s}={:s}'.format(n, p) for n, p in self.__params]
        if any(isinstance(p, Parameterized) for _, p in self.__params):
            sep = ',\n' + ' ' * (1+len(typename))
        else:
            sep = ', '
        return typename + '(' + sep.join(parts) + ')'

    def __deepcopy__(self, memo):
        # populate the memo with our param values so that these get copied
        # first. this is in order to work around the 0-dimensional array bug
        # noted in Parameter.
        for _, param in self.__params:
            copy.deepcopy(param, memo)
        return _deepcopy(self, memo)

    def copy(self, theta=None, transform=False):
        obj = copy.deepcopy(self)
        if theta is not None:
            obj.set_params(theta, transform)
        return obj

    @property
    def nparams(self):
        """
        Return the number of parameters for this object.
        """
        return sum(param.nparams for _, param in self.__params)

    def _register(self, name, param, req_class=None, ndim=None):
        """
        Register a parameter.
        """
        if req_class is not None and not isinstance(param, req_class):
            raise ValueError("parameter '{:s}' must be of type {:s}"
                             .format(name, req_class.__name__))

        if not isinstance(param, Parameterized):
            try:
                # create the parameter vector
                param = np.array(param,
                                 dtype=float,
                                 ndmin=(0 if ndim is None else ndim))
            except (TypeError, ValueError):
                raise ValueError("parameter '{:s}' must be array-like"
                                 .format(name))

            # check the size of the parameter
            if ndim is not None and param.ndim > ndim:
                raise ValueError("parameter '{:s}' must be {:d}-dimensional"
                                 .format(name, ndim))

            # create a parameter instance and save quick-access to the value
            param = Parameter(param)
            setattr(self, '_' + name, param._value)

        self.__params.append((name, param))
        self.__setattr__(name, param)

    def _walk_params(self):
        for name, param in self.__params:
            if isinstance(param, Parameterized):
                for name_, param_ in param._walk_params():
                    yield name + '.' + name_, param_
            else:
                yield name, param

    def describe(self):
        headers = ['name', 'value', 'prior', 'transform']
        table = []
        for name, param in self._walk_params():
            prior = param._prior
            trans = param._transform
            prior = '-' if prior is None else str(prior)
            trans = '-' if trans is None else type(trans).__name__
            table.append([name, str(param), prior, trans])
        print(tabulate.tabulate(table, headers))

    def get_params(self, transform=False):
        """
        Return a flattened vector consisting of the parameters for the object.
        """
        if len(self.__params) == 0:
            return np.array([])
        else:
            return np.hstack(param.get_params(transform)
                             for _, param in self.__params)

    def set_params(self, theta, transform=False):
        """
        Given a parameter vector of the appropriate size, assign the values of
        this vector to the internal parameters.
        """
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)

        if theta.shape != (self.nparams,):
            raise ValueError('incorrect number of parameters')

        for param, a, b in _get_offsets(self.__params):
            param.set_params(theta[a:b], transform)

    def transform_grad(self, theta, grad):
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        grad = np.array(grad, dtype=float, copy=False, ndmin=1)
        shape = (self.nparams,)

        if grad.shape != shape or theta.shape != shape:
            raise ValueError('incorrect number of parameters')

        if len(self.__params) == 0:
            return np.array([])
        else:
            return np.hstack(param.transform_grad(theta[a:b], grad[a:b])
                             for param, a, b in _get_offsets(self.__params))

    def get_logprior(self):
        """
        Return the log probability of parameter assignments to a parameterized
        object as well as the gradient with respect to those parameters.
        """
        logp = 0.0
        dlogp = []
        for _, param in self.__params:
            elem = param.get_logprior()
            logp += elem[0]
            dlogp.append(elem[1])
        return logp, (np.hstack(dlogp) if len(dlogp) > 0 else np.array([]))
