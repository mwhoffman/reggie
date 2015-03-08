"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import copy
import tabulate

from collections import OrderedDict

from .transforms import TRANSFORMS
from .priors import PRIORS

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


class Parameter(object):
    """
    Representation of a parameter vector.
    """
    def __init__(self, value, transform=None, prior=None, block=0):
        self.nparams = value.size
        self.value = value
        self.transform = transform
        self.prior = prior
        self.block = block

    def __repr__(self):
        if self.value.shape == ():
            return np.array2string(self.value.ravel(),
                                   precision=PRECISION,
                                   suppress_small=True)[1:-1].strip()
        else:
            return np.array2string(self.value,
                                   separator=',',
                                   precision=PRECISION,
                                   suppress_small=True)

    def __deepcopy__(self, memo):
        # this gets around a bug where copy.deepcopy(array) does not return an
        # array when called on a 0-dimensional object.
        memo[id(self.value)] = self.value.copy()
        return _deepcopy(self, memo)

    def copy(self, theta=None, transform=False):
        obj = copy.deepcopy(self)
        if theta is not None:
            obj.set_params(theta, transform)
        return obj

    def get_params(self, transform=False):
        """Return the parameters."""
        if transform and self.transform is not None:
            return self.transform.get_transform(self.value)
        else:
            return self.value.copy()

    def set_params(self, theta, transform=False):
        """Set the parameters."""
        if transform and self.transform is not None:
            theta = self.transform.get_inverse(theta)
        self.value.flat[:] = theta

    def set_transform(self, transform, *args, **kwargs):
        if transform is not None:
            transform = TRANSFORMS[transform](*args, **kwargs)
        self.transform = transform

    def set_prior(self, prior, *args, **kwargs):
        if prior is not None:
            prior = PRIORS[prior](*args, **kwargs)
            self.value.flat[:] = prior.project(self.value)
        self.prior = prior

    def get_gradfactor(self):
        if self.transform is None:
            return np.ones(self.nparams)
        else:
            return self.transform.get_gradfactor(self.value)

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
    Representation of a parameterized object.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Parameterized, cls).__new__(cls, *args, **kwargs)
        # pylint: disable=W0212
        self.__params = OrderedDict()
        self.__kwargs = OrderedDict()
        return self

    def __repr__(self):
        typename = self.__class__.__name__
        parts = self.__params.items() + self.__kwargs.items()
        parts = ['{:s}={:s}'.format(n, repr(p)) for n, p in parts]
        if any(isinstance(p, Parameterized) for p in self.__params.values()):
            sep = ',\n' + ' ' * (1+len(typename))
        else:
            sep = ', '
        return typename + '(' + sep.join(parts) + ')'

    def __deepcopy__(self, memo):
        # populate the memo with our param values so that these get copied
        # first. this is in order to work around the 0-dimensional array bug
        # noted in Parameter.
        for param in self.__params.values():
            copy.deepcopy(param, memo)
        return _deepcopy(self, memo)

    def __get_param(self, key):
        node = self
        try:
            for part in key.split('.'):
                # pylint: disable=W0212
                node = node.__params[part]
            if not isinstance(node, Parameter):
                raise KeyError
        except KeyError:
            raise ValueError('Unknown parameter: {:s}'.format(key))
        return node

    def __walk_params(self):
        for name, param in self.__params.items():
            if isinstance(param, Parameterized):
                # pylint: disable=W0212
                for name_, param_ in param.__walk_params():
                    yield name + '.' + name_, param_
            else:
                yield name, param

    def copy(self, theta=None, transform=False):
        obj = copy.deepcopy(self)
        if theta is not None:
            obj.set_params(theta, transform)
        return obj

    def _flatten(self, rename=None):
        rename = dict() if rename is None else rename
        params = []
        for name, param in self.__walk_params():
            params.append((rename.get(name, name), param))
        self.__params = OrderedDict(params)
        self.__kwargs = OrderedDict()

    def _kwarg(self, name, value, default=None):
        if value != default:
            self.__kwargs[name] = value

    def _register(self, name, param, klass=None, transform=None, shape=()):
        """
        Register a parameter.
        """
        if klass is not None and not isinstance(param, klass):
            raise ValueError("parameter '{:s}' must be of type {:s}"
                             .format(name, klass.__name__))

        if isinstance(param, Parameterized):
            # copy the parameterized object and store it.
            param = param.copy()
            self.__params[name] = param

        else:
            try:
                # create the parameter vector
                ndmin = len(shape)
                param = np.array(param, dtype=float, copy=True, ndmin=ndmin)

            except (TypeError, ValueError):
                raise ValueError("parameter '{:s}' must be array-like"
                                 .format(name))

            # construct the desired shape
            shapes = dict()
            shape_ = tuple(
                (shapes.setdefault(d, d_) if isinstance(d, str) else d)
                for (d, d_) in zip(shape, param.shape))

            # check the size of the parameter
            if param.shape != shape_:
                raise ValueError("parameter '{:s}' must have shape ({:s})"
                                 .format(name, ', '.join(map(str, shape))))

            # save the parameter
            self.__params[name] = Parameter(param, transform)

        # return either the value of a Parameter instance or the Parameterized
        # object so that it can be used by the actual model
        return param

    def _update(self):
        """
        Update any internal parameters (sufficient statistics, etc.).
        """
        pass

    @property
    def nparams(self):
        """
        Return the number of parameters for this object.
        """
        return sum(param.nparams for _, param in self.__walk_params())

    def describe(self):
        headers = ['name', 'value', 'transform', 'prior', 'block']
        table = []
        for name, param in self.__walk_params():
            trans = '-' if param.transform is None else str(param.transform)
            prior = '-' if param.prior is None else str(param.prior)
            table.append([name, str(param), trans, prior, param.block])
        print(tabulate.tabulate(table, headers, numalign=None))

    def set_param(self, key, theta):
        self.__get_param(key).set_params(theta)
        self._update()

    def set_prior(self, key, prior, *args, **kwargs):
        self.__get_param(key).set_prior(prior, *args, **kwargs)
        self._update()

    def set_transform(self, key, transform, *args, **kwargs):
        self.__get_param(key).set_transform(transform, *args, **kwargs)

    def set_block(self, key, block):
        self.__get_param(key).block = block

    def set_params(self, theta, transform=False):
        """
        Given a parameter vector of the appropriate size, assign the values of
        this vector to the internal parameters.
        """
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        if theta.shape != (self.nparams,):
            raise ValueError('incorrect number of parameters')
        a = 0
        for _, param in self.__walk_params():
            b = a + param.nparams
            param.set_params(theta[a:b], transform)
            a = b
        self._update()

    def get_params(self, transform=False):
        """
        Return a flattened vector consisting of the parameters for the object.
        """
        if self.nparams == 0:
            return np.array([])
        else:
            return np.hstack(param.get_params(transform)
                             for _, param in self.__walk_params())

    def get_gradfactor(self):
        if self.nparams == 0:
            return np.array([])
        else:
            return np.hstack(param.get_gradfactor()
                             for _, param in self.__walk_params())

    def get_logprior(self, grad=False):
        """
        Return the log probability of parameter assignments to a parameterized
        object as well as the gradient with respect to those parameters.
        """
        if not grad:
            return sum(param.get_logprior(False)
                       for _, param in self.__walk_params())

        elif self.nparams == 0:
            return 0, np.array([])

        else:
            logp = 0.0
            dlogp = []
            for _, param in self.__walk_params():
                elem = param.get_logprior(True)
                logp += elem[0]
                dlogp.append(elem[1])
            return logp, np.hstack(dlogp)

    def get_support(self, transform=False):
        support = np.tile((-np.inf, np.inf), (self.nparams, 1))
        a = 0
        for _, param in self.__walk_params():
            b = a + param.nparams
            if param.prior is not None and hasattr(param.prior, 'bounds'):
                if transform and param.transform is not None:
                    support[a:b] = np.array(map(param.transform.get_transform,
                                                param.prior.bounds.T)).T
                else:
                    support[a:b] = param.prior.bounds
            a = b
        return support

    def get_blocks(self):
        blocks = dict()
        a = 0
        for _, param in self.__walk_params():
            b = a + param.nparams
            blocks.setdefault(param.block, []).extend(range(a, b))
            a = b
        return blocks.values()

    def get_names(self):
        names = []
        for name, param in self.__walk_params():
            if param.nparams == 1:
                names.append(name)
            else:
                names.extend('{:s}_{:d}'.format(name, n)
                             for n in xrange(param.nparams))
        return names
