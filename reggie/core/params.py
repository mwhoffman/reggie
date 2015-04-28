"""
Interface for parameterized objects. The Parameterized class represents a
parameterized object which can get/set its values and evaluate a log-prior over
those values. Note, however, that this does not define any error or probability
terms for the object. For this, see the Model class.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import copy
import tabulate
import warnings
import collections

from .priors import PRIORS
from .domains import REAL
from .domains import BOUNDS, TRANSFORMS

__all__ = ['Parameterized']


# CONSTANTS FOR ADJUSTING PARAMETER FORMATTING
PRECISION = 2


def _outbounds(bounds, theta):
    """
    Check whether a vector is inside the given bounds.
    """
    bounds = np.array(bounds, ndmin=2)
    return np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1])


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


def _ndprint(arr):
    if arr.shape == ():
        value = '{:.2f}'.format(arr.flat[0])
    else:
        value = '[' * arr.ndim
        value += ', '.join('{:.2f}'.format(_) for _ in arr.flat[:2])
        if arr.size > 3:
            value += ', ..., '
        if arr.size > 2:
            value += '{:.2f}'.format(arr.flat[-1])
        value += ']' * arr.ndim
    return value


class Parameter(object):
    """
    Representation of a single parameter array.
    """
    def __init__(self, value, domain, prior=None, block=0):
        self.value = value
        self.prior = prior
        self.block = block
        self.domain = domain
        self.transform = TRANSFORMS[domain]
        self.bounds = BOUNDS[domain]

        # note this will raise an error if we're out of bounds.
        self.set_value(self.value.ravel())

    def __deepcopy__(self, memo):
        # this gets around a bug where copy.deepcopy(array) does not return an
        # array when called on a 0-dimensional object.
        memo[id(self.value)] = self.value.copy()
        return _deepcopy(self, memo)

    def __repr__(self):
        return _ndprint(self.value)

    @property
    def nparams(self):
        return self.value.size

    def get_value(self, transform=False):
        """
        Return the parameters. If `transform` is True return values in the
        transformed space.
        """
        if transform:
            return self.transform.get_transform(self.value).ravel()
        else:
            return self.value.copy().ravel()

    def set_value(self, theta, transform=False):
        """
        Set the parameters to values given by `theta`. If `transform` is true
        then theta lies in the transformed space.
        """
        # transform the parameters if necessary and ensure that they lie in the
        # correct domain (here we're using the transform as a domain
        # specification, although we may want to separate these later).
        if transform:
            theta = self.transform.get_inverse(theta)

        if _outbounds(self.bounds, theta):
            raise ValueError('value lies outside the parameter\'s bounds')

        self.value.flat[:] = theta

    def set_prior(self, prior, *args, **kwargs):
        """
        Set the prior of the parameter object. This should be given as a string
        identifier and any (fixed!) hyperparameters.
        """
        if prior is None:
            self.prior = prior
            self.bounds = BOUNDS[self.domain]

        else:
            prior = PRIORS[prior](*args, **kwargs)
            dbounds = np.array(BOUNDS[self.domain], ndmin=2, copy=False)
            pbounds = np.array(prior.bounds, ndmin=2, copy=False)

            if len(pbounds) > 1:
                dbounds = np.tile(dbounds, (len(pbounds), 1))

            if (_outbounds(dbounds, pbounds[:, 0]) or
                    _outbounds(dbounds, pbounds[:, 1])):
                raise ValueError('prior support lies outside of the '
                                 'parameter\'s domain')

            bounds = np.c_[
                np.max(np.c_[pbounds[:, 0], dbounds[:, 0]], axis=1),
                np.min(np.c_[pbounds[:, 1], dbounds[:, 1]], axis=1)]

            if _outbounds(bounds, self.value.ravel()):
                message = 'clipping parameter value outside prior support'
                warnings.warn(message, stacklevel=3)

            value = np.clip(self.value.ravel(), bounds[:, 0], bounds[:, 1])

            self.prior = prior
            self.bounds = bounds.squeeze()
            self.value.flat[:] = value

    def get_gradfactor(self):
        """
        Return a gradient factor which can be used to transform a gradient in
        the original space into a gradient in the transformed space, via the
        chain rule.
        """
        return self.transform.get_gradfactor(self.value).ravel()

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


class Parameters(object):
    """
    Representation of a set of parameters bound to a Parameterized object.
    """
    def __init__(self, obj):
        self.__obj = obj
        self.__params = collections.OrderedDict()

    def _register(self, name, param):
        if isinstance(param, Parameter):
            if name in self.__params:
                raise ValueError("parameter '{:s}' has already been registered"
                                 .format(name))
            self.__params[name] = param
        elif isinstance(param, Parameters):
            for n, p in param.__params.items():
                if name is not None:
                    n = name + '.' + n
                self._register(n, p)
        else:
            raise ValueError('unknown type passed to _register')

    def __deepcopy__(self, memo):
        # populate the memo with our param values so that these get copied
        # first. this is in order to work around the 0-dimensional array bug
        # noted in Parameter.
        for param in self.__params.values():
            copy.deepcopy(param, memo)
        return _deepcopy(self, memo)

    @property
    def nparams(self):
        """
        Return the number of parameters for this object.
        """
        return sum(param.nparams for param in self.__params.values())

    @property
    def gradfactor(self):
        """
        Return the gradient factor which should be multipled by any gradient in
        order to define a gradient in the transformed space.
        """
        if self.nparams == 0:
            return np.array([])
        else:
            return np.hstack(param.get_gradfactor()
                             for param in self.__params.values())

    @property
    def blocks(self):
        """
        Return a list whose ith element contains indices for the parameters
        which make up the ith block.
        """
        blocks = dict()
        a = 0
        for param in self.__params.values():
            b = a + param.nparams
            blocks.setdefault(param.block, []).extend(range(a, b))
            a = b
        return blocks.values()

    @property
    def block(self):
        """Get the block assignment of the parameter set."""
        return [param.block for param in self.__params.values()]

    @block.setter
    def block(self, value):
        """Set the block assignment of the parameter set."""
        if np.isscalar(value):
            value = [value] * len(self.__params)
        if len(value) != len(self.__params):
            raise ValueError('invalid block assignment')
        for block, param in zip(value, self.__params.values()):
            param.block = block

    @property
    def names(self):
        """
        Return a list of names for each parameter.
        """
        names = []
        for name, param in self.__params.items():
            if param.nparams == 1:
                names.append(name)
            else:
                names.extend('{:s}[{:d}]'.format(name, n)
                             for n in xrange(param.nparams))
        return names

    def describe(self):
        """
        Describe the structure of the object in terms of its hyperparameters.
        """
        headers = ['name', 'value', 'domain', 'prior', 'block']
        table = []
        for name, param in self.__params.items():
            prior = '-' if param.prior is None else str(param.prior)
            table.append([name, str(param), param.domain, prior, param.block])
        print(tabulate.tabulate(table, headers, numalign=None))

    def get_value(self, transform=False):
        """Get the value of the parameters."""
        if self.__obj.nparams == 0:
            return np.array([])
        else:
            return np.hstack(param.get_value(transform)
                             for param in self.__params.values())

    def set_value(self, theta, transform=False):
        """Set the value of the parameters."""
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        if theta.shape != (self.__obj.nparams,):
            raise ValueError('incorrect number of parameters')
        a = 0
        for param in self.__params.values():
            b = a + param.nparams
            param.set_value(theta[a:b], transform)
            a = b
        self.__obj._update()

    def get_bounds(self, transform=False):
        """
        Get bounds on the hyperparameters. If `transform` is True then these
        bounds are those in the transformed space.
        """
        bounds = np.tile((-np.inf, np.inf), (self.nparams, 1))
        a = 0
        for param in self.__params.values():
            b = a + param.nparams
            if transform:
                bounds[a:b] = [
                    param.transform.get_transform(_)
                    for _ in np.array(param.bounds, ndmin=2)]
            else:
                bounds[a:b] = param.bounds
            a = b
        return bounds

    def get_logprior(self, grad=False):
        """
        Return the log probability of parameter assignments to a parameterized
        object as well as the gradient with respect to those parameters.
        """
        if not grad:
            return sum(param.get_logprior(False)
                       for param in self.__params.values())

        elif self.nparams == 0:
            return 0, np.array([])

        else:
            logp = 0.0
            dlogp = []
            for param in self.__params.values():
                elem = param.get_logprior(True)
                logp += elem[0]
                dlogp.append(elem[1])
            return logp, np.hstack(dlogp)


class Parameterized(object):
    """
    Representation of a parameterized object.
    """
    def __init__(self):
        self.params = Parameters(self)

    def __info__(self):
        return []

    # def __repr__(self):
    #     typename = self.__class__.__name__
    #     items = self.__params.items() + self.__info__()
    #     parts = []
    #     for name, param in items:
    #         if isinstance(param, np.ndarray):
    #             value = _ndprint(param)
    #         else:
    #             value = repr(param)
    #         parts.append('{:s}={:s}'.format(name, value))
    #     nintro = len(typename) + 1
    #     nchars = nintro + 1 + sum(len(_)+2 for _ in parts)
    #     split = any('\n' in _ for _ in parts)
    #     if nchars > 80 or split:
    #         sep = '\n' + ' ' * nintro
    #         parts = [sep.join(_.split('\n')) for _ in parts]
    #         sep = ',' + sep
    #     else:
    #         sep = ', '
    #     return typename + '(' + sep.join(parts) + ')'

    def __deepcopy__(self, memo):
        # populate the memo with our param values so that these get copied
        # first. this is in order to work around the 0-dimensional array bug
        # noted in Parameter.
        copy.deepcopy(self.params, memo)
        return _deepcopy(self, memo)

    @property
    def nparams(self):
        return self.params.nparams

    def copy(self, theta=None, transform=False):
        """
        Return a copy of the object. If `theta` is given then update the
        parameters of the copy; if `transform` is True then these parameters
        are in the transformed space.
        """
        obj = copy.deepcopy(self)
        if theta is not None:
            obj.set_params(theta, transform)
        return obj

    def _update(self):
        """
        Update any internal parameters (sufficient statistics, etc.).
        """
        pass

    def _register(self, name, param, *args):
        """
        Register a parameter given a `(name, param)` pair. Here `param` should
        be either a Parameterized or an array-like object. If `param` is of
        class Parameterized then name can be None in order to hide the
        hierarchy; otherwise it should be a string identifier.
        """
        if isinstance(param, Parameterized):
            if len(args) > 0 and not isinstance(param, args[0]):
                raise ValueError("parameter '{:s}' must be of type {:s}"
                                 .format(name, args[0].__name__))

            if len(args) > 1:
                raise ValueError('unknown constraints passed to _register')

            # copy the parameterized object
            param = param.copy()
            self.params._register(name, param.params)

        else:
            domain = args[0] if len(args) > 0 else REAL
            shape = args[1] if len(args) > 1 else ()

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
            self.params._register(name, Parameter(param, domain))

        # return either the value of a Parameter instance or the Parameterized
        # object so that it can be used by the actual model
        return param

    # def set_prior(self, key, prior, *args, **kwargs):
    #     """
    #     Set the prior of the named parameter.
    #     """
    #     self.__get_param(key).set_prior(prior, *args, **kwargs)
