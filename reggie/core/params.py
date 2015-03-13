"""
Parameters and parameterizable objects.
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


class Parameter(object):
    """
    Representation of a parameter vector.
    """
    def __init__(self, value, domain, prior=None, block=0):
        self.nparams = value.size
        self.value = value
        self.prior = prior
        self.block = block
        self.domain = domain
        self.transform = TRANSFORMS[domain]
        self.bounds = BOUNDS[domain]

        # note this will raise an error if we're out of bounds.
        self.set_params(self.value.ravel())

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

    def get_params(self, transform=False):
        """
        Return the parameters. If `transform` is True return values in the
        transformed space.
        """
        if transform:
            return self.transform.get_transform(self.value).ravel()
        else:
            return self.value.copy().ravel()

    def set_params(self, theta, transform=False):
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


class Parameterized(object):
    """
    Representation of a parameterized object.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Parameterized, cls).__new__(cls, *args, **kwargs)
        # pylint: disable=W0212
        self.__params = collections.OrderedDict()
        return self

    def __repr__(self, **kwargs):
        typename = self.__class__.__name__
        parts = self.__params.items() + kwargs.items()
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
        """
        Return the parameter object associated with the given key.
        """
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
        """
        Walk the set of parameters, yielding the Parameter objects via a
        depth-first traversal.
        """
        for name, param in self.__params.items():
            if isinstance(param, Parameterized):
                # pylint: disable=W0212
                for name_, param_ in param.__walk_params():
                    yield name + '.' + name_, param_
            else:
                yield name, param

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

    def _flatten(self, rename=None):
        """
        Flatten the set of parameters associated with this object. Ultimately
        this should have no outward effect unless `rename` is given as a
        dictionary mapping requested parameters to new names. This allows for
        aliasing parameters (see BasicGP for an example).
        """
        rename = dict() if rename is None else rename
        if len(set(rename.values())) < len(rename):
            raise ValueError('assigning multiple parameters to the same name')
        params = []
        for name, param in self.__walk_params():
            params.append((rename.get(name, name), param))
        self.__params = collections.OrderedDict(params)

    def _register(self, name, param, klass=None, domain=REAL, shape=()):
        """
        Register a parameter given a `(name, param)` pair. If `klass` is given
        then this should be a Parameterized object of the given class.
        Otherwise `domain` and `shape` can be used to specify the domain and
        shape of a Parameter object.
        """
        if name in self.__params:
            raise ValueError("parameter '{:s}' has already been registered"
                             .format(name))

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
            self.__params[name] = Parameter(param, domain)

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
        """
        Describe the structure of the object in terms of its hyperparameters.
        """
        headers = ['name', 'value', 'domain', 'prior', 'block']
        table = []
        for name, param in self.__walk_params():
            prior = '-' if param.prior is None else str(param.prior)
            table.append([name, str(param), param.domain, prior, param.block])
        print(tabulate.tabulate(table, headers, numalign=None))

    def set_param(self, key, theta):
        """
        Set the value of the named parameter.
        """
        self.__get_param(key).set_params(np.array(theta, ndmin=1, copy=False))
        self._update()

    def set_prior(self, key, prior, *args, **kwargs):
        """
        Set the prior of the named parameter.
        """
        self.__get_param(key).set_prior(prior, *args, **kwargs)
        self._update()

    def set_block(self, key, block):
        """
        Set the block of the named parameter (used by sampling methods).
        """
        # pylint: disable=W0201
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
        """
        Return the gradient factor which should be multipled by any gradient in
        order to define a gradient in the transformed space.
        """
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

    def get_bounds(self, transform=False):
        """
        Get bounds on the hyperparameters. If `transform` is True then these
        bounds are those in the transformed space.
        """
        bounds = np.tile((-np.inf, np.inf), (self.nparams, 1))
        a = 0
        for _, param in self.__walk_params():
            b = a + param.nparams
            if transform:
                bounds[a:b] = [
                    param.transform.get_transform(_)
                    for _ in np.array(param.bounds, ndmin=2)]
            else:
                bounds[a:b] = param.bounds
            a = b
        return bounds

    def get_blocks(self):
        """
        Return a list whose ith element contains indices for the parameters
        which make up the ith block.
        """
        blocks = dict()
        a = 0
        for _, param in self.__walk_params():
            b = a + param.nparams
            blocks.setdefault(param.block, []).extend(range(a, b))
            a = b
        return blocks.values()

    def get_names(self):
        """
        Return a list of names for each parameter.
        """
        names = []
        for name, param in self.__walk_params():
            if param.nparams == 1:
                names.append(name)
            else:
                names.extend('{:s}[{:d}]'.format(name, n)
                             for n in xrange(param.nparams))
        return names
