"""
Parameters and parameterizable objects.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = []


class Parameterized(object):
    def __init__(self):
        self.nhyper = 0
        self.__names = []
        self.__hypers = []
        self.__transforms = []
        self.__priors = []

    def __repr__(self):
        substrings = []
        for name, hyper in zip(self.__names, self.__hypers):
            value = np.array2string(hyper, separator=',')
            substrings.append('{:s}={:s}'.format(name, value))
        return self.__class__.__name__ + '(' + ', '.join(substrings) + ')'

    def _link_hyper(self, name, hyper, transform=None, prior=None):
        # FIXME: check C-order, etc?
        # FIXME: add properties of some sort?
        if name in self.__names:
            raise ValueError('hyperparameter names must be unique')
        self.nhyper += hyper.size
        self.__names.append(name)
        self.__hypers.append(hyper)
        self.__transforms.append(transform)
        self.__priors.append(prior)

    def set_hyper(self, hyper):
        hyper = np.array(hyper, dtype=float, copy=False, ndmin=1)
        if hyper.shape != (self.nhyper,):
            raise ValueError('incorrect number of hyperparameters')
        offset = 0
        for hyper_ in self.__hypers:
            hyper_.flat[:] = hyper[offset:offset+hyper_.size]
            offset += hyper_.size

    def get_hyper(self):
        if self.nhyper == 0:
            return np.array([])
        else:
            return np.hstack(_ for _ in self.__hypers)
