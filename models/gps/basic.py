"""
Exact inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .exact import ExactGP
from ..kernels import SE
from ..functions import Constant

__all__ = ['BasicGP']


class BasicGP(ExactGP):
    def __init__(self, sn2, rho, ell, mean=0.0, ndim=None):
        super(BasicGP, self).__init__(sn2, SE(rho, ell, ndim), Constant(mean))

        # flatten the parameters and rename them
        self._flatten({'kernel.rho': 'rho',
                       'kernel.ell': 'ell',
                       'mean.bias': 'mean'})

    def __repr__(self):
        if self._kernel._iso:
            return super(BasicGP, self).__repr__(ndim=self._kernel.ndim)
        else:
            return super(BasicGP, self).__repr__()
