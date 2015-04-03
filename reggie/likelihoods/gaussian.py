"""
Implementation of the Gaussian likelihood.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ._core import Likelihood
from ..core.domains import POSITIVE

__all__ = ['Gaussian']


class Gaussian(Likelihood):
    def __init__(self, sn2):
        # register our parameters
        self._sn2 = self._register('sn2', sn2, domain=POSITIVE)
