"""
Objects which implement the function interface.
"""

# pylint: disable=wildcard-import

from .basic import *
from .gpmean import *

from . import basic
from . import gpmean

__all__ = []
__all__ += basic.__all__
__all__ += gpmean.__all__
