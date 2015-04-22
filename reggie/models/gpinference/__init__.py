"""
Objects which implement GP inference.
"""

# pylint: disable=wildcard-import

from .exact import *
from .fitc import *
from .laplace import *

from . import exact
from . import fitc
from . import laplace

__all__ = []
__all__ += exact.__all__
__all__ += fitc.__all__
__all__ += laplace.__all__

INFERENCE = dict((name.lower(), globals()[name]) for name in __all__)
