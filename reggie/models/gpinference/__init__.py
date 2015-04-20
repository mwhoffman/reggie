"""
Objects which implement GP inference.
"""

# pylint: disable=wildcard-import

from .exact import *
from .fitc import *

from . import exact
from . import fitc

__all__ = []
__all__ += exact.__all__
__all__ += fitc.__all__

INFERENCE = {
    'exact': Exact,
    'fitc': FITC
}
