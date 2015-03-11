"""
Tests for 
"""

# pylint: disable=missing-docstring

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import reggie.utils.pretty as pretty


def test_repr_args():
    class TmpClass(object):
        def __repr__(self):
            return pretty.repr_args(self, 1, 2, a=3, b=4)

    instance = TmpClass()
    assert repr(instance) == 'TmpClass(1, 2, a=3, b=4)'
