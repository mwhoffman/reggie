"""
Utilities for pretty-printing.
"""

__all__ = ['repr_args']


def repr_args(obj, *args, **kwargs):
    """
    Return a repr string for an object with args and kwargs.
    """
    typename = type(obj).__name__
    args = ['{:s}'.format(str(val)) for val in args]
    kwargs = ['{:s}={:s}'.format(name, str(val))
              for name, val in kwargs.items()]
    return '{:s}({:s})'.format(typename, ', '.join(args + kwargs))
