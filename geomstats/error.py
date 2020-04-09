"""Checks and associated errors."""

import geomstats.backend as gs


def check_strictly_positive_integer(n, n_name):
    """Raise an error if n is not a > 0 integer.

    Parameters
    ----------
    n : unspecified
       Parameter to be tested.
    n_name : string
       Name of the parameter.
    """
    if not(isinstance(n, int) and n > 0):
        raise ValueError(
            '{} is required to be a strictly positive integer,'
            ' got {}.'.format(n_name, n))


def check_belongs(point, manifold, manifold_name):
    """Raise an error if point does not belong to the input manifold.

    Parameters
    ----------
    points: array-like
        Point to be tested.
    manifold : Manifold
        Manifold to which the point should belong.
    manifold_name : string
        Name of the manifold for the error message.
    """
    if not gs.all(manifold.belongs(point)):
        raise RuntimeError(
            'Some points do not belong to manifold \'%s\'.' % manifold_name)
