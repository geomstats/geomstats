"""Checks and associated errors."""

import math

import geomstats.backend as gs


def check_integer(n, n_name):
    """Raise an error if n is not a > 0 integer.

    Parameters
    ----------
    n : unspecified
       Parameter to be tested.
    n_name : string
       Name of the parameter.
    """
    if not(isinstance(n, int) and n > 0):
        if n is not None and n != math.inf:
            raise ValueError(
                '{} is required to be either'
                ' None, math.inf or a strictly positive integer,'
                ' got {}.'.format(n_name, n))


def check_belongs(point, manifold, **kwargs):
    """Raise an error if point does not belong to the input manifold.

    Parameters
    ----------
    point : array-like
        Point to be tested.
    manifold : Manifold
        Manifold to which the point should belong.
    manifold_name : string
        Name of the manifold for the error message.
    """
    if not gs.all(manifold.belongs(point, **kwargs)):
        raise RuntimeError(
            'Some points do not belong to manifold \'%s\' of dimension %d.'
            % (type(manifold).__name__, manifold.dim))


def check_parameter_accepted_values(param, param_name, accepted_values):
    """Raise an error if parameter does not belong to a set of values.

    Parameters
    ----------
    param : unspecified
        Parameter to be tested.
    param_name : string
        Name of the parameter.
    accepted_values : list
        Accepted values that the parameter can take.
    """
    if param not in accepted_values:
        raise ValueError(
            'Parameter {} needs to be in {}, got: {}'.format(
                param_name, accepted_values, param))
