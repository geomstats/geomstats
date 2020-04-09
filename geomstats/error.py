"""Checks and associated errors."""


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
