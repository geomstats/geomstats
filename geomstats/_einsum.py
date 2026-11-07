"""Vectorized products written with einsum."""

import geomstats.backend as gs


def scalarvecmul(scalar, vec):
    """Vectorized scalar vector multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    vec : array-like, shape=[..., dim]
    """
    return gs.einsum("...,...i->...i", scalar, vec)


def scalarmatmul(scalar, mat):
    """Vectorized scalar matrix multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    mat : array-like, shape=[..., n, m]
    """
    return gs.einsum("...,...ij->...ij", scalar, mat)
