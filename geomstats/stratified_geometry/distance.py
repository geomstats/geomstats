"""Module for metrics/distances on a space."""

import abc

import geomstats.backend as gs


class Distance(abc.ABC):
    """Class for the distance on a length space.

    The argument ``kwargs`` can have information like the choice of a particular
    geometry that is used for computation of the distance of the length space, where one
    thus obtains a different length space for a different choice of geometry.

    Parameters
    ----------
    kwargs : dict
        Optional parameters.
    """

    def __init__(self, **kwargs):
        super(Distance, self).__init__(**kwargs)

    @abc.abstractmethod
    def dist(self, p, q, **kwargs):
        """Compute the distance between two points.

        Point-like means that points need to be instances of a class that inherits from
        class ``Point`` that is associated to the ``Space`` that this distance is
        associated to.

        Parameters
        ----------
        p : point-like
            The first input of the distance function (metric).
        q : point-like
            The second input of the distance function (metric).
        kwargs : dict
            Optional parameters.

        Returns
        -------
        distance : float
            The distance between points ``p`` and ``q``.
        """
