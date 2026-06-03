"""Quotient structure for a geodesic metric space."""

from abc import ABC

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import PointSetMetric


class Aligner(ABC):
    """Bundle structure.

    Parameters
    ----------
    total_space : PointSet
        Set with quotient structure.
    align_algo : AlignerAlgorithm
        Algorihtm performing alignment.
    """

    def __init__(self, total_space, align_algo=None):
        self._total_space = total_space
        if align_algo is not None:
            self.align_algo = align_algo

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point to align.
        base_point : array-like, shape=[..., *point_shape]
            Reference point.

        Returns
        -------
        aligned_point: list, shape = [..., *point_shape]
        """
        if hasattr(self, "align_algo"):
            return self.align_algo.align(point, base_point)

        raise NotImplementedError("`align` is not implemented")


class QuotientMetric(PointSetMetric, ABC):
    """Quotient metric.

    Parameters
    ----------
    space : PointSet
        Set to equip with metric.
    total_space : PointSet
        Set with quotient structure.
    """

    def __init__(self, space, total_space):
        self._total_space = total_space
        super().__init__(space)

    def squared_dist(self, point_a, point_b):
        """Compute distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., *point_shape]
        point_b : array-like, shape=[..., *point_shape]

        Returns
        -------
        distance : array-like, shape=[...]
            Distance between the points.
        """
        aligned_point_b = self._total_space.aligner.align(point_b, point_a)
        return self._total_space.metric.squared_dist(
            point_a,
            aligned_point_b,
        )

    def dist(self, point_a, point_b):
        """Compute distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., *point_shape]
        point_b : array-like, shape=[..., *point_shape]

        Returns
        -------
        distance : array-like, shape=[...]
            Distance between the points.
        """
        return gs.sqrt(self.squared_dist(point_a, point_b))

    def geodesic(self, initial_point, end_point):
        """Compute geodesic between two points.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *point_shape]
            Initial point.
        end_point : array-like, shape=[..., *point_shape]
            End point.

        Returns
        -------
        geodesic : callable
            Geodesic function.
        """
        aligned_end_point = self._total_space.aligner.align(end_point, initial_point)
        return self._total_space.metric.geodesic(
            initial_point=initial_point, end_point=aligned_end_point
        )
