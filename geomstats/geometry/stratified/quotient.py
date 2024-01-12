"""Quotient structure for a geodesic metric space."""

from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import PointSetMetric


class AlignerAlgorithm(ABC):
    """Base class for point to point aligner."""

    @abstractmethod
    def align(self, total_space, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        total_space : PointSet
            Set with quotient structure.
        point : array-like, shape=[..., *point_shape]
            Point to align.
        base_point : array-like, shape=[..., *point_shape]
            Reference point.

        Returns
        -------
        aligned_point : array-like, shape=[..., *point_shape]
            Aligned point.
        """


class Aligner(ABC):
    """Bundle structure."""

    def __init__(self, total_space, aligner_algorithm=None):
        self._total_space = total_space
        if aligner_algorithm is not None:
            self.aligner_algorithm = aligner_algorithm

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
        if hasattr(self, "aligner_algorithm"):
            return self.aligner_algorithm.align(self._total_space, point, base_point)

        raise NotImplementedError("`align` is not implemented")


class QuotientMetric(PointSetMetric, ABC):
    """Quotient metric."""

    def __init__(self, space, total_space):
        self.total_space = total_space
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
        aligned_point_b = self.total_space.aligner.align(point_b, point_a)
        return self.total_space.metric.squared_dist(
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
        aligned_end_point = self.total_space.aligner.align(end_point, initial_point)
        return self.total_space.metric.geodesic(
            initial_point=initial_point, end_point=aligned_end_point
        )
