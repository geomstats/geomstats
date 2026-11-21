import random

import geomstats.backend as gs
from geomstats.numerics.finite_differences import (
    centered_difference,
    forward_difference,
    second_centered_difference,
)
from geomstats.test.test_case import TestCase


class _FiniteDifferenceTestCase(TestCase):
    def _linear_path(self, base_point, tangent_vec, n_times):
        """Sample a straight line at uniformly spaced times in [0, 1].

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point at time 0.
        tangent_vec : array-like, shape=[..., *space.shape]
            Velocity of the line.
        n_times : int
            Number of sampled times.

        Returns
        -------
        path : array-like, shape=[..., n_times, *space.shape]
            Sampled line.
        """
        time = gs.linspace(0.0, 1.0, n_times)
        ijk = "ijk"[: self.space.point_ndim]
        if base_point.shape[: -self.space.point_ndim]:
            base_point = gs.expand_dims(base_point, axis=-(self.space.point_ndim + 1))

        return base_point + gs.einsum(f"t,...{ijk}->...t{ijk}", time, tangent_vec)


class ForwardDifferenceTestCase(_FiniteDifferenceTestCase):
    def test_forward_difference_last(
        self, base_point, tangent_vec, n_times, atol=gs.atol
    ):
        """Check the last point is reached from the last forward difference.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point at time 0 of a straight line.
        tangent_vec : array-like, shape=[..., *space.shape]
            Velocity of the straight line.
        n_times : int
            Number of sampled times.
        atol : float
            Absolute tolerance.
        """
        path = self._linear_path(base_point, tangent_vec, n_times)

        finite_diffs = forward_difference(path, axis=-(self.space.point_ndim + 1))

        delta = 1 / (n_times - 1)
        point_ndim_slc = tuple([slice(None)] * self.space.point_ndim)
        point_ = (
            path[(..., -2) + point_ndim_slc]
            + delta * finite_diffs[(..., -1) + point_ndim_slc]
        )
        self.assertAllClose(point_, path[(..., -1) + point_ndim_slc], atol=atol)


class CenteredDifferenceTestCase(_FiniteDifferenceTestCase):
    def test_centered_difference_random_index(
        self, base_point, tangent_vec, n_times, endpoints, atol=gs.atol
    ):
        """Check the centered difference at a random index.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point at time 0 of a straight line.
        tangent_vec : array-like, shape=[..., *space.shape]
            Velocity of the straight line.
        n_times : int
            Number of sampled times.
        endpoints : bool
            Whether the differences at the endpoints are computed.
        atol : float
            Absolute tolerance.
        """
        path = self._linear_path(base_point, tangent_vec, n_times)
        finite_diffs = centered_difference(
            path, axis=-(self.space.point_ndim + 1), endpoints=endpoints
        )

        index = random.randint(1, n_times - 2)
        fd_index = index if endpoints else index - 1

        point_ndim_slc = tuple([slice(None)] * self.space.point_ndim)

        previous_point = path[(..., index - 1) + point_ndim_slc]
        next_point = path[(..., index + 1) + point_ndim_slc]
        diff = finite_diffs[(..., fd_index) + point_ndim_slc]

        delta = 1 / (n_times - 1)
        expected_diff = (next_point - previous_point) / (2 * delta)
        self.assertAllClose(diff, expected_diff, atol=atol)


class SecondCenteredDifferenceTestCase(_FiniteDifferenceTestCase):
    def test_second_centered_difference_random_index(
        self, base_point, tangent_vec, n_times, atol=gs.atol
    ):
        """Check the second centered difference at a random index.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point at time 0 of a straight line.
        tangent_vec : array-like, shape=[..., *space.shape]
            Velocity of the straight line.
        n_times : int
            Number of sampled times.
        atol : float
            Absolute tolerance.
        """
        path = self._linear_path(base_point, tangent_vec, n_times)
        finite_diffs = second_centered_difference(
            path,
            axis=-(self.space.point_ndim + 1),
        )

        index = random.randint(1, n_times - 2)
        fd_index = index - 1

        point_ndim_slc = tuple([slice(None)] * self.space.point_ndim)

        previous_point = path[(..., index - 1) + point_ndim_slc]
        point = path[(..., index) + point_ndim_slc]
        next_point = path[(..., index + 1) + point_ndim_slc]
        diff = finite_diffs[(..., fd_index) + point_ndim_slc]

        delta = 1 / (n_times - 1)
        expected_diff = (next_point + previous_point - 2 * point) / (delta**2)
        self.assertAllClose(diff, expected_diff, atol=atol)
