import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.numerics.finite_differences import (
    centered_difference,
    forward_difference,
    second_centered_difference,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.finite_differences import FiniteDifferenceTestCase

from .data.finite_differences import (
    CenteredDifferenceTestData,
    ForwardDifferenceTestData,
    SecondCenteredDifferenceTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(dim=random.randint(2, 5), equip=False),
        Matrices(m=random.randint(2, 5), n=random.randint(2, 5)),
    ],
)
def spaces(request):
    request.cls.space = request.param


@pytest.mark.usefixtures("spaces")
class TestForwardDifference(FiniteDifferenceTestCase, metaclass=DataBasedParametrizer):
    testing_data = ForwardDifferenceTestData()

    @pytest.mark.random
    def test_forward_difference_last(self, n_points, n_times, atol):
        """Check if last point is reached by finite difference"""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        time = gs.linspace(0.0, 1.0, n_times)

        path = self._linear_path(base_point, tangent_vec, time)

        finite_diffs = forward_difference(path, axis=-(self.space.point_ndim + 1))

        delta = 1 / (n_times - 1)
        point_ndim_slc = tuple([slice(None)] * self.space.point_ndim)
        point_ = (
            path[(..., -2) + point_ndim_slc]
            + delta * finite_diffs[(..., -1) + point_ndim_slc]
        )
        self.assertAllClose(point_, path[(..., -1) + point_ndim_slc], atol=atol)


@pytest.mark.usefixtures("spaces")
class TestCenteredDifference(FiniteDifferenceTestCase, metaclass=DataBasedParametrizer):
    testing_data = CenteredDifferenceTestData()

    @pytest.mark.random
    def test_centered_difference_random_index(self, n_points, n_times, endpoints, atol):
        """Check if random index difference is obtained properly."""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        time = gs.linspace(0.0, 1.0, n_times)

        path = self._linear_path(base_point, tangent_vec, time)
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


@pytest.mark.usefixtures("spaces")
class TestSecondCenteredDifference(
    FiniteDifferenceTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SecondCenteredDifferenceTestData()

    @pytest.mark.random
    def test_second_centered_difference_random_index(self, n_points, n_times, atol):
        """Check if random index difference is obtained properly."""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        time = gs.linspace(0.0, 1.0, n_times)

        path = self._linear_path(base_point, tangent_vec, time)
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
