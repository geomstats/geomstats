import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.test.test_case import TestCase
from geomstats.test.utils import PointTransformer
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class HyperbolicTransformer(PointTransformer):
    def __init__(self, space, other_space):
        self.space = space
        self.other_space = other_space

    def transform_point(self, point):
        return self.space.change_coordinates_system(
            point,
            self.space.coords_type,
            self.other_space.coords_type,
        )

    def inverse_transform_point(self, other_point):
        return self.space.change_coordinates_system(
            other_point,
            self.other_space.coords_type,
            self.space.coords_type,
        )


class BallToHalfSpace(HyperbolicTransformer):
    def transform_tangent_vec(self, tangent_vec, base_point):
        return self.space.ball_to_half_space_tangent(
            tangent_vec,
            base_point,
        )

    def inverse_transform_tangent_vec(self, other_tangent_vec, other_base_point):
        return self.space.half_space_to_ball_tangent(
            other_tangent_vec,
            other_base_point,
        )


class HyperbolicCoordsTransformTestCase(TestCase):
    def _get_space(self, coords_type):
        return Hyperbolic(dim=self.dim, coords_type=coords_type)

    def _get_random_point(self, coords_type, n_points=1):
        return self._get_space(coords_type).random_point(n_points)

    def _test_from_space_to_other_space_tangent_is_tangent(
        self, n_points, from_, other, atol
    ):
        space = self._get_space(from_)
        base_point = space.random_point(n_points)
        tangent_vec = space.random_tangent_vec(base_point)

        func_to_tangent = getattr(
            self.space,
            f'{from_.replace("-", "_")}_to_{other.replace("-", "_")}_tangent',
        )
        tangent_vec_other = func_to_tangent(tangent_vec, base_point)
        base_point_other = self.space.change_coordinates_system(
            base_point, from_, other
        )

        other_space = self._get_space(other)
        res = other_space.is_tangent(tangent_vec_other, base_point_other, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def _test_from_to_other_space_tangent_vec(self, n_reps, from_, other, atol):
        space = self._get_space(from_)
        base_point = space.random_point()
        tangent_vec = space.random_tangent_vec(base_point)

        func_to_tangent = getattr(
            self.space,
            f'{from_.replace("-", "_")}_to_{other.replace("-", "_")}_tangent',
        )
        expected = func_to_tangent(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(
            vec_data,
            test_fnc_name=f'test_{from_.replace("-", "_")}_to_{other.replace("-", "_")}_tangent',
        )

    def test_half_space_to_ball_tangent(self, tangent_vec, base_point, expected, atol):
        res = self.space.half_space_to_ball_tangent(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_half_space_to_ball_tangent_vec(self, n_reps, atol):
        return self._test_from_to_other_space_tangent_vec(
            n_reps, "half-space", "ball", atol
        )

    @pytest.mark.random
    def test_half_space_to_ball_tangent_is_tangent(self, n_points, atol):
        return self._test_from_space_to_other_space_tangent_is_tangent(
            n_points, "half-space", "ball", atol
        )

    def test_ball_to_half_space_tangent(self, tangent_vec, base_point, expected, atol):
        res = self.space.ball_to_half_space_tangent(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_ball_to_half_space_tangent_vec(self, n_reps, atol):
        return self._test_from_to_other_space_tangent_vec(
            n_reps, "ball", "half-space", atol
        )

    @pytest.mark.random
    def test_ball_to_half_space_tangent_is_tangent(self, n_points, atol):
        return self._test_from_space_to_other_space_tangent_is_tangent(
            n_points, "ball", "half-space", atol
        )

    def test_change_coordinates_system(
        self, point, from_coordinates_system, to_coordinates_system, expected, atol
    ):
        res = self.space.change_coordinates_system(
            point, from_coordinates_system, to_coordinates_system
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_change_coordinates_system_vec(
        self, n_reps, from_coordinates_system, to_coordinates_system, atol
    ):
        point = self._get_random_point(from_coordinates_system)
        expected = self.space.change_coordinates_system(
            point, from_coordinates_system, to_coordinates_system
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    from_coordinates_system=from_coordinates_system,
                    to_coordinates_system=to_coordinates_system,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.vec
    def test_change_coordinates_system_after_change_coordinates_system(
        self, n_points, from_coordinates_system, to_coordinates_system, atol
    ):
        point = self._get_random_point(from_coordinates_system, n_points)
        point_other = self.space.change_coordinates_system(
            point, from_coordinates_system, to_coordinates_system
        )
        point_ = self.space.change_coordinates_system(
            point_other, to_coordinates_system, from_coordinates_system
        )
        self.assertAllClose(point_, point, atol=atol)


class HyperbolicMetricTestCase(RiemannianMetricTestCase):
    pass
