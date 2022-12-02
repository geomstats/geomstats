import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.geometry.base import LevelSetTestCase
from geomstats.test.vectorization import generate_vectorization_data


class FullRankCorrelationMatricesTestCase(LevelSetTestCase):
    # TODO: add smoke tests?

    def test_from_covariance(self, point, expected, atol):
        point_ = self.space.from_covariance(point)
        self.assertAllClose(point_, expected, atol=atol)

    @pytest.mark.random
    def test_from_covariance_belongs(self, n_points, atol):
        spd_point = self.space.embedding_space.random_point(n_points)

        point = self.space.from_covariance(spd_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_belongs(point, expected, atol)

    @pytest.mark.vec
    def test_from_covariance_vec(self, n_reps, atol):
        point = self.space.embedding_space.random_point()
        expected = self.space.from_covariance(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_from_covariance(**datum)

    def test_diag_action(self, diagonal_vec, point, expected, atol):
        res = self.space.diag_action(diagonal_vec, point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_diag_action_vec(self, n_reps, atol):
        point = self.space.embedding_space.random_point()
        diagonal_vec = Matrices.diagonal(point)

        expected = self.space.diag_action(diagonal_vec, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    diagonal_vec=diagonal_vec, point=point, expected=expected, atol=atol
                )
            ],
            arg_names=["diagonal_vec", "point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        for datum in vec_data:
            self.test_diag_action(**datum)
