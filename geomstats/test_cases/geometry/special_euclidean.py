import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    homogeneous_representation,
)
from geomstats.test.random import LieGroupVectorRandomDataGenerator
from geomstats.test.test_case import assert_allclose
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.lie_group import LieGroupTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins


def homogeneous_representation_test_case(
    rotation,
    translation,
    constant,
    expected,
    atol=gs.atol,
):
    out = homogeneous_representation(rotation, translation, constant)
    assert_allclose(out, expected, atol=atol)


def homogeneous_representation_vec_test_case(n, n_reps, atol):
    rotation = SquareMatrices(n).random_point()
    translation = Euclidean(n).random_point()
    constant = gs.array(1.0)

    expected = homogeneous_representation(rotation, translation, constant)

    vec_data = generate_vectorization_data(
        data=[
            dict(
                rotation=rotation,
                translation=translation,
                constant=constant,
                expected=expected,
                atol=atol,
            )
        ],
        arg_names=["rotation", "translation", "constant"],
        expected_name="expected",
        n_reps=n_reps,
    )
    for datum in vec_data:
        homogeneous_representation_test_case(**datum)


class SpecialEuclideanVectorsTestCase(ProjectionTestCaseMixins, LieGroupTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = LieGroupVectorRandomDataGenerator(self.space)

    def test_matrix_from_vector(self, vec, expected, atol):
        mat = self.space.matrix_from_vector(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_from_vector_vec(self, n_reps, atol):
        vec = self.data_generator.random_point()
        expected = self.space.matrix_from_vector(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_matrix_from_vector_belongs_to_matrices(self, n_points):
        point = self.data_generator.random_point(n_points)
        mat = self.space.matrix_from_vector(point)

        space = SpecialEuclidean(self.space.n, point_type="matrix")
        belongs = space.belongs(mat)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(expected, belongs)

    def test_regularize_tangent_vec(self, tangent_vec, base_point, expected, atol):
        regularized = self.space.regularize_tangent_vec(tangent_vec, base_point)
        self.assertAllClose(regularized, expected, atol=atol)

    def test_regularize_tangent_vec_at_identity(self, tangent_vec, expected, atol):
        regularized = self.space.regularize_tangent_vec_at_identity(tangent_vec)
        self.assertAllClose(regularized, expected, atol=atol)

    @pytest.mark.vec
    def test_regularize_tangent_vec_at_identity_vec(self, n_reps, atol):
        tangent_vec = self.data_generator.random_tangent_vec(self.space.identity)

        expected = self.space.regularize_tangent_vec_at_identity(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[dict(tangent_vec=tangent_vec, expected=expected, atol=atol)],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
