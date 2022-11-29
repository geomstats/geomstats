import pytest

import geomstats.backend as gs
from geomstats.test.geometry.base import VectorSpaceTestCase
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data

# TODO: mixins with MatrixVectorSpaces?
# TODO: use `self.space.ndim` to control vector dimension


class SymmetricMatricesTestCase(VectorSpaceTestCase):
    def test_to_vector(self, point, expected, atol):
        vec = self.space.to_vector(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_vector_vec(self, n_reps, atol):
        point = self.space.random_point()
        expected = self.space.to_vector(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_to_vector(**datum)

    def test_from_vector(self, vec, expected, atol):
        mat = self.space.from_vector(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_from_vector_vec(self, n_reps, atol):
        n = self.space.n
        vec = gs.random.rand(n * (n + 1) // 2)
        expected = self.space.from_vector(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_from_vector(**datum)

    @pytest.mark.random
    def test_from_vector_after_to_vector(self, n_points, atol):
        mat = self.space.random_point(n_points)

        vec = self.space.to_vector(mat)

        mat_ = self.space.from_vector(vec)
        self.assertAllClose(mat_, mat, atol=atol)

    @pytest.mark.random
    def test_to_vector_after_from_vector(self, n_points, atol):
        n = self.space.n
        vec_dim = n * (n + 1) // 2
        vec = gs.reshape(gs.random.rand(n_points * vec_dim), (n_points, -1))

        mat = self.space.from_vector(vec)

        vec_ = self.space.to_vector(mat)
        self.assertAllClose(vec_, vec, atol=atol)


class SymmetricMatricesOpsTestCase(TestCase):
    # TODO: check apply_func_to_eigenvals alone

    def test_expm(self, mat, expected, atol):
        res = self.Space.expm(mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_powerm(self, mat, power, expected, atol):
        # TODO: check vectorization
        res = self.Space.powerm(mat, power)
        self.assertAllClose(res, expected, atol=atol)
