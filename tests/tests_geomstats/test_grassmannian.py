"""Unit tests for the Grassmannian."""

import random

import geomstats.backend as gs
import geomstats.tests.helper as helper
from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.geometry.matrices import Matrices
from tests.conftest import MetricParametrizer, Parametrizer, TestCase, TestData

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

r_y = gs.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
pi_2 = gs.pi / 2
pi_4 = gs.pi / 4


class TestGrassmannian(TestCase, metaclass=Parametrizer):
    cls = Grassmannian

    class TestDataGrassmannian(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=3, k=2, point=p_xy, expected=True),
                dict(n=3, k=2, point=gs.array([p_yz, p_xz]), expected=[True, True]),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_shape_data(self):
            random_data = []
            n_list = random.sample(range(2, 50), 5)
            for n in n_list:
                k_list = random.sample(range(1, n), 5)
                for k in k_list:
                    n_samples = random.sample(range(2, 50))
                    random_data += [dict(n=n, k=k, n_samples=n_samples)]

            return self.generate_tests([], random_data)

        def to_tangent_is_tangent_data(self):
            random_data = []
            n_list = random.sample(range(2, 50), 5)
            for n in n_list:
                k_list = random.sample(range(1, n), 5)
                for k in k_list:
                    n_samples = random.sample(range(2, 50))
                    random_data += [dict(n=n, k=k, n_samples=n_samples)]

            return self.generate_tests([], random_data)

    def test_belongs(self, n, k, point, expected):
        self.assertAllClose(self.cls(n, k).belongs(point), gs.array(expected))

    def test_random_point_belongs_shape(self, n, k, n_samples):
        result = gs.all(self.cls(n, k).belongs(self.cls(n, k).random_point(n_samples)))
        self.assertAllClose(result, gs.array(True))

    def test_to_tangent_is_tangent(self, n, k, n_samples):
        space = self.cls(n, k)
        point = space.random_point(n_samples)
        vector = gs.random.rand(*point.shape) / 4
        tangent_vec = space.to_tangent(vector, point)
        result = space.is_tangent(tangent_vec, point)
        self.assertAllClose(result, gs.array(True))

    def test_projection_and_belongs(self, n, k):
        space = self.cls(n, k)
        shape = (2, n, k)
        result = helper.test_projection_and_belongs(space, shape)
        self.assertAllClose(gs.all(result), gs.array(True))


class TestGrassmannianCanonicalMetric(TestCase, metaclass=MetricParametrizer):
    cls = GrassmannianCanonicalMetric
    space = Grassmannian

    class TestDataGrassmannianCanonicalMetric(TestData):
        def exp_data(self):
            smoke_data = [
                dict(
                    n=3,
                    k=2,
                    tangent_vec=Matrices.bracket(pi_2 * r_y, gs.array([p_xy, p_yz])),
                    base_point=gs.array([p_xy, p_yz]),
                    expected=gs.array([p_yz, p_xy]),
                ),
                dict(
                    n=3,
                    k=2,
                    tangent_vec=Matrices.bracket(
                        pi_2 * gs.array([r_y, r_z]), gs.array([p_xy, p_yz])
                    ),
                    base_point=gs.array([p_xy, p_yz]),
                    expected=gs.array([p_yz, p_xz]),
                ),
            ]
            return self.generate_tests(smoke_data)

    def test_exp(self, n, k, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.cls(n, k).exp(gs.array(tangent_vec), gs.array(base_point)),
            gs.array(expected),
        )

    def test_parallel_transport(self, n, k):
        metric = self.cls(n, k)
        space = self.space(n, k)
        shape = (2, n, k)

        result = helper.test_parallel_transport(space, metric, shape)
        self.assertAllClose(gs.all(result), gs.array(True))
