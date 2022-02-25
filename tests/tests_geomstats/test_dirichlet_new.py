"""Unit tests for the Dirichlet manifold."""

import random

from scipy.stats import dirichlet

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.information_geometry.dirichlet import DirichletDistributions
from tests.conftest import TestCase
from tests.data_generation import OpenSetTestData
from tests.parametrizers import OpenSetParametrizer


class TestDirichlet(TestCase, metaclass=OpenSetParametrizer):
    space = DirichletDistributions

    class TestDataDirichlet(OpenSetTestData):
        space = DirichletDistributions
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n,) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(1, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(dim=3, vec=[0.1, 1.0, 0.3], expected=True),
                dict(dim=3, vec=[0.1, 1.0], expected=False),
                dict(dim=3, vec=[0.0, 1.0, 0.3], expected=False),
                dict(dim=2, vec=[-1.0, 0.3], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_data(self):
            smoke_data = [
                dict(dim=2, n_samples=1, expected=(2,)),
                dict(dim=3, n_samples=5, expected=(5, 3)),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                self.space,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def to_tangent_is_tangent_in_ambient_space_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_data(
                self.space,
                self.space_args_list,
                self.shape_list,
            )

        def sample_data(self):
            smoke_data = [
                dict(dim=2, point=gs.array([1.0, 1.0]), n_samples=1, expected=(1, 2)),
                dict(
                    dim=3, point=gs.array([0.1, 0.2, 0.3]), n_samples=2, expected=(2, 3)
                ),
            ]
            return self.generate_tests(smoke_data)

        def sample_belongs_data(self):
            random_data = [
                dict(dim=n, n_points=n_points, n_samples=n_samples)
                for n in self.n_list
                for n_points in self.n_points_list
                for n_samples in self.n_samples_list
            ]
            return self.generate_tests([], random_data)

        def point_to_pdf_data(self):
            n_samples = 10
            random_data = [
                dict(dim=n, n_points=n_points, n_samples=n_samples)
                for n in self.n_list
                for n_points in self.n_points_list
            ]
            return self.generate_tests([], random_data)

        def metric_matrix_shape_data(self):
            smoke_data = [
                dict(dim=2, n_points=1, expected=(2, 2)),
                dict(dim=2, n_points=3, expected=(3, 2, 2)),
                dict(dim=3, n_points=2, expected=(2, 3, 3)),
            ]
            return self.generate_tests(smoke_data)

        def metric_matrix_dim_2_data(self):
            random_data = [
                dict(point=self.space(2).random_point(n_points))
                for n_points in self.n_points_list
            ]
            return self.generate_tests([], random_data)

        def christoffels_shape_data(self):
            smoke_data = [
                dict(dim=2, n_points=1, expected=(2, 2, 2)),
                dict(dim=2, n_points=3, expected=(3, 2, 2, 2)),
                dict(dim=3, n_points=2, expected=(2, 3, 3, 3)),
            ]
            return self.generate_tests(smoke_data)

        def christoffels_dim_2_data(self):
            def coefficients(param_a, param_b):
                """Christoffel coefficients for the beta distributions."""
                poly1a = gs.polygamma(1, param_a)
                poly2a = gs.polygamma(2, param_a)
                poly1b = gs.polygamma(1, param_b)
                poly2b = gs.polygamma(2, param_b)
                poly1ab = gs.polygamma(1, param_a + param_b)
                poly2ab = gs.polygamma(2, param_a + param_b)
                metric_det = 2 * (poly1a * poly1b - poly1ab * (poly1a + poly1b))

                c1 = (poly2a * (poly1b - poly1ab) - poly1b * poly2ab) / metric_det
                c2 = -poly1b * poly2ab / metric_det
                c3 = (poly2b * poly1ab - poly1b * poly2ab) / metric_det
                return c1, c2, c3

            gs.random.seed(123)
            n_points = 3
            points = self.space(2).random_point(n_points)
            param_a, param_b = points[:, 0], points[:, 1]
            c1, c2, c3 = coefficients(param_a, param_b)
            c4, c5, c6 = coefficients(param_b, param_a)
            vector_0 = gs.stack([c1, c2, c3], axis=-1)
            vector_1 = gs.stack([c6, c5, c4], axis=-1)
            gamma_0 = SymmetricMatrices.from_vector(vector_0)
            gamma_1 = SymmetricMatrices.from_vector(vector_1)
            smoke_data = [
                dict(point=points, expected=gs.stack([gamma_0, gamma_1], axis=-3))
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataDirichlet()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.space(dim).belongs(gs.array(vec)), expected)

    def test_random_point(self, dim, n_samples, expected):
        self.assertAllClose(self.space(dim).random_point(n_samples).shape, expected)

    def test_sample(self, dim, point, n_samples, expected):
        self.assertAllClose(self.space(dim).sample(point, n_samples).shape, expected)

    def test_sample_belongs(self, dim, n_points, n_samples):
        points = self.space(dim).random_point(n_points)
        samples = self.space(dim).sample(points, n_samples)
        expected = gs.squeeze(gs.ones((n_points, n_samples)))
        self.assertAllClose(gs.sum(samples, axis=-1), expected)

    def test_point_to_pdf(self, dim, n_points, n_samples):
        point = self.space(dim).random_point(n_points)
        pdf = self.space(dim).point_to_pdf(point)
        alpha = gs.ones(dim)
        samples = self.space(dim).sample(alpha, n_samples)
        result = pdf(samples)
        pdf = []
        for i in range(n_points):
            pdf.append(gs.array([dirichlet.pdf(x, point[i, :]) for x in samples]))
        expected = gs.squeeze(gs.stack(pdf, axis=0))
        self.assertAllClose(result, expected)

    def test_metric_matrix_shape(self, dim, n_points, expected):
        points = self.space(dim).random_point(n_points)
        return self.assertAllClose(
            self.space(dim).metric.metric_matrix(points).shape, expected
        )

    def test_metric_matrix_dim_2(self, points):
        param_a = points[..., 0]
        param_b = points[..., 1]
        vector = gs.stack(
            [
                gs.polygamma(1, param_a) - gs.polygamma(1, param_a + param_b),
                -gs.polygamma(1, param_a + param_b),
                gs.polygamma(1, param_b) - gs.polygamma(1, param_a + param_b),
            ],
            axis=-1,
        )
        expected = SymmetricMatrices.from_vector(vector)
        return self.assertAllClose(self.space(2).metric.metric_matrix(points), expected)

    def test_christoffels_shape(self, dim, n_points, expected):
        points = self.space(dim).random_point(n_points)
        return self.assertAllClose(
            self.space(dim).metric.christoffels(points).shape, expected
        )

    def test_christoffels_dim_2(self, points, expected):
        return self.assertAllClose(self.space(2).metric.christoffels(points), expected)
