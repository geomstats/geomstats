import pytest

from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase


class ConnectionComparisonTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    @pytest.mark.random
    def test_christoffels(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.christoffels(base_point)
        res_ = self.other_space.metric.christoffels(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.exp(tangent_vec, base_point)
        res_ = self.other_space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        res = self.space.metric.log(point, base_point)
        res_ = self.other_space.metric.log(point, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_riemann_tensor(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.riemann_tensor(base_point)
        res_ = self.other_space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_curvature(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        res_ = self.other_space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_ricci_tensor(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.ricci_tensor(base_point)
        res_ = self.other_space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_directional_curvature(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_curvature_derivative(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_d = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        res_ = self.other_space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_directional_curvature_derivative(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        res = self.space.metric.geodesic(initial_point, end_point=end_point)(time)
        res_ = self.other_space.metric.geodesic(initial_point, end_point=end_point)(
            time
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        initial_tangent_vec = self.data_generator.random_tangent_vec(initial_point)
        time = get_random_times(n_times)

        res = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)

        res_ = self.other_space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_parallel_transport_with_direction(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        direction = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_parallel_transport_with_end_point(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_injectivity_radius(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.injectivity_radius(base_point)
        res_ = self.other_space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, res_, atol=atol)


class RiemannianMetricComparisonTestCase(ConnectionComparisonTestCase):
    @pytest.mark.random
    def test_metric_matrix(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.metric_matrix(base_point)
        res_ = self.other_space.metric.metric_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_cometric_matrix(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.cometric_matrix(base_point)
        res_ = self.other_space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inner_product_derivative_matrix(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.inner_product_derivative_matrix(base_point)
        res_ = self.other_space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inner_product(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        res_ = self.other_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inner_coproduct(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        cotangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        cotangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        res_ = self.other_space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_squared_norm(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        vector = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.squared_norm(vector, base_point)
        res_ = self.other_space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_norm(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        vector = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.norm(vector, base_point)
        res_ = self.other_space.metric.norm(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_normalize(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        vector = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.normalize(vector, base_point)
        res_ = self.other_space.metric.normalize(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_squared_dist(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        res = self.space.metric.squared_dist(point_a, point_b)
        res_ = self.other_space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_dist(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        res = self.space.metric.dist(point_a, point_b)
        res_ = self.other_space.metric.dist(point_a, point_b)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_covariant_riemann_tensor(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.covariant_riemann_tensor(base_point)
        res_ = self.other_space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_sectional_curvature(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_scalar_curvature(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.scalar_curvature(base_point)
        res_ = self.other_space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, res_, atol=atol)
