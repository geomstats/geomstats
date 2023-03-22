"""Unit tests for the pull-back metrics."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from tests.conftest import Parametrizer, TestCase
from tests.data.pullback_metric_data import PullbackMetricTestData


@tests.conftest.autograd_and_torch_only
class TestPullbackMetric(TestCase, metaclass=Parametrizer):

    testing_data = PullbackMetricTestData()
    Metric = testing_data.Metric

    def test_sphere_immersion(self, space, point, expected):
        result = space.immersion(point)
        self.assertAllClose(result, expected)

    def test_sphere_immersion_and_spherical_to_extrinsic(self, space, point):
        result = space.immersion(point)
        expected = Hypersphere(space.dim).spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_jacobian_immersion(self, space, pole, expected_func):
        result = space.jacobian_immersion(pole)
        expected = expected_func(pole)
        self.assertAllClose(result, expected)

    def test_tangent_immersion(self, space, tangent_vec, point, expected):
        result = space.tangent_immersion(tangent_vec, point)
        self.assertAllClose(result, expected)

    def test_metric_matrix(self, space, base_point, expected_func):
        space.equip_with_metric(self.Metric)
        result = space.metric.metric_matrix(base_point)
        expected = expected_func(base_point)
        self.assertAllClose(result, expected)

    def test_inverse_metric_matrix(self, space, base_point, expected_func):
        space.equip_with_metric(self.Metric)
        result = space.metric.cometric_matrix(base_point)
        expected = expected_func(base_point)
        self.assertAllClose(result, expected)

    def test_inner_product_and_hypersphere_inner_product(
        self,
        space,
        tangent_vec_a,
        tangent_vec_b,
        base_point,
    ):
        """Test consistency between sphere's inner-products.

        The inner-product of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The inner-product of pullback_metric is defined in terms
        of the spherical coordinates.
        """
        space.equip_with_metric(self.Metric)
        immersed_base_point = space.immersion(base_point)

        jac_immersion = space.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matvec(jac_immersion, tangent_vec_b)

        result = space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point=base_point
        )
        expected = Hypersphere(space.dim).metric.inner_product(
            immersed_tangent_vec_a,
            immersed_tangent_vec_b,
            base_point=immersed_base_point,
        )
        self.assertAllClose(result, expected)

    def test_sphere_inner_product_derivative_matrix(self, space, base_point):
        space.equip_with_metric(self.Metric)
        dim = space.dim

        theta, _ = base_point[0], base_point[1]

        derivative_matrix = space.metric.inner_product_derivative_matrix(base_point)

        assert ~gs.allclose(derivative_matrix, gs.zeros((dim, dim, dim)))

        # derivative with respect to theta
        expected_1 = gs.array([[0, 0], [0, 2 * gs.cos(theta) * gs.sin(theta)]])
        # derivative with respect to phi
        expected_2 = gs.zeros((2, 2))

        self.assertAllClose(derivative_matrix.shape, (2, 2, 2))
        self.assertAllClose(derivative_matrix[:, :, 0], expected_1)
        self.assertAllClose(derivative_matrix[:, :, 1], expected_2)

    def test_christoffels_and_hypersphere_christoffels(self, space, base_point):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        space.equip_with_metric(self.Metric)
        result = space.metric.christoffels(base_point)
        expected = Hypersphere(space.dim).metric.christoffels(base_point)
        self.assertAllClose(result, expected)

    def test_christoffels_sphere(self, space, base_point):
        space.equip_with_metric(self.Metric)
        theta, _ = base_point[0], base_point[1]

        christoffels = space.metric.christoffels(base_point)

        self.assertAllClose(christoffels.shape, (2, 2, 2))

        expected_1_11 = expected_2_11 = expected_2_22 = expected_1_12 = 0

        self.assertAllClose(christoffels[0, 0, 0], expected_1_11)
        self.assertAllClose(christoffels[1, 0, 0], expected_2_11)
        self.assertAllClose(christoffels[1, 1, 1], expected_2_22)
        self.assertAllClose(christoffels[0, 0, 1], expected_1_12)

        expected_1_22 = -gs.sin(theta) * gs.cos(theta)
        expected_2_12 = expected_2_21 = gs.cos(theta) / gs.sin(theta)

        self.assertAllClose(christoffels[0, 1, 1], expected_1_22)
        self.assertAllClose(christoffels[1, 0, 1], expected_2_12)
        self.assertAllClose(christoffels[1, 1, 0], expected_2_21)

    def test_christoffels_circle(self, space, base_point):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        space.equip_with_metric(self.Metric)
        result = space.metric.christoffels(base_point)

        self.assertAllClose(result.shape, (1, 1, 1))
        self.assertAllClose(result, gs.zeros((1, 1, 1)))

    @tests.conftest.autograd_and_torch_only
    def test_exp_and_hypersphere_exp(self, space, tangent_vec, base_point):
        """Test consistency between sphere's Riemannian exp.

        The exp map of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The exp map of pullback_metric is defined
        in terms of the spherical coordinates.
        """
        space.equip_with_metric(self.Metric)

        immersed_base_point = space.immersion(base_point)
        jac_immersion = space.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec)
        result = space.metric.exp(tangent_vec, base_point=base_point)
        result = Hypersphere(space.dim).spherical_to_extrinsic(result)
        expected = Hypersphere(space.dim).metric.exp(
            immersed_tangent_vec_a, base_point=immersed_base_point
        )
        self.assertAllClose(result, expected, atol=1e-1)

    @tests.conftest.torch_only
    def test_parallel_transport_and_hypersphere_parallel_transport(
        self, space, tangent_vec_a, tangent_vec_b, base_point
    ):
        """Test consistency between sphere's parallel transports.

        The parallel transport of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The parallel transport of pullback_metric is defined
        in terms of the spherical coordinates.

        Note: this test passes in autograd and in tf but takes a very long time.
        """
        space.equip_with_metric(self.Metric)
        immersed_base_point = space.immersion(base_point)
        jac_immersion = space.metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matvec(jac_immersion, tangent_vec_b)

        result_dict = space.metric.ladder_parallel_transport(
            tangent_vec_a, base_point=base_point, direction=tangent_vec_b
        )

        result = result_dict["transported_tangent_vec"]
        end_point = result_dict["end_point"]
        result = space.metric.tangent_immersion(v=result, x=end_point)

        expected = Hypersphere(space.dim).metric.parallel_transport(
            immersed_tangent_vec_a,
            base_point=immersed_base_point,
            direction=immersed_tangent_vec_b,
        )
        self.assertAllClose(result, expected, atol=5e-3)

    def test_hessian_sphere_immersion(self, space, base_point, expected_func):
        """Test the hessian immersion.

        The hessian immersion is the hessian of the immersion
        function.
        """
        result = space.hessian_immersion(base_point)
        expected = expected_func(base_point)
        self.assertAllClose(
            result.shape, (space.embedding_space.dim, space.dim, space.dim)
        )
        self.assertAllClose(result, expected)

    def test_second_fundamental_form_sphere(self, space, base_point):
        space.equip_with_metric(self.Metric)

        theta, phi = base_point[0], base_point[1]
        radius = 1
        result = space.metric.second_fundamental_form(base_point)

        expected_11 = gs.array(
            [
                -radius * gs.sin(theta) * gs.cos(phi),
                -radius * gs.sin(theta) * gs.sin(phi),
                -radius * gs.cos(theta),
            ]
        )
        expected_22 = gs.array(
            [
                -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.cos(phi),
                -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.sin(phi),
                -radius * gs.sin(theta) ** 2 * gs.cos(theta),
            ]
        )

        result_11 = result[:, 0, 0]
        result_22 = result[:, 1, 1]

        self.assertAllClose(result_11.shape, expected_11.shape)
        self.assertAllClose(result_22.shape, expected_22.shape)

        self.assertAllClose(result_11, expected_11)
        self.assertAllClose(result_22, expected_22, atol=1e-5)

    def test_second_fundamental_form_circle(self, space, base_point):
        space.equip_with_metric(self.Metric)
        result = space.metric.second_fundamental_form(base_point)

        expected = gs.array(
            [
                [-gs.cos(base_point)],
                [-gs.sin(base_point)],
            ]
        )

        self.assertAllClose(result.shape, expected.shape)
        self.assertAllClose(result, expected)

    def test_mean_curvature_vector_norm_sphere(self, space, base_point):
        space.equip_with_metric(self.Metric)
        radius = 1
        result = space.metric.mean_curvature_vector(base_point)
        result = gs.linalg.norm(result)
        expected = gs.array(2 / radius)
        self.assertAllClose(result, expected)

    def test_mean_curvature_vector_norm_circle(self, space, base_point):
        space.equip_with_metric(self.Metric)
        radius = 1
        result = space.metric.mean_curvature_vector(base_point)
        result = gs.linalg.norm(result)
        expected = gs.array(1 / radius)
        self.assertAllClose(result, expected)
