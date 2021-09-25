"""Unit tests for the pull-back metrics."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackMetric


class TestPullbackMetric(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        gs.random.seed(0)
        self.dim = 2
        self.sphere = Hypersphere(dim=self.dim)
        self.sphere_metric = self.sphere.metric

        def _sphere_immersion(spherical_coords):
            theta = spherical_coords[..., 0]
            phi = spherical_coords[..., 1]
            return gs.array(
                [
                    gs.cos(phi) * gs.sin(theta),
                    gs.sin(phi) * gs.sin(theta),
                    gs.cos(theta),
                ]
            )

        self.immersion = _sphere_immersion
        self.pullback_metric = PullbackMetric(
            dim=self.dim, embedding_dim=self.dim + 1, immersion=self.immersion
        )

    @geomstats.tests.autograd_tf_and_torch_only
    def test_immersion(self):
        expected = gs.array([0.0, 0.0, 1.0])
        result = self.immersion(gs.array([0.0, 0.0]))
        self.assertAllClose(result, expected)

        expected = gs.array([0.0, 0.0, -1.0])
        result = self.immersion(gs.array([gs.pi, 0.0]))
        self.assertAllClose(result, expected)

        expected = gs.array([-1.0, 0.0, 0.0])
        result = self.immersion(gs.array([gs.pi / 2.0, gs.pi]))
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_immersion_and_spherical_to_extrinsic(self):
        point = gs.array([0.0, 0.0])
        expected = self.immersion(point)
        result = self.sphere.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

        point = gs.array([0.2, 0.1])
        expected = self.immersion(point)
        result = self.sphere.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_jacobian_immersion(self):
        def _expected_jacobian_immersion(point):
            theta = point[..., 0]
            phi = point[..., 1]
            jacobian = gs.array(
                [
                    [gs.cos(phi) * gs.cos(theta), -gs.sin(phi) * gs.sin(theta)],
                    [gs.sin(phi) * gs.cos(theta), gs.cos(phi) * gs.sin(theta)],
                    [-gs.sin(theta), 0.0],
                ]
            )
            return jacobian

        pole = gs.array([0.0, 0.0])
        result = self.pullback_metric.jacobian_immersion(pole)
        expected = _expected_jacobian_immersion(pole)
        self.assertAllClose(result, expected)

        base_point = gs.array([0.22, 0.1])
        result = self.pullback_metric.jacobian_immersion(base_point)
        expected = _expected_jacobian_immersion(base_point)
        self.assertAllClose(result, expected)

        base_point = gs.array([0.1, 0.88])
        result = self.pullback_metric.jacobian_immersion(base_point)
        expected = _expected_jacobian_immersion(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_tangent_immersion(self):
        point = gs.array([gs.pi / 2.0, gs.pi / 2.0])

        tangent_vec = gs.array([1.0, 0.0])
        result = self.pullback_metric.tangent_immersion(tangent_vec, point)
        expected = gs.array([0.0, 0.0, -1.0])
        self.assertAllClose(result, expected)

        tangent_vec = gs.array([0.0, 1.0])
        result = self.pullback_metric.tangent_immersion(tangent_vec, point)
        expected = gs.array([-1.0, 0.0, 0.0])
        self.assertAllClose(result, expected)

        point = gs.array([gs.pi / 2.0, 0.0])

        tangent_vec = gs.array([1.0, 0.0])
        result = self.pullback_metric.tangent_immersion(tangent_vec, point)
        expected = gs.array([0.0, 0.0, -1.0])
        self.assertAllClose(result, expected)

        tangent_vec = gs.array([0.0, 1.0])
        result = self.pullback_metric.tangent_immersion(tangent_vec, point)
        expected = gs.array([0.0, 1.0, 0.0])
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_metric_matrix(self):
        def _expected_metric_matrix(point):
            theta = point[..., 0]
            mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
            return mat

        base_point = gs.array([0.0, 0.0])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)

        base_point = gs.array([1.0, 1.0])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)

        base_point = gs.array([0.3, 0.8])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_inverse_metric_matrix(self):
        def _expected_inverse_metric_matrix(point):
            theta = point[..., 0]
            mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** (-2)]])
            return mat

        base_point = gs.array([0.6, -1.0])
        result = self.pullback_metric.metric_inverse_matrix(base_point)
        expected = _expected_inverse_metric_matrix(base_point)
        self.assertAllClose(result, expected)

        base_point = gs.array([0.8, -0.8])
        result = self.pullback_metric.metric_inverse_matrix(base_point)
        expected = _expected_inverse_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_inner_product_and_sphere_inner_product(self):
        """Test consistency between sphere's inner-products.

        The inner-product of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The inner-product of pullback_metric is defined in terms
        of the spherical coordinates.
        """
        tangent_vec_a = gs.array([0.0, 1.0])
        tangent_vec_b = gs.array([0.0, 1.0])
        base_point = gs.array([gs.pi / 2.0, 0.0])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matmul(jac_immersion, tangent_vec_b)

        result = self.pullback_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point=base_point
        )
        expected = self.sphere_metric.inner_product(
            immersed_tangent_vec_a,
            immersed_tangent_vec_b,
            base_point=immersed_base_point,
        )
        self.assertAllClose(result, expected)

        tangent_vec_a = gs.array([0.4, 1.0])
        tangent_vec_b = gs.array([0.2, 0.6])
        base_point = gs.array([gs.pi / 2.0, 0.1])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matmul(jac_immersion, tangent_vec_b)

        result = self.pullback_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point=base_point
        )
        expected = self.sphere_metric.inner_product(
            immersed_tangent_vec_a,
            immersed_tangent_vec_b,
            base_point=immersed_base_point,
        )
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_tf_only
    def test_christoffels_and_sphere_christoffels(self):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        base_point = gs.array([0.1, 0.2])
        result = self.pullback_metric.christoffels(base_point=base_point)
        expected = self.sphere_metric.christoffels(point=base_point)
        self.assertAllClose(result, expected)

        base_point = gs.array([0.7, 0.233])
        result = self.pullback_metric.christoffels(base_point=base_point)
        expected = self.sphere_metric.christoffels(point=base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_exp_and_sphere_exp(self):
        """Test consistency between sphere's Riemannian exp.

        The exp map of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The exp map of pullback_metric is defined
        in terms of the spherical coordinates.
        """
        base_point = gs.array([gs.pi / 2.0, 0.0])
        tangent_vec_a = gs.array([0.0, 1.0])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        result = self.pullback_metric.exp(tangent_vec_a, base_point=base_point)
        result = self.sphere.spherical_to_extrinsic(result)
        expected = self.sphere.metric.exp(
            immersed_tangent_vec_a, base_point=immersed_base_point
        )
        self.assertAllClose(result, expected)

        base_point = gs.array([gs.pi / 2.0, 0.1])
        tangent_vec_a = gs.array([0.4, 1.0])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        result = self.pullback_metric.exp(tangent_vec_a, base_point=base_point)
        result = self.sphere.spherical_to_extrinsic(result)
        expected = self.sphere.metric.exp(
            immersed_tangent_vec_a, base_point=immersed_base_point
        )

        self.assertAllClose(result, expected, atol=1e-1)

    @geomstats.tests.autograd_and_torch_only
    def test_parallel_transport_and_sphere_parallel_transport(self):
        """Test consistency between sphere's parallel transports.

        The parallel transport of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The parallel transport of pullback_metric is defined
        in terms of the spherical coordinates.
        """
        tangent_vec_a = gs.array([0.0, 1.0])
        tangent_vec_b = gs.array([0.0, 1.0])
        base_point = gs.array([gs.pi / 2.0, 0.0])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matmul(jac_immersion, tangent_vec_b)

        result_dict = self.pullback_metric.ladder_parallel_transport(
            tangent_vec_a, tangent_vec_b, base_point=base_point
        )

        result = result_dict["transported_tangent_vec"]
        end_point = result_dict["end_point"]
        result = self.pullback_metric.tangent_immersion(v=result, x=end_point)

        expected = self.sphere_metric.parallel_transport(
            immersed_tangent_vec_a,
            immersed_tangent_vec_b,
            base_point=immersed_base_point,
        )
        self.assertAllClose(result, expected, atol=1e-5)
