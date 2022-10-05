"""Unit tests for the pull-back metrics."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from tests.conftest import Parametrizer, TestCase
from tests.data.pullback_metric_data import PullbackMetricTestData


def _circle_immersion(point):
    return gs.array(
        [
            gs.cos(point),
            gs.sin(point),
        ]
    )


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


def _expected_jacobian_circle_immersion(point):
    jacobian = gs.array(
        [
            [-gs.sin(point)],
            [gs.cos(point)],
        ]
    )
    return jacobian


def _expected_jacobian_sphere_immersion(point):
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


def _expected_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
    return mat


def _expected_inverse_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_inverse_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** (-2)]])
    return mat


@tests.conftest.autograd_tf_and_torch_only
class TestPullbackMetric(TestCase, metaclass=Parametrizer):

    testing_data = PullbackMetricTestData()
    Metric = testing_data.Metric

    def test_sphere_immersion(self, spherical_coords, expected):
        result = _sphere_immersion(spherical_coords)
        self.assertAllClose(result, expected)

    def test_sphere_immersion_and_spherical_to_extrinsic(self, dim, point):
        expected = _sphere_immersion(point)
        result = Hypersphere(dim).spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_jacobian_sphere_immersion(self, dim, pole):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        result = pullback_metric.jacobian_immersion(pole)
        expected = _expected_jacobian_sphere_immersion(pole)
        self.assertAllClose(result, expected)

    def test_jacobian_circle_immersion(self, dim, pole):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_circle_immersion
        )
        result = pullback_metric.jacobian_immersion(pole)
        expected = _expected_jacobian_circle_immersion(pole)
        self.assertAllClose(result, expected)

    def test_tangent_sphere_immersion(self, dim, tangent_vec, point, expected):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        result = pullback_metric.tangent_immersion(tangent_vec, point)
        self.assertAllClose(result, expected)

    def test_tangent_circle_immersion(self, dim, tangent_vec, point, expected):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_circle_immersion
        )
        result = pullback_metric.tangent_immersion(tangent_vec, point)
        self.assertAllClose(result, expected)

    def test_sphere_metric_matrix(self, dim, base_point):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        result = pullback_metric.metric_matrix(base_point)
        expected = _expected_sphere_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_circle_metric_matrix(self, dim, base_point):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_circle_immersion
        )
        result = pullback_metric.metric_matrix(base_point)
        expected = _expected_circle_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_inverse_sphere_metric_matrix(self, dim, base_point):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )

        result = pullback_metric.cometric_matrix(base_point)
        expected = _expected_inverse_sphere_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_inverse_circle_metric_matrix(self, dim, base_point):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_circle_immersion
        )

        result = pullback_metric.cometric_matrix(base_point)
        expected = _expected_inverse_circle_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_inner_product_and_sphere_inner_product(
        self,
        dim,
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
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        immersed_base_point = _sphere_immersion(base_point)
        jac_immersion = pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matvec(jac_immersion, tangent_vec_b)

        result = pullback_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point=base_point
        )
        expected = Hypersphere(dim).metric.inner_product(
            immersed_tangent_vec_a,
            immersed_tangent_vec_b,
            base_point=immersed_base_point,
        )
        self.assertAllClose(result, expected)

    def test_inner_product_derivative_matrix_s2(self, dim, base_point):
        metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        theta, _ = base_point[0], base_point[1]

        derivative_matrix = metric.inner_product_derivative_matrix(base_point)

        assert ~gs.allclose(derivative_matrix, gs.zeros((dim, dim, dim)))

        # derivative with respect to theta
        expected_1 = gs.array([[0, 0], [0, 2 * gs.cos(theta) * gs.sin(theta)]])
        # derivative with respect to phi
        expected_2 = gs.zeros((2, 2))

        self.assertAllClose(derivative_matrix.shape, (2, 2, 2))
        self.assertAllClose(derivative_matrix[:, :, 0], expected_1)
        self.assertAllClose(derivative_matrix[:, :, 1], expected_2)

    def test_christoffels_and_sphere_christoffels(self, dim, base_point):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        result = pullback_metric.christoffels(base_point)
        expected = Hypersphere(2).metric.christoffels(base_point)
        self.assertAllClose(result, expected)

    def test_christoffels_sphere(self, dim, base_point):
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        theta, _ = base_point[0], base_point[1]

        christoffels = pullback_metric.christoffels(base_point)

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

    def test_christoffels_circle(self, dim, base_point):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_circle_immersion
        )
        result = pullback_metric.christoffels(base_point)

        self.assertAllClose(result.shape, (1, 1, 1))
        self.assertAllClose(result, gs.zeros((1, 1, 1)))

    def test_exp_and_sphere_exp(self, dim, tangent_vec, base_point):
        """Test consistency between sphere's Riemannian exp.

        The exp map of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The exp map of pullback_metric is defined
        in terms of the spherical coordinates.
        """
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        immersed_base_point = _sphere_immersion(base_point)
        jac_immersion = pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec)
        result = pullback_metric.exp(tangent_vec, base_point=base_point)
        result = Hypersphere(dim).spherical_to_extrinsic(result)
        expected = Hypersphere(dim).metric.exp(
            immersed_tangent_vec_a, base_point=immersed_base_point
        )
        self.assertAllClose(result, expected, atol=1e-1)

    @tests.conftest.torch_only
    def test_parallel_transport_and_sphere_parallel_transport(
        self, dim, tangent_vec_a, tangent_vec_b, base_point
    ):
        """Test consistency between sphere's parallel transports.

        The parallel transport of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The parallel transport of pullback_metric is defined
        in terms of the spherical coordinates.

        Note: this test passes in autograd and in tf but takes a very long time.
        """
        pullback_metric = self.Metric(
            dim=dim, embedding_dim=dim + 1, immersion=_sphere_immersion
        )
        immersed_base_point = _sphere_immersion(base_point)
        jac_immersion = pullback_metric.jacobian_immersion(base_point)
        immersed_tangent_vec_a = gs.matvec(jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matvec(jac_immersion, tangent_vec_b)

        result_dict = pullback_metric.ladder_parallel_transport(
            tangent_vec_a, base_point=base_point, direction=tangent_vec_b
        )

        result = result_dict["transported_tangent_vec"]
        end_point = result_dict["end_point"]
        result = pullback_metric.tangent_immersion(v=result, x=end_point)

        expected = Hypersphere(dim).metric.parallel_transport(
            immersed_tangent_vec_a,
            base_point=immersed_base_point,
            direction=immersed_tangent_vec_b,
        )
        self.assertAllClose(result, expected, atol=5e-3)
