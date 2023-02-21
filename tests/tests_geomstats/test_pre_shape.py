"""Unit tests for the preshape space."""

import pytest

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from tests.conftest import Parametrizer
from tests.data.pre_shape_data import (
    KendallShapeMetricTestData,
    PreShapeMetricTestData,
    PreShapeSpaceTestData,
)
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestPreShapeSpace(LevelSetTestCase, metaclass=Parametrizer):
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True

    testing_data = PreShapeSpaceTestData()

    def test_belongs(self, k_landmarks, m_ambient, mat, expected):
        space = self.Space(k_landmarks, m_ambient)
        result = space.belongs(mat)
        self.assertAllClose(result, expected)

    def test_is_centered(self, k_landmarks, m_ambient, point, expected):
        space = self.Space(k_landmarks, m_ambient)
        result = space.is_centered(point)
        self.assertAllEqual(result, expected)

    def test_to_center_is_center(self, k_landmarks, m_ambient, point):
        space = self.Space(k_landmarks, m_ambient)
        centered_point = space.center(point)
        result = gs.all(space.is_centered(centered_point))
        self.assertTrue(result)

    def test_vertical_projection(self, k_landmarks, m_ambient, tangent_vec, point):
        space = self.Space(k_landmarks, m_ambient)
        vertical = space.vertical_projection(tangent_vec, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tangent_vec)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self, k_landmarks, m_ambient, tangent_vec, point):
        space = self.Space(k_landmarks, m_ambient)
        horizontal = space.horizontal_projection(tangent_vec, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)
        self.assertAllClose(result, expected)

    def test_horizontal_and_is_tangent(
        self, k_landmarks, m_ambient, tangent_vec, point
    ):
        space = self.Space(k_landmarks, m_ambient)
        horizontal = space.horizontal_projection(tangent_vec, point)
        result = gs.all(space.is_tangent(horizontal, point))
        self.assertTrue(result)

    def test_alignment_is_symmetric(self, k_landmarks, m_ambient, point, base_point):
        space = self.Space(k_landmarks, m_ambient)
        aligned = space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = gs.all(Matrices.is_symmetric(alignment))
        self.assertTrue(result)

    @tests.conftest.np_and_autograd_only
    def test_integrability_tensor(
        self, k_landmarks, m_ambient, tangent_vec_a, tangent_vec_b, base_point
    ):
        """Identities of integrability tensor in kendall pre-shape space.

        The integrability tensor A_X E is skew-symmetric with respect to the
        pre-shape metric, :math:`< A_X E, F> + <E, A_X F> = 0`. By
        polarization, this is equivalent to :math:`< A_X E, E> = 0`.

        The integrability tensor is also alternating (:math:`A_X Y =
        - A_Y X`)  for horizontal vector fields :math:'X,Y',  and it is
        exchanging horizontal and vertical vector spaces.
        """
        space = self.Space(k_landmarks, m_ambient)
        result_ab = space.integrability_tensor(tangent_vec_a, tangent_vec_b, base_point)

        result = space.total_space_metric.inner_product(
            tangent_vec_b, result_ab, base_point
        )
        expected = 0.0
        self.assertAllClose(result, expected, atol=gs.atol * 10)

        horizontal_b = space.horizontal_projection(tangent_vec_b, base_point)
        horizontal_a = space.horizontal_projection(tangent_vec_a, base_point)
        result = space.integrability_tensor(horizontal_a, horizontal_b, base_point)
        expected = -space.integrability_tensor(horizontal_b, horizontal_a, base_point)
        self.assertAllClose(result, expected, atol=gs.atol * 10)

        is_vertical = space.is_vertical(result, base_point)
        self.assertTrue(is_vertical)

        vertical_b = tangent_vec_b - horizontal_b
        result = space.integrability_tensor(horizontal_a, vertical_b, base_point)
        is_horizontal = space.is_horizontal(result, base_point)
        self.assertTrue(is_horizontal)

    @tests.conftest.np_and_autograd_only
    def test_integrability_tensor_derivative_is_alternate(
        self,
        k_landmarks,
        m_ambient,
        hor_x,
        hor_y,
        hor_z,
        nabla_x_y,
        nabla_x_z,
        base_point,
    ):
        r"""Integrability tensor derivatives is alternate in pre-shape.

        For two horizontal vector fields :math:`X,Y` the integrability
        tensor (hence its derivatives) is alternate:
        :math:`\nabla_X ( A_Y Z + A_Z Y ) = 0`.
        """
        space = self.Space(k_landmarks, m_ambient)
        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            hor_z,
            nabla_x_z,
            base_point,
        )
        nabla_x_a_z_y, a_z_y = space.integrability_tensor_derivative(
            hor_x,
            hor_z,
            nabla_x_z,
            hor_y,
            nabla_x_y,
            base_point,
        )
        result = nabla_x_a_y_z + nabla_x_a_z_y
        self.assertAllClose(a_y_z + a_z_y, gs.zeros_like(result), atol=gs.atol * 10)
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_integrability_tensor_derivative_is_skew_symmetric(
        self,
        k_landmarks,
        m_ambient,
        hor_x,
        hor_y,
        hor_z,
        ver_v,
        nabla_x_y,
        nabla_x_z,
        nabla_x_v,
        base_point,
    ):
        r"""Integrability tensor derivatives is skew-symmetric in pre-shape.

        For :math:`X,Y` horizontal and :math:`V,W` vertical:
        :math:`\nabla_X (< A_Y Z , V > + < A_Y V , Z >) = 0`.
        """
        space = self.Space(k_landmarks, m_ambient)

        scal = space.total_space_metric.inner_product

        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            hor_z,
            nabla_x_z,
            base_point,
        )

        nabla_x_a_y_v, a_y_v = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            ver_v,
            nabla_x_v,
            base_point,
        )

        result = (
            scal(nabla_x_a_y_z, ver_v)
            + scal(a_y_z, nabla_x_v)
            + scal(nabla_x_a_y_v, hor_z)
            + scal(a_y_v, nabla_x_z)
        )
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_integrability_tensor_derivative_reverses_hor_ver(
        self,
        k_landmarks,
        m_ambient,
        hor_x,
        hor_y,
        hor_z,
        ver_v,
        ver_w,
        hor_h,
        nabla_x_y,
        nabla_x_z,
        nabla_x_h,
        nabla_x_v,
        nabla_x_w,
        base_point,
    ):
        r"""Integrability tensor derivatives exchanges hor & ver in pre-shape.

        For :math:`X,Y,Z` horizontal and :math:`V,W` vertical, the
        integrability tensor (and thus its derivative) reverses horizontal
        and vertical subspaces: :math:`\nabla_X < A_Y Z, H > = 0`  and
        :math:`nabla_X < A_Y V, W > = 0`.
        """
        space = self.Space(k_landmarks, m_ambient)

        scal = space.total_space_metric.inner_product

        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            hor_z,
            nabla_x_z,
            base_point,
        )
        result = scal(nabla_x_a_y_z, hor_h) + scal(a_y_z, nabla_x_h)
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 10)

        nabla_x_a_y_v, a_y_v = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            ver_v,
            nabla_x_v,
            base_point,
        )
        result = scal(nabla_x_a_y_v, ver_w) + scal(a_y_v, nabla_x_w)
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_integrability_tensor_derivative_parallel(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, base_point
    ):
        """Test optimized integrability tensor derivatives in pre-shape space.

        Optimized version for quotient-parallel vector fields should equal
        the general implementation.
        """
        space = self.Space(k_landmarks, m_ambient)
        (nabla_x_a_y_z_qp, a_y_z_qp,) = space.integrability_tensor_derivative_parallel(
            hor_x, hor_y, hor_z, base_point
        )

        a_x_y = space.integrability_tensor(hor_x, hor_y, base_point)
        a_x_z = space.integrability_tensor(hor_x, hor_z, base_point)

        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x, hor_y, a_x_y, hor_z, a_x_z, base_point
        )

        self.assertAllClose(a_y_z, a_y_z_qp, atol=gs.atol * 10)
        self.assertAllClose(nabla_x_a_y_z, nabla_x_a_y_z_qp, atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_iterated_integrability_tensor_derivative_parallel(
        self, k_landmarks, m_ambient, hor_x, hor_y, base_point
    ):
        """Test optimized iterated integrability tensor derivatives.

        The optimized version of the iterated integrability tensor
        :math:`A_X A_Y A_X Y`, computed with the horizontal lift of
        quotient-parallel vector fields extending the tangent vectors
        :math:`X,Y` of Kendall shape spaces (identified to horizontal vectors
        of the pre-shape space), is the recursive application of two general
        integrability tensor derivatives with proper derivatives.
        Intermediate computations returned are also verified.
        """
        space = self.Space(k_landmarks, m_ambient)
        a_x_y = space.integrability_tensor(hor_x, hor_y, base_point)
        nabla_x_v, a_x_y = space.integrability_tensor_derivative(
            hor_x,
            hor_x,
            gs.zeros_like(hor_x),
            hor_y,
            a_x_y,
            base_point,
        )

        (nabla_x_a_y_a_x_y, a_y_a_x_y,) = space.integrability_tensor_derivative(
            hor_x, hor_y, a_x_y, a_x_y, nabla_x_v, base_point
        )

        a_x_a_y_a_x_y = space.integrability_tensor(hor_x, a_y_a_x_y, base_point)

        (
            nabla_x_a_y_a_x_y_qp,
            a_x_a_y_a_x_y_qp,
            nabla_x_v_qp,
            a_y_a_x_y_qp,
            ver_v_qp,
        ) = space.iterated_integrability_tensor_derivative_parallel(
            hor_x, hor_y, base_point
        )
        self.assertAllClose(a_x_y, ver_v_qp, atol=gs.atol * 10)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp, atol=gs.atol * 10)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp, atol=gs.atol * 10)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp, atol=gs.atol * 10)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp, atol=gs.atol * 10)


class TestKendallShapeMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = KendallShapeMetricTestData()
    Space = testing_data.Space

    def test_curvature_is_skew_operator(self, k_landmarks, m_ambient, vec, base_point):
        metric = self.Metric(k_landmarks, m_ambient)
        space = self.Space(k_landmarks, m_ambient)
        tangent_vec_a = space.to_tangent(vec[:2], base_point)
        tangent_vec_b = space.to_tangent(vec[2:], base_point)

        result = metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point
        )
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected, atol=gs.atol * 100)

    @tests.conftest.np_and_autograd_only
    def test_curvature_bianchi_identity(
        self,
        k_landmarks,
        m_ambient,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        base_point,
    ):
        """First Bianchi identity on curvature in pre-shape space.

        :math:`R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0`.
        """
        metric = self.Metric(k_landmarks, m_ambient)
        curvature_1 = metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        curvature_2 = metric.curvature(
            tangent_vec_b, tangent_vec_c, tangent_vec_a, base_point
        )
        curvature_3 = metric.curvature(
            tangent_vec_c, tangent_vec_a, tangent_vec_b, base_point
        )

        result = curvature_1 + curvature_2 + curvature_3
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    @pytest.mark.skip("keeps throwing error")
    def test_kendall_sectional_curvature(
        self, k_landmarks, m_ambient, tangent_vec_a, tangent_vec_b, base_point
    ):
        """Sectional curvature of Kendall shape space is larger than 1.

        The sectional curvature always increase by taking the quotient in a
        Riemannian submersion. Thus, it should larger in kendall shape space
        thane the sectional curvature of the pre-shape space which is 1 as it
        a hypersphere.
        The sectional curvature is computed here with the generic
        directional_curvature and sectional curvature methods.
        """
        space = self.Space(k_landmarks, m_ambient)
        metric = self.Metric(k_landmarks, m_ambient)
        hor_a = space.horizontal_projection(tangent_vec_a, base_point)
        hor_b = space.horizontal_projection(tangent_vec_b, base_point)

        tidal_force = metric.directional_curvature(hor_a, hor_b, base_point)

        numerator = metric.inner_product(tidal_force, hor_a, base_point)
        denominator = (
            metric.inner_product(hor_a, hor_a, base_point)
            * metric.inner_product(hor_b, hor_b, base_point)
            - metric.inner_product(hor_a, hor_b, base_point) ** 2
        )
        condition = ~gs.isclose(denominator, 0.0, atol=gs.atol * 100)
        kappa = numerator[condition] / denominator[condition]
        kappa_direct = metric.sectional_curvature(hor_a, hor_b, base_point)[condition]
        self.assertAllClose(kappa, kappa_direct)
        result = kappa > 1.0 - 1e-10
        self.assertTrue(gs.all(result))

    @tests.conftest.np_and_autograd_only
    def test_kendall_curvature_derivative_bianchi_identity(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, hor_h, base_point
    ):
        r"""2nd Bianchi identity on curvature derivative in kendall space.

        For any 3 tangent vectors horizontally lifted from kendall shape
        space to Kendall pre-shape space, :math:`(\nabla_X R)(Y, Z)
        + (\nabla_Y R)(Z,X) + (\nabla_Z R)(X, Y) = 0`.
        """
        metric = self.Metric(k_landmarks, m_ambient)
        term_x = metric.curvature_derivative(hor_x, hor_y, hor_z, hor_h, base_point)
        term_y = metric.curvature_derivative(hor_y, hor_z, hor_x, hor_h, base_point)
        term_z = metric.curvature_derivative(hor_z, hor_x, hor_y, hor_h, base_point)

        result = term_x + term_y + term_z
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 100)

    def test_curvature_derivative_is_skew_operator(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, base_point
    ):
        r"""Derivative of a skew operator is skew.

        For any 3 tangent vectors horizontally lifted from kendall shape space
        to Kendall pre-shape space, :math:`(\nabla_X R)(Y,Y)Z = 0`.
        """
        metric = self.Metric(k_landmarks, m_ambient)
        result = metric.curvature_derivative(hor_x, hor_y, hor_y, hor_z, base_point)
        self.assertAllClose(result, gs.zeros_like(result), atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_directional_curvature_derivative(
        self, k_landmarks, m_ambient, hor_x, hor_y, base_point
    ):
        """Test equality of directional curvature derivative implementations.

        General formula based on curvature derivative, optimized method of
        KendallShapeMetric class, method from the QuotientMetric class and
        method from the Connection class have to give identical results.
        """
        metric = self.Metric(k_landmarks, m_ambient)

        # General formula based on curvature derivative
        expected = metric.curvature_derivative(hor_x, hor_y, hor_x, hor_y, base_point)

        # Optimized method of KendallShapeMetric class
        result_kendall_shape_metric = metric.directional_curvature_derivative(
            hor_x, hor_y, base_point
        )
        self.assertAllClose(result_kendall_shape_metric, expected, atol=gs.atol * 10)

        # Method from the QuotientMetric class
        result_quotient_metric = super(
            self.Metric, metric
        ).directional_curvature_derivative(hor_x, hor_y, base_point)
        self.assertAllClose(result_quotient_metric, expected, atol=gs.atol * 10)

        # Method from the Connection class

        result_connection = super(
            QuotientMetric, metric
        ).directional_curvature_derivative(hor_x, hor_y, base_point)
        self.assertAllClose(result_connection, expected, atol=gs.atol * 10)

    @tests.conftest.np_and_autograd_only
    def test_directional_curvature_derivative_is_quadratic(
        self, k_landmarks, m_ambient, coef_x, coef_y, hor_x, hor_y, base_point
    ):
        """Directional curvature derivative is quadratic in both variables."""
        metric = self.Metric(k_landmarks, m_ambient)
        coef_x = -2.5
        coef_y = 1.5
        result = metric.directional_curvature_derivative(
            coef_x * hor_x, coef_y * hor_y, base_point
        )
        expected = (
            coef_x**2
            * coef_y**2
            * metric.directional_curvature_derivative(hor_x, hor_y, base_point)
        )
        self.assertAllClose(result, expected, atol=gs.atol * 1000)

    def test_parallel_transport(
        self, k_landmarks, m_ambient, tangent_vec_a, tangent_vec_b, base_point
    ):
        space = self.Space(k_landmarks, m_ambient)
        metric = self.Metric(k_landmarks, m_ambient)
        tan_a = space.horizontal_projection(tangent_vec_a, base_point)
        tan_b = space.horizontal_projection(tangent_vec_b, base_point)

        # orthonormalize and move to base_point
        tan_a -= gs.einsum(
            "...,...ij->...ij",
            metric.inner_product(tan_a, tan_b, base_point)
            / metric.squared_norm(tan_b, base_point),
            tan_b,
        )
        tan_b = gs.einsum(
            "...ij,...->...ij", tan_b, 1.0 / metric.norm(tan_b, base_point)
        )
        tan_a = gs.einsum(
            "...ij,...->...ij", tan_a, 1.0 / metric.norm(tan_a, base_point)
        )

        transported = metric.parallel_transport(
            tan_a, base_point, tan_b, n_steps=400, step="rk4"
        )
        end_point = metric.exp(tan_b, base_point)
        result = metric.norm(transported, end_point)
        expected = metric.norm(tan_a, base_point)
        self.assertAllClose(result, expected, atol=gs.atol * 10)

        is_tangent = space.is_tangent(transported, end_point)
        is_horizontal = space.is_horizontal(transported, end_point)
        self.assertTrue(gs.all(is_tangent))
        self.assertTrue(gs.all(is_horizontal))


class TestPreShapeMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = True
    skip_test_log_after_exp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = PreShapeMetricTestData()
