"""Unit tests for the preshape space."""

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace
from geomstats.geometry.quotient_metric import QuotientMetric
from tests.conftest import TestCase
from tests.data_generation import LevelSetTestData, RiemannianMetricTestData, TestData
from tests.parametrizers import (
    LevelSetParametrizer,
    Parametrizer,
    RiemannianMetricParametrizer,
)

smoke_space = PreShapeSpace(4, 3)
vector = gs.random.rand(11, 4, 3)
base_point = smoke_space.random_point()
tg_vec_0 = smoke_space.to_tangent(vector[0], base_point)
hor_x = smoke_space.horizontal_projection(tg_vec_0, base_point)
tg_vec_1 = smoke_space.to_tangent(vector[1], base_point)
hor_y = smoke_space.horizontal_projection(tg_vec_1, base_point)
tg_vec_2 = smoke_space.to_tangent(vector[2], base_point)
hor_z = smoke_space.horizontal_projection(tg_vec_2, base_point)
tg_vec_3 = smoke_space.to_tangent(vector[3], base_point)
hor_h = smoke_space.horizontal_projection(tg_vec_3, base_point)
tg_vec_4 = smoke_space.to_tangent(vector[4], base_point)
ver_v = smoke_space.vertical_projection(tg_vec_4, base_point)
tg_vec_5 = smoke_space.to_tangent(vector[5], base_point)
ver_w = smoke_space.vertical_projection(tg_vec_5, base_point)
tg_vec_6 = smoke_space.to_tangent(vector[6], base_point)
hor_dy = smoke_space.horizontal_projection(tg_vec_6, base_point)
tg_vec_7 = smoke_space.to_tangent(vector[7], base_point)
hor_dz = smoke_space.horizontal_projection(tg_vec_7, base_point)
tg_vec_8 = smoke_space.to_tangent(vector[8], base_point)
ver_dv = smoke_space.vertical_projection(tg_vec_8, base_point)
tg_vec_9 = smoke_space.to_tangent(vector[9], base_point)
ver_dw = smoke_space.vertical_projection(tg_vec_9, base_point)
tg_vec_10 = smoke_space.to_tangent(vector[10], base_point)
hor_dh = smoke_space.horizontal_projection(tg_vec_10, base_point)

# generate valid derivatives of horizontal / vertical vector fields.
a_x_y = smoke_space.integrability_tensor(hor_x, hor_y, base_point)
nabla_x_y = hor_dy + a_x_y
a_x_z = smoke_space.integrability_tensor(hor_x, hor_z, base_point)
nabla_x_z = hor_dz + a_x_z
a_x_v = smoke_space.integrability_tensor(hor_x, ver_v, base_point)
nabla_x_v = ver_dv + a_x_v
a_x_w = smoke_space.integrability_tensor(hor_x, ver_w, base_point)
nabla_x_w = ver_dw + a_x_w
a_x_h = smoke_space.integrability_tensor(hor_x, hor_h, base_point)
nabla_x_h = hor_dh + a_x_h


class TestPreShapeSpace(TestCase, metaclass=Parametrizer):
    space = PreShapeSpace

    class TestDataPreShapeSpace(TestData):
        def belongs_data(self):
            random_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    mat=gs.random.rand(2, 4),
                    expected=gs.array(False),
                ),
                dict(
                    m_ambient=3,
                    k_landmarks=4,
                    mat=gs.random.rand(10, 2, 4),
                    expected=gs.array([False] * 10),
                ),
            ]
            return self.generate_tests([], random_data)

        def is_centered_data(self):
            random_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    point=gs.ones((4, 3)),
                    expected=gs.array(False),
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    point=gs.zeros((4, 3)),
                    expected=gs.array(True),
                ),
            ]
            return self.generate_tests([], random_data)

        def to_center_is_center_data(self):
            smoke_data = [
                dict(k_landmarks=4, m_ambient=3, point=gs.ones((4, 3))),
                dict(k_landmarks=4, m_ambient=3, point=gs.ones((10, 4, 3))),
            ]
            return self.generate_tests(smoke_data)

        def vertical_projection_data(self):
            vector = gs.random.rand(10, 4, 3)
            space = PreShapeSpace(4, 3)
            point = space.random_point()
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector[0], point),
                    point=point,
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector, point),
                    point=point,
                ),
            ]
            return self.generate_tests(smoke_data)

        def horizontal_projection_data(self):
            vector = gs.random.rand(10, 4, 3)
            space = PreShapeSpace(4, 3)
            point = space.random_point()
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector[0], point),
                    point=point,
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector, point),
                    point=point,
                ),
            ]
            return self.generate_tests(smoke_data)

        def horizontal_and_is_tangent_data(self):
            vector = gs.random.rand(10, 4, 3)
            space = PreShapeSpace(4, 3)
            point = space.random_point()
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector[0], point),
                    point=point,
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec=space.to_tangent(vector, point),
                    point=point,
                ),
            ]
            return self.generate_tests(smoke_data)

        def alignment_is_symmetric_data(self):
            space = PreShapeSpace(4, 3)
            random_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    point=space.random_point(),
                    base_point=space.random_point(),
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    point=space.random_point(),
                    base_point=space.random_point(2),
                ),
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    point=space.random_point(2),
                    base_point=space.random_point(2),
                ),
            ]
            return self.generate_tests([], random_data)

        def integrability_tensor_data(self):
            space = PreShapeSpace(4, 3)
            vector = gs.random.rand(2, 4, 3)
            base_point = space.random_point()
            random_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec_a=space.to_tangent(vector[0], base_point),
                    tangent_vec_b=space.to_tangent(vector[1], base_point),
                    base_point=base_point,
                )
            ]
            return self.generate_tests(random_data)

        def integrability_tensor_old_data(self):
            return self.integrability_tensor_data()

        def integrability_tensor_derivative_is_alternate_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    nabla_x_y=nabla_x_y,
                    nabla_x_z=nabla_x_z,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def integrability_tensor_derivative_is_skew_symmetric_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    ver_v=ver_v,
                    nabla_x_y=nabla_x_y,
                    nabla_x_z=nabla_x_z,
                    nabla_x_v=nabla_x_v,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def integrability_tensor_derivative_reverses_hor_ver_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    ver_v=ver_v,
                    ver_w=ver_w,
                    hor_h=hor_h,
                    nabla_x_y=nabla_x_y,
                    nabla_x_z=nabla_x_z,
                    nabla_x_h=nabla_x_h,
                    nabla_x_v=nabla_x_v,
                    nabla_x_w=nabla_x_w,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def integrability_tensor_derivative_parallel_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def iterated_integrability_tensor_derivative_parallel_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataPreShapeSpace()

    def test_belongs(self, k_landmarks, m_ambient, mat, expected):
        space = self.space(k_landmarks, m_ambient)
        result = space.belongs(mat)
        self.assertAllClose(result, expected)

    def test_is_centered(self, k_landmarks, m_ambient, point, expected):
        space = self.space(k_landmarks, m_ambient)
        result = space.is_centered(point)
        self.assertAllClose(result, expected)

    def test_to_center_is_center(self, k_landmarks, m_ambient, point):
        space = self.space(k_landmarks, m_ambient)
        centered_point = space.center(point)
        result = gs.all(space.is_centered(centered_point))
        self.assertAllClose(result, gs.array(True))

    def test_vertical_projection(self, k_landmarks, m_ambient, tangent_vec, point):
        space = self.space(k_landmarks, m_ambient)
        vertical = space.vertical_projection(tangent_vec, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tangent_vec)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self, k_landmarks, m_ambient, tangent_vec, point):
        space = self.space(k_landmarks, m_ambient)
        horizontal = space.horizontal_projection(tangent_vec, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)
        self.assertAllClose(result, expected)

    def test_horizontal_and_is_tangent(
        self, k_landmarks, m_ambient, tangent_vec, point
    ):
        space = self.space(k_landmarks, m_ambient)
        horizontal = space.horizontal_projection(tangent_vec, point)
        result = gs.all(space.is_tangent(horizontal, point))
        self.assertAllClose(result, gs.array(True))

    def test_alignment_is_symmetric(self, k_landmarks, m_ambient, point, base_point):
        space = self.space(k_landmarks, m_ambient)
        aligned = space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = gs.all(Matrices.is_symmetric(alignment))
        self.assertAllClose(result, gs.array(True))

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
        space = self.space(k_landmarks, m_ambient)
        result_ab = space.integrability_tensor(tangent_vec_a, tangent_vec_b, base_point)

        result = space.ambient_metric.inner_product(
            tangent_vec_b, result_ab, base_point
        )
        expected = 0.0
        self.assertAllClose(result, expected)

        horizontal_b = space.horizontal_projection(tangent_vec_b, base_point)
        horizontal_a = space.horizontal_projection(tangent_vec_a, base_point)
        result = space.integrability_tensor(horizontal_a, horizontal_b, base_point)
        expected = -space.integrability_tensor(horizontal_b, horizontal_a, base_point)
        self.assertAllClose(result, expected)

        is_vertical = space.is_vertical(result, base_point)
        self.assertTrue(is_vertical)

        vertical_b = tangent_vec_b - horizontal_b
        result = space.integrability_tensor(horizontal_a, vertical_b, base_point)
        is_horizontal = space.is_horizontal(result, base_point)
        self.assertTrue(is_horizontal)

    def test_integrability_tensor_old(
        self, k_landmarks, m_ambient, tangent_vec_x, tangent_vec_e, base_point
    ):
        """Test if old and new implementation give the same result."""

        space = self.space(k_landmarks, m_ambient)
        result = space.integrability_tensor_old(
            tangent_vec_x, tangent_vec_e, base_point
        )
        expected = space.integrability_tensor(tangent_vec_x, tangent_vec_e, base_point)
        self.assertAllClose(result, expected)

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
        space = self.space(k_landmarks, m_ambient)
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
        self.assertAllClose(a_y_z + a_z_y, gs.zeros_like(result))
        self.assertAllClose(result, gs.zeros_like(result))

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

        space = self.space(k_landmarks, m_ambient)

        scal = space.ambient_metric.inner_product

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
        self.assertAllClose(result, gs.zeros_like(result))

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
        space = self.space(k_landmarks, m_ambient)

        scal = space.ambient_metric.inner_product

        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            hor_z,
            nabla_x_z,
            base_point,
        )
        result = scal(nabla_x_a_y_z, hor_h) + scal(a_y_z, nabla_x_h)
        self.assertAllClose(result, gs.zeros_like(result))

        nabla_x_a_y_v, a_y_v = space.integrability_tensor_derivative(
            hor_x,
            hor_y,
            nabla_x_y,
            ver_v,
            nabla_x_v,
            base_point,
        )
        result = scal(nabla_x_a_y_v, ver_w) + scal(a_y_v, nabla_x_w)
        self.assertAllClose(result, gs.zeros_like(result))

    def test_integrability_tensor_derivative_parallel(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, base_point
    ):
        """Test optimized integrability tensor derivatives in pre-shape space.

        Optimized version for quotient-parallel vector fields should equal
        the general implementation.
        """
        space = self.space(k_landmarks, m_ambient)
        (nabla_x_a_y_z_qp, a_y_z_qp,) = space.integrability_tensor_derivative_parallel(
            hor_x, hor_y, hor_z, base_point
        )

        a_x_y = space.integrability_tensor(hor_x, hor_y, base_point)
        a_x_z = space.integrability_tensor(hor_x, hor_z, base_point)

        nabla_x_a_y_z, a_y_z = space.integrability_tensor_derivative(
            hor_x, hor_y, a_x_y, hor_z, a_x_z, base_point
        )

        self.assertAllClose(a_y_z, a_y_z_qp)
        self.assertAllClose(nabla_x_a_y_z, nabla_x_a_y_z_qp)

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
        space = self.space(k_landmarks, m_ambient)
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
        self.assertAllClose(a_x_y, ver_v_qp)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp)


class TestKendasllShapeMetric(TestCase, metaclass=Parametrizer):
    metric = connection = KendallShapeMetric
    space = PreShapeSpace

    class TestDataKendallShapeMetric(TestData):
        def curvature_is_skew_operator_data(self):
            base_point = smoke_space.random_point(2)
            vec = gs.random.rand(4, 4, 3)
            smoke_data = [
                dict(k_landmarks=4, m_ambient=3, vec=vec, base_point=base_point)
            ]
            return self.generate_tests(smoke_data)

        def curvature_bianchi_identity_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec_a=tg_vec_0,
                    tangent_vec_b=tg_vec_1,
                    tangent_vec_cs=tg_vec_2,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def kendall_sectional_curvature_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    tangent_vec_a=tg_vec_0,
                    tangent_vec_b=tg_vec_1,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def kendall_curvature_derivative_bianchi_identity_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    hor_h=hor_h,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def curvature_derivative_is_skew_operator_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    hor_z=hor_z,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def directional_curvature_derivative_data(self):
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def directional_curvature_derivative_is_quadratic_data(self):
            coef_x = -2.5
            coef_y = 1.5
            smoke_data = [
                dict(
                    k_landmarks=4,
                    m_ambient=3,
                    coef_x=coef_x,
                    coef_y=coef_y,
                    hor_x=hor_x,
                    hor_y=hor_y,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

        def parallel_transport_data(self):
            k_landmarks = 4
            m_ambient = 3
            n_samples = 10
            space = PreShapeSpace(4, 3)
            base_point = space.projection(gs.eye(4)[:, :3])
            vec_a = gs.random.rand(n_samples, k_landmarks, m_ambient)
            tangent_vec_a = space.to_tangent(space.center(vec_a), base_point)

            vec_b = gs.random.rand(n_samples, k_landmarks, m_ambient)
            tangent_vec_b = space.to_tangent(space.center(vec_b), base_point)
            smoke_data = [
                dict(
                    k_landmarks=k_landmarks,
                    m_ambient=m_ambient,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataKendallShapeMetric()

    def test_curvature_is_skew_operator(self, k_landmarks, m_ambient, vec, base_point):
        metric = self.metric(k_landmarks, m_ambient)
        space = self.space(k_landmarks, m_ambient)
        tangent_vec_a = space.to_tangent(vec[:2], base_point)
        tangent_vec_b = space.to_tangent(vec[2:], base_point)

        result = metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point
        )
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

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
        metric = self.metric(k_landmarks, m_ambient)
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

        space = self.space(k_landmarks, m_ambient)
        metric = self.metric(k_landmarks, m_ambient)
        hor_a = space.horizontal_projection(tangent_vec_a, base_point)
        hor_b = space.horizontal_projection(tangent_vec_b, base_point)

        tidal_force = metric.directional_curvature(hor_a, hor_b, base_point)

        numerator = metric.inner_product(tidal_force, hor_a, base_point)
        denominator = (
            metric.inner_product(hor_a, hor_a, base_point)
            * metric.inner_product(hor_b, hor_b, base_point)
            - metric.inner_product(hor_a, hor_b, base_point) ** 2
        )
        condition = ~gs.isclose(denominator, 0.0)
        kappa = numerator[condition] / denominator[condition]
        kappa_direct = metric.sectional_curvature(hor_a, hor_b, base_point)[condition]
        self.assertAllClose(kappa, kappa_direct)
        result = kappa > 1.0 - 1e-12
        self.assertTrue(gs.all(result))

    def test_kendall_curvature_derivative_bianchi_identity(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, hor_h, base_point
    ):
        r"""2nd Bianchi identity on curvature derivative in kendall space.

        For any 3 tangent vectors horizontally lifted from kendall shape
        space to Kendall pre-shape space, :math:`(\nabla_X R)(Y, Z)
        + (\nabla_Y R)(Z,X) + (\nabla_Z R)(X, Y) = 0`.
        """
        metric = self.metric(k_landmarks, m_ambient)
        term_x = metric.curvature_derivative(hor_x, hor_y, hor_z, hor_h, base_point)
        term_y = metric.curvature_derivative(hor_y, hor_z, hor_x, hor_h, base_point)
        term_z = metric.curvature_derivative(hor_z, hor_x, hor_y, hor_h, base_point)

        result = term_x + term_y + term_z
        self.assertAllClose(result, gs.zeros_like(result))

    def test_curvature_derivative_is_skew_operator(
        self, k_landmarks, m_ambient, hor_x, hor_y, hor_z, base_point
    ):
        r"""Derivative of a skew operator is skew.

        For any 3 tangent vectors horizontally lifted from kendall shape space
        to Kendall pre-shape space, :math:`(\nabla_X R)(Y,Y)Z = 0`.
        """
        metric = self.metric(k_landmarks, m_ambient)
        result = metric.curvature_derivative(hor_x, hor_y, hor_y, hor_z, base_point)
        self.assertAllClose(result, gs.zeros_like(result))

    def test_directional_curvature_derivative(
        self, k_landmarks, m_ambient, hor_x, hor_y, base_point
    ):
        """Test equality of directional curvature derivative implementations.

        General formula based on curvature derivative, optimized method of
        KendallShapeMetric class, method from the QuotientMetric class and
        method from the Connection class have to give identical results.
        """
        metric = self.metric(k_landmarks, m_ambient)

        # General formula based on curvature derivative
        expected = metric.curvature_derivative(hor_x, hor_y, hor_x, hor_y, base_point)

        # Optimized method of KendallShapeMetric class
        result_kendall_shape_metric = metric.directional_curvature_derivative(
            hor_x, hor_y, base_point
        )
        self.assertAllClose(result_kendall_shape_metric, expected)

        # Method from the QuotientMetric class
        result_quotient_metric = super(
            KendallShapeMetric, metric
        ).directional_curvature_derivative(hor_x, hor_y, base_point)
        self.assertAllClose(result_quotient_metric, expected)

        # Method from the Connection class

        result_connection = super(
            QuotientMetric, metric
        ).directional_curvature_derivative(hor_x, hor_y, base_point)
        self.assertAllClose(result_connection, expected)

    def test_directional_curvature_derivative_is_quadratic(
        self, k_landmarks, m_ambient, coef_x, coef_y, hor_x, hor_y, base_point
    ):
        """Directional curvature derivative is quadratic in both variables."""
        metric = self.metric(k_landmarks, m_ambient)
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
        self.assertAllClose(result, expected)

    def test_parallel_transport(
        self, k_landmarks, m_ambient, tangent_vec_a, tangent_vec_b, base_point
    ):
        space = self.space(k_landmarks, m_ambient)
        metric = self.metric(k_landmarks, m_ambient)
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
        self.assertAllClose(result, expected)

        is_tangent = space.is_tangent(transported, end_point)
        is_horizontal = space.is_horizontal(transported, end_point)
        self.assertTrue(gs.all(is_tangent))
        self.assertTrue(gs.all(is_horizontal))

        transported = metric.parallel_transport(
            tan_a[0], base_point, end_point=end_point[0]
        )
        result = metric.norm(transported, end_point[0])
        self.assertAllClose(result, expected[0])

