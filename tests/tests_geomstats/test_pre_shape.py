"""Unit tests for the preshape space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace


class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.k_landmarks = 4
        self.m_ambient = 3
        self.space = PreShapeSpace(self.k_landmarks, self.m_ambient)
        self.matrices = self.space.embedding_space
        self.n_samples = 10
        self.shape_metric = KendallShapeMetric(self.k_landmarks, self.m_ambient)

        self.base_point = self.space.random_point()
        vector = gs.random.rand(11, self.k_landmarks, self.m_ambient)
        tg_vec_0 = self.space.to_tangent(vector[0], self.base_point)
        self.hor_x = self.space.horizontal_projection(tg_vec_0, self.base_point)
        tg_vec_1 = self.space.to_tangent(vector[1], self.base_point)
        self.hor_y = self.space.horizontal_projection(tg_vec_1, self.base_point)
        tg_vec_2 = self.space.to_tangent(vector[2], self.base_point)
        self.hor_z = self.space.horizontal_projection(tg_vec_2, self.base_point)
        tg_vec_3 = self.space.to_tangent(vector[3], self.base_point)
        self.hor_h = self.space.horizontal_projection(tg_vec_3, self.base_point)
        tg_vec_4 = self.space.to_tangent(vector[4], self.base_point)
        self.ver_v = self.space.vertical_projection(tg_vec_4, self.base_point)
        tg_vec_5 = self.space.to_tangent(vector[5], self.base_point)
        self.ver_w = self.space.vertical_projection(tg_vec_5, self.base_point)
        tg_vec_6 = self.space.to_tangent(vector[6], self.base_point)
        hor_dy = self.space.horizontal_projection(tg_vec_6, self.base_point)
        tg_vec_7 = self.space.to_tangent(vector[7], self.base_point)
        hor_dz = self.space.horizontal_projection(tg_vec_7, self.base_point)
        tg_vec_8 = self.space.to_tangent(vector[8], self.base_point)
        ver_dv = self.space.vertical_projection(tg_vec_8, self.base_point)
        tg_vec_9 = self.space.to_tangent(vector[9], self.base_point)
        ver_dw = self.space.vertical_projection(tg_vec_9, self.base_point)
        tg_vec_10 = self.space.to_tangent(vector[10], self.base_point)
        hor_dh = self.space.horizontal_projection(tg_vec_10, self.base_point)

        # generate valid derivatives of horizontal / vertical vector fields.
        a_x_y = self.space.integrability_tensor(self.hor_x, self.hor_y, self.base_point)
        self.nabla_x_y = hor_dy + a_x_y
        a_x_z = self.space.integrability_tensor(self.hor_x, self.hor_z, self.base_point)
        self.nabla_x_z = hor_dz + a_x_z
        a_x_v = self.space.integrability_tensor(self.hor_x, self.ver_v, self.base_point)
        self.nabla_x_v = ver_dv + a_x_v
        a_x_w = self.space.integrability_tensor(self.hor_x, self.ver_w, self.base_point)
        self.nabla_x_w = ver_dw + a_x_w
        a_x_h = self.space.integrability_tensor(self.hor_x, self.hor_h, self.base_point)
        self.nabla_x_h = hor_dh + a_x_h

    def test_belongs(self):
        point = gs.random.rand(self.m_ambient - 1, self.k_landmarks)
        result = self.space.belongs(point)
        self.assertFalse(result)

        point = gs.random.rand(self.n_samples, self.m_ambient - 1, self.k_landmarks)
        result = self.space.belongs(point)
        self.assertFalse(gs.all(result))

    def test_random_point_and_belongs(self):
        """Test random uniform and belongs.

        Test that the random uniform method samples
        on the pre-shape space.
        """
        n_samples = self.n_samples
        point = self.space.random_point(n_samples)
        result = self.space.belongs(point)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    def test_random_point_shape(self):
        point = self.space.random_point()
        result = gs.shape(point)
        expected = (
            self.k_landmarks,
            self.m_ambient,
        )

        self.assertAllClose(result, expected)

        point = self.space.random_point(self.n_samples)
        result = gs.shape(point)
        expected = (
            self.n_samples,
            self.k_landmarks,
            self.m_ambient,
        )
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        point = Matrices.transpose(
            gs.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            )
        )
        proj = self.space.projection(point)
        result = self.space.belongs(proj)
        expected = True

        self.assertAllClose(expected, result)

    def test_is_centered(self):
        point = gs.ones((self.k_landmarks, self.m_ambient))
        result = self.space.is_centered(point)
        self.assertFalse(result)

        point = gs.zeros((self.k_landmarks, self.m_ambient))
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_center(self):
        point = gs.ones((self.k_landmarks, self.m_ambient))
        point = self.space.center(point)
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_centered_vectorization(self):
        point = gs.ones((self.n_samples, self.k_landmarks, self.m_ambient))
        point = self.space.center(point)
        result = gs.all(self.space.is_centered(point))
        self.assertTrue(result)

    def test_is_tangent_to_tangent(self):
        point, vector = self.matrices.random_point(2)
        point = self.space.projection(point)

        result = self.space.is_tangent(vector, point)
        self.assertFalse(result)

        tangent_vec = self.space.to_tangent(vector, point)
        result = self.space.is_tangent(tangent_vec, point)
        self.assertTrue(result)

        vec = gs.array([tangent_vec, vector])
        result = self.space.is_tangent(vec, point)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_vertical_projection(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tan)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_vertical_projection_vectorization(self):
        vector = gs.random.rand(self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tan)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_projection_vectorized(self):
        vector = gs.random.rand(self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_and_is_tangent(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)

        horizontal = gs.stack([horizontal, vector])
        result = self.space.is_tangent(horizontal, point)
        expected = gs.array([True, False])

        self.assertAllClose(result, expected)

    def test_align(self):
        point, base_point = self.space.random_point(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(result)

    def test_align_vectorization(self):
        base_point = self.space.random_point()
        point = self.space.random_point(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

        base_point = self.space.random_point(2)
        point = self.space.random_point()
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

    def test_inner_product_shape(self):
        vector = gs.random.rand(self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        inner = self.space.ambient_metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_exp_and_belongs(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(result)

        exp = self.space.ambient_metric.exp(gs.zeros_like(point), point)
        result = gs.isclose(point, exp)
        self.assertTrue(gs.all(result))

    def test_exp_and_belongs_vectorization(self):
        vector = gs.random.rand(self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

        point = point[0]
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

    def test_log_and_exp(self):
        point, base_point = self.space.random_point(2)
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        exp = self.space.ambient_metric.exp(log, base_point)
        self.assertAllClose(exp, point)

    def test_exp_and_log(self):
        base_point = self.space.random_point()
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        tangent_vec = self.space.to_tangent(vector, base_point)
        point = self.space.ambient_metric.exp(tangent_vec, base_point)
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        self.assertAllClose(tangent_vec, log)

    def test_log_vectorization(self):
        point = self.space.random_point(self.n_samples)
        base_point = self.space.random_point()
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(gs.all(result))

        exp = self.space.ambient_metric.exp(log, base_point)
        self.assertAllClose(exp, point)

        log = self.space.ambient_metric.log(base_point, point)
        result = self.space.is_tangent(log, point)
        self.assertTrue(gs.all(result))

        exp = self.space.ambient_metric.exp(log, point)
        expected = gs.stack([base_point] * self.n_samples)
        self.assertAllClose(exp, expected)

    def test_kendall_inner_product_shape(self):
        vector = gs.random.rand(self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        inner = self.shape_metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_kendall_log_and_exp(self):
        point, base_point = self.space.random_point(2)
        expected = self.space.align(point, base_point)
        log = self.shape_metric.log(expected, base_point)
        result = self.space.is_horizontal(log, base_point)
        self.assertTrue(result)

        exp = self.shape_metric.exp(log, base_point)
        self.assertAllClose(exp, expected)

    def test_kendall_exp_and_log(self):
        base_point = self.space.random_point()
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        tangent_vec = self.space.to_tangent(vector, base_point)
        point = self.shape_metric.exp(tangent_vec, base_point)
        log = self.shape_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        expected = self.space.horizontal_projection(tangent_vec, base_point)
        self.assertAllClose(expected, log, rtol=1e-3)

    def test_dist_extreme_case(self):
        point = self.space.projection(gs.eye(self.k_landmarks, self.m_ambient))
        result = self.shape_metric.dist(point, point)
        expected = 0.0
        self.assertAllClose(result, expected)

    def test_dist(self):
        point, base_point = self.space.random_point(2)
        result = self.shape_metric.dist(point, base_point)
        log = self.shape_metric.log(point, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)

    def test_dist_vectorization(self):
        point = self.space.random_point(self.n_samples)
        base_point = self.space.random_point(self.n_samples)
        aligned = self.space.align(point, base_point)
        result = self.shape_metric.dist(aligned, base_point)
        log = self.shape_metric.log(aligned, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)

    def test_curvature_is_skew_operator(self):
        """Pre-shape space curvature tensor is skew in the first two arguments.

        :math:`R(X,X)Y = 0`.
        """
        space = self.space
        base_point = space.random_point(2)
        vector = gs.random.rand(4, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[:2], base_point)
        tangent_vec_b = space.to_tangent(vector[2:], base_point)

        result = self.shape_metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point
        )
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_curvature_bianchi_identity(self):
        """First Bianchi identity on curvature in pre-shape space.

        :math:`R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0`.
        """
        space = self.space
        base_point = space.random_point()
        vector = gs.random.rand(3, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[0], base_point)
        tangent_vec_b = space.to_tangent(vector[1], base_point)
        tangent_vec_c = space.to_tangent(vector[2], base_point)

        curvature_1 = self.shape_metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        curvature_2 = self.shape_metric.curvature(
            tangent_vec_b, tangent_vec_c, tangent_vec_a, base_point
        )
        curvature_3 = self.shape_metric.curvature(
            tangent_vec_c, tangent_vec_a, tangent_vec_b, base_point
        )

        result = curvature_1 + curvature_2 + curvature_3
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_integrability_tensor(self):
        """Identities of integrability tensor in kendall pre-shape space.

        The integrability tensor A_X E is skew-symmetric with respect to the
        pre-shape metric, :math:`< A_X E, F> + <E, A_X F> = 0`. By
        polarization, this is equivalent to :math:`< A_X E, E> = 0`.

        The integrability tensor is also alternating (:math:`A_X Y =
        - A_Y X`)  for horizontal vector fields :math:'X,Y',  and it is
        exchanging horizontal and vertical vector spaces.
        """
        space = self.space
        base_point = space.random_point()
        vector = gs.random.rand(2, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[0], base_point)
        tangent_vec_b = space.to_tangent(vector[1], base_point)
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

    def test_integrability_tensor_old(self):
        """Test if old and new implementation give the same result."""
        space = self.space
        base_point = space.random_point()
        vector = gs.random.rand(2, self.k_landmarks, self.m_ambient)
        tangent_vec_x = space.to_tangent(vector[0], base_point)
        tangent_vec_e = space.to_tangent(vector[1], base_point)

        result = space.integrability_tensor_old(
            tangent_vec_x, tangent_vec_e, base_point
        )
        expected = space.integrability_tensor(tangent_vec_x, tangent_vec_e, base_point)
        self.assertAllClose(result, expected)

    def test_kendall_sectional_curvature(self):
        """Sectional curvature of Kendall shape space is larger than 1.

        The sectional curvature always increase by taking the quotient in a
        Riemannian submersion. Thus, it should larger in kendall shape space
        thane the sectional curvature of the pre-shape space which is 1 as it
        a hypersphere.
        The sectional curvature is computed here with the generic
        directional_curvature and sectional curvature methods.
        """
        space = self.space
        metric = self.shape_metric
        n_samples = 4 * self.k_landmarks * self.m_ambient
        base_point = self.space.random_point(1)

        vec_a = gs.random.rand(n_samples, self.k_landmarks, self.m_ambient)
        tg_vec_a = space.to_tangent(space.center(vec_a), base_point)
        hor_a = space.horizontal_projection(tg_vec_a, base_point)

        vec_b = gs.random.rand(n_samples, self.k_landmarks, self.m_ambient)
        tg_vec_b = space.to_tangent(space.center(vec_b), base_point)
        hor_b = space.horizontal_projection(tg_vec_b, base_point)

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

    def test_integrability_tensor_derivative_is_alternate(self):
        r"""Integrability tensor derivatives is alternate in pre-shape.

        For two horizontal vector fields :math:`X,Y` the integrability
        tensor (hence its derivatives) is alternate:
        :math:`\nabla_X ( A_Y Z + A_Z Y ) = 0`.
        """
        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_y,
            self.nabla_x_y,
            self.hor_z,
            self.nabla_x_z,
            self.base_point,
        )
        nabla_x_a_z_y, a_z_y = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_z,
            self.nabla_x_z,
            self.hor_y,
            self.nabla_x_y,
            self.base_point,
        )
        result = nabla_x_a_y_z + nabla_x_a_z_y
        self.assertAllClose(a_y_z + a_z_y, gs.zeros_like(result))
        self.assertAllClose(result, gs.zeros_like(result))

    def test_integrability_tensor_derivative_is_skew_symmetric(self):
        r"""Integrability tensor derivatives is skew-symmetric in pre-shape.

        For :math:`X,Y` horizontal and :math:`V,W` vertical:
        :math:`\nabla_X (< A_Y Z , V > + < A_Y V , Z >) = 0`.
        """
        scal = self.space.ambient_metric.inner_product

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_y,
            self.nabla_x_y,
            self.hor_z,
            self.nabla_x_z,
            self.base_point,
        )

        nabla_x_a_y_v, a_y_v = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_y,
            self.nabla_x_y,
            self.ver_v,
            self.nabla_x_v,
            self.base_point,
        )

        result = (
            scal(nabla_x_a_y_z, self.ver_v)
            + scal(a_y_z, self.nabla_x_v)
            + scal(nabla_x_a_y_v, self.hor_z)
            + scal(a_y_v, self.nabla_x_z)
        )
        self.assertAllClose(result, gs.zeros_like(result))

    def test_integrability_tensor_derivative_reverses_hor_ver(self):
        r"""Integrability tensor derivatives exchanges hor & ver in pre-shape.

        For :math:`X,Y,Z` horizontal and :math:`V,W` vertical, the
        integrability tensor (and thus its derivative) reverses horizontal
        and vertical subspaces: :math:`\nabla_X < A_Y Z, H > = 0`  and
        :math:`nabla_X < A_Y V, W > = 0`.
        """
        scal = self.space.ambient_metric.inner_product

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_y,
            self.nabla_x_y,
            self.hor_z,
            self.nabla_x_z,
            self.base_point,
        )
        result = scal(nabla_x_a_y_z, self.hor_h) + scal(a_y_z, self.nabla_x_h)
        self.assertAllClose(result, gs.zeros_like(result))

        nabla_x_a_y_v, a_y_v = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_y,
            self.nabla_x_y,
            self.ver_v,
            self.nabla_x_v,
            self.base_point,
        )
        result = scal(nabla_x_a_y_v, self.ver_w) + scal(a_y_v, self.nabla_x_w)
        self.assertAllClose(result, gs.zeros_like(result))

    def test_integrability_tensor_derivative_parallel(self):
        """Test optimized integrability tensor derivatives in pre-shape space.

        Optimized version for quotient-parallel vector fields should equal
        the general implementation.
        """
        (
            nabla_x_a_y_z_qp,
            a_y_z_qp,
        ) = self.space.integrability_tensor_derivative_parallel(
            self.hor_x, self.hor_y, self.hor_z, self.base_point
        )

        a_x_y = self.space.integrability_tensor(self.hor_x, self.hor_y, self.base_point)
        a_x_z = self.space.integrability_tensor(self.hor_x, self.hor_z, self.base_point)

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            self.hor_x, self.hor_y, a_x_y, self.hor_z, a_x_z, self.base_point
        )

        self.assertAllClose(a_y_z, a_y_z_qp)
        self.assertAllClose(nabla_x_a_y_z, nabla_x_a_y_z_qp)

    def test_iterated_integrability_tensor_derivative_parallel(self):
        """Test optimized iterated integrability tensor derivatives.

        The optimized version of the iterated integrability tensor
        :math:`A_X A_Y A_X Y`, computed with the horizontal lift of
        quotient-parallel vector fields extending the tangent vectors
        :math:`X,Y` of Kendall shape spaces (identified to horizontal vectors
        of the pre-shape space), is the recursive application of two general
        integrability tensor derivatives with proper derivatives.
        Intermediate computations returned are also verified.
        """
        a_x_y = self.space.integrability_tensor(self.hor_x, self.hor_y, self.base_point)
        nabla_x_v, a_x_y = self.space.integrability_tensor_derivative(
            self.hor_x,
            self.hor_x,
            gs.zeros_like(self.hor_x),
            self.hor_y,
            a_x_y,
            self.base_point,
        )

        (nabla_x_a_y_a_x_y, a_y_a_x_y,) = self.space.integrability_tensor_derivative(
            self.hor_x, self.hor_y, a_x_y, a_x_y, nabla_x_v, self.base_point
        )

        a_x_a_y_a_x_y = self.space.integrability_tensor(
            self.hor_x, a_y_a_x_y, self.base_point
        )

        (
            nabla_x_a_y_a_x_y_qp,
            a_x_a_y_a_x_y_qp,
            nabla_x_v_qp,
            a_y_a_x_y_qp,
            ver_v_qp,
        ) = self.space.iterated_integrability_tensor_derivative_parallel(
            self.hor_x, self.hor_y, self.base_point
        )
        self.assertAllClose(a_x_y, ver_v_qp)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp)

    def test_kendall_curvature_derivative_bianchi_identity(self):
        r"""2nd Bianchi identity on curvature derivative in kendall space.

        For any 3 tangent vectors horizontally lifted from kendall shape
        space to Kendall pre-shape space, :math:`(\nabla_X R)(Y, Z)
        + (\nabla_Y R)(Z,X) + (\nabla_Z R)(X, Y) = 0`.
        """
        term_x = self.shape_metric.curvature_derivative(
            self.hor_x, self.hor_y, self.hor_z, self.hor_h, self.base_point
        )
        term_y = self.shape_metric.curvature_derivative(
            self.hor_y, self.hor_z, self.hor_x, self.hor_h, self.base_point
        )
        term_z = self.shape_metric.curvature_derivative(
            self.hor_z, self.hor_x, self.hor_y, self.hor_h, self.base_point
        )

        result = term_x + term_y + term_z
        self.assertAllClose(result, gs.zeros_like(result))

    def test_curvature_derivative_is_skew_operator(self):
        r"""Derivative of a skew operator is skew.

        For any 3 tangent vectors horizontally lifted from kendall shape space
        to Kendall pre-shape space, :math:`(\nabla_X R)(Y,Y)Z = 0`.
        """
        result = self.shape_metric.curvature_derivative(
            self.hor_x, self.hor_y, self.hor_y, self.hor_z, self.base_point
        )
        self.assertAllClose(result, gs.zeros_like(result))

    def test_directional_curvature_derivative(self):
        """Test equality of directional curvature derivative implementations.

        General formula based on curvature derivative, optimized method of
        KendallShapeMetric class, method from the QuotientMetric class and
        method from the Connection class have to give identical results.
        """
        metric = self.shape_metric

        # General formula based on curvature derivative
        expected = metric.curvature_derivative(
            self.hor_x, self.hor_y, self.hor_x, self.hor_y, self.base_point
        )

        # Optimized method of KendallShapeMetric class
        result_kendall_shape_metric = metric.directional_curvature_derivative(
            self.hor_x, self.hor_y, self.base_point
        )
        self.assertAllClose(result_kendall_shape_metric, expected)

        # Method from the QuotientMetric class
        result_quotient_metric = super(
            KendallShapeMetric, metric
        ).directional_curvature_derivative(self.hor_x, self.hor_y, self.base_point)
        self.assertAllClose(result_quotient_metric, expected)

        # Method from the Connection class
        from geomstats.geometry.quotient_metric import QuotientMetric

        result_connection = super(
            QuotientMetric, metric
        ).directional_curvature_derivative(self.hor_x, self.hor_y, self.base_point)
        self.assertAllClose(result_connection, expected)

    def test_directional_curvature_derivative_is_quadratic(self):
        """Directional curvature derivative is quadratic in both variables."""
        coef_x = -2.5
        coef_y = 1.5
        result = self.shape_metric.directional_curvature_derivative(
            coef_x * self.hor_x, coef_y * self.hor_y, self.base_point
        )
        expected = (
            coef_x ** 2
            * coef_y ** 2
            * self.shape_metric.directional_curvature_derivative(
                self.hor_x, self.hor_y, self.base_point
            )
        )
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_parallel_transport(self):
        space = self.space
        metric = self.shape_metric
        shape = (self.n_samples, self.k_landmarks, self.m_ambient)

        point = space.projection(gs.eye(4)[:, :3])
        tan_b = gs.random.rand(*shape)
        tan_b = space.to_tangent(tan_b, point)
        tan_b = space.horizontal_projection(tan_b, point)

        # use a vector orthonormal to tan_b
        tan_a = gs.random.rand(*shape)
        tan_a = space.to_tangent(tan_a, point)
        tan_a = space.horizontal_projection(tan_a, point)

        # orthonormalize and move to base_point
        tan_a -= gs.einsum(
            "...,...ij->...ij",
            metric.inner_product(tan_a, tan_b, point)
            / metric.squared_norm(tan_b, point),
            tan_b,
        )
        tan_b = gs.einsum("...ij,...->...ij", tan_b, 1.0 / metric.norm(tan_b, point))
        tan_a = gs.einsum("...ij,...->...ij", tan_a, 1.0 / metric.norm(tan_a, point))

        transported = metric.parallel_transport(
            tan_a, tan_b, point, n_steps=150, step="rk4"
        )
        end_point = metric.exp(tan_b, point)
        result = metric.norm(transported, end_point)
        expected = metric.norm(tan_a, point)
        self.assertAllClose(result, expected)

        is_tangent = space.is_tangent(transported, end_point)
        is_horizontal = space.is_horizontal(transported, end_point)
        self.assertTrue(gs.all(is_tangent))
        self.assertTrue(gs.all(is_horizontal))
