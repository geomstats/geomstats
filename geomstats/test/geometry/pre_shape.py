import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.geometry.base import FiberBundleTestCase, LevelSetTestCase
from geomstats.test.vectorization import generate_vectorization_data


def integrability_tensor_alt(space, tangent_vec_a, tangent_vec_b, base_point):
    r"""Compute the fundamental tensor A of the submersion.

    The fundamental tensor A is defined for tangent vectors of the total
    space by [O'Neill]_ :math:`A_X Y = ver\nabla^M_{hor X} (hor Y)
    + hor \nabla^M_{hor X}( ver Y)` where :math:`hor,ver` are the
    horizontal and vertical projections.

    For the pre-shape space, we have closed-form expressions and the result
    does not depend on the vertical part of :math:`X`.

    (This is an alternative implementation.)

    Parameters
    ----------
    space : PreShapeSpace
    tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
        Tangent vector at `base_point`.
    tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
        Tangent vector at `base_point`.
    base_point : array-like, shape=[..., k_landmarks, m_ambient]
        Point of the total space.

    Returns
    -------
    vector : array-like, shape=[..., k_landmarks, m_ambient]
        Tangent vector at `base_point`, result of the A tensor applied to
        `tangent_vec_a` and `tangent_vec_b`.

    References
    ----------
    .. [O'Neill]  O’Neill, Barrett. The Fundamental Equations of a
        Submersion, Michigan Mathematical Journal 13, no. 4
        (December 1966): 459–69. https://doi.org/10.1307/mmj/1028999604.
    """
    # Only the horizontal part of a counts
    horizontal_a = space.horizontal_projection(tangent_vec_a, base_point)
    vertical_b, skew = space.vertical_projection(
        tangent_vec_b, base_point, return_skew=True
    )
    horizontal_b = tangent_vec_b - vertical_b

    # For the horizontal part of b
    transposed_point = Matrices.transpose(base_point)
    sigma = gs.matmul(transposed_point, base_point)
    alignment = gs.matmul(Matrices.transpose(horizontal_a), horizontal_b)
    right_term = alignment - Matrices.transpose(alignment)
    skew_hor = gs.linalg.solve_sylvester(sigma, sigma, right_term)
    vertical = -gs.matmul(base_point, skew_hor)

    # For the vertical part of b
    vert_part = -gs.matmul(horizontal_a, skew)
    tangent_vert = space.to_tangent(vert_part, base_point)
    horizontal_ = space.horizontal_projection(tangent_vert, base_point)

    return vertical + horizontal_


class PreShapeSpaceTestCase(LevelSetTestCase):
    def _get_random_matrix_point(self, n_points=1):
        return self.space.embedding_space.random_point(n_points)

    def test_is_centered(self, point, expected, atol):
        res = self.space.is_centered(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_centered_vec(self, n_reps, atol):
        point = self._get_random_matrix_point()
        expected = self.space.is_centered(point, atol=atol)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_center(self, point, expected, atol):
        res = self.space.center(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_center_vec(self, n_reps, atol):
        point = self._get_random_matrix_point()
        expected = self.space.center(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_center_is_centered(self, n_points, atol):
        point = self._get_random_matrix_point(n_points)
        res = self.space.center(point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_centered(res, expected, atol)


class PreShapeSpaceBundleTestCase(FiberBundleTestCase, PreShapeSpaceTestCase):
    def _get_horizontal_vec(self, base_point, return_tangent=False):
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        horizontal = self.space.horizontal_projection(
            tangent_vec,
            base_point,
        )
        if return_tangent:
            return tangent_vec, horizontal

        return horizontal

    def _get_vertical_vec(self, base_point, return_tangent=False):
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        vertical = self.space.vertical_projection(
            tangent_vec,
            base_point,
        )
        if return_tangent:
            return tangent_vec, vertical

        return vertical

    def _get_nabla_x_y(self, horizontal_vec_x, vec_y, base_point, horizontal=True):
        a_x_y = self.space.integrability_tensor(horizontal_vec_x, vec_y, base_point)

        dy = (
            self._get_horizontal_vec(base_point)
            if horizontal
            else self._get_vertical_vec(base_point)
        )

        return a_x_y + dy

    @pytest.mark.random
    def test_vertical_projection_correctness(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transposed_point = Matrices.transpose(base_point)
        tmp_expected = gs.matmul(transposed_point, tangent_vec)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        vertical = self.space.vertical_projection(tangent_vec, base_point)
        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result

        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_horizontal_projection_correctness(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.space.horizontal_projection(tangent_vec, base_point)

        result = gs.matmul(Matrices.transpose(base_point), horizontal)
        self.assertAllClose(result, Matrices.transpose(result), atol=atol)

    @pytest.mark.random
    def test_horizontal_projection_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.space.horizontal_projection(tangent_vec, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_tangent(horizontal, base_point, expected, atol)

    @pytest.mark.random
    def test_alignment_is_symmetric(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        aligned_point = self.space.align(point, base_point)
        alignment = Matrices.mul(Matrices.transpose(aligned_point), base_point)
        self.assertAllClose(alignment, Matrices.transpose(alignment), atol=atol)

    @pytest.mark.random
    def test_integrability_tensor_identity_1(self, n_points, atol):
        """Test integrability tensor identity.

        The integrability tensor A_X E is skew-symmetric with respect to the
        pre-shape metric, :math:`< A_X E, F> + <E, A_X F> = 0`. By
        polarization, this is equivalent to :math:`< A_X E, E> = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        result_ab = self.space.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )

        result = self.space.metric.inner_product(tangent_vec_b, result_ab, base_point)
        expected = gs.zeros(n_points)
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_integrability_tensor_identity_2(self, n_points, atol):
        """Test integrability tensor identity.

        The integrability tensor is also alternating (:math:`A_X Y =
        - A_Y X`)  for horizontal vector fields :math:'X,Y',  and it is
        exchanging horizontal and vertical vector spaces.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_a = self._get_horizontal_vec(base_point)
        tangent_vec_b, horizontal_b = self._get_horizontal_vec(
            base_point, return_tangent=True
        )

        result = self.space.integrability_tensor(horizontal_a, horizontal_b, base_point)
        expected = -self.space.integrability_tensor(
            horizontal_b, horizontal_a, base_point
        )
        self.assertAllClose(result, expected, atol=atol)

        is_vertical = self.space.is_vertical(result, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_vertical, expected)

        vertical_b = tangent_vec_b - horizontal_b
        result = self.space.integrability_tensor(horizontal_a, vertical_b, base_point)
        is_horizontal = self.space.is_horizontal(result, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_horizontal, expected)

    @pytest.mark.random
    def test_integrability_tensor_alt(self, n_points, atol):
        """Test if alternative and new implementation give the same result."""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )
        alt_expected = integrability_tensor_alt(
            self.space, tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(expected, alt_expected)

    @pytest.mark.vec
    def test_integrability_tensor_derivative_vec(self, n_reps, atol):
        # TODO: need to be reviewed
        # TODO: can this be moved up?
        base_point = self.data_generator.random_point()
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        nabla_x_y = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_y, base_point)

        tangent_vec_e = self.data_generator.random_tangent_vec(base_point)
        nabla_x_e = self.data_generator.random_tangent_vec(base_point)

        nabla_x_a_y_e, a_y_e = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            tangent_vec_e,
            nabla_x_e,
            base_point,
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    horizontal_vec_x=horizontal_vec_x,
                    horizontal_vec_y=horizontal_vec_y,
                    nabla_x_y=nabla_x_y,
                    tangent_vec_e=tangent_vec_e,
                    nabla_x_e=nabla_x_e,
                    base_point=base_point,
                    expected_nabla_x_a_y_e=nabla_x_a_y_e,
                    expected_a_y_e=a_y_e,
                    atol=atol,
                )
            ],
            arg_names=[
                "horizontal_vec_x",
                "horizontal_vec_y",
                "nabla_x_y",
                "tangent_vec_e",
                "nabla_x_e",
                "base_point",
            ],
            expected_name=["expected_nabla_x_a_y_e", "expected_a_y_e"],
            n_reps=n_reps,
            vectorization_type="basic",
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_integrability_tensor_derivative_is_alternate(self, n_points, atol):
        r"""Integrability tensor derivatives is alternate in pre-shape.

        For two horizontal vector fields :math:`X,Y` the integrability
        tensor (hence its derivatives) is alternate:
        :math:`\nabla_X ( A_Y Z + A_Z Y ) = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        horizontal_vec_z = self._get_horizontal_vec(base_point)
        nabla_x_y = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_y, base_point)
        nabla_x_z = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_z, base_point)

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )
        nabla_x_a_z_y, a_z_y = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_z,
            nabla_x_z,
            horizontal_vec_y,
            nabla_x_y,
            base_point,
        )

        shape = (n_points,) + self.space.shape if n_points > 1 else self.space.shape
        expected = gs.zeros(shape)

        result = nabla_x_a_y_z + nabla_x_a_z_y
        self.assertAllClose(result, expected, atol=atol)

        result = a_y_z + a_z_y
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_integrability_tensor_derivative_is_skew_symmetric(self, n_points, atol):
        r"""Integrability tensor derivatives is alternate in pre-shape.

        For two horizontal vector fields :math:`X,Y` the integrability
        tensor (hence its derivatives) is alternate:
        :math:`\nabla_X ( A_Y Z + A_Z Y ) = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        horizontal_vec_z = self._get_horizontal_vec(base_point)
        nabla_x_y = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_y, base_point)
        nabla_x_z = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_z, base_point)
        ver_v = self.data_generator.random_tangent_vec(base_point)
        nabla_x_v = self.data_generator.random_tangent_vec(base_point)

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )

        nabla_x_a_y_v, a_y_v = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            ver_v,
            nabla_x_v,
            base_point,
        )

        inner = self.space.metric.inner_product
        result = (
            inner(nabla_x_a_y_z, ver_v)
            + inner(a_y_z, nabla_x_v)
            + inner(nabla_x_a_y_v, horizontal_vec_z)
            + inner(a_y_v, nabla_x_z)
        )
        expected = gs.zeros(n_points)
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_integrability_tensor_derivative_reverses_hor(self, n_points, atol):
        r"""Integrability tensor derivatives exchanges hor & ver in pre-shape.

        For :math:`X,Y,Z` horizontal and :math:`V,W` vertical, the
        integrability tensor (and thus its derivative) reverses horizontal
        and vertical subspaces: :math:`\nabla_X < A_Y Z, H > = 0`  and
        :math:`nabla_X < A_Y V, W > = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        horizontal_vec_z = self._get_horizontal_vec(base_point)
        nabla_x_y = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_y, base_point)
        nabla_x_z = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_z, base_point)

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )

        horizontal_vec_h = self._get_horizontal_vec(base_point)
        nabla_x_h = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_h, base_point)

        inner = self.space.metric.inner_product
        result = inner(nabla_x_a_y_z, horizontal_vec_h) + inner(a_y_z, nabla_x_h)
        expected = gs.zeros(n_points)
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_integrability_tensor_derivative_reverses_ver(self, n_points, atol):
        r"""Integrability tensor derivatives exchanges hor & ver in pre-shape.

        For :math:`X,Y,Z` horizontal and :math:`V,W` vertical, the
        integrability tensor (and thus its derivative) reverses horizontal
        and vertical subspaces: :math:`\nabla_X < A_Y Z, H > = 0`  and
        :math:`nabla_X < A_Y V, W > = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        vertical_vec_z = self._get_vertical_vec(base_point)
        nabla_x_y = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_y, base_point)
        nabla_x_z = self._get_nabla_x_y(
            horizontal_vec_x, vertical_vec_z, base_point, horizontal=False
        )

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            vertical_vec_z,
            nabla_x_z,
            base_point,
        )

        vertical_vec_w = self._get_vertical_vec(base_point)
        nabla_x_w = self._get_nabla_x_y(
            horizontal_vec_x, vertical_vec_w, base_point, horizontal=False
        )

        inner = self.space.metric.inner_product
        result = inner(nabla_x_a_y_z, vertical_vec_w) + inner(a_y_z, nabla_x_w)
        expected = gs.zeros(n_points)
        self.assertAllClose(result, expected, atol=atol)

    def test_integrability_tensor_derivative_parallel(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        base_point,
        expected_nabla_x_a_y_z,
        expected_a_y_z,
        atol,
    ):
        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
        )
        self.assertAllClose(nabla_x_a_y_z, expected_nabla_x_a_y_z, atol=atol)
        self.assertAllClose(a_y_z, expected_a_y_z, atol=atol)

    @pytest.mark.vec
    def test_integrability_tensor_derivative_parallel_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        horizontal_vec_z = self._get_horizontal_vec(base_point)

        (
            expected_nabla_x_a_y_z,
            expected_a_y_z,
        ) = self.space.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    horizontal_vec_x=horizontal_vec_x,
                    horizontal_vec_y=horizontal_vec_y,
                    horizontal_vec_z=horizontal_vec_z,
                    base_point=base_point,
                    expected_nabla_x_a_y_z=expected_nabla_x_a_y_z,
                    expected_a_y_z=expected_a_y_z,
                    atol=atol,
                )
            ],
            arg_names=["base_point"],
            expected_name=[
                "expected_nabla_x_a_y_z",
                "expected_a_y_z",
            ],
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_integrability_tensor_derivative_parallel_optimized(self, n_points, atol):
        """Test optimized integrability tensor derivatives in pre-shape space.

        Optimized version for quotient-parallel vector fields should equal
        the general implementation.
        """
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        horizontal_vec_z = self._get_horizontal_vec(base_point)

        nabla_x_a_y_z, a_y_z = self.space.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
        )

        a_x_y = self.space.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        a_x_z = self.space.integrability_tensor(
            horizontal_vec_x, horizontal_vec_z, base_point
        )
        (
            expected_nabla_x_a_y_z,
            expected_a_y_z,
        ) = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            a_x_y,
            horizontal_vec_z,
            a_x_z,
            base_point,
        )

        self.assertAllClose(nabla_x_a_y_z, expected_nabla_x_a_y_z, atol=atol)
        self.assertAllClose(a_y_z, expected_a_y_z, atol=atol)

    def test_iterated_integrability_tensor_derivative_parallel(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        base_point,
        expected_nabla_x_a_y_v,
        expected_a_x_a_y_a_x_y,
        expected_nabla_x_v,
        expected_a_y_a_x_y,
        expected_vertical_vec_v,
        atol,
    ):
        (
            nabla_x_a_y_v,
            a_x_a_y_a_x_y,
            nabla_x_v,
            a_y_a_x_y,
            vertical_vec_v,
        ) = self.space.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        self.assertAllClose(nabla_x_a_y_v, expected_nabla_x_a_y_v, atol=atol)
        self.assertAllClose(a_x_a_y_a_x_y, expected_a_x_a_y_a_x_y, atol=atol)
        self.assertAllClose(nabla_x_v, expected_nabla_x_v, atol=atol)
        self.assertAllClose(a_y_a_x_y, expected_a_y_a_x_y, atol=atol)
        self.assertAllClose(vertical_vec_v, expected_vertical_vec_v, atol=atol)

    @pytest.mark.vec
    def test_iterated_integrability_tensor_derivative_parallel_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)
        (
            expected_nabla_x_a_y_v,
            expected_a_x_a_y_a_x_y,
            expected_nabla_x_v,
            expected_a_y_a_x_y,
            expected_vertical_vec_v,
        ) = self.space.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    horizontal_vec_x=horizontal_vec_x,
                    horizontal_vec_y=horizontal_vec_y,
                    base_point=base_point,
                    expected_nabla_x_a_y_v=expected_nabla_x_a_y_v,
                    expected_a_x_a_y_a_x_y=expected_a_x_a_y_a_x_y,
                    expected_nabla_x_v=expected_nabla_x_v,
                    expected_a_y_a_x_y=expected_a_y_a_x_y,
                    expected_vertical_vec_v=expected_vertical_vec_v,
                    atol=atol,
                )
            ],
            arg_names=["base_point"],
            expected_name=[
                "expected_nabla_x_a_y_v",
                "expected_a_x_a_y_a_x_y",
                "expected_nabla_x_v",
                "expected_a_y_a_x_y",
                "expected_vertical_vec_v",
            ],
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_iterated_integrability_tensor_derivative_parallel_optimized(
        self, n_points, atol
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
        base_point = self.data_generator.random_point(n_points)
        horizontal_vec_x = self._get_horizontal_vec(base_point)
        horizontal_vec_y = self._get_horizontal_vec(base_point)

        a_x_y = self.space.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        nabla_x_v, a_x_y = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_x,
            gs.zeros_like(horizontal_vec_x),
            horizontal_vec_y,
            a_x_y,
            base_point,
        )

        nabla_x_a_y_a_x_y, a_y_a_x_y = self.space.integrability_tensor_derivative(
            horizontal_vec_x, horizontal_vec_y, a_x_y, a_x_y, nabla_x_v, base_point
        )

        a_x_a_y_a_x_y = self.space.integrability_tensor(
            horizontal_vec_x, a_y_a_x_y, base_point
        )

        (
            nabla_x_a_y_a_x_y_qp,
            a_x_a_y_a_x_y_qp,
            nabla_x_v_qp,
            a_y_a_x_y_qp,
            ver_v_qp,
        ) = self.space.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        self.assertAllClose(a_x_y, ver_v_qp, atol=atol)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp, atol=atol)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp, atol=atol)
