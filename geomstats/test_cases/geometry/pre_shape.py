import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.random import KendalShapeRandomDataGenerator, get_random_times
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase
from geomstats.vectorization import get_batch_shape


def integrability_tensor_alt(bundle, tangent_vec_a, tangent_vec_b, base_point):
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
    bundle : PreShapeSpaceBundle
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
    horizontal_a = bundle.horizontal_projection(tangent_vec_a, base_point)
    vertical_b, skew = bundle.vertical_projection(
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
    tangent_vert = bundle.total_space.to_tangent(vert_part, base_point)
    horizontal_ = bundle.horizontal_projection(tangent_vert, base_point)

    return vertical + horizontal_


class PreShapeSpaceTestCase(LevelSetTestCase):
    def test_is_centered(self, point, expected, atol):
        res = self.space.is_centered(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_centered_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
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
        point = self.data_generator.random_point()
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
        point = self.data_generator.random_point(n_points)
        res = self.space.center(point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_centered(res, expected, atol)


class PreShapeSpaceBundleTestCase(FiberBundleTestCase):
    def _test_is_tangent(self, vector, base_point, expected, atol):
        res = self.total_space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    def _get_horizontal_vec(self, base_point, return_tangent=False):
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        horizontal = self.bundle.horizontal_projection(
            tangent_vec,
            base_point,
        )
        if return_tangent:
            return tangent_vec, horizontal

        return horizontal

    def _get_vertical_vec(self, base_point, return_tangent=False):
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        vertical = self.bundle.vertical_projection(
            tangent_vec,
            base_point,
        )
        if return_tangent:
            return tangent_vec, vertical

        return vertical

    def _get_nabla_x_y(self, horizontal_vec_x, vec_y, base_point, horizontal=True):
        a_x_y = self.bundle.integrability_tensor(horizontal_vec_x, vec_y, base_point)

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

        vertical = self.bundle.vertical_projection(tangent_vec, base_point)
        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result

        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_horizontal_projection_correctness(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.bundle.horizontal_projection(tangent_vec, base_point)

        result = gs.matmul(Matrices.transpose(base_point), horizontal)
        self.assertAllClose(result, Matrices.transpose(result), atol=atol)

    @pytest.mark.random
    def test_horizontal_projection_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.bundle.horizontal_projection(tangent_vec, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self._test_is_tangent(horizontal, base_point, expected, atol)

    @pytest.mark.random
    def test_alignment_is_symmetric(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        aligned_point = self.bundle.align(point, base_point)
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

        result_ab = self.bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )

        result = self.total_space.metric.inner_product(
            tangent_vec_b, result_ab, base_point
        )
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
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

        result = self.bundle.integrability_tensor(
            horizontal_a, horizontal_b, base_point
        )
        expected = -self.bundle.integrability_tensor(
            horizontal_b, horizontal_a, base_point
        )
        self.assertAllClose(result, expected, atol=atol)

        is_vertical = self.bundle.is_vertical(result, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_vertical, expected)

        vertical_b = tangent_vec_b - horizontal_b
        result = self.bundle.integrability_tensor(horizontal_a, vertical_b, base_point)
        is_horizontal = self.bundle.is_horizontal(result, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_horizontal, expected)

    @pytest.mark.random
    def test_integrability_tensor_alt(self, n_points, atol):
        """Test if alternative and new implementation give the same result."""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )
        alt_expected = integrability_tensor_alt(
            self.bundle, tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(expected, alt_expected, atol=atol)

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

        nabla_x_a_y_e, a_y_e = self.bundle.integrability_tensor_derivative(
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

        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )
        nabla_x_a_z_y, a_z_y = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_z,
            nabla_x_z,
            horizontal_vec_y,
            nabla_x_y,
            base_point,
        )

        shape = (
            (n_points,) + self.total_space.shape
            if n_points > 1
            else self.total_space.shape
        )
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

        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )

        nabla_x_a_y_v, a_y_v = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            ver_v,
            nabla_x_v,
            base_point,
        )

        inner = self.total_space.metric.inner_product
        result = (
            inner(nabla_x_a_y_z, ver_v)
            + inner(a_y_z, nabla_x_v)
            + inner(nabla_x_a_y_v, horizontal_vec_z)
            + inner(a_y_v, nabla_x_z)
        )
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
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

        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            horizontal_vec_z,
            nabla_x_z,
            base_point,
        )

        horizontal_vec_h = self._get_horizontal_vec(base_point)
        nabla_x_h = self._get_nabla_x_y(horizontal_vec_x, horizontal_vec_h, base_point)

        inner = self.total_space.metric.inner_product
        result = inner(nabla_x_a_y_z, horizontal_vec_h) + inner(a_y_z, nabla_x_h)
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
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

        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative(
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

        inner = self.total_space.metric.inner_product
        result = inner(nabla_x_a_y_z, vertical_vec_w) + inner(a_y_z, nabla_x_w)
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
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
        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative_parallel(
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
        ) = self.bundle.integrability_tensor_derivative_parallel(
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

        nabla_x_a_y_z, a_y_z = self.bundle.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
        )

        a_x_y = self.bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        a_x_z = self.bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_z, base_point
        )
        (
            expected_nabla_x_a_y_z,
            expected_a_y_z,
        ) = self.bundle.integrability_tensor_derivative(
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
        ) = self.bundle.iterated_integrability_tensor_derivative_parallel(
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
        ) = self.bundle.iterated_integrability_tensor_derivative_parallel(
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

        a_x_y = self.bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        nabla_x_v, a_x_y = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_x,
            gs.zeros_like(horizontal_vec_x),
            horizontal_vec_y,
            a_x_y,
            base_point,
        )

        nabla_x_a_y_a_x_y, a_y_a_x_y = self.bundle.integrability_tensor_derivative(
            horizontal_vec_x, horizontal_vec_y, a_x_y, a_x_y, nabla_x_v, base_point
        )

        a_x_a_y_a_x_y = self.bundle.integrability_tensor(
            horizontal_vec_x, a_y_a_x_y, base_point
        )

        (
            nabla_x_a_y_a_x_y_qp,
            a_x_a_y_a_x_y_qp,
            nabla_x_v_qp,
            a_y_a_x_y_qp,
            ver_v_qp,
        ) = self.bundle.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        self.assertAllClose(a_x_y, ver_v_qp, atol=atol)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp, atol=atol)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp, atol=atol)


class KendallShapeMetricTestCase(QuotientMetricTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = KendalShapeRandomDataGenerator(self.space)
        super().setup_method()

    def _cmp_points(self, point, point_, atol):
        dists = self.space.metric.dist(point, point_)
        batch_shape = get_batch_shape(self.space.point_ndim, point, point_)

        self.assertAllClose(dists, gs.zeros(batch_shape), atol=atol)

    @pytest.mark.random
    def test_exp_after_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(end_point, base_point)
        end_point_ = self.space.metric.exp(tangent_vec, base_point)

        self._cmp_points(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_log_after_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        end_point_ = self.space.metric.exp(tangent_vec_, base_point)

        self._cmp_points(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp_reverse(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        points = gs.reshape(geod_func(time), (-1, *self.space.shape))

        points_ = gs.reshape(geod_func_reverse(1.0 - time), (-1, *self.space.shape))

        self._cmp_points(points, points_, atol=atol)

    @pytest.mark.random
    def test_geodesic_boundary_points(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        points = gs.reshape(geod_func(time), (-1, *self.space.shape))
        expected_points = gs.reshape(
            gs.stack([initial_point, end_point], axis=-(self.space.point_ndim + 1)),
            (-1, *self.space.shape),
        )
        self._cmp_points(points, expected_points, atol=atol)

    @pytest.mark.random
    def test_curvature_is_skew_operator(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point
        )

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.zeros(batch_shape + self.space.shape)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_curvature_bianchi_identity(self, n_points, atol):
        """First Bianchi identity on curvature in pre-shape space.

        :math:`R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        curvature_1 = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        curvature_2 = self.space.metric.curvature(
            tangent_vec_b, tangent_vec_c, tangent_vec_a, base_point
        )
        curvature_3 = self.space.metric.curvature(
            tangent_vec_c, tangent_vec_a, tangent_vec_b, base_point
        )

        result = curvature_1 + curvature_2 + curvature_3

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.zeros(batch_shape + self.space.shape)
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.random
    def test_sectional_curvature_lower_bound(self, n_points, atol):
        """Sectional curvature of Kendall shape space is larger than 1.

        The sectional curvature always increase by taking the quotient in a
        Riemannian submersion. Thus, it should larger in kendall shape space
        thane the sectional curvature of the pre-shape space which is 1 as it
        a hypersphere.
        The sectional curvature is computed here with the generic
        directional_curvature and sectional curvature methods.
        """
        base_point = self.data_generator.random_point(n_points)
        hor_a = self.data_generator.random_horizontal_vec(base_point)
        hor_b = self.data_generator.random_horizontal_vec(base_point)

        tidal_force = self.space.metric.directional_curvature(hor_a, hor_b, base_point)

        numerator = self.space.metric.inner_product(tidal_force, hor_a, base_point)
        denominator = (
            self.space.metric.inner_product(hor_a, hor_a, base_point)
            * self.space.metric.inner_product(hor_b, hor_b, base_point)
            - self.space.metric.inner_product(hor_a, hor_b, base_point) ** 2
        )

        condition = ~gs.isclose(denominator, 0.0, atol=atol)
        kappa = numerator[condition] / denominator[condition]
        kappa_direct = self.space.metric.sectional_curvature(hor_a, hor_b, base_point)[
            condition
        ]
        self.assertAllClose(kappa, kappa_direct)

        result = kappa > 1.0 - atol
        self.assertTrue(gs.all(result))

    @pytest.mark.random
    def test_curvature_derivative_bianchi_identity(self, n_points, atol):
        r"""2nd Bianchi identity on curvature derivative in kendall space.

        For any 3 tangent vectors horizontally lifted from kendall shape
        space to Kendall pre-shape space, :math:`(\nabla_X R)(Y, Z)
        + (\nabla_Y R)(Z,X) + (\nabla_Z R)(X, Y) = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        hor_x = self.data_generator.random_horizontal_vec(base_point)
        hor_y = self.data_generator.random_horizontal_vec(base_point)
        hor_z = self.data_generator.random_horizontal_vec(base_point)
        hor_h = self.data_generator.random_horizontal_vec(base_point)

        term_x = self.space.metric.curvature_derivative(
            hor_x, hor_y, hor_z, hor_h, base_point
        )
        term_y = self.space.metric.curvature_derivative(
            hor_y, hor_z, hor_x, hor_h, base_point
        )
        term_z = self.space.metric.curvature_derivative(
            hor_z, hor_x, hor_y, hor_h, base_point
        )

        result = term_x + term_y + term_z
        self.assertAllClose(result, gs.zeros_like(result), atol=atol)

    @pytest.mark.random
    def test_curvature_derivative_is_skew_operator(self, n_points, atol):
        r"""Derivative of a skew operator is skew.

        For any 3 tangent vectors horizontally lifted from kendall shape space
        to Kendall pre-shape space, :math:`(\nabla_X R)(Y,Y)Z = 0`.
        """
        base_point = self.data_generator.random_point(n_points)
        hor_x = self.data_generator.random_horizontal_vec(base_point)
        hor_y = self.data_generator.random_horizontal_vec(base_point)
        hor_z = self.data_generator.random_horizontal_vec(base_point)

        result = self.space.metric.curvature_derivative(
            hor_x, hor_y, hor_y, hor_z, base_point
        )
        self.assertAllClose(result, gs.zeros_like(result), atol=atol)

    @pytest.mark.random
    def test_directional_curvature_derivative_is_quadratic(self, n_points, atol):
        """Directional curvature derivative is quadratic in both variables."""
        base_point = self.data_generator.random_point(n_points)
        hor_x = self.data_generator.random_horizontal_vec(base_point)
        hor_y = self.data_generator.random_horizontal_vec(base_point)

        coef_x, coef_y = gs.random.uniform(size=(2,))

        res = self.space.metric.directional_curvature_derivative(
            coef_x * hor_x, coef_y * hor_y, base_point
        )
        expected = (
            coef_x**2
            * coef_y**2
            * self.space.metric.directional_curvature_derivative(
                hor_x, hor_y, base_point
            )
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_parallel_transport_ivp_transported_is_horizontal(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        direction = self.data_generator.random_tangent_vec(base_point)
        tangent_vec = self.data_generator.random_horizontal_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        fiber_bundle = self.space.metric.fiber_bundle
        res = fiber_bundle.is_horizontal(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)
