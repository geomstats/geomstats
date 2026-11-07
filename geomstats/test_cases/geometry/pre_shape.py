import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.vectorization import get_batch_shape


def integrability_tensor_alt(total_space, tangent_vec_a, tangent_vec_b, base_point):
    r"""Compute the fundamental tensor A of the submersion.

    The fundamental tensor A is defined for tangent vectors of the total
    space by [ONeill]_ :math:`A_X Y = ver\nabla^M_{hor X} (hor Y)
    + hor \nabla^M_{hor X}( ver Y)` where :math:`hor,ver` are the
    horizontal and vertical projections.

    For the pre-shape space, we have closed-form expressions and the result
    does not depend on the vertical part of :math:`X`.

    (This is an alternative implementation.)

    Parameters
    ----------
    total_space : Manifold
        Total space with fiber bundle structure.
    tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
        Tangent vector at ``base_point``.
    tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
        Tangent vector at ``base_point``.
    base_point : array-like, shape=[..., k_landmarks, ambient_dim]
        Point of the total space.

    Returns
    -------
    vector : array-like, shape=[..., k_landmarks, ambient_dim]
        Tangent vector at ``base_point``, result of the A tensor applied to
        ``tangent_vec_a`` and ``tangent_vec_b``.

    References
    ----------
    .. [ON1966]  O’Neill, Barrett. The Fundamental Equations of a
        Submersion, Michigan Mathematical Journal 13, no. 4
        (December 1966): 459–69. https://doi.org/10.1307/mmj/1028999604.
    """
    # Only the horizontal part of a counts
    horizontal_a = total_space.fiber_bundle.horizontal_projection(
        tangent_vec_a, base_point
    )
    vertical_b, skew = total_space.fiber_bundle.vertical_projection(
        tangent_vec_b, base_point, return_skew=True
    )
    horizontal_b = tangent_vec_b - vertical_b

    # For the horizontal part of b
    transposed_point = gs.transpose(base_point)
    sigma = gs.matmul(transposed_point, base_point)
    alignment = gs.matmul(gs.transpose(horizontal_a), horizontal_b)
    right_term = alignment - gs.transpose(alignment)
    skew_hor = gs.linalg.solve_sylvester(sigma, sigma, right_term)
    vertical = -gs.matmul(base_point, skew_hor)

    # For the vertical part of b
    vert_part = -gs.matmul(horizontal_a, skew)
    tangent_vert = total_space.to_tangent(vert_part, base_point)
    horizontal_ = total_space.fiber_bundle.horizontal_projection(
        tangent_vert, base_point
    )

    return vertical + horizontal_


class PreShapeSpaceTestCase(LevelSetTestCase):
    def test_is_centered(self, point, expected, atol=gs.atol):
        """Check is_centered.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point.
        expected : array-like, shape=[...]
            Expected boolean.
        atol : float
            Absolute tolerance.
        """
        res = self.space.is_centered(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_center(self, point, expected, atol=gs.atol):
        """Check center.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point.
        expected : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected centered point.
        atol : float
            Absolute tolerance.
        """
        res = self.space.center(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_center_is_centered(self, point, atol=gs.atol):
        """Check centering a point gives a centered point.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point.
        atol : float
            Absolute tolerance.
        """
        res = self.space.center(point)

        expected = gs.ones(get_batch_shape(self.space.point_ndim, point), dtype=bool)
        self.test_is_centered(res, expected, atol=atol)


class PreShapeBundleTestCase(FiberBundleTestCase):
    def _integrability_tensor_derivative(
        self, horizontal_vec_x, vec_y, vec_dy, vec_z, vec_dz, base_point
    ):
        r"""Compute the integrability tensor derivative.

        The derivatives :math:`\nabla_X Y` and :math:`\nabla_X Z` are built as
        :math:`A_X Y + dY` and :math:`A_X Z + dZ`.
        """
        fiber_bundle = self.total_space.fiber_bundle
        nabla_x_y = (
            fiber_bundle.integrability_tensor(horizontal_vec_x, vec_y, base_point)
            + vec_dy
        )
        nabla_x_z = (
            fiber_bundle.integrability_tensor(horizontal_vec_x, vec_z, base_point)
            + vec_dz
        )
        nabla_x_a_y_z, a_y_z = fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x, vec_y, nabla_x_y, vec_z, nabla_x_z, base_point
        )
        return nabla_x_a_y_z, a_y_z, nabla_x_z

    def test_integrability_tensor_derivative_parallel(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        base_point,
        expected_nabla_x_a_y_z,
        expected_a_y_z,
        atol=gs.atol,
    ):
        """Check integrability tensor derivative for parallel vector fields.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        expected_nabla_x_a_y_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected derivative of the integrability tensor.
        expected_a_y_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected integrability tensor.
        atol : float
            Absolute tolerance.
        """
        (
            nabla_x_a_y_z,
            a_y_z,
        ) = self.total_space.fiber_bundle.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
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
        atol=gs.atol,
    ):
        """Check iterated integrability tensor derivative for parallel fields.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        expected_nabla_x_a_y_v : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected derivative of :math:`A_Y V`.
        expected_a_x_a_y_a_x_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected :math:`A_X A_Y A_X Y`.
        expected_nabla_x_v : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected derivative of :math:`V`.
        expected_a_y_a_x_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected :math:`A_Y A_X Y`.
        expected_vertical_vec_v : array-like, shape=[..., k_landmarks, ambient_dim]
            Expected vertical vector :math:`V = A_X Y`.
        atol : float
            Absolute tolerance.
        """
        fiber_bundle = self.total_space.fiber_bundle
        (
            nabla_x_a_y_v,
            a_x_a_y_a_x_y,
            nabla_x_v,
            a_y_a_x_y,
            vertical_vec_v,
        ) = fiber_bundle.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        self.assertAllClose(nabla_x_a_y_v, expected_nabla_x_a_y_v, atol=atol)
        self.assertAllClose(a_x_a_y_a_x_y, expected_a_x_a_y_a_x_y, atol=atol)
        self.assertAllClose(nabla_x_v, expected_nabla_x_v, atol=atol)
        self.assertAllClose(a_y_a_x_y, expected_a_y_a_x_y, atol=atol)
        self.assertAllClose(vertical_vec_v, expected_vertical_vec_v, atol=atol)

    def test_vertical_projection_correctness(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        r"""Check vertical projection keeps the skew part of :math:`X^T W`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        transposed_point = gs.transpose(base_point)
        tmp_expected = gs.matmul(transposed_point, tangent_vec)
        expected = gs.transpose(tmp_expected) - tmp_expected

        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )
        tmp_result = gs.matmul(transposed_point, vertical)
        result = gs.transpose(tmp_result) - tmp_result

        self.assertAllClose(result, expected, atol=atol)

    def test_horizontal_projection_correctness(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        r"""Check :math:`X^T H` is symmetric for the horizontal projection H.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        result = gs.matmul(gs.transpose(base_point), horizontal)
        self.assertAllClose(result, gs.transpose(result), atol=atol)

    def test_horizontal_projection_is_tangent(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check horizontal projection is tangent to the pre-shape space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        res = self.total_space.is_tangent(horizontal, base_point, atol=atol)
        expected = gs.ones(
            get_batch_shape(self.total_space.point_ndim, base_point), dtype=bool
        )
        self.assertAllEqual(res, expected)

    def test_alignment_is_symmetric(self, point, base_point, atol=gs.atol):
        """Check the aligned point and base point have a symmetric product.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        alignment = Matrices.mul(gs.transpose(aligned_point), base_point)
        self.assertAllClose(alignment, gs.transpose(alignment), atol=atol)

    def test_integrability_tensor_identity_1(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        r"""Check integrability tensor is skew-symmetric.

        The integrability tensor :math:`A_X E` is skew-symmetric with respect
        to the pre-shape metric, :math:`< A_X E, F> + <E, A_X F> = 0`. By
        polarization, this is equivalent to :math:`< A_X E, E> = 0`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        result_ab = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )

        result = self.total_space.metric.inner_product(
            tangent_vec_b, result_ab, base_point
        )
        expected = gs.zeros(get_batch_shape(self.total_space.point_ndim, base_point))
        self.assertAllClose(result, expected, atol=atol)

    def test_integrability_tensor_identity_2(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        r"""Check integrability tensor is alternating and exchanges subspaces.

        The integrability tensor is alternating (:math:`A_X Y = - A_Y X`) for
        horizontal vector fields :math:`X,Y`, and it exchanges horizontal and
        vertical vector spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        fiber_bundle = self.total_space.fiber_bundle
        horizontal_a = fiber_bundle.horizontal_projection(tangent_vec_a, base_point)
        horizontal_b = fiber_bundle.horizontal_projection(tangent_vec_b, base_point)

        result = fiber_bundle.integrability_tensor(
            horizontal_a, horizontal_b, base_point
        )
        expected = -fiber_bundle.integrability_tensor(
            horizontal_b, horizontal_a, base_point
        )
        self.assertAllClose(result, expected, atol=atol)

        expected = gs.ones(
            get_batch_shape(self.total_space.point_ndim, base_point), dtype=bool
        )

        is_vertical = fiber_bundle.is_vertical(result, base_point, atol=atol)
        self.assertAllEqual(is_vertical, expected)

        vertical_b = tangent_vec_b - horizontal_b
        result = fiber_bundle.integrability_tensor(horizontal_a, vertical_b, base_point)
        is_horizontal = fiber_bundle.is_horizontal(result, base_point, atol=atol)
        self.assertAllEqual(is_horizontal, expected)

    def test_integrability_tensor_alt(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check integrability tensor against an alternative implementation.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        expected = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )
        alt_expected = integrability_tensor_alt(
            self.total_space, tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(expected, alt_expected, atol=atol)

    def test_integrability_tensor_derivative_is_alternate(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        horizontal_vec_dy,
        horizontal_vec_dz,
        base_point,
        atol=gs.atol,
    ):
        r"""Check integrability tensor derivative is alternate.

        For horizontal vector fields :math:`X,Y,Z` the integrability tensor
        (hence its derivatives) is alternate:
        :math:`\nabla_X ( A_Y Z + A_Z Y ) = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_dy : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Y`.
        horizontal_vec_dz : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Z`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        nabla_x_a_y_z, a_y_z, _ = self._integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_dy,
            horizontal_vec_z,
            horizontal_vec_dz,
            base_point,
        )
        nabla_x_a_z_y, a_z_y, _ = self._integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_z,
            horizontal_vec_dz,
            horizontal_vec_y,
            horizontal_vec_dy,
            base_point,
        )

        expected = gs.zeros_like(nabla_x_a_y_z)
        self.assertAllClose(nabla_x_a_y_z + nabla_x_a_z_y, expected, atol=atol)
        self.assertAllClose(a_y_z + a_z_y, expected, atol=atol)

    def test_integrability_tensor_derivative_is_skew_symmetric(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        horizontal_vec_dy,
        horizontal_vec_dz,
        tangent_vec_v,
        tangent_vec_dv,
        base_point,
        atol=gs.atol,
    ):
        r"""Check integrability tensor derivative is skew-symmetric.

        For horizontal vector fields :math:`X,Y,Z` and a vector field
        :math:`V`,
        :math:`\nabla_X (< A_Y Z, V> + < Z, A_Y V>) = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_dy : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Y`.
        horizontal_vec_dz : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Z`.
        tangent_vec_v : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_dv : array-like, shape=[..., k_landmarks, ambient_dim]
            Derivative :math:`\nabla_X V`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        nabla_x_a_y_z, a_y_z, nabla_x_z = self._integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_dy,
            horizontal_vec_z,
            horizontal_vec_dz,
            base_point,
        )

        fiber_bundle = self.total_space.fiber_bundle
        nabla_x_y = (
            fiber_bundle.integrability_tensor(
                horizontal_vec_x, horizontal_vec_y, base_point
            )
            + horizontal_vec_dy
        )
        nabla_x_a_y_v, a_y_v = fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            tangent_vec_v,
            tangent_vec_dv,
            base_point,
        )

        inner = self.total_space.metric.inner_product
        result = (
            inner(nabla_x_a_y_z, tangent_vec_v)
            + inner(a_y_z, tangent_vec_dv)
            + inner(nabla_x_a_y_v, horizontal_vec_z)
            + inner(a_y_v, nabla_x_z)
        )
        expected = gs.zeros(get_batch_shape(self.total_space.point_ndim, base_point))
        self.assertAllClose(result, expected, atol=atol)

    def test_integrability_tensor_derivative_reverses_hor(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        horizontal_vec_h,
        horizontal_vec_dy,
        horizontal_vec_dz,
        horizontal_vec_dh,
        base_point,
        atol=gs.atol,
    ):
        r"""Check integrability tensor derivative maps horizontal to vertical.

        For horizontal vector fields :math:`X,Y,Z,H`,
        :math:`\nabla_X < A_Y Z, H > = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_h : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_dy : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Y`.
        horizontal_vec_dz : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Z`.
        horizontal_vec_dh : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X H`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        nabla_x_a_y_z, a_y_z, _ = self._integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_dy,
            horizontal_vec_z,
            horizontal_vec_dz,
            base_point,
        )
        nabla_x_h = (
            self.total_space.fiber_bundle.integrability_tensor(
                horizontal_vec_x, horizontal_vec_h, base_point
            )
            + horizontal_vec_dh
        )

        inner = self.total_space.metric.inner_product
        result = inner(nabla_x_a_y_z, horizontal_vec_h) + inner(a_y_z, nabla_x_h)
        expected = gs.zeros(get_batch_shape(self.total_space.point_ndim, base_point))
        self.assertAllClose(result, expected, atol=atol)

    def test_integrability_tensor_derivative_reverses_ver(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        vertical_vec_z,
        vertical_vec_w,
        horizontal_vec_dy,
        vertical_vec_dz,
        vertical_vec_dw,
        base_point,
        atol=gs.atol,
    ):
        r"""Check integrability tensor derivative maps vertical to horizontal.

        For horizontal vector fields :math:`X,Y` and vertical vector fields
        :math:`V,W`, :math:`\nabla_X < A_Y V, W > = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        vertical_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Vertical tangent vector at `base_point`.
        vertical_vec_w : array-like, shape=[..., k_landmarks, ambient_dim]
            Vertical tangent vector at `base_point`.
        horizontal_vec_dy : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal part of :math:`\nabla_X Y`.
        vertical_vec_dz : array-like, shape=[..., k_landmarks, ambient_dim]
            Vertical part of :math:`\nabla_X Z`.
        vertical_vec_dw : array-like, shape=[..., k_landmarks, ambient_dim]
            Vertical part of :math:`\nabla_X W`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        nabla_x_a_y_z, a_y_z, _ = self._integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_dy,
            vertical_vec_z,
            vertical_vec_dz,
            base_point,
        )
        nabla_x_w = (
            self.total_space.fiber_bundle.integrability_tensor(
                horizontal_vec_x, vertical_vec_w, base_point
            )
            + vertical_vec_dw
        )

        inner = self.total_space.metric.inner_product
        result = inner(nabla_x_a_y_z, vertical_vec_w) + inner(a_y_z, nabla_x_w)
        expected = gs.zeros(get_batch_shape(self.total_space.point_ndim, base_point))
        self.assertAllClose(result, expected, atol=atol)

    def test_integrability_tensor_derivative_parallel_optimized(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        base_point,
        atol=gs.atol,
    ):
        """Check optimized integrability tensor derivative for parallel fields.

        The optimized version for quotient-parallel vector fields should equal
        the general implementation.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        fiber_bundle = self.total_space.fiber_bundle
        (
            nabla_x_a_y_z,
            a_y_z,
        ) = fiber_bundle.integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
        )

        a_x_y = fiber_bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        a_x_z = fiber_bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_z, base_point
        )
        (
            expected_nabla_x_a_y_z,
            expected_a_y_z,
        ) = fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            a_x_y,
            horizontal_vec_z,
            a_x_z,
            base_point,
        )

        self.assertAllClose(nabla_x_a_y_z, expected_nabla_x_a_y_z, atol=atol)
        self.assertAllClose(a_y_z, expected_a_y_z, atol=atol)

    def test_iterated_integrability_tensor_derivative_parallel_optimized(
        self, horizontal_vec_x, horizontal_vec_y, base_point, atol=gs.atol
    ):
        r"""Check optimized iterated integrability tensor derivatives.

        The optimized version of the iterated integrability tensor
        :math:`A_X A_Y A_X Y`, computed with the horizontal lift of
        quotient-parallel vector fields extending the tangent vectors
        :math:`X,Y` of Kendall shape spaces (identified to horizontal vectors
        of the pre-shape space), is the recursive application of two general
        integrability tensor derivatives with proper derivatives.
        Intermediate computations returned are also verified.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the pre-shape space.
        atol : float
            Absolute tolerance.
        """
        fiber_bundle = self.total_space.fiber_bundle
        a_x_y = fiber_bundle.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        (
            nabla_x_v,
            a_x_y,
        ) = fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_x,
            gs.zeros_like(horizontal_vec_x),
            horizontal_vec_y,
            a_x_y,
            base_point,
        )

        (
            nabla_x_a_y_a_x_y,
            a_y_a_x_y,
        ) = fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x, horizontal_vec_y, a_x_y, a_x_y, nabla_x_v, base_point
        )

        a_x_a_y_a_x_y = fiber_bundle.integrability_tensor(
            horizontal_vec_x, a_y_a_x_y, base_point
        )

        (
            nabla_x_a_y_a_x_y_qp,
            a_x_a_y_a_x_y_qp,
            nabla_x_v_qp,
            a_y_a_x_y_qp,
            ver_v_qp,
        ) = fiber_bundle.iterated_integrability_tensor_derivative_parallel(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        self.assertAllClose(a_x_y, ver_v_qp, atol=atol)
        self.assertAllClose(a_y_a_x_y, a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_v, nabla_x_v_qp, atol=atol)
        self.assertAllClose(a_x_a_y_a_x_y, a_x_a_y_a_x_y_qp, atol=atol)
        self.assertAllClose(nabla_x_a_y_a_x_y, nabla_x_a_y_a_x_y_qp, atol=atol)


class KendallShapeMetricTestCase(RiemannianMetricTestCase):
    """Kendall shape metric test case.

    Points of the shape space are represented by pre-shapes, so points are
    compared through their squared distance instead of their coordinates.
    The squared distance is used as the square root amplifies rounding
    errors close to zero.

    Requires: ``space``, ``total_space``.
    """

    def _cmp_points(self, point, point_, atol):
        squared_dists = self.space.metric.squared_dist(point, point_)
        batch_shape = get_batch_shape(self.space.point_ndim, point, point_)

        self.assertAllClose(squared_dists, gs.zeros(batch_shape), atol=atol)

    def test_exp_after_log(self, point, base_point, atol=gs.atol):
        """Check exp after log gives the same shape.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.log(point, base_point)
        point_ = self.space.metric.exp(tangent_vec, base_point)

        self._cmp_points(point_, point, atol=atol)

    def test_log_after_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check exp after log after exp gives the same shape.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        end_point_ = self.space.metric.exp(tangent_vec_, base_point)

        self._cmp_points(end_point_, end_point, atol=atol)

    def test_geodesic_bvp_reverse(self, initial_point, end_point, time, atol=gs.atol):
        """Check the reversed geodesic gives the same shapes.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Initial point on the shape space.
        end_point : array-like, shape=[..., k_landmarks, ambient_dim]
            End point on the shape space.
        time : array-like, shape=[n_times]
            Times.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        points = gs.reshape(geod_func(time), (-1, *self.space.shape))
        points_ = gs.reshape(geod_func_reverse(1.0 - time), (-1, *self.space.shape))

        self._cmp_points(points, points_, atol=atol)

    def test_geodesic_boundary_points(self, initial_point, end_point, atol=gs.atol):
        """Check the geodesic starts and ends at the given shapes.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Initial point on the shape space.
        end_point : array-like, shape=[..., k_landmarks, ambient_dim]
            End point on the shape space.
        atol : float
            Absolute tolerance.
        """
        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        points = gs.reshape(geod_func(time), (-1, *self.space.shape))
        expected_points = gs.reshape(
            gs.stack([initial_point, end_point], axis=-(self.space.point_ndim + 1)),
            (-1, *self.space.shape),
        )
        self._cmp_points(points, expected_points, atol=atol)

    def test_curvature_is_skew_operator(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check curvature is a skew operator, i.e. :math:`R(X, X)Y = 0`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, gs.zeros_like(res), atol=atol)

    def test_curvature_bianchi_identity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol=gs.atol
    ):
        r"""Check first Bianchi identity.

        :math:`R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
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
        self.assertAllClose(result, gs.zeros_like(result), atol=atol)

    def test_sectional_curvature_lower_bound(
        self, horizontal_vec_a, horizontal_vec_b, base_point, atol=gs.atol
    ):
        """Check sectional curvature of Kendall shape space is larger than 1.

        The sectional curvature always increases by taking the quotient in a
        Riemannian submersion. Thus, it should be larger in Kendall shape space
        than the sectional curvature of the pre-shape space, which is 1 as it
        is a hypersphere.
        The sectional curvature is computed here with the generic
        directional_curvature and sectional curvature methods.

        Parameters
        ----------
        horizontal_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        metric = self.space.metric
        tidal_force = metric.directional_curvature(
            horizontal_vec_a, horizontal_vec_b, base_point
        )

        numerator = metric.inner_product(tidal_force, horizontal_vec_a, base_point)
        denominator = (
            metric.inner_product(horizontal_vec_a, horizontal_vec_a, base_point)
            * metric.inner_product(horizontal_vec_b, horizontal_vec_b, base_point)
            - metric.inner_product(horizontal_vec_a, horizontal_vec_b, base_point)
            ** 2
        )

        condition = ~gs.isclose(denominator, 0.0, atol=atol)
        kappa = numerator[condition] / denominator[condition]
        kappa_direct = metric.sectional_curvature(
            horizontal_vec_a, horizontal_vec_b, base_point
        )[condition]
        self.assertAllClose(kappa, kappa_direct)

        self.assertTrue(gs.all(kappa > 1.0 - atol))

    def test_curvature_derivative_bianchi_identity(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        horizontal_vec_h,
        base_point,
        atol=gs.atol,
    ):
        r"""Check second Bianchi identity.

        For horizontal lifts of tangent vectors of the shape space,
        :math:`(\nabla_X R)(Y, Z) + (\nabla_Y R)(Z,X) + (\nabla_Z R)(X, Y) = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_h : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        curvature_derivative = self.space.metric.curvature_derivative
        term_x = curvature_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_z,
            horizontal_vec_h,
            base_point,
        )
        term_y = curvature_derivative(
            horizontal_vec_y,
            horizontal_vec_z,
            horizontal_vec_x,
            horizontal_vec_h,
            base_point,
        )
        term_z = curvature_derivative(
            horizontal_vec_z,
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_h,
            base_point,
        )

        result = term_x + term_y + term_z
        self.assertAllClose(result, gs.zeros_like(result), atol=atol)

    def test_curvature_derivative_is_skew_operator(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        horizontal_vec_z,
        base_point,
        atol=gs.atol,
    ):
        r"""Check curvature derivative is a skew operator.

        For horizontal lifts of tangent vectors of the shape space,
        :math:`(\nabla_X R)(Y,Y)Z = 0`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        result = self.space.metric.curvature_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            horizontal_vec_y,
            horizontal_vec_z,
            base_point,
        )
        self.assertAllClose(result, gs.zeros_like(result), atol=atol)

    def test_directional_curvature_derivative_is_quadratic(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        base_point,
        coef_x,
        coef_y,
        atol=gs.atol,
    ):
        """Check directional curvature derivative is quadratic in both variables.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        coef_x : float
            Coefficient of `horizontal_vec_x`.
        coef_y : float
            Coefficient of `horizontal_vec_y`.
        atol : float
            Absolute tolerance.
        """
        directional_curvature_derivative = (
            self.space.metric.directional_curvature_derivative
        )
        res = directional_curvature_derivative(
            coef_x * horizontal_vec_x, coef_y * horizontal_vec_y, base_point
        )
        expected = (
            coef_x**2
            * coef_y**2
            * directional_curvature_derivative(
                horizontal_vec_x, horizontal_vec_y, base_point
            )
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_parallel_transport_ivp_transported_is_horizontal(
        self, horizontal_vec, direction, base_point, atol=gs.atol
    ):
        """Check parallel transport of a horizontal vector is horizontal.

        Parameters
        ----------
        horizontal_vec : array-like, shape=[..., k_landmarks, ambient_dim]
            Horizontal tangent vector at `base_point`.
        direction : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector at `base_point` defining the geodesic.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Point on the shape space.
        atol : float
            Absolute tolerance.
        """
        transported = self.space.metric.parallel_transport(
            horizontal_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        res = self.total_space.fiber_bundle.is_horizontal(
            transported, end_point, atol=atol
        )
        expected = gs.ones(
            get_batch_shape(self.space.point_ndim, base_point), dtype=bool
        )
        self.assertAllEqual(res, expected)
