import pytest

import geomstats.backend as gs
from geomstats.test.assertions import (
    assert_belongs,
    assert_is_horizontal,
    assert_is_tangent,
    assert_is_vertical,
)
from geomstats.test.test_case import TestCase


class FiberBundleTestCase(TestCase):
    """Test case for fiber bundles.

    The following tests only depend on the bundle structure, i.e. they
    do not need a metric on the total space:

    - `riemannian_submersion`, `lift`, `tangent_riemannian_submersion`
      and `is_vertical`;
    - `riemannian_submersion_belongs_to_base`,
      `lift_belongs_to_total_space`, `riemannian_submersion_after_lift`
      and `tangent_riemannian_submersion_is_tangent`.

    The remaining tests depend on the metric on the total space, either
    because they call it (e.g. `log_after_align_is_horizontal`,
    `projections_are_orthogonal`) or because the tested methods are
    defined through it (alignment, projections, horizontal lift and
    integrability tensor).
    """

    def _test_belongs_to_base(self, point, expected, atol=gs.atol):
        """Check membership to the base space.

        Parameters
        ----------
        point : array-like, shape=[..., *base_space.shape]
            Point.
        expected : array-like, shape=[...]
            Boolean evaluating if `point` belongs to the base space.
        atol : float
            Absolute tolerance.
        """
        res = self.base_space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def _test_belongs_to_total_space(self, point, expected, atol=gs.atol):
        """Check membership to the total space.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point.
        expected : array-like, shape=[...]
            Boolean evaluating if `point` belongs to the total space.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_riemannian_submersion(self, point, expected, atol=gs.atol):
        """Check Riemannian submersion.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[..., *base_space.shape]
            Expected point on the base space.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.riemannian_submersion(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lift(self, point, expected, atol=gs.atol):
        """Check lift.

        Parameters
        ----------
        point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        expected : array-like, shape=[..., *total_space.shape]
            Expected point on the total space.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.lift(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_riemannian_submersion(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        """Check tangent Riemannian submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[..., *base_space.shape]
            Expected tangent vector to the base space.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_align(self, point, base_point, expected, atol=gs.atol):
        """Check alignment.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space to align to.
        expected : array-like, shape=[..., *total_space.shape]
            Expected aligned point.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.align(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_horizontal_projection(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        """Check horizontal projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[..., *total_space.shape]
            Expected horizontal component of `tangent_vec`.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_vertical_projection(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check vertical projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[..., *total_space.shape]
            Expected vertical component of `tangent_vec`.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.vertical_projection(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_is_horizontal(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check if tangent vector is horizontal.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[...]
            Boolean evaluating if `tangent_vec` is horizontal.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.is_horizontal(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    def test_is_vertical(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check if tangent vector is vertical.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[...]
            Boolean evaluating if `tangent_vec` is vertical.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.is_vertical(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    def test_horizontal_lift(
        self, tangent_vec, expected, atol=gs.atol, base_point=None, fiber_point=None
    ):
        """Check horizontal lift.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *base_space.shape]
            Tangent vector to the base space.
        expected : array-like, shape=[..., *total_space.shape]
            Expected horizontal tangent vector to the total space.
        atol : float
            Absolute tolerance.
        base_point : array-like, shape=[..., *base_space.shape]
            Point on the base space at which to lift.
            Optional, default: None.
        fiber_point : array-like, shape=[..., *total_space.shape]
            Point on the total space at which to lift.
            Optional, default: None.
        """
        res = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_integrability_tensor(
        self, tangent_vec_x, tangent_vec_e, base_point, expected, atol=gs.atol
    ):
        """Check integrability tensor.

        Parameters
        ----------
        tangent_vec_x : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        tangent_vec_e : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected : array-like, shape=[..., *total_space.shape]
            Expected tensor applied to `tangent_vec_x` and `tangent_vec_e`.
        atol : float
            Absolute tolerance.
        """
        res = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_x, tangent_vec_e, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_integrability_tensor_derivative(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        nabla_x_y,
        tangent_vec_e,
        nabla_x_e,
        base_point,
        expected_nabla_x_a_y_e,
        expected_a_y_e,
        atol=gs.atol,
    ):
        """Check covariant derivative of the integrability tensor.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., *total_space.shape]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., *total_space.shape]
            Horizontal tangent vector at `base_point`.
        nabla_x_y : array-like, shape=[..., *total_space.shape]
            Tangent vector at `base_point`.
        tangent_vec_e : array-like, shape=[..., *total_space.shape]
            Tangent vector at `base_point`.
        nabla_x_e : array-like, shape=[..., *total_space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        expected_nabla_x_a_y_e : array-like, shape=[..., *total_space.shape]
            Expected covariant derivative of the tensor.
        expected_a_y_e : array-like, shape=[..., *total_space.shape]
            Expected tensor.
        atol : float
            Absolute tolerance.
        """
        (
            nabla_x_a_y_e,
            a_y_e,
        ) = self.total_space.fiber_bundle.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            tangent_vec_e,
            nabla_x_e,
            base_point,
        )
        self.assertAllClose(nabla_x_a_y_e, expected_nabla_x_a_y_e, atol=atol)
        self.assertAllClose(a_y_e, expected_a_y_e, atol=atol)

    def test_riemannian_submersion_belongs_to_base(self, point, atol=gs.atol):
        """Check Riemannian submersion maps to the base space.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        proj_point = self.total_space.fiber_bundle.riemannian_submersion(point)

        assert_belongs(self.base_space, proj_point, atol=atol)

    def test_lift_belongs_to_total_space(self, point, atol=gs.atol):
        """Check lift maps to the total space.

        Parameters
        ----------
        point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        atol : float
            Absolute tolerance.
        """
        lifted_point = self.total_space.fiber_bundle.lift(point)

        assert_belongs(self.total_space, lifted_point, atol=atol)

    def test_riemannian_submersion_after_lift(self, point, atol=gs.atol):
        """Check lift is a right inverse of the Riemannian submersion.

        That is, the lifted point lies in the fiber above `point`.

        Parameters
        ----------
        point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        atol : float
            Absolute tolerance.
        """
        lifted_point = self.total_space.fiber_bundle.lift(point)
        point_ = self.total_space.fiber_bundle.riemannian_submersion(lifted_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_riemannian_submersion_is_tangent(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check tangent submersion maps to the tangent space of the base space.

        The image of a tangent vector at `base_point` must be tangent to the
        base space at the image of `base_point` under the Riemannian
        submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        proj_tangent_vector = (
            self.total_space.fiber_bundle.tangent_riemannian_submersion(
                tangent_vec, base_point
            )
        )
        proj_point = self.total_space.fiber_bundle.riemannian_submersion(base_point)

        assert_is_tangent(self.base_space, proj_tangent_vector, proj_point, atol=atol)

    def test_log_after_align_is_horizontal(self, point, base_point, atol=gs.atol):
        """Check log from an aligned point is horizontal.

        Aligning `point` to `base_point` finds the point of its fiber that is
        closest to `base_point`. The geodesic between them is then orthogonal
        to the fiber, i.e. its initial velocity is horizontal.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        log = self.total_space.metric.log(aligned_point, base_point)

        assert_is_horizontal(self.total_space, log, base_point, atol=atol)

    def test_align_in_same_fiber(self, point, base_point, atol=gs.atol):
        """Check aligning a point of the fiber of a base point gives it back.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space, in the same fiber as `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        self.assertAllClose(aligned_point, base_point, atol=atol)

    def test_projections_are_orthogonal(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check horizontal and vertical projections are orthogonal.

        The horizontal component of a tangent vector is orthogonal to the
        vertical component of any tangent vector at the same point.
        Together with `vertical_projection_is_vertical`, this characterizes
        vertical projection as the orthogonal projection onto the kernel
        of the tangent submersion.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        tangent_vec_b : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.

        Notes
        -----
        This test makes others redundant in the following cases:

        1. If horizontal projection is the identity minus vertical
           projection (i.e. at most one of them is implemented
           independently), then both projections are idempotent.
        2. If 1. holds and `is_horizontal` is not overridden (it compares
           a vector with its horizontal projection), then
           `horizontal_projection_is_horizontal` is redundant.
        3. If 2. holds and horizontal lift is computed through horizontal
           projection, then `horizontal_lift_is_horizontal` is redundant.
        """
        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec_a, base_point
        )
        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec_b, base_point
        )
        res = self.total_space.metric.inner_product(horizontal, vertical, base_point)

        self.assertAllClose(res, gs.zeros_like(res), atol=atol)

    def test_horizontal_projection_is_horizontal(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check horizontal projection maps to horizontal vectors.

        With the default `is_horizontal`, which compares a vector with its
        horizontal projection, this checks that horizontal projection is
        idempotent.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        assert_is_horizontal(self.total_space, horizontal, base_point, atol=atol)

    def test_vertical_projection_is_vertical(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check vertical projection maps to vertical vectors.

        With the default `is_vertical`, which checks that the tangent
        submersion vanishes, this checks that vertical projection maps to
        the kernel of the tangent submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )

        assert_is_vertical(self.total_space, vertical, base_point, atol=atol)

    def test_tangent_riemannian_submersion_after_vertical_projection(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check tangent submersion vanishes on vertical projection.

        Vertical projection maps to the kernel of the tangent submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *total_space.shape]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.

        Notes
        -----
        If `is_vertical` is not overridden (it checks the tangent
        submersion vanishes), then this test is the same as
        `vertical_projection_is_vertical`.
        """
        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            vertical, base_point
        )
        expected = gs.zeros_like(res)

        self.assertAllClose(res, expected, atol=atol)

    def test_horizontal_lift_is_horizontal(self, tangent_vec, base_point, atol=gs.atol):
        """Check horizontal lift maps to horizontal vectors.

        The tangent vector is lifted at the lift of `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *base_space.shape]
            Tangent vector to the base space at `base_point`.
        base_point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        atol : float
            Absolute tolerance.

        Notes
        -----
        Both `base_point` and `fiber_point` are passed to
        `horizontal_lift`. Implementations that give precedence to
        `base_point` do not exercise the `fiber_point` branch.
        """
        fiber_point = self.total_space.fiber_bundle.lift(base_point)
        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )

        assert_is_horizontal(self.total_space, horizontal, fiber_point, atol=atol)

    def test_tangent_riemannian_submersion_after_horizontal_lift(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check horizontal lift is a right inverse of the tangent submersion.

        The tangent vector is lifted at the lift of `base_point`, passed
        as `fiber_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *base_space.shape]
            Tangent vector to the base space at `base_point`.
        base_point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        atol : float
            Absolute tolerance.

        Notes
        -----
        Fiber points are only lifts of base points. If `lift` is the
        identity, then horizontal lift is not checked at general points
        of the total space.
        """
        fiber_point = self.total_space.fiber_bundle.lift(base_point)

        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, fiber_point=fiber_point
        )
        tangent_vec_ = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            horizontal, fiber_point
        )

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.api
    def test_horizontal_lift_with_base_point(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check horizontal lift with base point.

        Lifting with `base_point` is the same as lifting with
        `fiber_point` set to the lift of `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *base_space.shape]
            Tangent vector to the base space at `base_point`.
        base_point : array-like, shape=[..., *base_space.shape]
            Point on the base space.
        atol : float
            Absolute tolerance.
        """
        fiber_point = self.total_space.fiber_bundle.lift(base_point)

        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, fiber_point=fiber_point
        )
        horizontal_ = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point
        )

        self.assertAllClose(horizontal_, horizontal, atol=atol)



class AlignComparisonTestCase(TestCase):
    """Test case comparing two alignment maps.

    Each test checks `align` and `other_align` align a point to a base point
    in the same way.
    """

    def test_align(self, point, base_point, atol=gs.atol):
        """Check alignment is the same for both maps.

        Parameters
        ----------
        point : array-like, shape=[..., *total_space.shape]
            Point on the total space to align.
        base_point : array-like, shape=[..., *total_space.shape]
            Point on the total space.
        atol : float
            Absolute tolerance.
        """
        aligned = self.align(point, base_point)
        other_aligned = self.other_align(point, base_point)

        self.assertAllClose(aligned, other_aligned, atol=atol)
