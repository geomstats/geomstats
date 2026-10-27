import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.manifold import register_quotient
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.test.test_case import TestCase
from geomstats.vectorization import get_batch_shape


class FiberBundleTestCase(TestCase):
    def _test_belongs_to_base(self, point, expected, atol=gs.atol):
        res = self.base_space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def _test_belongs_to_total_space(self, point, expected, atol=gs.atol):
        res = self.total_space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_riemannian_submersion(self, point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.riemannian_submersion(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_riemannian_submersion_belongs_to_base(self, point, atol=gs.atol):
        proj_point = self.total_space.fiber_bundle.riemannian_submersion(point)

        expected_shape = get_batch_shape(self.total_space.point_ndim, point)
        expected = gs.ones(expected_shape, dtype=bool)

        self._test_belongs_to_base(proj_point, expected, atol)

    def test_lift(self, point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.lift(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lift_belongs_to_total_space(self, point, atol=gs.atol):
        lifted_point = self.total_space.fiber_bundle.lift(point)

        expected_shape = get_batch_shape(self.base_space.point_ndim, point)
        expected = gs.ones(expected_shape, dtype=bool)

        self._test_belongs_to_total_space(lifted_point, expected, atol)

    def test_riemannian_submersion_after_lift(self, point, atol=gs.atol):
        lifted_point = self.total_space.fiber_bundle.lift(point)
        point_ = self.total_space.fiber_bundle.riemannian_submersion(lifted_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_riemannian_submersion(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_riemannian_submersion_is_tangent(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        proj_tangent_vector = (
            self.total_space.fiber_bundle.tangent_riemannian_submersion(
                tangent_vec, base_point
            )
        )
        proj_point = self.total_space.fiber_bundle.riemannian_submersion(base_point)

        res = self.base_space.is_tangent(proj_tangent_vector, proj_point, atol=atol)

        expected_shape = get_batch_shape(
            self.total_space.point_ndim, tangent_vec, base_point
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_align(self, point, base_point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.align(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_log_after_align_is_horizontal(self, point, base_point, atol=gs.atol):
        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        log = self.total_space.metric.log(aligned_point, base_point)

        expected_shape = get_batch_shape(self.total_space.point_ndim, point, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.test_is_horizontal(log, base_point, expected, atol)

    def test_horizontal_projection(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        res = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_horizontal_projection_is_horizontal(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        expected_shape = get_batch_shape(
            self.total_space.point_ndim, tangent_vec, base_point
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.test_is_horizontal(horizontal, base_point, expected, atol)

    def test_vertical_projection(self, tangent_vec, base_point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.vertical_projection(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_vertical_projection_is_vertical(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )

        expected_shape = get_batch_shape(
            self.total_space.point_ndim, tangent_vec, base_point
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.test_is_vertical(vertical, base_point, expected, atol)

    def test_tangent_riemannian_submersion_after_vertical_projection(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            vertical, base_point
        )
        expected = gs.zeros_like(res)

        self.assertAllClose(res, expected, atol=atol)

    def test_is_horizontal(self, tangent_vec, base_point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.is_horizontal(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    def test_is_vertical(self, tangent_vec, base_point, expected, atol=gs.atol):
        res = self.total_space.fiber_bundle.is_vertical(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    def test_horizontal_lift(
        self, tangent_vec, expected, atol=gs.atol, base_point=None, fiber_point=None
    ):
        res = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_horizontal_lift_is_horizontal(self, tangent_vec, base_point, atol=gs.atol):
        fiber_point = self.total_space.fiber_bundle.lift(base_point)
        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )

        expected_shape = get_batch_shape(
            self.base_space.point_ndim, tangent_vec, base_point
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.test_is_horizontal(horizontal, fiber_point, expected, atol)

    def test_tangent_riemannian_submersion_after_horizontal_lift(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        fiber_point = self.total_space.fiber_bundle.lift(base_point)

        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, fiber_point=fiber_point
        )
        tangent_vec_ = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            horizontal, fiber_point
        )

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_integrability_tensor(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        res = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
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


class GeneralLinearBuresWassersteinBundle(FiberBundle):
    def __init__(self, total_space):
        super().__init__(total_space=total_space, aligner=True)

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, gs.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, gs.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None and fiber_point is None:
            raise ValueError(
                "Either a point (of the total space) or a "
                "base point (of the base manifold) must be "
                "given."
            )

        if base_point is None:
            base_point = self.riemannian_submersion(fiber_point)

        if fiber_point is None:
            fiber_point = self.lift(base_point)

        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


register_quotient(
    Space=GeneralLinear,
    Metric=MatricesMetric,
    GroupAction=SpecialOrthogonalComposeAction,
    FiberBundle=GeneralLinearBuresWassersteinBundle,
)
