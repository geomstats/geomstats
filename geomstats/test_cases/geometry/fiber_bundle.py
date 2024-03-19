import pytest

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.group_action import LieAlgebraBasedGroupAction
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class FiberBundleTestCase(TestCase):
    tangent_to_multiple = False

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.total_space)

        if not hasattr(self, "base_data_generator"):
            self.base_data_generator = RandomDataGenerator(self.base)

    def _test_belongs_to_base(self, point, expected, atol):
        res = self.base.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def _test_belongs_to_total_space(self, point, expected, atol):
        res = self.total_space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_riemannian_submersion(self, point, expected, atol):
        res = self.total_space.fiber_bundle.riemannian_submersion(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_riemannian_submersion_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.total_space.fiber_bundle.riemannian_submersion(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_riemannian_submersion_belongs_to_base(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        proj_point = self.total_space.fiber_bundle.riemannian_submersion(point)
        expected = gs.ones(n_points, dtype=bool)

        self._test_belongs_to_base(proj_point, expected, atol)

    def test_lift(self, point, expected, atol):
        res = self.total_space.fiber_bundle.lift(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_lift_vec(self, n_reps, atol):
        point = self.base_data_generator.random_point()
        expected = self.total_space.fiber_bundle.lift(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_lift_belongs_to_total_space(self, n_points, atol):
        point = self.base_data_generator.random_point(n_points)
        lifted_point = self.total_space.fiber_bundle.lift(point)

        expected = gs.ones(n_points, dtype=bool)
        self._test_belongs_to_total_space(lifted_point, expected, atol)

    @pytest.mark.random
    def test_riemannian_submersion_after_lift(self, n_points, atol):
        point = self.base_data_generator.random_point(n_points)
        lifted_point = self.total_space.fiber_bundle.lift(point)
        point_ = self.total_space.fiber_bundle.riemannian_submersion(lifted_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_riemannian_submersion(
        self, tangent_vec, base_point, expected, atol
    ):
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_riemannian_submersion_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            tangent_vec, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        proj_tangent_vector = (
            self.total_space.fiber_bundle.tangent_riemannian_submersion(
                tangent_vec, base_point
            )
        )
        proj_point = self.total_space.fiber_bundle.riemannian_submersion(base_point)

        res = self.base.is_tangent(proj_tangent_vector, proj_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_align(self, point, base_point, expected, atol):
        res = self.total_space.fiber_bundle.align(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_align_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        base_point = self.data_generator.random_point()

        expected = self.total_space.fiber_bundle.align(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_log_after_align_is_horizontal(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        log = self.total_space.metric.log(aligned_point, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(log, base_point, expected, atol)

    def test_horizontal_projection(self, tangent_vec, base_point, expected, atol):
        res = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_horizontal_projection_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_horizontal_projection_is_horizontal(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, base_point, expected, atol)

    def test_vertical_projection(self, tangent_vec, base_point, expected, atol):
        res = self.total_space.fiber_bundle.vertical_projection(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_vertical_projection_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_vertical_projection_is_vertical(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_vertical(vertical, base_point, expected, atol)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_after_vertical_projection(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        vertical = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )
        res = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            vertical, base_point
        )
        expected = gs.zeros_like(res)

        self.assertAllClose(res, expected, atol=atol)

    def test_is_horizontal(self, tangent_vec, base_point, expected, atol):
        res = self.total_space.fiber_bundle.is_horizontal(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_horizontal_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.is_horizontal(
            tangent_vec, base_point, atol=atol
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_is_vertical(self, tangent_vec, base_point, expected, atol):
        res = self.total_space.fiber_bundle.is_vertical(
            tangent_vec, base_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_vertical_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.is_vertical(
            tangent_vec, base_point, atol=atol
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_horizontal_lift(
        self, tangent_vec, expected, atol, base_point=None, fiber_point=None
    ):
        res = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_horizontal_lift_vec(self, n_reps, atol):
        base_point = self.base_data_generator.random_point()
        tangent_vec = self.base_data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_horizontal_lift_is_horizontal(self, n_points, atol):
        base_point = self.base_data_generator.random_point(n_points)
        tangent_vec = self.base_data_generator.random_tangent_vec(base_point)

        fiber_point = self.total_space.fiber_bundle.lift(base_point)
        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, fiber_point, expected, atol)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_after_horizontal_lift(self, n_points, atol):
        base_point = self.base_data_generator.random_point(n_points)
        tangent_vec = self.base_data_generator.random_tangent_vec(base_point)
        fiber_point = self.total_space.fiber_bundle.lift(base_point)

        horizontal = self.total_space.fiber_bundle.horizontal_lift(
            tangent_vec, fiber_point=fiber_point
        )
        tangent_vec_ = self.total_space.fiber_bundle.tangent_riemannian_submersion(
            horizontal, fiber_point
        )

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_integrability_tensor(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_integrability_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.total_space.fiber_bundle.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

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
        atol,
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

    @pytest.mark.vec
    def test_integrability_tensor_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        horizontal_vec_x = self.total_space.fiber_bundle.horizontal_lift(
            self.data_generator.random_tangent_vec(base_point),
            fiber_point=base_point,
        )
        horizontal_vec_y = self.total_space.fiber_bundle.horizontal_lift(
            self.data_generator.random_tangent_vec(base_point),
            fiber_point=base_point,
        )
        nabla_x_y = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_e = self.data_generator.random_tangent_vec(base_point)
        nabla_x_e = self.data_generator.random_tangent_vec(base_point)

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


class GeneralLinearBuresWassersteinBundle(FiberBundle):
    def __init__(self, total_space):
        if not hasattr(total_space, "group_action"):
            total_space.equip_with_group_action(
                LieAlgebraBasedGroupAction(
                    SpecialOrthogonal(total_space.n, equip=False)
                )
            )

        super().__init__(total_space=total_space, aligner=True)

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
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
