import pytest

import geomstats.backend as gs
from geomstats.test.geometry.base import LevelSetTestCase, ManifoldTestCase
from geomstats.test.random import get_random_tangent_vec
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.vectorization import get_batch_shape


def _belongs_intrinsic(space, point, atol=gs.atol):
    # TODO: use this concept in other places
    shape = point.shape[: -space.point_ndim]

    if point.shape[-1] == space.dim:
        return gs.ones(shape, dtype=bool)

    return gs.zeros(shape, dtype=bool)


def _is_tangent_intrinsic(space, tangent_vec, point, atol=gs.atol):
    shape = get_batch_shape(space, point, tangent_vec)
    if tangent_vec.shape[-1] == space.dim:
        return gs.ones(shape, dtype=bool)

    return gs.zeros(shape, dtype=bool)


class HypersphereCoordsTransformTestCase(TestCase):
    def _test_belongs_intrinsic(self, point, expected):
        res = _belongs_intrinsic(self.space_intrinsic, point)
        self.assertAllEqual(res, expected)

    def test_intrinsic_to_extrinsic_coords(self, point_intrinsic, expected, atol):
        res = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)
        self.assertAllClose(res, expected, atol=atol)
        self.assertEqual(
            point_intrinsic.ndim,
            res.ndim,
            msg=f"`point_intrinsic.shape` is {point_intrinsic.shape} and "
            f"`point_extrinsic.shape` is {res.shape}",
        )

    @pytest.mark.vec
    def test_intrinsic_to_extrinsic_coords_vec(self, n_reps, atol):
        point_intrinsic = self.space_intrinsic.random_point()
        expected = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)

        vec_data = generate_vectorization_data(
            data=[dict(point_intrinsic=point_intrinsic, expected=expected, atol=atol)],
            arg_names=["point_intrinsic"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_intrinsic_to_extrinsic_coords_belongs(self, n_points, atol):
        point_intrinsic = self.space_intrinsic.random_point(n_points)
        point_extrinsic = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)

        res = self.space_extrinsic.belongs(point_extrinsic, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_extrinsic_to_intrinsic_coords(self, point_extrinsic, expected, atol):
        res = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)
        self.assertAllClose(res, expected, atol=atol)
        self.assertEqual(
            point_extrinsic.ndim,
            res.ndim,
            msg=f"`point_extrinsic.shape` is {point_extrinsic.shape} and "
            f"`point_intrinsic.shape` is {res.shape}",
        )

    @pytest.mark.vec
    def test_extrinsic_to_intrinsic_coords_vec(self, n_reps, atol):
        point_extrinsic = self.space_extrinsic.random_point()
        expected = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)

        vec_data = generate_vectorization_data(
            data=[dict(point_extrinsic=point_extrinsic, expected=expected, atol=atol)],
            arg_names=["point_extrinsic"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_extrinsic_to_intrinsic_coords_belongs(self, n_points):
        point_extrinsic = self.space_extrinsic.random_point(n_points)
        point_intrinsic = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)

        expected = gs.ones(n_points, dtype=bool)
        self._test_belongs_intrinsic(point_intrinsic, expected)

    @pytest.mark.random
    def test_intrinsic_to_extrinsic_coords_after_extrinsic_to_intrinsic(
        self, n_points, atol
    ):
        point_extrinsic = self.space_extrinsic.random_point(n_points)
        point_intrinsic = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)
        point_extrinsic_ = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)

        self.assertAllClose(point_extrinsic_, point_extrinsic, atol=atol)

    @pytest.mark.random
    def test_extrinsic_to_intrinsic_coords_after_intrinsic_to_extrinsic_coords(
        self, n_points, atol
    ):
        point_intrinsic = self.space_intrinsic.random_point(n_points)
        point_extrinsic = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)
        point_intrinsic_ = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)

        self.assertAllClose(point_intrinsic_, point_intrinsic, atol=atol)

    def test_tangent_spherical_to_extrinsic(
        self,
        tangent_vec_spherical,
        base_point_spherical,
        expected,
        atol,
    ):

        res = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_spherical_to_extrinsic_vec(self, n_reps, atol):
        if self.space.dim != 2:
            # TODO: check it raises not implemented (also in other places)?
            return

        base_point_spherical = self.space_intrinsic.random_point()
        tangent_vec_spherical = get_random_tangent_vec(
            self.space_intrinsic, base_point_spherical
        )

        expected = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_spherical=tangent_vec_spherical,
                    base_point_spherical=base_point_spherical,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_spherical", "base_point_spherical"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_spherical_to_extrinsic_is_tangent(self, n_points, atol):
        if self.space.dim != 2:
            return

        base_point_spherical = self.space_intrinsic.random_point(n_points)
        tangent_vec_spherical = get_random_tangent_vec(
            self.space_intrinsic, base_point_spherical
        )

        tangent_vec = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical,
            base_point_spherical,
        )
        base_point = self.space.intrinsic_to_extrinsic_coords(base_point_spherical)

        res = self.space_extrinsic.is_tangent(tangent_vec, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_tangent_extrinsic_to_spherical(
        self, tangent_vec, base_point, expected, atol
    ):
        if self.space.dim != 2:
            return

        res = self.space.tangent_extrinsic_to_spherical(
            tangent_vec,
            base_point=base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_extrinsic_to_spherical_vec(self, n_reps, atol):
        if self.space.dim != 2:
            return

        base_point = self.space_extrinsic.random_point()
        tangent_vec = get_random_tangent_vec(self.space, base_point)

        expected = self.space.tangent_extrinsic_to_spherical(
            tangent_vec,
            base_point=base_point,
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
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_extrinsic_to_spherical_is_tangent(self, n_points, atol):
        if self.space.dim != 2:
            return

        base_point = self.space_extrinsic.random_point(n_points)
        tangent_vec = get_random_tangent_vec(self.space_extrinsic, base_point)

        tangent_vec_spherical = self.space.tangent_extrinsic_to_spherical(
            tangent_vec,
            base_point=base_point,
        )
        base_point_spherical = self.space.extrinsic_to_intrinsic_coords(base_point)

        res = _is_tangent_intrinsic(
            self.space_intrinsic, tangent_vec_spherical, base_point_spherical
        )
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_tangent_extrinsic_to_spherical_after_tangent_spherical_to_extrinsic(
        self, n_points, atol
    ):
        if self.space.dim != 2:
            return

        base_point_spherical = self.space_intrinsic.random_point(n_points)
        tangent_vec_spherical = get_random_tangent_vec(
            self.space_intrinsic, base_point_spherical
        )

        tangent_vec = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical=base_point_spherical
        )
        base_point = self.space.intrinsic_to_extrinsic_coords(base_point_spherical)

        tangent_vec_spherical_ = self.space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point
        )

        self.assertAllClose(tangent_vec_spherical_, tangent_vec_spherical, atol=atol)

    @pytest.mark.random
    def test_tangent_spherical_to_extrinsic_after_tangent_extrinsic_to_spherical(
        self, n_points, atol
    ):
        if self.space.dim != 2:
            return

        base_point = self.space_extrinsic.random_point(n_points)
        tangent_vec = get_random_tangent_vec(self.space_extrinsic, base_point)

        tangent_vec_spherical = self.space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point
        )
        base_point_spherical = self.space.extrinsic_to_intrinsic_coords(base_point)

        tangent_vec_ = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical=base_point_spherical
        )

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)


class _HypersphereTestCaseMixins:
    pass


class HypersphereExtrinsicTestCase(_HypersphereTestCaseMixins, LevelSetTestCase):
    pass


class HypersphereIntrinsicTestCase(_HypersphereTestCaseMixins, ManifoldTestCase):
    # TODO: update after refactoring

    @pytest.mark.random
    def test_random_point_belongs(self, n_points, atol):
        point = self.space.random_point(n_points)
        expected = gs.ones(n_points, dtype=bool)

        res = _belongs_intrinsic(self.space, point)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.shape
    def test_random_point_shape(self, n_points):
        point = self.space.random_point(n_points)

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], (self.space.dim))

        if n_points > 1:
            self.assertEqual(gs.shape(point)[0], n_points)

    def test_is_tangent(self, vector, base_point, expected, atol):
        res = _is_tangent_intrinsic(self.space, vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_tangent_vec(self, n_reps, atol):
        point = self.space.random_point()
        tangent_vec = get_random_tangent_vec(self.space, point)

        res = _is_tangent_intrinsic(self.space, tangent_vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=tangent_vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
