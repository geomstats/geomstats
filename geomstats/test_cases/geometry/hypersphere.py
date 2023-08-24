import random

import pytest
import scipy

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.test.random import (
    EmbeddedSpaceRandomDataGenerator,
    HypersphereIntrinsicRandomDataGenerator,
)
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.vectorization import get_batch_shape


def _estimate_von_mises_kappa(X, n_steps=100):
    n_points = X.shape[0]
    dim = X.shape[1] - 1

    sum_points = gs.sum(X, axis=0)
    mean_norm = gs.linalg.norm(sum_points) / n_points

    kappa_estimate = mean_norm * (dim + 1.0 - mean_norm**2) / (1.0 - mean_norm**2)
    kappa_estimate = gs.cast(kappa_estimate, gs.float64)
    p = dim + 1
    for _ in range(n_steps):
        bessel_func_1 = scipy.special.iv(p / 2.0, kappa_estimate)
        bessel_func_2 = scipy.special.iv(p / 2.0 - 1.0, kappa_estimate)
        ratio = bessel_func_1 / bessel_func_2
        denominator = 1.0 - ratio**2 - (p - 1.0) * ratio / kappa_estimate
        kappa_estimate -= (ratio - mean_norm) / denominator

    return kappa_estimate


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
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.extrinsic_data_generator = EmbeddedSpaceRandomDataGenerator(self.space)
            self.intrinsic_data_generator = HypersphereIntrinsicRandomDataGenerator(
                self.space_intrinsic
            )

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
        point_intrinsic = self.intrinsic_data_generator.random_point()

        try:
            expected = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)
        except NotImplementedError:
            return

        vec_data = generate_vectorization_data(
            data=[dict(point_intrinsic=point_intrinsic, expected=expected, atol=atol)],
            arg_names=["point_intrinsic"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_intrinsic_to_extrinsic_coords_belongs(self, n_points, atol):
        point_intrinsic = self.intrinsic_data_generator.random_point(n_points)

        try:
            point_extrinsic = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)
        except NotImplementedError:
            return

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
        point_extrinsic = self.extrinsic_data_generator.random_point()

        try:
            expected = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)
        except NotImplementedError:
            return

        vec_data = generate_vectorization_data(
            data=[dict(point_extrinsic=point_extrinsic, expected=expected, atol=atol)],
            arg_names=["point_extrinsic"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_extrinsic_to_intrinsic_coords_belongs(self, n_points):
        point_extrinsic = self.extrinsic_data_generator.random_point(n_points)

        try:
            point_intrinsic = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)
        except NotImplementedError:
            return

        expected = gs.ones(n_points, dtype=bool)
        self._test_belongs_intrinsic(point_intrinsic, expected)

    @pytest.mark.random
    def test_intrinsic_to_extrinsic_coords_after_extrinsic_to_intrinsic(
        self, n_points, atol
    ):
        point_extrinsic = self.extrinsic_data_generator.random_point(n_points)

        try:
            point_intrinsic = self.space.extrinsic_to_intrinsic_coords(point_extrinsic)
        except NotImplementedError:
            return

        point_extrinsic_ = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)

        self.assertAllClose(point_extrinsic_, point_extrinsic, atol=atol)

    @pytest.mark.random
    def test_extrinsic_to_intrinsic_coords_after_intrinsic_to_extrinsic_coords(
        self, n_points, atol
    ):
        point_intrinsic = self.intrinsic_data_generator.random_point(n_points)

        try:
            point_extrinsic = self.space.intrinsic_to_extrinsic_coords(point_intrinsic)
        except NotImplementedError:
            return

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
        base_point_spherical = self.intrinsic_data_generator.random_point()
        tangent_vec_spherical = self.intrinsic_data_generator.random_tangent_vec(
            base_point_spherical
        )

        try:
            expected = self.space.tangent_spherical_to_extrinsic(
                tangent_vec_spherical, base_point_spherical
            )
        except NotImplementedError:
            return

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
        base_point_spherical = self.intrinsic_data_generator.random_point(n_points)
        tangent_vec_spherical = self.intrinsic_data_generator.random_tangent_vec(
            base_point_spherical
        )

        try:
            tangent_vec = self.space.tangent_spherical_to_extrinsic(
                tangent_vec_spherical,
                base_point_spherical,
            )
        except NotImplementedError:
            return

        base_point = self.space.intrinsic_to_extrinsic_coords(base_point_spherical)

        res = self.space_extrinsic.is_tangent(tangent_vec, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_tangent_extrinsic_to_spherical(
        self, tangent_vec, base_point, expected, atol
    ):
        res = self.space.tangent_extrinsic_to_spherical(
            tangent_vec,
            base_point=base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_extrinsic_to_spherical_vec(self, n_reps, atol):
        base_point = self.extrinsic_data_generator.random_point()
        tangent_vec = self.extrinsic_data_generator.random_tangent_vec(base_point)

        try:
            expected = self.space.tangent_extrinsic_to_spherical(
                tangent_vec,
                base_point=base_point,
            )
        except NotImplementedError:
            return

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
        base_point = self.extrinsic_data_generator.random_point(n_points)
        tangent_vec = self.extrinsic_data_generator.random_tangent_vec(base_point)

        try:
            tangent_vec_spherical = self.space.tangent_extrinsic_to_spherical(
                tangent_vec,
                base_point=base_point,
            )
        except NotImplementedError:
            return

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
        base_point_spherical = self.intrinsic_data_generator.random_point(n_points)
        tangent_vec_spherical = self.intrinsic_data_generator.random_tangent_vec(
            base_point_spherical
        )

        try:
            tangent_vec = self.space.tangent_spherical_to_extrinsic(
                tangent_vec_spherical, base_point_spherical=base_point_spherical
            )
        except NotImplementedError:
            return

        base_point = self.space.intrinsic_to_extrinsic_coords(base_point_spherical)

        tangent_vec_spherical_ = self.space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point
        )

        self.assertAllClose(tangent_vec_spherical_, tangent_vec_spherical, atol=atol)

    @pytest.mark.random
    def test_tangent_spherical_to_extrinsic_after_tangent_extrinsic_to_spherical(
        self, n_points, atol
    ):
        base_point = self.extrinsic_data_generator.random_point(n_points)
        tangent_vec = self.extrinsic_data_generator.random_tangent_vec(base_point)

        try:
            tangent_vec_spherical = self.space.tangent_extrinsic_to_spherical(
                tangent_vec, base_point
            )
        except NotImplementedError:
            return

        base_point_spherical = self.space.extrinsic_to_intrinsic_coords(base_point)

        tangent_vec_ = self.space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical=base_point_spherical
        )

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)


class HypersphereExtrinsicTestCase(LevelSetTestCase):
    def _get_random_kappa(self, size=1):
        sample = gs.random.uniform(low=1.0, high=10000.0, size=(size,))
        if size == 1:
            return sample[0]
        return sample

    def _get_random_precision(self, precision_type="array"):
        if precision_type is None:
            return None

        precision = gs.random.uniform(low=1.0, high=10.0, size=(1,))[0]
        if precision_type is float:
            return precision

        return precision * gs.eye(self.space.dim)

    def _fit_frechet_mean(self, sample):
        if self.space.dim == 1:
            estimator = FrechetMean(self.space)
        else:
            estimator = FrechetMean(self.space, method="adaptive")

        estimator.fit(sample)

        return estimator.estimate_

    @pytest.mark.random
    def test_replace_values(self, n_points, atol):
        points = self.data_generator.random_point(n_points)
        new_points = self.data_generator.random_point(n_points)

        n_indices = random.randint(1, n_points)
        indices_int = [
            (index,) for index in random.sample(range(0, n_points), n_indices)
        ]
        if n_indices:
            indices = gs.array_from_sparse(
                indices_int, [True for _ in range(n_indices)], (n_points,)
            )
        else:
            indices = gs.zeros(n_points, dtype=bool)

        replaced_points = self.space._replace_values(
            points, new_points[indices], indices
        )

        self.assertAllClose(replaced_points[indices], new_points[indices], atol=atol)

    @pytest.mark.random
    def test_random_von_mises_fisher_sample_mean(
        self, n_samples, atol, random_mu=True, max_iter=100
    ):
        if random_mu:
            mu = self.space.random_point()
            expected = mu
        else:
            mu = None
            expected = gs.array([1.0] + [0.0] * self.space.dim)

        kappa = self._get_random_kappa()

        try:
            point = self.space.random_von_mises_fisher(
                n_samples=n_samples, mu=mu, kappa=kappa, max_iter=max_iter
            )
        except NotImplementedError:
            return

        sum_point = gs.sum(point, axis=0)
        res = sum_point / gs.linalg.norm(sum_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.validation
    def test_random_von_mises_fisher_sample_kappa(
        self,
        n_samples,
        atol,
        max_iter=100,
    ):
        kappa = 1.0

        try:
            X = self.space.random_von_mises_fisher(
                n_samples=n_samples, kappa=kappa, max_iter=max_iter
            )
        except NotImplementedError:
            return

        kappa_estimate = _estimate_von_mises_kappa(X, n_steps=100)
        self.assertAllClose(kappa_estimate, gs.array(kappa), atol=atol)

    @pytest.mark.random
    def test_random_von_mises_fisher_belongs(
        self, n_points, random_mu, atol, max_iter=100
    ):
        mu = self.space.random_point() if random_mu else None
        kappa = self._get_random_kappa()

        try:
            point = self.space.random_von_mises_fisher(
                n_samples=n_points, mu=mu, kappa=kappa, max_iter=max_iter
            )
        except NotImplementedError:
            return

        expected = gs.ones(n_points, dtype=bool)
        self.test_belongs(point, expected, atol)

    @pytest.mark.shape
    def test_random_von_mises_fisher_shape(self, n_points, random_mu, max_iter=100):
        mu = self.space.random_point() if random_mu else None
        kappa = self._get_random_kappa()

        try:
            point = self.space.random_von_mises_fisher(
                n_samples=n_points, mu=mu, kappa=kappa, max_iter=max_iter
            )
        except NotImplementedError:
            return

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], self.space.shape)

        if n_points > 1:
            self.assertEqual(gs.shape(point)[0], n_points)

    @pytest.mark.random
    def test_random_riemannian_normal_belongs(
        self, n_samples, random_mean, precision_type, atol, max_iter=100
    ):
        mean = self.space.random_point() if random_mean else None
        precision = self._get_random_precision(precision_type)

        sample = self.space.random_riemannian_normal(
            mean=mean, precision=precision, n_samples=n_samples, max_iter=max_iter
        )

        expected = gs.ones(n_samples, dtype=bool)
        self.test_belongs(sample, expected, atol)

    @pytest.mark.shape
    def test_random_riemannian_normal_shape(
        self, n_samples, random_mean, precision_type, atol, max_iter=100
    ):
        mean = self.space.random_point() if random_mean else None
        precision = self._get_random_precision(precision_type)

        point = self.space.random_riemannian_normal(
            mean=mean, precision=precision, n_samples=n_samples, max_iter=max_iter
        )

        expected_ndim = self.space.point_ndim + int(n_samples > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], self.space.shape)

        if n_samples > 1:
            self.assertEqual(gs.shape(point)[0], n_samples)

    @pytest.mark.random
    def test_random_riemannian_normal_frechet_mean(
        self, n_samples, random_mean, atol, max_iter=100
    ):
        if random_mean:
            expected = mean = self.space.random_point()
        else:
            mean = None
            expected = gs.array([0.0] * self.space.dim + [1.0])  # north pole

        precision = self._get_random_precision()

        sample = self.space.random_riemannian_normal(
            mean=mean, precision=precision, n_samples=n_samples, max_iter=max_iter
        )
        estimate_ = self._fit_frechet_mean(sample)

        self.assertAllClose(estimate_, expected, atol=atol)


class HypersphereIntrinsicTestCase(ManifoldTestCase):
    # TODO: update after refactoring

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = HypersphereIntrinsicRandomDataGenerator(self.space)

    @pytest.mark.random
    def test_random_point_belongs(self, n_points, atol):
        point = self.space.random_point(n_points)
        expected = gs.ones(n_points, dtype=bool)

        res = _belongs_intrinsic(self.space, point)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.shape
    def test_random_point_shape(self, n_points):
        point = self.data_generator.random_point(n_points)

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
        point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(point)

        res = _is_tangent_intrinsic(self.space, tangent_vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=tangent_vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
