import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.test.geometry.base import (
    FiberBundleTestCase,
    LevelSetTestCase,
    ManifoldTestCase,
    _ProjectionTestCaseMixins,
)
from geomstats.test.random import get_random_tangent_vec
from geomstats.test.vectorization import generate_vectorization_data


class DiscreteCurvesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points=1):
        return Matrices(
            self.space.k_sampling_points, self.space.ambient_manifold.dim
        ).random_point(n_points)


class SRVShapeBundleTestCase(DiscreteCurvesTestCase, FiberBundleTestCase):
    # TODO: need to review random_point and tangent_vec (based on existing tests)

    def _get_random_point(self, n_points=1):
        sampling_times = gs.linspace(0.0, 1.0, self.space.k_sampling_points)

        initial_point = self.sphere.random_point(n_points)
        initial_tangent_vec = get_random_tangent_vec(self.sphere, initial_point)

        return self.sphere.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(sampling_times)

    def _get_random_tangent_vec(self, base_point, n_discretized_curves=5):
        n_points = base_point.shape[0] if base_point.ndim > 2 else 1
        point = self._get_random_point(n_points=n_points)

        geo = self.space.total_space_metric.geodesic(
            initial_point=base_point, end_point=point
        )

        times = gs.linspace(0.0, 1.0, n_discretized_curves)

        geod = geo(times)
        # TODO: need to fix geodesic
        if geod.ndim == 4:
            geod = gs.moveaxis(geod, 1, 0)

        return n_discretized_curves * (geod[..., 1, :, :] - geod[..., 0, :, :])

    @pytest.mark.random
    def test_horizontal_projection_is_horizontal(self, n_points, atol):
        # TODO: can we avoid override?
        base_point = self._get_random_point(n_points)
        tangent_vec = self._get_random_tangent_vec(base_point)

        horizontal = self.space.horizontal_projection(tangent_vec, base_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, base_point, expected, atol)

    @pytest.mark.random
    def test_vertical_projection_is_vertical(self, n_points, atol):
        # TODO: can we avoid override?
        base_point = self._get_random_point(n_points)
        tangent_vec = self._get_random_tangent_vec(base_point)

        vertical = self.space.vertical_projection(tangent_vec, base_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_vertical(vertical, base_point, expected, atol)

    @pytest.mark.random
    def test_horizontal_lift_is_horizontal(self, n_points, atol):
        fiber_point = self._get_random_point(n_points)
        tangent_vec = self._get_random_tangent_vec(fiber_point)

        horizontal = self.space.horizontal_lift(tangent_vec, fiber_point=fiber_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, fiber_point, expected, atol)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_after_vertical_projection(
        self, n_points, atol
    ):
        # TODO: can we avoid override?
        base_point = self._get_random_point(n_points)
        tangent_vec = self._get_random_tangent_vec(base_point)

        vertical = self.space.vertical_projection(tangent_vec, base_point)
        res = self.space.tangent_riemannian_submersion(vertical, base_point)
        expected = gs.zeros_like(res)

        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_tangent_vector_projections_orthogonality_with_metric(self, n_points, atol):
        """Test horizontal and vertical projections.

        Check that horizontal and vertical parts of any tangent
        vector are orthogonal with respect to the SRVMetric inner
        product.
        """
        base_point = self._get_random_point(n_points)
        tangent_vec = self._get_random_tangent_vec(base_point)

        tangent_vec_hor = self.space.horizontal_projection(tangent_vec, base_point)
        tangent_vec_ver = self.space.vertical_projection(tangent_vec, base_point)

        res = self.space.total_space_metric.inner_product(
            tangent_vec_hor, tangent_vec_ver, base_point
        )
        expected = gs.zeros(n_points)
        self.assertAllClose(res, expected, atol=atol)

    def test_horizontal_geodesic(
        self, initial_point, end_point, times, expected, atol, threshold=1e-3
    ):
        res = self.space.horizontal_geodesic(
            initial_point, end_point, threshold=threshold
        )(times)

        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_horizontal_geodesic_vec(self, n_reps, n_times, atol):
        times = gs.linspace(0.0, 1.0, n_times)

        initial_point = self.space.random_point()
        end_point = self.space.random_point()

        expected = self.space.horizontal_geodesic(initial_point, end_point)(times)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    times=times,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_horizontal_geodesic_has_horizontal_derivative(
        self, n_points, n_times, atol
    ):
        """Test horizontal geodesic.

        Check that the time derivative of the geodesic is horizontal at all time.
        """
        if n_points > 1:
            raise NotImplementedError("Not implemented for n_points > 1")

        initial_point = self._get_random_point(n_points)
        end_point = self._get_random_point(n_points)

        hor_geo_fun = self.space.horizontal_geodesic(initial_point, end_point)

        n_times = 20
        times = gs.linspace(0, 1, n_times)

        horizontal_geo = hor_geo_fun(times)
        velocity_vec = n_times * (horizontal_geo[1:] - horizontal_geo[:-1])

        _, vertical_norms = self.space.vertical_projection(
            velocity_vec, horizontal_geo[:-1], return_norm=True
        )

        res = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(res, expected, atol=atol)


class ClosedDiscreteCurvesTestCase(LevelSetTestCase):
    def _get_srv_point(self, n_points=1):
        return DiscreteCurves(
            ambient_manifold=self.space.ambient_manifold,
            k_sampling_points=self.space.k_sampling_points,
        ).random_point(n_points)

    def _is_planar(self):
        is_euclidean = isinstance(self.space.ambient_manifold, Euclidean)
        return is_euclidean and self.space.ambient_manifold.dim == 2

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        if not self._is_planar():
            return

        super().test_projection_vec(n_reps, atol)

    @pytest.mark.random
    def test_projection_belongs(self, n_points, atol):
        if not self._is_planar():
            return

        super().test_projection_belongs(n_points, atol)

    def test_srv_projection(self, srv, expected, atol, max_iter=1000):
        res = self.space.srv_projection(srv, atol=atol, max_iter=max_iter)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_srv_projection_vec(self, n_reps, atol, max_iter=1000):
        if not self._is_planar():
            return

        srv = self._get_srv_point()
        expected = self.space.srv_projection(srv, atol=atol, max_iter=max_iter)

        vec_data = generate_vectorization_data(
            data=[dict(srv=srv, expected=expected, max_iter=max_iter, atol=atol)],
            arg_names=["srv"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_projection_is_itself(self, n_points, atol):
        # TODO: should this go up?
        if not self._is_planar():
            return

        point = self.space.random_point(n_points)
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, point, atol=atol)

    @pytest.mark.random
    def test_random_point_is_closed(self, n_points, atol):
        point = self.space.random_point(n_points)
        self.assertAllClose(point[..., 0, :], point[..., -1, :], atol=atol)
