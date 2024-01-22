import pytest

import geomstats.backend as gs
from geomstats.test.random import get_random_times
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.nfold_manifold import NFoldManifoldTestCase
from geomstats.vectorization import get_batch_shape


class DiscreteCurvesStartingAtOriginTestCase(NFoldManifoldTestCase):
    def test_interpolate(self, point, param, expected, atol):
        points = self.space.interpolate(point)(param)
        self.assertAllClose(points, expected, atol=atol)

    def test_interpolate_vec(self, n_reps, n_times, atol):
        point = self.data_generator.random_point()
        param = get_random_times(n_times)

        expected = self.space.interpolate(point)(param)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    param=param,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_length(self, point, expected, atol):
        length = self.space.length(point)
        self.assertAllClose(length, expected, atol=atol)

    def test_normalize(self, point, expected, atol):
        normalized_point = self.space.normalize(point)
        self.assertAllClose(normalized_point, expected, atol)

    @pytest.mark.random
    def test_normalize_is_unit_length(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        normalized_point = self.space.normalize(point)
        normalize_lengths = self.space.length(normalized_point)
        self.assertAllClose(
            normalize_lengths, gs.ones_like(normalize_lengths), atol=atol
        )


class SRVReparametrizationBundleTestCase(FiberBundleTestCase):
    @pytest.mark.random
    def test_tangent_vector_projections_orthogonality_with_metric(self, n_points, atol):
        """Test horizontal and vertical projections.

        Check that horizontal and vertical parts of any tangent
        vector are orthogonal with respect to the SRVMetric inner
        product.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_hor = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        tangent_vec_ver = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )

        res = self.total_space.metric.inner_product(
            tangent_vec_hor, tangent_vec_ver, base_point
        )
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
        self.assertAllClose(res, expected, atol=atol)
