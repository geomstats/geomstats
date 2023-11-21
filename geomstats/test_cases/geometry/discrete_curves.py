import pytest

import geomstats.backend as gs
from geomstats.test.random import ShapeBundleRandomDataGenerator
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.vectorization import get_batch_shape


class SRVTranslationReparametrizationBundleTestCase(FiberBundleTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            n_discretized_curves = (
                5
                if not hasattr(self, "n_discretized_curves")
                else self.n_discretized_curves
            )
            self.data_generator = ShapeBundleRandomDataGenerator(
                self.total_space,
                n_discretized_curves=n_discretized_curves,
            )

    @pytest.mark.random
    def test_tangent_vector_projections_orthogonality_with_metric(self, n_points, atol):
        """Test horizontal and vertical projections.

        Check that horizontal and vertical parts of any tangent
        vector are orthogonal with respect to the SRVMetric inner
        product.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_hor = self.bundle.horizontal_projection(tangent_vec, base_point)
        tangent_vec_ver = self.bundle.vertical_projection(tangent_vec, base_point)

        res = self.total_space.metric.inner_product(
            tangent_vec_hor, tangent_vec_ver, base_point
        )
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
        self.assertAllClose(res, expected, atol=atol)
