import pytest

import geomstats.backend as gs
from geomstats.test.random import NFoldManifoldRandomDataGenerator, RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class NFoldManifoldTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = NFoldManifoldRandomDataGenerator(self.space)


class NFoldMetricTestCase(RiemannianMetricTestCase):
    pass


class NFoldMetricScalesTestCase(TestCase):
    def setup_method(self):
        if self.space.n_copies != 1:
            raise ValueError("Test case works only for n_copies=1")

        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.other_space)

    @pytest.mark.random
    def test_inner_product(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        axis = -(self.other_space.point_ndim + 1)
        tangent_vec_a_ = gs.expand_dims(tangent_vec_a, axis=axis)
        tangent_vec_b_ = gs.expand_dims(tangent_vec_b, axis=axis)
        base_point_ = gs.expand_dims(base_point, axis=axis)

        res = self.space.metric.inner_product(
            tangent_vec_a_, tangent_vec_b_, base_point_
        )
        other_res = self.other_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, other_res, atol=atol)
