from geomstats.geometry.matrices import Matrices
from geomstats.test.geometry.base import ManifoldTestCase, _ProjectionTestCaseMixins


class DiscreteCurvesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points=1):
        return Matrices(
            self.space.k_sampling_points, self.space.ambient_manifold.dim
        ).random_point(n_points)
