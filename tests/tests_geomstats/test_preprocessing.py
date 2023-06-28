"""Unit tests for pre-processing transformers."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDLogEuclideanMetric, SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.preprocessing import ToTangentSpace


class TestToTangentSpace(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        gs.random.seed(123)
        self.sphere = Hypersphere(dim=4)
        self.hyperbolic = Hyperboloid(dim=3)
        self.euclidean = Euclidean(dim=2)
        self.minkowski = Minkowski(dim=2)
        self.so3 = SpecialOrthogonal(n=3, point_type="vector")
        self.so_matrix = SpecialOrthogonal(n=3, point_type="matrix")

    def test_estimate_transform_sphere(self):
        point = gs.array([0.0, 0.0, 0.0, 0.0, 1.0])
        points = gs.array([point, point])
        transformer = ToTangentSpace(geometry=self.sphere)
        transformer.fit(X=points)
        result = transformer.transform(points)
        expected = gs.zeros_like(points)
        self.assertAllClose(expected, result)

    def test_inverse_transform_no_fit_sphere(self):
        point = self.sphere.random_uniform(3)
        base_point = point[0]
        point = point[1:]
        transformer = ToTangentSpace(geometry=self.sphere)
        X = transformer.transform(point, base_point=base_point)
        result = transformer.inverse_transform(X, base_point=base_point)
        expected = point
        self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_estimate_transform_so_group(self):
        point = self.so_matrix.random_uniform()
        points = gs.array([point, point])

        transformer = ToTangentSpace(geometry=self.so_matrix)
        transformer.fit(X=points)
        result = transformer.transform(points)
        expected = gs.zeros((2, 6))
        self.assertAllClose(expected, result)

    def test_estimate_transform_spd(self):
        space = SPDMatrices(3)
        point = space.random_point()
        points = gs.stack([point, point])
        transformer = ToTangentSpace(geometry=space)
        transformer.fit(X=points)
        result = transformer.transform(points)
        expected = gs.zeros((2, 6))
        self.assertAllClose(expected, result, atol=1e-5)

    def test_fit_transform_hyperbolic(self):
        point = gs.array([2.0, 1.0, 1.0, 1.0])
        points = gs.array([point, point])
        transformer = ToTangentSpace(geometry=self.hyperbolic)
        result = transformer.fit_transform(X=points)
        expected = gs.zeros_like(points)
        self.assertAllClose(expected, result)

    def test_inverse_transform_hyperbolic(self):
        points = self.hyperbolic.random_point(10)
        transformer = ToTangentSpace(geometry=self.hyperbolic)
        X = transformer.fit_transform(X=points)
        result = transformer.inverse_transform(X)
        expected = points
        self.assertAllClose(expected, result)

    def test_inverse_transform_spd(self):
        space = SPDMatrices(3, equip=False)
        space.equip_with_metric(SPDLogEuclideanMetric)
        point = space.random_point(10)
        transformer = ToTangentSpace(geometry=space)
        X = transformer.fit_transform(X=point)
        result = transformer.inverse_transform(X)
        expected = point
        self.assertAllClose(expected, result, atol=1e-4)

        space = SPDMatrices(3, equip=True)
        transformer = ToTangentSpace(geometry=space)
        X = transformer.fit_transform(X=point)
        result = transformer.inverse_transform(X)
        expected = point
        self.assertAllClose(expected, result, atol=1e-4)

    @tests.conftest.np_and_autograd_only
    def test_inverse_transform_so(self):
        point = self.so_matrix.random_uniform(10)
        transformer = ToTangentSpace(geometry=self.so_matrix)
        X = transformer.transform(X=point, base_point=self.so_matrix.identity)
        result = transformer.inverse_transform(X, base_point=self.so_matrix.identity)
        expected = point
        self.assertAllClose(expected, result)
