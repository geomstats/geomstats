"""Unit tests for pre-processing transformers."""

import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.preprocessing import ToTangentSpace


class TestToTangentSpace(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(123)
        self.sphere = Hypersphere(dim=4)
        self.hyperbolic = Hyperboloid(dim=3)
        self.euclidean = Euclidean(dim=2)
        self.minkowski = Minkowski(dim=2)
        self.so3 = SpecialOrthogonal(n=3, point_type='vector')
        self.so_matrix = SpecialOrthogonal(n=3, point_type='matrix')

    def test_estimate_transform_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
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

    @geomstats.tests.np_and_tf_only
    def test_estimate_transform_so_group(self):
        point = self.so_matrix.random_uniform()
        points = gs.array([point, point])

        transformer = ToTangentSpace(geometry=self.so_matrix)
        transformer.fit(X=points)
        result = transformer.transform(points)
        expected = gs.zeros((2, 6))
        self.assertAllClose(expected, result)

    def test_estimate_transform_spd(self):
        point = spd.SPDMatrices(3).random_uniform()
        points = gs.stack([point, point])
        transformer = ToTangentSpace(geometry=spd.SPDMetricAffine(3))
        transformer.fit(X=points)
        result = transformer.transform(points)
        expected = gs.zeros((2, 6))
        self.assertAllClose(expected, result, atol=1e-5)

    def test_fit_transform_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        transformer = ToTangentSpace(geometry=self.hyperbolic.metric)
        result = transformer.fit_transform(X=points)
        expected = gs.zeros_like(points)
        self.assertAllClose(expected, result)

    def test_inverse_transform_hyperbolic(self):
        points = self.hyperbolic.random_uniform(10)
        transformer = ToTangentSpace(geometry=self.hyperbolic.metric)
        X = transformer.fit_transform(X=points)
        result = transformer.inverse_transform(X)
        expected = points
        self.assertAllClose(expected, result)

    def test_inverse_transform_spd(self):
        point = spd.SPDMatrices(3).random_uniform(10)
        transformer = ToTangentSpace(geometry=spd.SPDMetricLogEuclidean(3))
        X = transformer.fit_transform(X=point)
        result = transformer.inverse_transform(X)
        expected = point
        self.assertAllClose(expected, result, atol=1e-4)

        transformer = ToTangentSpace(geometry=spd.SPDMetricAffine(3))
        X = transformer.fit_transform(X=point)
        result = transformer.inverse_transform(X)
        expected = point
        self.assertAllClose(expected, result, atol=1e-4)

    @geomstats.tests.np_only
    def test_inverse_transform_so(self):
        # FIXME: einsum vectorization error for invariant_metric log in tf
        point = self.so_matrix.random_uniform(10)
        transformer = ToTangentSpace(
            geometry=self.so_matrix.bi_invariant_metric)
        X = transformer.transform(X=point, base_point=self.so_matrix.identity)
        result = transformer.inverse_transform(
            X, base_point=self.so_matrix.identity)
        expected = point
        self.assertAllClose(expected, result)
