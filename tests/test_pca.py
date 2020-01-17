"""Unit tests for Tangent PCA."""


import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.learning.pca import TangentPCA


class TestTangentPCA(geomstats.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.so3 = SpecialOrthogonalGroup(n=3)
        self.n_samples = 10

        self.X = self.so3.random_uniform(n_samples=self.n_samples)
        self.metric = self.so3.bi_invariant_metric
        self.n_components = 2

    def test_tangent_pca_error(self):
        X = self.X
        trans = TangentPCA(self.metric, n_components=self.n_components)
        trans.fit(X)
        X_diff_size = gs.ones((self.n_samples, gs.shape(X)[1] + 1))
        self.assertRaises(trans.transform(X_diff_size), ValueError)

    def test_tangent_pca(self):
        X = self.X
        trans = TangentPCA(self.metric, n_components=self.n_components)

        trans.fit(X)
        self.assertEquals(trans.n_features_, gs.shape(X)[1])

        X_trans = trans.transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))

        X_trans = trans.fit_transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))
