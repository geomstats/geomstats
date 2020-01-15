import unittest
import numpy as np

import geomstats.backend as gs
assert_allclose = gs.testing.assert_allclose
from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.learning.pca import TangentPCA


SO3 = SpecialOrthogonalGroup(n=3)
metric = SO3.bi_invariant_metric
N_SAMPLES = 10
N_COMPONENTS = 2


# XXX: Should these tests run on all backends?

class TestTangentPCA(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.X = SO3.random_uniform(n_samples=N_SAMPLES)

    def test_tangent_pca_error(self):
        X = self.X
        trans = TangentPCA(metric, n_components=N_COMPONENTS)
        trans.fit(X)
        X_diff_size = np.ones((10, X.shape[1] + 1))
        self.assertRaises(trans.transform(X_diff_size), ValueError)

    def test_tangent_pca(self):
        X = self.X
        trans = TangentPCA(metric, n_components=N_COMPONENTS)
        self.assertEquals(trans.demo_param, 'demo')

        trans.fit(X)
        self.assertEquals(trans.n_features_, X.shape[1])

        X_trans = trans.transform(X)
        assert_allclose(X_trans, np.sqrt(X))

        X_trans = trans.fit_transform(X)
        assert_allclose(X_trans, np.sqrt(X))
