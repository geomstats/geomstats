import pytest
import numpy as np

from sklearn.utils.testing import assert_allclose

from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.learning.pca import TangentPCA


SO3_GROUP = SpecialOrthogonalGroup(n=3)
METRIC = SO3_GROUP.bi_invariant_metric
N_SAMPLES = 10
N_COMPONENTS = 2


@pytest.fixture
def data():
    data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)
    return data


def test_tangent_pca_error(data):
    X = data
    trans = TangentPCA(n_components=N_COMPONENTS)
    trans.fit(X)
    with pytest.raises(ValueError, match="Shape of input is different"):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        trans.transform(X_diff_size)


def test_tangent_pca(data):
    X = data
    trans = TangentPCA(n_components=N_COMPONENTS)
    assert trans.demo_param == 'demo'

    trans.fit(X)
    assert trans.n_features_ == X.shape[1]

    X_trans = trans.transform(X)
    assert_allclose(X_trans, np.sqrt(X))

    X_trans = trans.fit_transform(X)
    assert_allclose(X_trans, np.sqrt(X))
