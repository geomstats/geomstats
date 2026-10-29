import numpy as np
import pytest
from scipy.linalg import expm

import geomstats.backend as gs
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.spd_matrices import SPDLogEuclideanMetric, SPDMatrices


@pytest.mark.parametrize(
    "mean",
    [[[2.0, 0.5], [0.5, 1.0]], [[0.5, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]],
)
@pytest.mark.parametrize("n_samples", [1, 4])
def test_affine_lognormal_samples_match_centered_matrix_normal(mean, n_samples):
    """Whitened logarithms follow the supplied zero-mean normal law."""
    mean = gs.array(mean)
    cov = gs.array([[0.4, 0.1, 0.0], [0.1, 0.3, 0.05], [0.0, 0.05, 0.2]])
    original_mean, original_cov = gs.copy(mean), gs.copy(cov)
    gs.random.seed(216)
    vectors = gs.to_numpy(gs.random.multivariate_normal(gs.zeros(3), cov, (n_samples,)))
    mean_np = gs.to_numpy(mean)
    values, basis = np.linalg.eigh(mean_np)
    half_mean = (basis * np.sqrt(values)) @ basis.T
    expected = []
    for a, b, c in vectors:
        tangent = np.array([[a, c / np.sqrt(2.0)], [c / np.sqrt(2.0), b]])
        expected.append(half_mean @ expm(tangent) @ half_mean)

    space = SPDMatrices(2)
    gs.random.seed(216)
    actual = LogNormal(space, mean, cov).sample(n_samples)

    np.testing.assert_allclose(gs.to_numpy(actual), expected, rtol=1e-10, atol=1e-10)
    assert actual.shape == (n_samples, 2, 2)
    assert gs.all(space.belongs(actual))
    np.testing.assert_allclose(mean, original_mean)
    np.testing.assert_allclose(cov, original_cov)


@pytest.mark.parametrize("metric", [None, SPDLogEuclideanMetric])
def test_spd_lognormal_concentrates_at_its_mean(metric):
    """Both supported metrics concentrate at mean as covariance vanishes."""
    space = SPDMatrices(2)
    if metric is not None:
        space.equip_with_metric(metric)
    mean = gs.array([[2.0, 0.5], [0.5, 1.0]])
    gs.random.seed(18)

    samples = LogNormal(space, mean, 1e-20 * gs.eye(3)).sample(3)

    np.testing.assert_allclose(samples, gs.broadcast_to(mean, samples.shape), atol=1e-8)
