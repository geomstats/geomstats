"""Regression tests for whitening tangent principal component scores."""

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.pca import TangentPCA
from geomstats.test.test_case import assert_allclose, np_and_autograd_only

pytestmark = np_and_autograd_only


def _data():
    """Return full-rank, anisotropic data without using the global RNG."""
    return gs.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0]])


@pytest.mark.parametrize("matrix_points", [False, True])
def test_whitened_scores_have_identity_sample_covariance(matrix_points):
    """Whitening removes component scales for vector and SPD data."""
    data = _data()
    if matrix_points:
        space = SPDMatrices(2)
        data = gs.vec_to_diag(gs.exp(data))
        base_point = gs.eye(2)
    else:
        space = Euclidean(2)
        base_point = gs.zeros(2)
    estimator = TangentPCA(space, n_components=2, whiten=True)

    scores = estimator.fit_transform(data, base_point=base_point)

    covariance = gs.transpose(scores) @ scores / (len(data) - 1)
    assert_allclose(covariance, gs.eye(2), atol=1e-10)
    assert_allclose(scores, estimator.transform(data), atol=1e-10)
    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-9)


@pytest.mark.parametrize("whiten", [False, True])
def test_transform_and_inverse_use_the_fitted_component_scales(whiten):
    """Held-out scores and reconstructions obey the fitted PCA equations."""
    data = _data()
    estimator = TangentPCA(Euclidean(2), n_components=1, whiten=whiten)
    estimator.fit(data, base_point=gs.zeros(2))
    held_out = gs.array([[-2.0, 0.5], [6.0, 3.0]])
    unscaled = (held_out - estimator.mean_) @ gs.transpose(estimator.components_)
    expected = unscaled
    if whiten:
        expected = expected / gs.sqrt(estimator.explained_variance_)

    scores = estimator.transform(held_out)

    assert_allclose(scores, expected, atol=1e-10)
    expected_reconstruction = estimator.mean_ + unscaled @ estimator.components_
    assert_allclose(
        estimator.inverse_transform(expected), expected_reconstruction, atol=1e-10
    )


@pytest.mark.parametrize("whiten", [False, True])
def test_fit_transform_preserves_unwhitened_behavior(whiten):
    """Both training entry points agree and the default retains variances."""
    data = _data()
    estimator = TangentPCA(Euclidean(2), n_components=2, whiten=whiten)
    scores = estimator.fit_transform(data, base_point=gs.zeros(2))

    assert_allclose(scores, estimator.transform(data), atol=1e-10)
    expected_variance = gs.ones(2) if whiten else estimator.explained_variance_
    variance = gs.sum(scores**2, axis=0) / (len(data) - 1)
    assert_allclose(variance, expected_variance, atol=1e-10)
    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)


def test_whitening_keeps_zero_variance_components_finite():
    """An exactly zero singular value does not introduce NaN scores."""
    data = gs.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    estimator = TangentPCA(Euclidean(2), n_components=2, whiten=True)
    scores = estimator.fit_transform(data, base_point=gs.zeros(2))

    assert gs.all(gs.abs(scores) < gs.inf)
    assert_allclose(scores[:, 1], gs.zeros(len(data)))
    assert_allclose(scores, estimator.transform(data), atol=1e-10)
    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)
