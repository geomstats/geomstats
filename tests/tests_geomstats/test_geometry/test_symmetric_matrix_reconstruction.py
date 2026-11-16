"""Regression tests for differentiable symmetric matrix reconstruction."""

import math

import numpy as np
import pytest
from scipy.special import polygamma

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.test.test_case import np_backend


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (2, 1, 3), (0,)])
def test_symmetric_matrix_reconstruction_preserves_all_batches(n, batch_shape):
    """Place every upper-triangular coordinate and its reflection exactly."""
    size = n * (n + 1) // 2
    values = np.arange(math.prod(batch_shape) * size, dtype=float).reshape(
        batch_shape + (size,)
    )
    expected = np.zeros(batch_shape + (n, n))
    rows, columns = np.triu_indices(n)
    expected[..., rows, columns] = values
    expected[..., columns, rows] = values

    result = SymmetricMatrices.matrix_representation(gs.array(values))

    assert result.shape == expected.shape
    assert gs.allclose(result, gs.array(expected))
    assert gs.allclose(SymmetricMatrices.basis_representation(result), gs.array(values))


@pytest.mark.parametrize("dtype", [gs.float32, gs.float64, gs.complex64, gs.complex128])
def test_symmetric_matrix_reconstruction_preserves_dtype(dtype):
    """Reconstruct coordinates without changing their scalar type."""
    coordinates = gs.array([1.0, 2.0, 3.0], dtype=dtype)
    if gs.is_complex(coordinates):
        coordinates = coordinates + 1j

    result = SymmetricMatrices.matrix_representation(coordinates)

    assert result.dtype == coordinates.dtype
    assert gs.allclose(result, coordinates[gs.array([[0, 1], [1, 2]])])


@pytest.mark.skipif(np_backend(), reason="Requires automatic differentiation.")
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_symmetric_reconstruction_gradient_matches_frobenius_norm(batch_shape):
    """Off-diagonal coordinates contribute twice to the squared norm."""
    coordinates = gs.reshape(
        gs.arange(math.prod(batch_shape) * 6, dtype=gs.float64) + 1.0,
        batch_shape + (6,),
    )

    _, gradient = gs.autodiff.value_and_grad(
        lambda vector: gs.sum(SymmetricMatrices.matrix_representation(vector) ** 2)
    )(coordinates)

    assert gradient.shape == coordinates.shape
    assert gs.allclose(gradient, coordinates * gs.array([2.0, 4.0, 4.0, 2.0, 4.0, 2.0]))


@pytest.mark.skipif(np_backend(), reason="Requires automatic differentiation.")
def test_symmetric_reconstruction_preserves_second_derivatives():
    """The squared Frobenius norm has a constant diagonal Hessian."""
    hessian = gs.autodiff.hessian(
        lambda vector: gs.sum(SymmetricMatrices.matrix_representation(vector) ** 2)
    )(gs.array([1.0, 2.0, 3.0]))

    assert gs.allclose(
        hessian, gs.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    )


@pytest.mark.skipif(np_backend(), reason="Requires automatic differentiation.")
@pytest.mark.parametrize("batched", [False, True])
def test_beta_metric_derivative_matches_polygamma_formula(batched):
    """Differentiate the Fisher matrix at fixed positive beta parameters."""
    parameters = np.array([[1.5, 2.5], [3.0, 4.0]]) if batched else np.array([1.5, 2.5])
    alpha, beta = parameters[..., 0], parameters[..., 1]
    common = -polygamma(2, alpha + beta)
    expected = np.stack(
        [
            np.stack([polygamma(2, alpha) + common, common], axis=-1),
            np.stack([common, common], axis=-1),
            np.stack([common, common], axis=-1),
            np.stack([common, polygamma(2, beta) + common], axis=-1),
        ],
        axis=-2,
    ).reshape(parameters.shape[:-1] + (2, 2, 2))

    result = BetaDistributions().metric.inner_product_derivative_matrix(
        gs.array(parameters)
    )

    assert result.shape == expected.shape
    assert gs.allclose(result, gs.array(expected))
