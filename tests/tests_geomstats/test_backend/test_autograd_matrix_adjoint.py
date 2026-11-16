"""Regression tests for matrix-function adjoints with several batch axes."""

import math

import pytest

import geomstats.backend as gs
from geomstats.test.test_case import autograd_only

pytestmark = autograd_only


@pytest.mark.parametrize("function_name", ["expm", "logm"])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,), (2, 1), (2, 1, 3)])
def test_matrix_function_gradient_matches_diagonal_frechet_derivative(
    function_name, batch_shape
):
    """Match the closed-form derivative at diagonal matrices in every batch."""
    scales = gs.reshape(gs.linspace(0.5, 1.5, math.prod(batch_shape)), batch_shape)
    matrices = scales[..., None, None] * gs.array([[1.0, 0.0], [0.0, 2.0]])
    weights = gs.reshape(
        gs.arange(math.prod(matrices.shape), dtype=gs.float64) + 1.0,
        matrices.shape,
    )
    function = getattr(gs.linalg, function_name)

    _, gradient = gs.autodiff.value_and_grad(
        lambda array: gs.sum(weights * function(array))
    )(matrices)

    if function_name == "expm":
        diagonal_a = gs.exp(scales)
        diagonal_b = gs.exp(2.0 * scales)
        off_diagonal = (diagonal_b - diagonal_a) / scales
    else:
        diagonal_a = 1.0 / scales
        diagonal_b = 1.0 / (2.0 * scales)
        off_diagonal = math.log(2.0) / scales
    divided_differences = gs.stack(
        [
            gs.stack([diagonal_a, off_diagonal], axis=-1),
            gs.stack([off_diagonal, diagonal_b], axis=-1),
        ],
        axis=-2,
    )

    assert gradient.shape == matrices.shape
    assert gs.allclose(gradient, weights * divided_differences)


@pytest.mark.parametrize("function_name", ["expm", "logm"])
def test_batched_nonsymmetric_matrix_function_gradient_matches_difference(
    function_name,
):
    """Check the adjoint on noncommuting directions and nonsymmetric inputs."""
    matrices = gs.array(
        [
            [[[1.0, 0.4], [0.0, 2.0]]],
            [[[1.5, -0.2], [0.0, 2.5]]],
        ]
    )
    weights = gs.array([[[[0.2, -0.3], [0.7, 1.1]]], [[[0.5, 0.4], [-0.2, 0.8]]]])
    direction = gs.array([[[[0.2, -0.4], [0.7, 0.1]]], [[[-0.3, 0.5], [0.1, -0.2]]]])
    function = getattr(gs.linalg, function_name)

    def objective(array):
        return gs.sum(weights * function(array))

    _, gradient = gs.autodiff.value_and_grad(objective)(matrices)
    epsilon = 1e-6
    expected = (
        objective(matrices + epsilon * direction)
        - objective(matrices - epsilon * direction)
    ) / (2.0 * epsilon)

    assert gradient.shape == matrices.shape
    assert gs.allclose(gs.sum(gradient * direction), expected, atol=1e-6)
