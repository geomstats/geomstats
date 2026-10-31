import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.distributions.brownian_motion import BrownianMotion
from geomstats.geometry.euclidean import Euclidean


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("n_steps", [1, 4])
def test_zero_time_brownian_path_keeps_batch_axes(batch_shape, n_steps):
    """Batch axes precede time, including a single-point path."""
    shape = batch_shape + (2,)
    initial = gs.reshape(gs.arange(int(gs.prod(gs.array(shape)))), shape) * 1.0
    original = gs.copy(initial)

    path = BrownianMotion(Euclidean(2)).sample_path(0.0, n_steps, initial)

    assert path.shape == batch_shape + (n_steps, 2)
    expected = gs.broadcast_to(gs.expand_dims(initial, -2), path.shape)
    np.testing.assert_allclose(path, expected)
    np.testing.assert_allclose(initial, original)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_batched_euclidean_brownian_path_has_expected_increments(batch_shape):
    """Flat-space increments use each batch's own normal draws."""
    initial = gs.ones(batch_shape + (2,))
    n_steps = 5
    end_time = 1.25
    gs.random.seed(318)
    expected_positions = [initial]
    for _ in range(1, n_steps):
        increment = gs.random.normal(size=initial.shape) * gs.sqrt(end_time / n_steps)
        expected_positions.append(expected_positions[-1] + increment)
    expected = gs.stack(expected_positions, axis=-2)

    gs.random.seed(318)
    path = BrownianMotion(Euclidean(2)).sample_path(end_time, n_steps, initial)

    assert path.shape == expected.shape
    np.testing.assert_allclose(path, expected)
