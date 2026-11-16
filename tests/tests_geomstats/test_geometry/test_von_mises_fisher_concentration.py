"""Regression tests for numerically stable von Mises-Fisher rejection sampling."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry import hypersphere


def _reference_coordinates_and_criterion(dim, kappa, draw):
    """Evaluate the original Wood formulas with enough decimal precision."""
    with localcontext() as context:
        context.prec = 450
        dim, kappa, draw = [Decimal(str(value)) for value in (dim, kappa, draw)]
        envelope = ((4 * kappa**2 + dim**2).sqrt() - 2 * kappa) / dim
        node = (1 - envelope) / (1 + envelope)
        coordinate = (1 - (1 + envelope) * draw) / (1 - (1 - envelope) * draw)
        radial = (1 - coordinate**2).sqrt()
        criterion = (
            kappa * coordinate
            + dim * (1 - node * coordinate).ln()
            - kappa * node
            - dim * (1 - node**2).ln()
        )
        return float(coordinate), float(radial), float(criterion)


def _fixed_draws(monkeypatch, draws, uniforms):
    monkeypatch.setattr(
        hypersphere.beta,
        "rvs",
        lambda *args, **kwargs: np.array(draws),
    )
    monkeypatch.setattr(gs.random, "rand", lambda *shape: gs.array(uniforms))


@pytest.mark.parametrize("dim", [3, 5])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 100.0, 1e8, 1e12, 1e20, 1e200])
def test_von_mises_fisher_coordinates_match_high_precision(monkeypatch, dim, kappa):
    """Keep the requested samples and their transverse spread at high concentration."""
    draws = [0.1, 0.25, 0.5, 0.75, 0.9]
    _fixed_draws(monkeypatch, draws, [1e-20] * len(draws))
    distribution = hypersphere.VonMisesFisher(
        hypersphere.Hypersphere(dim, equip=False), kappa=kappa, max_iter=1
    )

    sample = gs.to_numpy(distribution.sample(len(draws)))

    reference = np.array(
        [_reference_coordinates_and_criterion(dim, kappa, draw) for draw in draws]
    )
    assert sample.shape == (len(draws), dim + 1)
    assert np.all(np.isfinite(sample))
    np.testing.assert_allclose(sample[:, 0], reference[:, 0], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(sample[:, 1:], axis=-1), reference[:, 1], rtol=2e-12, atol=0.0
    )
    np.testing.assert_allclose(np.linalg.norm(sample, axis=-1), 1.0, atol=1e-14)


@pytest.mark.parametrize("kappa", [1.0, 1e8, 1e20])
def test_von_mises_fisher_rejection_matches_high_precision(monkeypatch, kappa):
    """Retain the rejection law when large terms cancel in its log criterion."""
    dim = 5
    draws = [0.1, 0.3, 0.7, 0.9]
    uniforms = [0.1, 0.9, 0.1, 0.5]
    _fixed_draws(monkeypatch, draws, uniforms)
    distribution = hypersphere.VonMisesFisher(
        hypersphere.Hypersphere(dim, equip=False), kappa=kappa, max_iter=1
    )

    sample = gs.to_numpy(distribution.sample(len(draws)))

    reference = np.array(
        [_reference_coordinates_and_criterion(dim, kappa, draw) for draw in draws]
    )
    accepted = reference[:, 2] > np.log(uniforms)
    assert sample.shape == (int(np.sum(accepted)), dim + 1)
    np.testing.assert_allclose(sample[:, 0], reference[accepted, 0], atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(sample[:, 1:], axis=-1),
        reference[accepted, 1],
        rtol=2e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("dim", [3, 5])
@pytest.mark.parametrize("kappa", [1e12, 1e200])
@pytest.mark.parametrize("rotate", [False, True])
def test_von_mises_fisher_high_concentration_returns_complete_samples(
    dim, kappa, rotate
):
    """Exercise genuine rejection draws and both mean-direction branches."""
    gs.random.seed(0)
    mu = gs.eye(dim + 1)[1] if rotate else None
    distribution = hypersphere.VonMisesFisher(
        hypersphere.Hypersphere(dim, equip=False), mu=mu, kappa=kappa
    )

    sample = gs.to_numpy(distribution.sample(64))

    assert sample.shape == (64, dim + 1)
    assert np.all(np.isfinite(sample))
    np.testing.assert_allclose(np.linalg.norm(sample, axis=-1), 1.0, atol=1e-14)
    axis = 1 if rotate else 0
    assert np.all(sample[:, axis] > 0.99)
    transverse = np.delete(sample, axis, axis=-1)
    assert np.all(np.linalg.norm(transverse, axis=-1) > 0.0)
