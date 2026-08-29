import os
os.environ["GEOMSTATS_BACKEND"] = "autograd"
import geomstats.backend as gs
import pytest
from geomstats.information_geometry.normal import (
    UnivariateNormalDistributions, UnivariateNormalAlpha
)


def _connection(alpha):
    space = UnivariateNormalDistributions()
    return space, UnivariateNormalAlpha(space, alpha)

@pytest.mark.parametrize("base_point", [[1.5, 2.0], [-0.3, 0.8], [0.0, 1.0], [0.5, 1.5], [1.0, 8.0], [2.0, 10.0], [0.1, 0.5], [0.2, 0.8], [0.3, 1.2], [0.4, 1.8]])
def test_alpha_vs_fisher(base_point):
    space, alpha_conn = _connection(alpha=0.0)
    base_point = gs.array(base_point)

    gamma_alpha = alpha_conn.christoffels(base_point)
    gamma_fisher = space.metric.christoffels(base_point)

    assert gs.allclose(gamma_alpha, gamma_fisher, atol=1e-8)


@pytest.mark.parametrize("alpha", [-1.0, -0.3, 0.0, 0.5, 1.0])
def test_alpha_connection_symetry(alpha):
    space , alpha_conn = _connection(alpha=alpha)
    base_point = gs.array([0.8, 1.3])

    gamma = alpha_conn.christoffels(base_point)
    gamma_fisher = space.metric.christoffels(base_point)
    # Symetrie sur les indices covariants : Γ^k_{ij} = Γ^k_{ji}
    assert gs.allclose(gamma, gs.transpose(gamma, axes=(0, 2, 1)), atol=1e-8)
    assert gs.allclose(gamma_fisher, gs.transpose(gamma_fisher, axes=(0, 2, 1)), atol=1e-8)


# commandes pour tester 
# powershell avec bon backend activé : $env:GEOMSTATS_BACKEND="autograd" pytest tests/test_univariate_alpha.py -q