"""Regression tests for backend-valued Hessian-vector callbacks."""

import pytest

import geomstats.backend as gs
from geomstats.numerics.optimization import ScipyMinimize


@pytest.mark.parametrize("method", ["Newton-CG", "trust-ncg", "trust-krylov"])
@pytest.mark.parametrize("combined_gradient", [False, True])
def test_scipy_minimize_hessian_vector_uses_backend_inputs(method, combined_gradient):
    """Solve an anisotropic quadratic without constructing a Hessian callback."""
    matrix = gs.array([[3.0, 1.0], [1.0, 7.0]])
    linear = gs.array([2.0, -1.0])
    calls = []

    def objective(point):
        return 0.5 * gs.dot(point, gs.matvec(matrix, point)) - gs.dot(linear, point)

    def jacobian(point):
        return gs.matvec(matrix, point) - linear

    def value_and_jacobian(point):
        return objective(point), jacobian(point)

    def hessian_vector(point, vector):
        backend_point = gs.zeros_like(point) + point
        backend_vector = gs.zeros_like(vector) + vector
        calls.append((gs.copy(backend_point), gs.copy(backend_vector)))
        return gs.matvec(matrix, backend_vector)

    result = ScipyMinimize(method=method, tol=1e-8).minimize(
        value_and_jacobian if combined_gradient else objective,
        gs.array([3.0, -4.0]),
        fun_jac=True if combined_gradient else jacobian,
        hessp=hessian_vector,
    )

    assert result.success
    assert calls
    assert all(gs.is_array(point) and gs.is_array(vector) for point, vector in calls)
    assert gs.allclose(result.x, gs.linalg.solve(matrix, linear), atol=1e-6)
    assert gs.allclose(jacobian(result.x), gs.zeros(2), atol=1e-6)


def test_scipy_minimize_without_hessian_vector_callback():
    """Keep methods that do not use Hessian-vector products unchanged."""
    result = ScipyMinimize(method="BFGS").minimize(
        lambda point: gs.sum((point - 2.0) ** 2),
        gs.array([3.0, -4.0]),
        fun_jac=lambda point: 2.0 * (point - 2.0),
    )

    assert result.success
    assert gs.allclose(result.x, gs.array([2.0, 2.0]))
