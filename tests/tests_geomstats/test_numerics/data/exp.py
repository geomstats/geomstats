from .geodesic import ExpSolverComparisonTestData


class ExpODESolverComparisonTestData(ExpSolverComparisonTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "exp": {"atol": 1e-4},
        "geodesic_ivp": {"atol": 1e-4},
    }
