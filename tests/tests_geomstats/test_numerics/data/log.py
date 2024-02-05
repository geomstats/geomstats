from .geodesic import LogSolverTestData


class LogODESolverMatrixTestData(LogSolverTestData):
    fail_for_autodiff_exceptions = True
    fail_for_not_implemented_errors = True

    skip_vec = True

    tolerances = {
        "geodesic_bvp_known_geod": {"atol": 1e-3},
        "log_known_tangent_vec": {"atol": 5e-4},
    }
