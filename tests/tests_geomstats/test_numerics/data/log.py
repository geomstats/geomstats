from .geodesic import LogSolverAgainstMetricTestData, LogSolverTestData


class LogSolverAgainstClosedFormTestData(LogSolverAgainstMetricTestData):
    fail_for_not_implemented_errors = False

    tolerances = {
        "log": {"atol": 1e-3},
        "geodesic_bvp": {"atol": 1e-4},
    }


class PathStraighteningAgainstClosedFormTestData(LogSolverAgainstMetricTestData):
    tolerances = {
        "log": {"atol": 1e-2},
        "geodesic_bvp": {"atol": 1e-2},
    }


class LogODESolverMatrixTestData(LogSolverTestData):
    skip_vec = True

    tolerances = {
        "geodesic_bvp_known_geod": {"atol": 1e-3},
        "log_known_tangent_vec": {"atol": 5e-4},
    }
