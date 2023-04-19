from tests2.data.geodesic_solvers_data import ExpSolverComparisonTestData


class ExpIVPSolverComparisonTestData(ExpSolverComparisonTestData):
    tolerances = {
        "exp": {"atol": 1e-4},
        "geodesic_ivp": {"atol": 1e-4},
    }
