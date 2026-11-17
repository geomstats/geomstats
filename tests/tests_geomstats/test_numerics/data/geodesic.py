from polpo.testing.geometric import GeometricCaseData


class _SolverCaseData(GeometricCaseData):
    """Case data whose expected values are computed by a solver.

    Operations used to compute expected values are taken from ``solver``.
    """

    def __init__(self, space=None, solver=None, **kwargs):
        super().__init__(space, **kwargs)
        self.solver = solver

    def _get_operation(self, op_name, op_target=None):
        return getattr(self.solver, op_name)


class ExpSolverAgainstMetricTestData(GeometricCaseData):
    def exp_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def geodesic_ivp_test_data(self):
        return self.generate_timed_random_data(
            arg_names=("initial_point", "initial_tangent_vec")
        )


class ExpSolverComparisonTestData(GeometricCaseData):
    def exp_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def geodesic_ivp_test_data(self):
        return self.generate_timed_random_data(
            arg_names=("initial_point", "initial_tangent_vec")
        )


class ExpSolverTestData(_SolverCaseData):
    def exp_vec_test_data(self):
        return self.generate_vectorization_data(arg_names=("base_point", "tangent_vec"))

    def geodesic_ivp_vec_test_data(self):
        return self.generate_timed_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
            op_name="geodesic_ivp",
        )


class LogSolverAgainstMetricTestData(GeometricCaseData):
    def log_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def geodesic_bvp_test_data(self):
        return self.generate_timed_random_data(arg_names=("initial_point", "end_point"))


class LogSolverTestData(_SolverCaseData):
    def geodesic_bvp_vec_test_data(self):
        return self.generate_timed_vectorization_data(
            arg_names=("base_point", "point"),
            op_name="geodesic_bvp",
        )

    def log_after_exp_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def geodesic_bvp_known_geod_test_data(self):
        return self.generate_timed_random_data(arg_names=("base_point", "tangent_vec"))
