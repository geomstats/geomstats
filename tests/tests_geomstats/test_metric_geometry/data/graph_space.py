from .quotient import AlignerCaseData


class PointToGeodesicAlignerTestData(AlignerCaseData):
    def _geodesic(self, initial_point, end_point):
        return self.space.quotient.metric.geodesic(initial_point, end_point)

    def align_with_endpoints_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("initial_point", "end_point", "point"),
            op_name="align_point_to_geodesic",
            op_evaluator=lambda op, initial_point, end_point, point: op(
                self._geodesic(initial_point, end_point), point
            ),
        )

    def dist_with_endpoints_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("initial_point", "end_point", "point"),
            op_name="point_to_geodesic_aligner",
            op_evaluator=lambda op, initial_point, end_point, point: op.dist(
                self._geodesic(initial_point, end_point), point
            ),
        )

    def dist_along_geodesic_is_zero_test_data(self):
        return self.generate_random_data(arg_names=("initial_point", "end_point"))
