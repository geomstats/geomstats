from polpo.testing.geometric import GeometricCaseData

from .point_set import PointSetMetricWithArrayTestData


class AlignerCaseData(GeometricCaseData):
    """Data for the aligner of `space`.

    Operations are taken from ``space.aligner``.
    """

    def _get_operation(self, op_name, op_target="space"):
        return getattr(self.space.aligner, op_name)


class AlignerTestData(AlignerCaseData):
    def align_vec_test_data(self):
        return self.generate_vectorization_data(arg_names=("base_point", "point"))


class QuotientMetricWithArrayTestData(PointSetMetricWithArrayTestData):
    pass
