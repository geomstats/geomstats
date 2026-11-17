import pytest
from polpo.testing.data import CaseData


class InterpolatorTestData(CaseData):
    @pytest.mark.random
    def interpolate_at_nodes_test_data(self):
        return [dict()]

    @pytest.mark.random
    def interpolate_half_interval_test_data(self):
        return [dict()]
