import pytest
from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.test_cases.metric_geometry.trees import (
    BaseTopologyTestCase,
    GenerateSplitsTestCase,
    SplitTestCase,
)

from .data.trees import BaseTopologyTestData, GenerateSplitsTestData, SplitTestData


@pytest.mark.smoke
class TestGenerateSplits(GenerateSplitsTestCase, metaclass=DataBasedParametrizer):
    testing_data = GenerateSplitsTestData()


@pytest.mark.smoke
class TestSplit(SplitTestCase, metaclass=DataBasedParametrizer):
    testing_data = SplitTestData()


@pytest.mark.smoke
class TestBaseTopology(BaseTopologyTestCase, metaclass=DataBasedParametrizer):
    testing_data = BaseTopologyTestData()
