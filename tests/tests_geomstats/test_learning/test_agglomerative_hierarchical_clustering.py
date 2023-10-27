import pytest

from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.agglomerative_hierarchical_clustering import (
    AgglomerativeHierarchicalClusteringTestCase,
)

from .data.agglomerative_hierarchical_clustering import (
    AgglomerativeHierarchicalClusteringTestData,
)


@pytest.mark.smoke
class TestAgglomerativeHierarchicalClustering(
    AgglomerativeHierarchicalClusteringTestCase, metaclass=DataBasedParametrizer
):
    testing_data = AgglomerativeHierarchicalClusteringTestData()
