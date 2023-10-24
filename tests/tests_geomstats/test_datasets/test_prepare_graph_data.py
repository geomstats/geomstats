from geomstats.datasets.prepare_graph_data import HyperbolicEmbedding
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.datasets.prepare_graph_data import HyperbolicEmbeddingTestCase

from .data.prepare_graph_data import HyperbolicEmbeddingTestData


class TestHyperbolicEmbedding(
    HyperbolicEmbeddingTestCase, metaclass=DataBasedParametrizer
):
    embedding = HyperbolicEmbedding(
        dim=2,
        max_epochs=3,
        lr=0.05,
        n_context=1,
        n_negative=2,
    )

    testing_data = HyperbolicEmbeddingTestData()
