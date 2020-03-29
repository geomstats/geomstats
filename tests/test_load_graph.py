"""Unit tests for loading Graph dataset."""

import geomstats.tests
from geomstats.datasets.graph_data_preparation import Graph


class TestLoadDefaultGraph(geomstats.tests.TestCase):
    """Test for loading graph-structured data."""

    @geomstats.tests.np_only
    def setUp(self):
        """Declare the graph by default and the Karate club graph."""
        self.g1 = Graph()
        self.g2 = Graph(
            graph_matrix_path='examples/data'
                              '/graph_karate/karate.txt',
            labels_path='examples/data'
                        '/graph_karate/karate_labels.txt')

    @geomstats.tests.np_only
    def test_graph_load(self):
        """Test the correct number of edges and nodes for each graph."""
        result = [len(self.g1.edges) + len(self.g1.labels),
                  len(self.g2.edges) + len(self.g2.labels)]
        expected = [20, 68]

        self.assertAllClose(result, expected)

    def test_random_walks(self):
        """Test that random walks have the right length and number."""
        walk_length_g1 = 3
        walk_length_g2 = 6

        n_walks_per_node_g1 = 1
        n_walks_per_node_g2 = 2

        paths1 = self.g1.random_walk(walk_length=walk_length_g1,
                                     n_walks_per_node=n_walks_per_node_g1)
        paths2 = self.g2.random_walk(walk_length=walk_length_g2,
                                     n_walks_per_node=n_walks_per_node_g2)

        result = [len(paths1), len(paths1[0]), len(paths2), len(paths2[0])]
        expected = [len(self.g1.edges) * n_walks_per_node_g1,
                    walk_length_g1 + 1,
                    len(self.g2.edges) * n_walks_per_node_g2,
                    walk_length_g2 + 1]

        self.assertAllClose(result, expected)
