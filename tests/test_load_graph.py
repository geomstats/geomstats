"""Unit tests for loading Graph dataset."""

import geomstats.tests
from geomstats.learning.graph_data_preparation import Graph


class TestLoadDefaultGraph(geomstats.tests.TestCase):
    """Test for loading graph-structured data."""

    @geomstats.tests.np_only
    def setUp(self):
        """Declare the graph by default and the Karate club graph."""
        self.G1 = Graph()
        self.G2 = Graph(
            Graph_Matrix_Path='examples\\data_example\\'
                              'graph_karate\\Karate.txt',
            Labels_Path='examples\\data_example\\'
                        'graph_karate\\Karate_Labels.txt')

    @geomstats.tests.np_only
    def test_graph_load(self):
        """Test the correct number of edges and nodes for each graph."""
        result = [len(self.G1.edges) + len(self.G1.labels),
                  len(self.G2.edges) + len(self.G2.labels)]
        expected = [20, 68]

        self.assertAllClose(result, expected)

    def test_random_walks(self):
        """Test that random walks have the right length and number."""
        paths1 = self.G1.random_walk(walk_length=3)
        paths2 = self.G2.random_walk(walk_length=6)

        result = [len(paths1), len(paths1[0]), len(paths2), len(paths2[0])]
        expected = [len(self.G1.edges) * self.G1.number_walks_per_node,
                    self.G1.walk_length + 1,
                    len(self.G2.edges) * self.G2.number_walks_per_node,
                    self.G2.walk_length + 1]

        self.assertAllClose(result, expected)
