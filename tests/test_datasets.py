"""Unit tests for loading datasets."""

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestDatasets(geomstats.tests.TestCase):
    """Test for data-loading utilities."""

    def test_load_cities(self):
        """Test that the cities coordinates belong to the sphere."""
        sphere = Hypersphere(dim=2)
        data, _ = data_utils.load_cities()
        self.assertAllClose(gs.shape(data), (50, 3))

        tokyo = data[0]
        self.assertAllClose(
            tokyo, gs.array([0.61993792, -0.52479018, 0.58332859]))

        result = sphere.belongs(data)
        self.assertTrue(gs.all(result))

    def test_load_poses_only_rotations(self):
        """Test that the poses belong to SO(3)."""
        so3 = SpecialOrthogonal(n=3, point_type='vector')
        data, _ = data_utils.load_poses()
        result = so3.belongs(data)

        self.assertTrue(gs.all(result))

    def test_load_poses(self):
        """Test that the poses belong to SE(3)."""
        se3 = SpecialEuclidean(n=3, point_type='vector')
        data, _ = data_utils.load_poses(only_rotations=False)
        result = se3.belongs(data)

        self.assertTrue(gs.all(result))

    @geomstats.tests.np_and_pytorch_only
    def test_karate_graph(self):
        """Test the correct number of edges and nodes for each graph."""
        graph = data_utils.load_karate_graph()
        result = len(graph.edges) + len(graph.labels)
        expected = 68
        self.assertTrue(result == expected)

    @geomstats.tests.np_and_pytorch_only
    def test_random_graph(self):
        """Test the correct number of edges and nodes for each graph."""
        graph = data_utils.load_random_graph()
        result = len(graph.edges) + len(graph.labels)
        expected = 20
        self.assertTrue(result == expected)

    @geomstats.tests.np_and_pytorch_only
    def test_random_walks_random_graph(self):
        """Test that random walks have the right length and number."""
        graph = data_utils.load_random_graph()
        walk_length = 3
        n_walks_per_node = 1

        paths = graph.random_walk(walk_length=walk_length,
                                  n_walks_per_node=n_walks_per_node)

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node,
                    walk_length + 1]

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_random_walks_karate_graph(self):
        """Test that random walks have the right length and number."""
        graph = data_utils.load_karate_graph()
        walk_length = 6
        n_walks_per_node = 2

        paths = graph.random_walk(walk_length=walk_length,
                                  n_walks_per_node=n_walks_per_node)

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node,
                    walk_length + 1]

        self.assertAllClose(result, expected)

    def test_load_connectomes(self):
        """Test that the connectomes belong to SPD"""
        spd = SPDMatrices(28)
        data, _, _ = data_utils.load_connectomes(as_vectors=True)
        result = data.shape
        expected = (86, 27 * 14)
        self.assertAllClose(result, expected)

        data, _, labels = data_utils.load_connectomes()
        result = spd.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))
