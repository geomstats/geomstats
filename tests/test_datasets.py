"""Unit tests for loading Graph dataset."""

import os

import geomstats.backend as gs
import geomstats.datasets.utils as utils
import geomstats.tests
from geomstats.datasets.graph_data_preparation import Graph
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestDatasets(geomstats.tests.TestCase):
    """Test for loading graph-structured data."""

    @geomstats.tests.np_only
    def setUp(self):
        self.data_folder = utils.DATA_FOLDER

        self.g1 = Graph()
        self.g2 = Graph(
            graph_matrix_path=os.path.join(
                self.data_folder, utils.KARATE_PATH),
            labels_path=os.path.join(
                self.data_folder, utils.KARATE_LABELS_PATH))

    def test_load_cities(self):
        """Test that the cities coordinates belong to the sphere."""
        sphere = Hypersphere(dim=2)
        data, _ = utils.load_cities()
        data = sphere.spherical_to_extrinsic(data)
        result = sphere.belongs(data)

        self.assertTrue(gs.all(result))

    def test_load_poses(self):
        """Test that the poses belong to SO(3)."""
        so3 = SpecialOrthogonal(n=3)
        data, _ = utils.load_poses()
        result = so3.belongs(data)

        self.assertTrue(gs.all(result))

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
