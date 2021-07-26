"""Unit tests for loading datasets."""

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import geomstats.tests
from geomstats.geometry.discrete_curves import DiscreteCurves, R2
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.information_geometry.beta import BetaDistributions


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
        so3 = SpecialOrthogonal(n=3, point_type="vector")
        data, _ = data_utils.load_poses()
        result = so3.belongs(data)

        self.assertTrue(gs.all(result))

    def test_load_poses(self):
        """Test that the poses belong to SE(3)."""
        se3 = SpecialEuclidean(n=3, point_type="vector")
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

        paths = graph.random_walk(
            walk_length=walk_length, n_walks_per_node=n_walks_per_node
        )

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node, walk_length + 1]

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_random_walks_karate_graph(self):
        """Test that random walks have the right length and number."""
        graph = data_utils.load_karate_graph()
        walk_length = 6
        n_walks_per_node = 2

        paths = graph.random_walk(
            walk_length=walk_length, n_walks_per_node=n_walks_per_node
        )

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node, walk_length + 1]

        self.assertAllClose(result, expected)

    def test_load_connectomes(self):
        """Test that the connectomes belong to SPD."""
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

    @geomstats.tests.np_only
    def test_leaves(self):
        """Test that leaves data are beta distribution parameters."""
        beta = BetaDistributions()
        beta_param, distrib_type = data_utils.load_leaves()
        result = beta.belongs(beta_param)
        self.assertTrue(gs.all(result))

        result = len(distrib_type)
        expected = beta_param.shape[0]
        self.assertAllClose(result, expected)

    def test_load_emg(self):
        """Test that data have the correct column names."""
        data_emg = data_utils.load_emg()
        expected_col_name = [
            'time',
            'c0',
            'c1',
            'c2',
            'c3',
            'c4',
            'c5',
            'c6',
            'c7',
            'label',
            'exp',
        ]
        good_col_name = (expected_col_name == data_emg.keys()).all()
        self.assertTrue(good_col_name)

    def test_load_optical_nerves(self):
        """Test that optical nerves belong to space of landmarks."""
        data, labels, monkeys = data_utils.load_optical_nerves()
        result = data.shape
        n_monkeys = 11
        n_eyes_per_monkey = 2
        k_landmarks = 5
        dim = 3
        expected = (n_monkeys * n_eyes_per_monkey, k_landmarks, dim)
        self.assertAllClose(result, expected)

        landmarks_space = Landmarks(
            ambient_manifold=Euclidean(dim=dim), k_landmarks=k_landmarks
        )

        result = landmarks_space.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(monkeys >= 0, monkeys <= 11)
        self.assertTrue(gs.all(result))

    def test_hands(self):
        """Test that hands belong to space of landmarks."""
        data, labels, _ = data_utils.load_hands()
        result = data.shape
        n_hands = 52
        k_landmarks = 22
        dim = 3
        expected = (n_hands, k_landmarks, dim)
        self.assertAllClose(result, expected)

        landmarks_space = Landmarks(
            ambient_manifold=Euclidean(dim=3), k_landmarks=k_landmarks
        )

        result = landmarks_space.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))

    def test_cells(self):
        """Test that cells belong to space of planar curves."""
        cells, cell_lines, treatments = data_utils.load_cells()
        expected = 650
        result = len(cells)
        self.assertAllClose(result, expected)
        result = len(cell_lines)
        self.assertAllClose(result, expected)
        result = len(treatments)
        self.assertAllClose(result, expected)

        planar_curves_space = DiscreteCurves(R2)

        result = planar_curves_space.belongs(cells)
        self.assertTrue(gs.all(result))

        result = [line in ["dlm8", "dunn"] for line in cell_lines]
        self.assertTrue(gs.all(result))

        result = [treatment in ["control", "cytd", "jasp"]
                  for treatment in treatments]
        self.assertTrue(gs.all(result))
