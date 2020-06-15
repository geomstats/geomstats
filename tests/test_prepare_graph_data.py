"""Unit tests for embedding data class."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.datasets.prepare_graph_data import HyperbolicEmbedding
from geomstats.datasets.utils import load_karate_graph


class TestPrepareGraphData(geomstats.tests.TestCase):
    """Class for testing embedding."""

    def setUp(self):
        """Set up function."""
        gs.random.seed(1234)
        dim = 2
        max_epochs = 3
        lr = .05
        n_negative = 2
        context_size = 1
        self.karate_graph = load_karate_graph()

        self.embedding = HyperbolicEmbedding(
            dim=dim,
            max_epochs=max_epochs,
            lr=lr,
            n_context=context_size,
            n_negative=n_negative)

    def test_log_sigmoid(self):
        """Test log_sigmoid."""
        point = gs.array([0.1, 0.3])
        result = self.embedding.log_sigmoid(point)

        expected = gs.array([-0.644397, -0.554355])
        self.assertAllClose(result, expected)

    def test_grad_log_sigmoid(self):
        """Test grad_log_sigmoid."""
        point = gs.array([0.1, 0.3])
        result = self.embedding.grad_log_sigmoid(point)

        expected = gs.array([0.47502081, 0.42555748])
        self.assertAllClose(result, expected)

    def test_loss(self):
        """Test loss function."""
        point = gs.array([0.5, 0.5])
        point_context = gs.array([0.6, 0.6])
        point_negative = gs.array([-0.4, -0.4])

        loss_value, loss_grad = self.embedding.loss(
            point, point_context, point_negative)

        expected_loss = 1.00322045
        expected_grad = gs.array([-0.16565083, -0.16565083])

        self.assertAllClose(loss_value[0], expected_loss, rtol=1e-3)
        self.assertAllClose(gs.squeeze(loss_grad), expected_grad, rtol=1e-3)

    def test_embed(self):
        """Test embedding function."""
        embeddings = self.embedding.embed(self.karate_graph)
        self.assertTrue(
            gs.all(self.embedding.manifold.belongs(embeddings)))
