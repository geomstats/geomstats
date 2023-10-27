import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class HyperbolicEmbeddingTestCase(TestCase):
    def test_log_sigmoid(self, point, expected, atol):
        res = self.embedding.log_sigmoid(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_grad_log_sigmoid(self, point, expected, atol):
        res = self.embedding.grad_log_sigmoid(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_loss(
        self, point, point_context, point_negative, expected_loss, expected_grad, atol
    ):
        loss_value, loss_grad = self.embedding.loss(
            point, point_context, point_negative
        )
        self.assertAllClose(loss_value, expected_loss, atol=atol)
        self.assertAllClose(loss_grad, expected_grad, atol=atol)

    def test_embed(self, graph, atol):
        embeddings = self.embedding.embed(graph)
        res = self.embedding.manifold.belongs(embeddings, atol)
        self.assertAllEqual(res, gs.ones_like(res))
