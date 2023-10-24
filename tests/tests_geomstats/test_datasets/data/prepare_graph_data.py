import geomstats.backend as gs
from geomstats.datasets.utils import load_karate_graph
from geomstats.test.data import TestData


class HyperbolicEmbeddingTestData(TestData):
    def log_sigmoid_test_data(self):
        data = [
            dict(point=gs.array([0.1, 0.3]), expected=gs.array([-0.644397, -0.554355]))
        ]
        return self.generate_tests(data)

    def grad_log_sigmoid_test_data(self):
        data = [
            dict(
                point=gs.array([0.1, 0.3]), expected=gs.array([0.47502081, 0.42555748])
            )
        ]
        return self.generate_tests(data)

    def loss_test_data(self):
        data = [
            dict(
                point=gs.array([0.5, 0.5]),
                point_context=gs.array([0.6, 0.6]),
                point_negative=gs.array([-0.4, -0.4]),
                expected_loss=gs.array([1.00322045]),
                expected_grad=gs.array([[-0.16565083, -0.16565083]]),
            ),
        ]
        return self.generate_tests(data)

    def embed_test_data(self):
        data = [
            dict(
                graph=load_karate_graph(),
            )
        ]
        return self.generate_tests(data)
