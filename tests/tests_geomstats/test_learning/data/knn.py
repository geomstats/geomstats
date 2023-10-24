import geomstats.backend as gs
from geomstats.test.data import TestData


class KNearestNeighborsClassifierEuclideanTestData(TestData):
    def predict_test_data(self):
        data = [
            dict(
                X_train=gs.array([[0.0], [1.0], [2.0], [3.0]]),
                y_train=gs.array([0, 0, 1, 1]),
                X_test=gs.array([[1.1]]),
                y_test=gs.array([0]),
            )
        ]
        return self.generate_tests(data)

    def predict_proba_test_data(self):
        data = [
            dict(
                X_train=gs.array([[0.0], [1.0], [2.0], [3.0]]),
                y_train=gs.array([0, 0, 1, 1]),
                X_test=gs.array([[0.9]]),
                expected_proba=gs.array([[2 / 3, 1 / 3]]),
            ),
        ]

        return self.generate_tests(data)
