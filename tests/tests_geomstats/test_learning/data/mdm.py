import geomstats.backend as gs
from geomstats.test.data import TestData

EULER = gs.exp(1.0)


class RiemannianMinimumDistanceToMeanSPDTestData(TestData):
    def _get_X_train(self):
        X_train_a = gs.array([[[EULER**2, 0], [0, 1]], [[1, 0], [0, 1]]])
        X_train_b = gs.array([[[EULER**8, 0], [0, 1]], [[1, 0], [0, 1]]])
        return gs.concatenate([X_train_a, X_train_b])

    def _get_X_train_2(self):
        X_train_a = gs.array([[EULER, 0], [0, 1]])[None, ...]
        X_train_b = gs.array([[EULER**4, 0], [0, 1]])[None, ...]
        return gs.concatenate([X_train_a, X_train_b])

    def _get_X_train_3(self):
        X_train_a = gs.array([[1.0, 0], [0, 1]])[None, ...]
        X_train_b = gs.array([[EULER**10, 0], [0, 1]])[None, ...]
        return gs.concatenate([X_train_a, X_train_b])

    def fit_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train(),
                y_train=gs.array([0, 0, 1, 1]),
                expected=gs.array(
                    [
                        [[EULER, 0], [0, 1]],
                        [[EULER**4, 0], [0, 1]],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)

    def predict_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_2(),
                y_train=gs.array([42, 17]),
                X_test=gs.array([[EULER**2, 0], [0, 1]])[None, ...],
                y_test=gs.array([42]),
            )
        ]

        return self.generate_tests(data)

    def predict_proba_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_3(),
                y_train=gs.array([1, 2]),
                X_test=gs.array([[[1.0, 0], [0, 1]], [[EULER**5, 0], [0, 1]]]),
                expected_proba=gs.array([[1.0, 0.0], [0.5, 0.5]]),
            )
        ]

        return self.generate_tests(data)

    def transform_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_3(),
                y_train=gs.array([1, 2]),
                X_test=gs.array([[[1.0, 0], [0, 1]], [[EULER**5, 0], [0, 1]]]),
                expected=gs.array([[0.0, 10.0], [5.0, 5.0]]),
            )
        ]
        return self.generate_tests(data)

    def score_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_2(),
                y_train=gs.array([-1, 1]),
                X_test=gs.array([[[EULER**3, 0], [0, 1]], [[EULER**2, 0], [0, 1]]]),
                y_expected=gs.array([1, -1]),
            )
        ]
        return self.generate_tests(data)


class RiemannianMinimumDistanceToMeanSPDEuclideanTestData(
    RiemannianMinimumDistanceToMeanSPDTestData
):
    def fit_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train(),
                y_train=gs.array([0, 0, 1, 1]),
                expected=gs.array(
                    [
                        [[0.5 * EULER**2 + 0.5, 0], [0, 1]],
                        [[0.5 * EULER**8 + 0.5, 0], [0, 1]],
                    ]
                ),
            )
        ]

        return self.generate_tests(data)

    def predict_proba_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_3(),
                y_train=gs.array([1, 2]),
                X_test=gs.array([[[1.0, 0], [0, 1]], [[EULER**5, 0], [0, 1]]]),
                expected_proba=gs.array([[1.0, 0.0], [1.0, 0.0]]),
            )
        ]

        return self.generate_tests(data)

    def transform_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_3(),
                y_train=gs.array([1, 2]),
                X_test=gs.array([[[1.0, 0], [0, 1]], [[EULER**5, 0], [0, 1]]]),
                expected=gs.array([[0.0, 22025.465795], [147.413159, 21878.052636]]),
            )
        ]
        return self.generate_tests(data)

    def score_test_data(self):
        data = [
            dict(
                X_train=self._get_X_train_2(),
                y_train=gs.array([-1, 1]),
                X_test=gs.array([[[EULER**3, 0], [0, 1]], [[EULER**2, 0], [0, 1]]]),
                y_expected=gs.array([-1, -1]),
            )
        ]
        return self.generate_tests(data)
