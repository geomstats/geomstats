import geomstats.backend as gs
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifier,
)
from geomstats.learning.radial_kernel_functions import triangular_radial_kernel
from geomstats.test.data import TestData


class KernelDensityEstimationClassifierTestData(TestData):
    def _get_1d_dataset(self):
        X_train = gs.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = gs.array([0, 0, 1, 1])
        return X_train, y_train

    def _get_2d_dataset(self):
        X_train = gs.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        y_train = gs.array([0, 0, 1, 1])
        return X_train, y_train

    def _get_hyperbolic_dataset(self, intrinsic=True):
        X_train = gs.array(
            [
                [1 / 2, 1 / 4],
                [1 / 2, 0],
                [1 / 2, -1 / 4],
                [-1 / 2, 1 / 4],
                [-1 / 2, 0],
                [-1 / 2, -1 / 4],
            ]
        )
        if not intrinsic:
            X_train = _Hyperbolic.change_coordinates_system(
                X_train,
                from_coordinates_system="intrinsic",
                to_coordinates_system="extrinsic",
            )

        return X_train, gs.array([0, 0, 0, 1, 1, 1])

    def predict_test_data(self):
        X_train_1d, y_train_1d = self._get_1d_dataset()
        X_train_2d, y_train_2d = self._get_2d_dataset()
        X_train_poincare, y_train_poincare = self._get_hyperbolic_dataset()
        X_train_hyperboloid, y_train_hyperboloid = self._get_hyperbolic_dataset(
            intrinsic=False
        )
        X_test_poincare = gs.array(
            [
                [1 / 2, 1 / 5],
                [1 / 2, 0],
                [1 / 2, -1 / 5],
                [-1 / 2, 1 / 5],
                [-1 / 2, 0],
                [-1 / 2, -1 / 5],
            ]
        )

        data = [
            dict(
                estimator=KernelDensityEstimationClassifier(Euclidean(dim=1)),
                X_train=X_train_1d,
                y_train=y_train_1d,
                X_test=gs.array([[1.1]]),
                y_test=gs.array([0.0]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(Euclidean(dim=2)),
                X_train=X_train_2d,
                y_train=y_train_2d,
                X_test=gs.array([[1.1, 0.0]]),
                y_test=gs.array([0]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(Hypersphere(dim=2)),
                X_train=gs.array(
                    [
                        [1, 0, 0],
                        [3 ** (1 / 2) / 2, 1 / 2, 0],
                        [3 ** (1 / 2) / 2, -1 / 2, 0],
                        [0, 0, 1],
                        [0, 1 / 2, 3 ** (1 / 2) / 2],
                        [0, -1 / 2, 3 ** (1 / 2) / 2],
                    ]
                ),
                y_train=gs.array([0, 0, 0, 1, 1, 1]),
                X_test=gs.array(
                    [
                        [2 ** (1 / 2) / 2, 2 ** (1 / 2) / 2, 0],
                        [0, 1 / 2, -(3 ** (1 / 2)) / 2],
                        [0, -1 / 2, -(3 ** (1 / 2)) / 2],
                        [-(3 ** (1 / 2)) / 2, 1 / 2, 0],
                        [-(3 ** (1 / 2)) / 2, -1 / 2, 0],
                        [0, 2 ** (1 / 2) / 2, 2 ** (1 / 2) / 2],
                    ]
                ),
                y_test=gs.array([0, 0, 0, 1, 1, 1]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(
                    PoincareBall(dim=2), kernel="distance"
                ),
                X_train=X_train_poincare,
                y_train=y_train_poincare,
                X_test=X_test_poincare,
                y_test=gs.array([0, 0, 0, 1, 1, 1]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(
                    Hyperboloid(dim=2), kernel="distance"
                ),
                X_train=X_train_hyperboloid,
                y_train=y_train_hyperboloid,
                X_test=_Hyperbolic.change_coordinates_system(
                    X_test_poincare,
                    from_coordinates_system="intrinsic",
                    to_coordinates_system="extrinsic",
                ),
                y_test=gs.array([0, 0, 0, 1, 1, 1]),
            ),
        ]
        return self.generate_tests(data)

    def predict_proba_test_data(self):
        X_train_1d, y_train_1d = self._get_1d_dataset()
        X_train_2d, y_train_2d = self._get_2d_dataset()
        data = [
            dict(
                estimator=KernelDensityEstimationClassifier(
                    Euclidean(dim=1),
                    kernel="uniform",
                ),
                X_train=X_train_1d,
                y_train=y_train_1d,
                X_test=gs.array([[0.9]]),
                expected=gs.array([[1 / 2, 1 / 2]]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(
                    Euclidean(dim=2),
                    kernel="uniform",
                ),
                X_train=X_train_2d,
                y_train=y_train_2d,
                X_test=gs.array([[0.9, 0.0]]),
                expected=gs.array([[1 / 2, 1 / 2]]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(
                    Euclidean(dim=2),
                    kernel="distance",
                ),
                X_train=X_train_2d,
                y_train=y_train_2d,
                X_test=gs.array([[1.0, 0.0]]),
                expected=gs.array([[1.0, 0.0]]),
            ),
            dict(
                estimator=KernelDensityEstimationClassifier(
                    Euclidean(dim=2),
                    kernel=triangular_radial_kernel,
                    bandwidth=2.0,
                ),
                X_train=X_train_2d,
                y_train=y_train_2d,
                X_test=gs.array([[1.0, 0.0]]),
                expected=gs.array([[3 / 4, 1 / 4]]),
            ),
        ]
        return self.generate_tests(data)
