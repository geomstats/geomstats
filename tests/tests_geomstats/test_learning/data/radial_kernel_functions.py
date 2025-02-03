import geomstats.backend as gs
from geomstats.learning.radial_kernel_functions import (
    biweight_radial_kernel,
    bump_radial_kernel,
    cosine_radial_kernel,
    gaussian_radial_kernel,
    inverse_multiquadric_radial_kernel,
    inverse_quadratic_radial_kernel,
    laplacian_radial_kernel,
    logistic_radial_kernel,
    parabolic_radial_kernel,
    sigmoid_radial_kernel,
    triangular_radial_kernel,
    tricube_radial_kernel,
    triweight_radial_kernel,
    uniform_radial_kernel,
)
from geomstats.test.data import TestData


class RadialKernelFunctionsTestData(TestData):
    def kernel_test_data(self):
        data = [
            dict(
                kernel=uniform_radial_kernel,
                distance=gs.array([[1 / 2], [2.0]]),
                bandwidth=1.0,
                expected=gs.array([[1.0], [0.0]]),
            ),
            dict(
                kernel=uniform_radial_kernel,
                distance=gs.array([[1 / 2], [2.0]]),
                bandwidth=1 / 4,
                expected=gs.array([[0.0], [0.0]]),
            ),
            dict(
                kernel=triangular_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[1 / 2], [0.0]]),
            ),
            dict(
                kernel=parabolic_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array(
                    [[3 / 4], [0.0]],
                ),
            ),
            dict(
                kernel=biweight_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[9 / 16], [0.0]]),
            ),
            dict(
                kernel=triweight_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[(3 / 4) ** 3], [0.0]]),
            ),
            dict(
                kernel=tricube_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[(7 / 8) ** 3], [0.0]]),
            ),
            dict(
                kernel=gaussian_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[gs.exp(-1 / 8)], [gs.exp(-1 / 2)]]),
            ),
            dict(
                kernel=cosine_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array([[2 ** (1 / 2) / 2], [0.0]]),
            ),
            dict(
                kernel=logistic_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array(
                    [
                        [1 / (gs.exp(1 / 2) + 2 + gs.exp(-1 / 2))],
                        [1 / (gs.exp(1.0) + 2 + gs.exp(-1.0))],
                    ]
                ),
            ),
            dict(
                kernel=sigmoid_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2.0,
                expected=gs.array(
                    [
                        [1 / (gs.exp(1 / 2) + gs.exp(-1 / 2))],
                        [1 / (gs.exp(1.0) + gs.exp(-1.0))],
                    ],
                ),
            ),
            dict(
                kernel=bump_radial_kernel,
                distance=gs.array([[1 / 2], [2.0]]),
                bandwidth=1.0,
                expected=gs.array([[gs.exp(-1 / (3 / 4))], [0.0]]),
            ),
            dict(
                kernel=inverse_quadratic_radial_kernel,
                distance=gs.array([[1.0], [2.0]]),
                bandwidth=2.0,
                expected=gs.array([[4 / 5], [1 / 2]]),
            ),
            dict(
                kernel=inverse_multiquadric_radial_kernel,
                distance=gs.array([[1.0], [2]]),
                bandwidth=2,
                expected=gs.array([[2 / 5 ** (1 / 2)], [1 / 2 ** (1 / 2)]]),
            ),
            dict(
                kernel=laplacian_radial_kernel,
                distance=gs.array([[1.0], [2.0]]),
                bandwidth=2,
                expected=gs.array([[gs.exp(-1 / 2)], [gs.exp(-1.0)]]),
            ),
        ]
        return self.generate_tests(data)
