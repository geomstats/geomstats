import random

import geomstats.backend as gs
from geomstats.test.data import TestData


def _sum_squares(x):
    return gs.sum(x**2)


def _jacobian_sum_squares(x):
    return 2 * x


def _hessian_sum_squares(x):
    return 2 * gs.eye(x.shape[0])


class OptimizationSmokeTestData(TestData):
    trials = 1

    def minimize_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return self.generate_tests(data)


class OptimizationJacSmokeTestData(TestData):
    trials = 1

    def minimize_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_sum_squares,
                fun_jac=_jacobian_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return self.generate_tests(data)


class OptimizationHessSmokeTestData(TestData):
    trials = 1

    def minimize_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_sum_squares,
                fun_jac=_jacobian_sum_squares,
                fun_hess=_hessian_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return self.generate_tests(data)
