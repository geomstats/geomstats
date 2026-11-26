import random

from polpo.testing.data import CaseData

import geomstats.backend as gs


def _sum_squares(x):
    return gs.sum(x**2)


def _jacobian_sum_squares(x):
    return 2 * x


def _sum_squares_val_and_grad(x):
    return _sum_squares(x), _jacobian_sum_squares(x)


def _hessian_sum_squares(x):
    return 2 * gs.eye(x.shape[0])


class OptimizationSmokeTestData(CaseData):
    def minimize_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return data


class OptimizationJacSmokeTestData(CaseData):
    def minimize_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_sum_squares,
                fun_jac=_jacobian_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            ),
            dict(
                fun=_sum_squares_val_and_grad,
                fun_jac=True,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            ),
        ]

        return data


class OptimizationHessSmokeTestData(CaseData):
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

        return data


class RootFindingSmokeTestData(CaseData):
    def root_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_jacobian_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return data


class RootFindingJacSmokeTestData(CaseData):
    def root_test_data(self):
        dim = random.randint(1, 10)

        data = [
            dict(
                fun=_jacobian_sum_squares,
                fun_jac=_hessian_sum_squares,
                x0=gs.random.uniform(size=dim),
                expected=gs.zeros(dim),
            )
        ]

        return data
