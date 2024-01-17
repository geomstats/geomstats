import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData


class NumpyRaisesTestData(TestData):
    def raises_test_data(self):
        data = [
            dict(autodiff_func=gs.autodiff.value_and_grad),
            dict(autodiff_func=gs.autodiff.jacobian),
            dict(autodiff_func=gs.autodiff.jacobian_vec),
            dict(autodiff_func=gs.autodiff.hessian),
            dict(autodiff_func=gs.autodiff.hessian_vec),
            dict(autodiff_func=gs.autodiff.jacobian_and_hessian),
            dict(autodiff_func=gs.autodiff.value_jacobian_and_hessian),
        ]
        return self.generate_tests(data)


class NewAutodiffTestData(TestData):
    trials = 1

    def value_and_grad_test_data(self):
        data = [
            dict(
                func=lambda x: x**2,
                inputs=(gs.array(3.0),),
                point_ndims=0,
                expected_value=9.0,
                expected_grad=6.0,
            ),
            dict(
                func=lambda x: x[..., 0] ** 2 + x[..., 1],
                inputs=(gs.array([3.0, 2.0]),),
                expected_value=11.0,
                expected_grad=gs.array([6.0, 1.0]),
            ),
            dict(
                func=lambda x: x[..., 0, 0] ** 2 + x[..., 0, 1] + 2 * x[..., 1, 0],
                inputs=(gs.array([[4.0, 3.0], [2.0, 1.0]]),),
                point_ndims=2,
                expected_value=23.0,
                expected_grad=gs.array([[8.0, 1.0], [2.0, 0.0]]),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=-1),
                inputs=(
                    gs.array([1.0, 2.0]),
                    gs.array([2.0, 3.0]),
                ),
                argnums=0,
                expected_value=2.0,
                expected_grad=gs.array([-2.0, -2.0]),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=-1),
                inputs=(gs.array([1.0, 2.0]), gs.array([2.0, 3.0])),
                argnums=(0, 1),
                expected_value=2.0,
                expected_grad=(
                    gs.array([-2.0, -2.0]),
                    gs.array([2.0, 2.0]),
                ),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=(-2, -1)),
                inputs=(
                    gs.array([[1.0, 3.0], [2.0, 3.0]]),
                    gs.array([[2.0, 5.0], [0.0, 4.0]]),
                ),
                argnums=1,
                point_ndims=2,
                expected_value=10.0,
                expected_grad=gs.array([[2.0, 4.0], [-4.0, 2.0]]),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=(-2, -1)),
                inputs=(
                    gs.array([[1.0, 3.0], [2.0, 3.0]]),
                    gs.array([[2.0, 5.0], [0.0, 4.0]]),
                ),
                argnums=(0, 1),
                point_ndims=2,
                expected_value=10.0,
                expected_grad=(
                    gs.array([[-2.0, -4.0], [4.0, -2.0]]),
                    gs.array([[2.0, 4.0], [-4.0, 2.0]]),
                ),
            ),
            dict(
                func=lambda X, y: gs.sum(gs.matvec(X, y), axis=-1),
                inputs=(
                    gs.array([[1.0, 2.0], [3.0, 4.0]]),
                    gs.array(
                        [5.0, 6.0],
                    ),
                ),
                point_ndims=(2, 1),
                argnums=(0, 1),
                expected_value=56.0,
                expected_grad=(
                    gs.array([[5.0, 6.0], [5.0, 6.0]]),
                    gs.array([4.0, 6.0]),
                ),
            ),
            dict(
                func=lambda X, y: gs.sum(gs.matvec(X, y), axis=-1),
                inputs=(
                    gs.array([[1.0, 2.0], [3.0, 4.0]]),
                    gs.array(
                        [5.0, 6.0],
                    ),
                ),
                point_ndims=(2, 1),
                argnums=0,
                expected_value=56.0,
                expected_grad=gs.array([[5.0, 6.0], [5.0, 6.0]]),
            ),
        ]

        return self.generate_tests(data, marks=[pytest.mark.smoke])

    def value_and_grad_vec_test_data(self):
        data = [
            dict(
                func=lambda x: x[..., 0] ** 2 + x[..., 1],
                input_shape_x=(2,),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=-1),
                input_shape_x=(2,),
                input_shape_y=(2,),
                argnums=1,
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=-1),
                input_shape_x=(2,),
                input_shape_y=(2,),
                argnums=(0, 1),
            ),
            dict(
                func=lambda x, y: gs.sum(x * y**2, axis=-1),
                input_shape_x=(2,),
                input_shape_y=(2,),
                argnums=(0, 1),
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=(-2, -1)),
                input_shape_x=(2, 2),
                input_shape_y=(2, 2),
                argnums=(0, 1),
                point_ndims=2,
            ),
            dict(
                func=lambda x, y: gs.sum((x - y) ** 2, axis=(-2, -1)),
                input_shape_x=(2, 2),
                input_shape_y=(2, 2),
                argnums=0,
                point_ndims=2,
            ),
            dict(
                func=lambda X, y: gs.sum(gs.matvec(X, y), axis=-1),
                input_shape_x=(2, 2),
                input_shape_y=(2,),
                point_ndims=(2, 1),
                argnums=(0, 1),
            ),
            dict(
                func=lambda X, y: gs.sum(gs.matvec(X, y), axis=-1),
                input_shape_x=(2, 2),
                input_shape_y=(2,),
                point_ndims=(2, 1),
                argnums=0,
            ),
        ]

        data_with_reps = []
        for datum in data:
            for n_reps in self.N_VEC_REPS:
                new_datum = datum.copy()
                new_datum["n_reps"] = n_reps
                data_with_reps.append(new_datum)

        return self.generate_tests(data_with_reps)


class MetricDistGradTestData(TestData):
    trials = 1

    def value_and_grad_sdist_test_data(self):
        return self.generate_random_data()


class CustomGradientTestData(TestData):
    trials = 1

    def _func_1(self):
        def fake_grad_func(x):
            return 6 * x

        @gs.autodiff.custom_gradient(fake_grad_func)
        def func(x):
            return gs.sum(x**2, axis=-1)

        return func

    def _func_2(self, axis=-1):
        def fake_grad_func_x(x, y):
            return 6 * (x - y)

        def fake_grad_func_y(x, y):
            return 6 * (y - x)

        @gs.autodiff.custom_gradient(fake_grad_func_x, fake_grad_func_y)
        def func(x, y):
            return gs.sum((x - y) ** 2, axis=axis)

        return func

    def _func_3(self, axis=-1):
        def fake_grad_func_1(x):
            return 6 * x

        @gs.autodiff.custom_gradient(fake_grad_func_1)
        def func_1(x):
            return gs.sum(x, axis=axis) ** 2

        def func_2(x):
            return func_1(x) ** 3

        return func_2

    def value_and_grad_test_data(self):
        """Test value_and_grad with custom gradient.

        NB: fake gradients are used to ensure the code goes through
        the expected path.
        """
        data = [
            dict(
                func=self._func_1(),
                inputs=(gs.array([1.0, 3.0]),),
                expected_value=10.0,
                expected_grad=gs.array([6.0, 18.0]),
            ),
            dict(
                func=self._func_2(),
                inputs=(gs.array([1.0, 2.0]), gs.array([2.0, 3.0])),
                argnums=(0, 1),
                expected_value=2.0,
                expected_grad=(
                    gs.array([-6.0, -6.0]),
                    gs.array([6.0, 6.0]),
                ),
            ),
            dict(
                func=self._func_2(axis=(-2, -1)),
                inputs=(
                    gs.array([[1.0, 3.0], [2.0, 3.0]]),
                    gs.array([[2.0, 5.0], [0.0, 4.0]]),
                ),
                argnums=(0, 1),
                point_ndims=2,
                expected_value=10.0,
                expected_grad=(
                    gs.array([[-6.0, -12.0], [12.0, -6]]),
                    gs.array([[6.0, 12.0], [-12.0, 6.0]]),
                ),
            ),
            dict(
                func=self._func_3(),
                inputs=(gs.array([1.0, 2.0]),),
                expected_value=729.0,
                expected_grad=gs.array([1458.0, 2916.0]),
            ),
        ]

        return self.generate_tests(data)

    def value_and_grad_vec_test_data(self):
        data = [
            dict(
                func=self._func_1(),
                input_shape_x=(2,),
            ),
            dict(
                func=self._func_2(),
                input_shape_x=(2,),
                input_shape_y=(2,),
                argnums=(0, 1),
            ),
            dict(
                func=self._func_2(),
                input_shape_x=(2,),
                input_shape_y=(2,),
                argnums=0,
            ),
            dict(
                func=self._func_2(axis=(-2, -1)),
                input_shape_x=(2, 2),
                input_shape_y=(2, 2),
                argnums=0,
                point_ndims=2,
            ),
            dict(
                func=self._func_2(axis=(-2, -1)),
                input_shape_x=(2, 2),
                input_shape_y=(2, 2),
                argnums=(0, 1),
                point_ndims=2,
            ),
            dict(
                func=self._func_3(),
                input_shape_x=(2,),
            ),
            dict(
                func=self._func_3(axis=(-2, -1)),
                input_shape_x=(2, 2),
                point_ndim=2,
            ),
        ]

        data_with_reps = []
        for datum in data:
            for n_reps in self.N_VEC_REPS:
                new_datum = datum.copy()
                new_datum["n_reps"] = n_reps
                data_with_reps.append(new_datum)

        return self.generate_tests(data_with_reps)
