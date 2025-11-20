import pytest

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class NumpyRaisesTestCase(TestCase):
    def test_raises(self, autodiff_func):
        with pytest.raises(AutodiffNotImplementedError):
            autodiff_func(self.dummy_func)


class AutodiffTestCase(TestCase):
    def test_value_and_grad(
        self,
        func,
        inputs,
        expected_value,
        expected_grad,
        atol,
        argnums=0,
        point_ndims=1,
    ):
        value_and_grad = gs.autodiff.value_and_grad(
            func, argnums=argnums, point_ndims=point_ndims
        )
        value, grad = value_and_grad(*inputs)
        self.assertAllClose(value, expected_value, atol=atol)

        if isinstance(argnums, int):
            self.assertAllClose(grad, expected_grad, atol=atol)
        else:
            self.assertEqual(len(expected_grad), len(argnums))
            for grad_, expected_grad_ in zip(grad, expected_grad):
                self.assertAllClose(grad_, expected_grad_, atol=atol)

    def _test_value_and_grad(
        self,
        func,
        x,
        expected_value,
        expected_grad_x,
        atol,
        argnums=0,
        point_ndims=1,
        y=None,
        expected_grad_y=None,
    ):
        """Auxiliar func for vectorization test."""
        inputs = (x,) if y is None else (x, y)
        if isinstance(argnums, int):
            expected_grad = expected_grad_x if argnums == 0 else expected_grad_y
        elif len(argnums) == 1:
            argnum = argnums[0]
            expected_grad = (expected_grad_x if argnum == 0 else expected_grad_y,)
        else:
            expected_grad = (expected_grad_x, expected_grad_y)

        return self.test_value_and_grad(
            func,
            inputs,
            expected_value,
            expected_grad,
            atol,
            argnums=argnums,
            point_ndims=point_ndims,
        )

    @pytest.mark.vec
    def test_value_and_grad_vec(
        self,
        n_reps,
        func,
        input_shape_x,
        atol,
        argnums=0,
        point_ndims=1,
        input_shape_y=None,
    ):
        has_y = input_shape_y is not None

        inputs = [gs.random.rand(*input_shape_x)]
        if has_y:
            inputs.append(gs.random.rand(*input_shape_y))

        expected_value, expected_grad = gs.autodiff.value_and_grad(
            func,
            argnums=argnums,
            point_ndims=point_ndims,
        )(*inputs)

        argnums_ = (argnums,) if isinstance(argnums, int) else argnums
        expected_grad = (
            (expected_grad,) if not isinstance(expected_grad, tuple) else expected_grad
        )

        arg_names = ["x"]
        if has_y:
            arg_names.append("y")
        expected_names = ["expected_value"]

        expected_grad_ = []
        k = 0
        map_index_varname = {0: "x", 1: "y"}
        for index in range(2):
            if index not in argnums_:
                expected_grad_.append(None)
                continue

            expected_names.append(f"expected_grad_{map_index_varname[index]}")
            expected_grad_.append(expected_grad[k])
            k += 1

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    func=func,
                    x=inputs[0],
                    y=inputs[1] if input_shape_y is not None else None,
                    expected_value=expected_value,
                    expected_grad_x=expected_grad_[0],
                    expected_grad_y=expected_grad_[1],
                    argnums=argnums,
                    point_ndims=point_ndims,
                    atol=atol,
                )
            ],
            arg_names=arg_names,
            expected_name=expected_names,
            n_reps=n_reps,
        )

        self._test_vectorization(vec_data, test_fnc_name="_test_value_and_grad")


class MetricDistGradTestCase(TestCase):
    @pytest.mark.random
    def test_value_and_grad_sdist(self, n_points, atol):
        point_a = self.space.random_point(n_points)
        point_b = self.space.random_point(n_points)

        value_and_grad = gs.autodiff.value_and_grad(
            lambda v: self.space.metric.squared_dist(v, point_b),
            point_ndims=(self.space.point_ndim,),
        )
        sdist, sdist_grad = value_and_grad(point_a)

        expected_sdist = self.space.metric.squared_dist(point_a, point_b)

        tangent_vec = self.space.metric.log(point_b, point_a)
        expected_sdist_grad = -2 * tangent_vec

        self.assertAllClose(sdist, expected_sdist, atol=atol)
        self.assertAllClose(sdist_grad, expected_sdist_grad, atol=atol)
