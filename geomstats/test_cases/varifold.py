import math
import random

import pytest

import geomstats.backend as gs
from geomstats._mesh import Surface
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.mixins import DistTestCaseMixins

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Genred
else:
    from pykeops.numpy import Genred


class VarifoldRandomDataGenerator:
    def __init__(self, point, amplitude=2.0, max_vertices_removal=2):
        self.point = point
        self.max_vertices_removal = max_vertices_removal
        self.amplitude = amplitude

    def random_point(self, n_points=1):
        if n_points != 1:
            raise ValueError("Can only generate one point at a time.")

        n_remove = random.randint(0, self.max_vertices_removal)
        new_vertices = self.point.vertices
        new_faces = self.point.faces
        for _ in range(n_remove):
            new_vertices, new_faces = self._remove_last_vertex(new_vertices, new_faces)

        new_vertices += self._get_deformation(new_vertices.shape[0])
        return Surface(new_vertices, new_faces)

    def _remove_last_vertex(self, vertices, faces):
        index = vertices.shape[0] - 1
        indices = gs.array([index not in face for face in faces])
        return vertices[:-1], faces[indices]

    def _get_deformation(self, n_vertices):
        rand_shape = (n_vertices, 3)
        return gs.reshape(
            gs.random.rand(math.prod(rand_shape)),
            rand_shape,
        )


class KernelTestCase(TestCase):
    def test_against_other(self, point_a, point_b, atol):
        res = self.kernel(point_a, point_b)
        res_ = self.kernel_(point_a, point_b)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_against_other_random(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        self.test_against_other(point_a, point_b, atol)


class VarifoldMetricTestCase(DistTestCaseMixins, TestCase):
    pass


def GaussianKernel_(sigma):
    expr = Genred(
        "Exp(-SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def CauchyKernel_(sigma):
    expr = Genred(
        "IntCst(1)/(IntCst(1)+SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def LinearKernel_():
    expr = Genred(
        "(u|v)",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def BinetKernel_():
    expr = Genred(
        "Square((u|v))",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def OrientedGaussianKernel_(sigma=1.0):
    expr = Genred(
        "Exp(IntCst(2)*b*((u|v)-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def UnorientedGaussianKernel_(sigma=1.0):
    expr = Genred(
        "Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval
