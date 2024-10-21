import pytest

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats._mesh import Surface
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.varifold import (
    BinetKernel_,
    CauchyKernel_,
    GaussianKernel_,
    KernelTestCase,
    LinearKernel_,
    OrientedGaussianKernel_,
    UnorientedGaussianKernel_,
    VarifoldMetricTestCase,
    VarifoldRandomDataGenerator,
)
from geomstats.varifold import (
    BinetKernel,
    CauchyKernel,
    GaussianKernel,
    LinearKernel,
    RestrictedGaussianKernel,
    VarifoldMetric,
)

from .data.varifold import KernelTestData, VarifoldMetricTestData


class _DummySpace:
    def __init__(self, metric):
        self.metric = metric


class RandomDataGenerator_(RandomDataGenerator):
    def random_point(self, n_points=1):
        if n_points == 1:
            return gs.expand_dims(self.space.random_point(n_points), axis=0)

        return self.space.random_point(n_points)


def _wrap_kernel(kernel):
    """Wrap a kernel function.

    Because a function as a class attribute becomes a bound method (see
    https://stackoverflow.com/questions/35321744/python-function-as-class-attribute-becomes-a-bound-method #noqa
    ).
    """

    def func(self, point_a, point_b):
        return kernel(point_a, point_b)

    return func


@pytest.fixture(
    scope="class",
    params=[
        (GaussianKernel(0.5).sum_reduction(axis=1), _wrap_kernel(GaussianKernel_(0.5))),
        (CauchyKernel(0.7).sum_reduction(axis=1), _wrap_kernel(CauchyKernel_(0.7))),
    ],
)
def position_kernels(request):
    request.cls.kernel, request.cls.kernel_ = request.param


@pytest.mark.usefixtures("position_kernels")
class TestPositionKernel(KernelTestCase, metaclass=DataBasedParametrizer):
    testing_data = KernelTestData()
    data_generator = RandomDataGenerator_(Euclidean(dim=3, equip=False))


@pytest.fixture(
    scope="class",
    params=[
        (LinearKernel().sum_reduction(axis=1), _wrap_kernel(LinearKernel_())),
        (BinetKernel().sum_reduction(axis=1), _wrap_kernel(BinetKernel_())),
        (
            RestrictedGaussianKernel(0.6, oriented=False).sum_reduction(axis=1),
            _wrap_kernel(UnorientedGaussianKernel_(0.6)),
        ),
        (
            RestrictedGaussianKernel(0.3, oriented=True).sum_reduction(axis=1),
            _wrap_kernel(OrientedGaussianKernel_(0.3)),
        ),
    ],
)
def tangent_kernels(request):
    request.cls.kernel, request.cls.kernel_ = request.param


@pytest.mark.usefixtures("tangent_kernels")
class TestTangentKernel(KernelTestCase, metaclass=DataBasedParametrizer):
    testing_data = KernelTestData()
    data_generator = RandomDataGenerator_(Hypersphere(dim=2, equip=False))


class TestVarifoldMetric(VarifoldMetricTestCase, metaclass=DataBasedParametrizer):
    space = _DummySpace(VarifoldMetric())

    _vertices, _faces = data_utils.load_cube()
    _vertices = gs.array(_vertices, dtype=gs.float64)
    _faces = gs.array(_faces)
    testing_data = VarifoldMetricTestData()

    data_generator = VarifoldRandomDataGenerator(Surface(_vertices, _faces))
